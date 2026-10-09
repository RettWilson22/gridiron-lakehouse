"""Data-quality expectations for silver and gold datasets.

The same rule dictionaries drive Lakeflow expectations on Databricks
(``@dp.expect_all_or_drop`` / ``@dp.expect_all``) and the local helpers below, so the rules
are written once as SQL boolean expressions. Every rule is null-safe (it evaluates to TRUE
or FALSE, never NULL) so local and pipeline behaviour cannot diverge.

``DROP`` rules remove rows that cannot be used at all (missing keys). ``WARN`` rules keep
the row and count violations in the pipeline event log.

Player names, team codes and positions come from third-party feeds and end up on screen,
so silver also checks their shape: a team is one of the 32 nflverse codes, a position (in
datasets limited to fantasy positions) is QB, RB, WR or TE, and a name is at most 64
letters (accented ones included), spaces, periods, apostrophes and hyphens. These are
WARN rules: real names do break the pattern ("Kenneth Murray, Jr." in the box scores), so
violations are counted, not dropped. The app escapes every name it renders either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from gridiron.config import POSITIONS
from gridiron.player_search import TEAMS

Rules = dict[str, str]


def _one_of(column: str, values: Iterable[str]) -> str:
    listed = ", ".join(f"'{value}'" for value in sorted(values))
    return f"coalesce({column} IN ({listed}), FALSE)"


def known_teams(*columns: str) -> str:
    """Every column holds one of the 32 nflverse team codes."""
    return " AND ".join(_one_of(column, TEAMS) for column in columns)


FANTASY_POSITION: Final = _one_of("position", POSITIONS)
# Java regex: letters and combining marks (accents), space, period, apostrophe (\x27),
# hyphen; 1 to 64 characters.
PLAIN_PLAYER_NAME: Final = r"coalesce(player_name RLIKE '^[\\p{L}\\p{M} .\\x27-]{1,64}$', FALSE)"

DROP: Final[dict[str, Rules]] = {
    "plays": {
        "has_game_id": "game_id IS NOT NULL",
        "has_play_id": "play_id IS NOT NULL",
        "has_season_week": "season IS NOT NULL AND week IS NOT NULL",
        "valid_season_type": "season_type IS NOT NULL AND season_type IN ('REG', 'POST')",
    },
    "player_stats": {
        "has_player_id": "player_id IS NOT NULL",
        "has_game_id": "game_id IS NOT NULL",
        "has_season_week": "season IS NOT NULL AND week IS NOT NULL",
    },
    "snap_counts": {
        "has_game_id": "game_id IS NOT NULL",
        "has_pfr_player_id": "pfr_player_id IS NOT NULL",
    },
    "rosters": {
        "has_player_id": "player_id IS NOT NULL",
        "has_season_week": "season IS NOT NULL AND week IS NOT NULL",
    },
    "injuries": {
        "has_player_id": "player_id IS NOT NULL",
        "has_season_week": "season IS NOT NULL AND week IS NOT NULL",
    },
    "depth_charts": {
        "has_player_id": "player_id IS NOT NULL",
        "has_team": "team IS NOT NULL",
        "has_rank": "depth_rank IS NOT NULL AND depth_rank >= 1",
        "dated": "week IS NOT NULL OR published_at IS NOT NULL",
    },
    "ff_opportunity": {
        # ffopportunity also publishes team-level rows without a player id.
        "has_player_id": "player_id IS NOT NULL",
        "has_game_id": "game_id IS NOT NULL",
    },
    "schedules": {
        "has_game_id": "game_id IS NOT NULL",
        "has_teams": "home_team IS NOT NULL AND away_team IS NOT NULL",
        "has_kickoff": "kickoff_at IS NOT NULL",
    },
    "players": {"has_player_id": "player_id IS NOT NULL"},
    "player_ids": {"has_fantasypros_id": "fantasypros_id IS NOT NULL"},
    "ecr": {
        "has_fantasypros_id": "fantasypros_id IS NOT NULL",
        "mapped_to_week": "week IS NOT NULL",
        "has_ecr": "ecr IS NOT NULL",
    },
}

WARN: Final[dict[str, Rules]] = {
    "plays": {
        "yardline_in_range": "yardline_100 IS NULL OR yardline_100 BETWEEN 1 AND 99",
        "down_in_range": "down IS NULL OR down BETWEEN 1 AND 4",
    },
    "player_stats": {
        # nflverse PPR = standard + 1 per reception; anything else means a scoring change.
        "ppr_is_standard_plus_receptions": (
            "coalesce(abs(fantasy_points_ppr - fantasy_points - receptions) < 0.01, FALSE)"
        ),
        "known_teams": known_teams("team", "opponent_team"),
        "plain_player_name": PLAIN_PLAYER_NAME,
    },
    "players": {"plain_player_name": PLAIN_PLAYER_NAME},
    "snap_counts": {
        "mapped_to_gsis_id": "player_id IS NOT NULL",
        "offense_pct_in_range": "offense_pct IS NULL OR offense_pct BETWEEN 0 AND 1",
        "known_teams": known_teams("team", "opponent"),
    },
    "rosters": {"known_team": known_teams("team")},
    "injuries": {
        "known_report_status": (
            "report_status IS NULL OR report_status IN "
            "('Out', 'Doubtful', 'Questionable', 'Probable', 'Note')"
        ),
        "known_team": known_teams("team"),
    },
    "depth_charts": {
        "known_team": known_teams("team"),
        "fantasy_position": FANTASY_POSITION,
        "plain_player_name": PLAIN_PLAYER_NAME,
    },
    "schedules": {
        "completed_games_have_lines": (
            "home_score IS NULL OR (spread_line IS NOT NULL AND total_line IS NOT NULL)"
        ),
        "known_teams": known_teams("home_team", "away_team"),
    },
    # FantasyPros uses its own team codes (JAC, LAR, FA), so ECR teams are not checked.
    "ecr": {
        "mapped_to_gsis_id": "player_id IS NOT NULL",
        "fantasy_position": FANTASY_POSITION,
        "plain_player_name": PLAIN_PLAYER_NAME,
    },
    "player_week": {
        "shares_are_fractions": (
            "coalesce(target_share BETWEEN 0 AND 1, TRUE) "
            "AND coalesce(carry_share BETWEEN 0 AND 1, TRUE) "
            "AND coalesce(snap_share BETWEEN 0 AND 1, TRUE)"
        ),
        "ppr_at_least_standard": "fantasy_points_ppr >= fantasy_points_std",
        "has_snap_count": "snap_share IS NOT NULL",
        "has_opponent": "opponent IS NOT NULL",
    },
    "team_week": {
        "final_games_have_volume": "NOT is_final OR offensive_plays IS NOT NULL",
    },
}


def combined(rules: Rules) -> str:
    return " AND ".join(f"({expr})" for expr in rules.values())


def apply_drop_rules(df: DataFrame, rules: Rules) -> DataFrame:
    """Local equivalent of ``expect_all_or_drop``."""
    if not rules:
        return df
    return df.where(F.coalesce(F.expr(combined(rules)), F.lit(False)))


def count_failures(df: DataFrame, rules: Rules) -> dict[str, int]:
    """Local equivalent of the event-log metrics for ``expect_all``."""
    if not rules:
        return {}
    aggs = [
        F.sum(F.when(F.coalesce(F.expr(expr), F.lit(False)), 0).otherwise(1)).alias(name)
        for name, expr in rules.items()
    ]
    row = df.agg(*aggs).first()
    assert row is not None
    return {name: int(row[name] or 0) for name in rules}
