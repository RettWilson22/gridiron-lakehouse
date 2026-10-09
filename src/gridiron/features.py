"""Model features for player-weeks, built only from information available before kickoff.

Everything here is plain pandas so it runs identically in the Databricks job and locally.

Leakage rules, enforced by construction and by tests (``tests/test_features.py``):

* Player, team and opponent history is attached with ``merge_asof`` on the time key
  ``season * 100 + week`` with ``allow_exact_matches=False``: a row for week ``w`` only
  sees games from earlier weeks, never week ``w`` itself or anything later.
* Week-``w`` inputs are limited to what is published before the game: the schedule and
  betting lines, the depth chart as published for the week (``depth_chart_week`` already
  applies the before-kickoff cut for snapshot charts), that week's final injury report, and
  the roster as of that week (position and years of experience; never a later listing).
* The candidate pool is decided before kickoff too: players on the week's depth chart plus
  players who played for the team in its previous two games, minus anyone ruled Out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

import numpy as np
import pandas as pd

from gridiron.config import POSITIONS
from gridiron.scoring import STATS

TIME = "t"
KEY = ["season", "week", "player_id"]

# Per-game player stats smoothed with an exponentially weighted mean over previous games.
EWM_HALFLIFE_GAMES: Final = 3.0
EWM_STATS: Final = (
    "fantasy_points_ppr",
    "expected_ppr",
    "offense_snaps",
    "snap_share",
    "target_share",
    "carry_share",
    "air_yards_share",
    "red_zone_share",
    "targets",
    "carries",
    "receptions",
    "red_zone_targets",
    "red_zone_carries",
    "goal_line_carries",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
)
TEAM_EWM_HALFLIFE_GAMES: Final = 4.0
TEAM_STATS: Final = ("points_for", "offensive_plays", "pass_rate", "red_zone_plays")
DEFENSE_WINDOW_GAMES: Final = 6

# Stat-line components the model projects: every stat ``gridiron.scoring`` scores except
# special teams touchdowns, which are too rare to project. Fantasy points under any scoring
# are computed from them. Names match ``player_week`` columns.
NOT_PROJECTED: Final = ("special_teams_tds",)
COMPONENTS: Final = tuple(stat for stat in STATS if stat not in NOT_PROJECTED)

INJURY_LEVELS: Final = {"Questionable": 1, "Doubtful": 2, "Out": 3}
ROSTER_POSITIONS: Final = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
PRACTICE_LEVELS: Final = {
    "Limited Participation in Practice": 1,
    "Did Not Participate In Practice": 2,
}

FEATURES: Final = (
    "week",
    "depth_rank",
    "on_depth_chart",
    "injury_level",
    "practice_level",
    "is_home",
    "implied_points",
    "team_spread",
    "total_line",
    "games_played",
    "season_games",
    "games_missed_season",
    "missed_last_game",
    "season_ppr_mean",
    "prev_season_ppr_mean",
    "ppr_last1",
    "ppr_last3",
    *(f"ewm_{s}" for s in EWM_STATS),
    *(f"team_ewm_{s}" for s in TEAM_STATS),
    "opp_ppr_allowed_l6",
    "opp_matchup_index",
    "rookie",
    "years_exp",
    "age",
)


def time_key(frame: pd.DataFrame) -> pd.Series:
    return (frame["season"].astype("int64") * 100 + frame["week"].astype("int64")).rename(TIME)


def _asof(left: pd.DataFrame, right: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Attach the latest ``right`` row strictly before each ``left`` row's time key."""
    left = left.assign(**{TIME: time_key(left)}).sort_values(TIME, kind="stable")
    right = right.sort_values(TIME, kind="stable")
    merged = pd.merge_asof(
        left, right, on=TIME, by=by, allow_exact_matches=False, direction="backward"
    )
    return merged.drop(columns=TIME)


# ---------------------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------------------


def team_game_index(team_week: pd.DataFrame) -> pd.DataFrame:
    """Order of each team's regular-season games within a season (0 = first game)."""
    games = team_week[["season", "week", "team", "game_id"]].sort_values(["team", "season", "week"])
    games["game_index"] = games.groupby(["team", "season"]).cumcount()
    return games


def injury_report(injuries: pd.DataFrame) -> pd.DataFrame:
    """Final regular-season injury report status per player-week (most severe if repeated)."""
    report = injuries[injuries["game_type"] == "REG"].copy()
    report["injury_level"] = report["report_status"].map(INJURY_LEVELS).fillna(0).astype("int64")
    report["practice_level"] = (
        report["practice_status"].map(PRACTICE_LEVELS).fillna(0).astype("int64")
    )
    report = report.sort_values("injury_level", ascending=False, kind="stable")
    report = report.drop_duplicates(KEY)
    return report[
        [*KEY, "report_status", "report_primary_injury", "injury_level", "practice_level"]
    ]


def _recent_players(player_week: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """Players who played for each team in its previous two games, with the position they
    were listed at in the more recent of those games."""
    appeared = player_week[["season", "week", "team", "player_id", "position"]].merge(
        order[["season", "week", "team", "game_index"]], on=["season", "week", "team"]
    )
    recent = [
        appeared.assign(game_index=appeared["game_index"] + lag, _lag=lag)[
            ["season", "team", "game_index", "player_id", "position", "_lag"]
        ]
        for lag in (1, 2)
    ]
    return (
        pd.concat(recent)
        .sort_values("_lag", kind="stable")
        .drop_duplicates(["season", "team", "game_index", "player_id"])
        .merge(order[["season", "week", "team", "game_index"]], on=["season", "team", "game_index"])
        .rename(columns={"position": "recent_position"})[
            ["season", "week", "team", "player_id", "recent_position"]
        ]
    )


def roster_positions(rosters: pd.DataFrame) -> pd.DataFrame:
    """Each player's listed position per regular-season week (fullbacks count as RB)."""
    listed = rosters[rosters["game_type"] == "REG"][[*KEY, "position"]].dropna()
    listed = listed.assign(roster_position=listed["position"].map(ROSTER_POSITIONS))
    return listed.dropna(subset=["roster_position"]).drop_duplicates(KEY)[[*KEY, "roster_position"]]


def build_candidates(
    tables: Mapping[str, pd.DataFrame], *, weeks: Iterable[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """Players to project for each team-game, decided with pre-kickoff information only.

    Uses ``depth_chart_week``, ``player_week``, ``team_week``, ``injuries``, ``players``
    and ``rosters``. ``weeks`` limits the output to the given ``(season, week)`` pairs.
    A player's position is the one on that week's roster, else on that week's depth chart,
    else the one he played in the game that put him in the pool; never a later listing.
    """
    team_week = tables["team_week"]
    games = team_week[
        [
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
            "is_home",
            "kickoff_at",
            "implied_points",
            "team_spread",
            "total_line",
            "is_final",
        ]
    ]
    if weeks is not None:
        wanted = pd.DataFrame(list(weeks), columns=["season", "week"])
        games = games.merge(wanted, on=["season", "week"])

    charted = tables["depth_chart_week"][
        ["season", "week", "team", "player_id", "player_name", "depth_rank", "position"]
    ].rename(columns={"position": "chart_position", "player_name": "chart_name"})
    recent = _recent_players(tables["player_week"], team_game_index(team_week))

    pool = pd.concat(
        [
            charted[["season", "week", "team", "player_id"]],
            recent[["season", "week", "team", "player_id"]],
        ]
    ).drop_duplicates()
    pool = pool.merge(games, on=["season", "week", "team"])
    pool = pool.merge(charted, on=["season", "week", "team", "player_id"], how="left")
    pool = pool.merge(recent, on=["season", "week", "team", "player_id"], how="left")
    pool["on_depth_chart"] = pool["depth_rank"].notna().astype("int64")

    pool = pool.merge(roster_positions(tables["rosters"]), on=KEY, how="left")
    pool["position"] = (
        pool["roster_position"].fillna(pool["chart_position"]).fillna(pool["recent_position"])
    )
    identity = tables["players"][["player_id", "player_name", "birth_date", "rookie_season"]]
    pool = pool.merge(identity, on="player_id", how="left")
    pool["player_name"] = pool["player_name"].fillna(pool["chart_name"])
    pool = pool[pool["position"].isin(POSITIONS)]

    pool = pool.merge(injury_report(tables["injuries"]), on=KEY, how="left")
    pool[["injury_level", "practice_level"]] = (
        pool[["injury_level", "practice_level"]].fillna(0).astype("int64")
    )
    pool = pool[pool["injury_level"] < INJURY_LEVELS["Out"]]

    # A player traded during the week can appear for two teams: keep the depth chart entry
    # with the better rank (then the team listed first alphabetically, for determinism).
    pool = pool.sort_values(["depth_rank", "team"], na_position="last", kind="stable")
    pool = pool.drop_duplicates(KEY)
    columns = [
        *KEY,
        "game_id",
        "player_name",
        "position",
        "team",
        "opponent",
        "is_home",
        "kickoff_at",
        "is_final",
        "implied_points",
        "team_spread",
        "total_line",
        "depth_rank",
        "on_depth_chart",
        "report_status",
        "report_primary_injury",
        "injury_level",
        "practice_level",
        "birth_date",
        "rookie_season",
    ]
    return pool[columns].sort_values(KEY).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# History states
# ---------------------------------------------------------------------------------------


def player_states(player_week: pd.DataFrame) -> pd.DataFrame:
    """Each player's form after every game played (to be attached to later weeks)."""
    games = player_week.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = games.groupby("player_id", sort=False)
    states = pd.DataFrame(
        {
            "player_id": games["player_id"],
            TIME: time_key(games),
            "state_season": games["season"],
            "state_week": games["week"],
            "state_team": games["team"],
            "games_played": grouped.cumcount() + 1,
            "ppr_last1": games["fantasy_points_ppr"],
        }
    )
    states["ppr_last3"] = (
        grouped["fantasy_points_ppr"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ewm = (
        grouped[list(EWM_STATS)]
        .ewm(halflife=EWM_HALFLIFE_GAMES)
        .mean()
        .reset_index(level=0, drop=True)
    )
    states = states.join(ewm.add_prefix("ewm_"))
    season = games.groupby(["player_id", "season"], sort=False)
    states["season_games"] = season.cumcount() + 1
    states["season_ppr_mean"] = (
        season["fantasy_points_ppr"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )
    return states


def season_means(player_week: pd.DataFrame) -> pd.DataFrame:
    """Each player's full-season PPR average, keyed to the following season."""
    means = (
        player_week.groupby(["player_id", "season"])
        .agg(prev_season_ppr_mean=("fantasy_points_ppr", "mean"))
        .reset_index()
    )
    means["season"] += 1
    return means


def team_states(team_week: pd.DataFrame) -> pd.DataFrame:
    played = team_week[team_week["is_final"]].sort_values(["team", "season", "week"]).copy()
    played["pass_rate"] = played["dropbacks"] / played["offensive_plays"].where(
        played["offensive_plays"] > 0
    )
    ewm = (
        played.groupby("team", sort=False)[list(TEAM_STATS)]
        .ewm(halflife=TEAM_EWM_HALFLIFE_GAMES)
        .mean()
        .reset_index(level=0, drop=True)
    )
    states = pd.DataFrame({"team": played["team"], TIME: time_key(played)})
    return states.join(ewm.add_prefix("team_ewm_"))


def defense_states(defense_vs_position: pd.DataFrame) -> pd.DataFrame:
    """PPR allowed per position over each defense's last six games, after every game.

    Matches ``defense_vs_position.ppr_allowed_l6`` (gold), which is the same average as of
    the start of a week rather than after it; a test checks they agree.
    """
    games = defense_vs_position.sort_values(["team", "position", "season", "week"])
    rolled = (
        games.groupby(["team", "position"], sort=False)["ppr_allowed"]
        .rolling(DEFENSE_WINDOW_GAMES, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return pd.DataFrame(
        {
            "opponent": games["team"],
            "position": games["position"],
            TIME: time_key(games),
            "opp_ppr_allowed_l6": rolled,
        }
    )


def matchup_table(defense_vs_position: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Every defense's recent PPR allowed per position, as of the start of each week.

    ``opp_matchup_index`` divides by the league average for the position that week (above
    1 = a softer matchup than average); ``opp_matchup_rank`` 1 = allows the most points.
    """
    defense = defense_states(defense_vs_position)
    defenses = defense[["opponent", "position"]].drop_duplicates()
    grid = weeks[["season", "week"]].drop_duplicates().merge(defenses, how="cross")
    table = _asof(grid, defense, by=["opponent", "position"])
    allowed = table.groupby(["season", "week", "position"])["opp_ppr_allowed_l6"]
    table["opp_matchup_index"] = table["opp_ppr_allowed_l6"] / allowed.transform("mean")
    table["opp_matchup_rank"] = allowed.rank(ascending=False, method="min")
    return table


def roster_experience(rosters: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Years of experience for each ``rows`` player-week, from his latest roster row in the
    same season at or before that week (never a later week's listing)."""
    listed = rosters[["player_id", "season", "week", "years_exp"]].dropna()
    listed = listed.assign(**{TIME: time_key(listed)}).sort_values(TIME, kind="stable")
    wanted = rows[KEY].drop_duplicates()
    wanted = wanted.assign(**{TIME: time_key(wanted)}).sort_values(TIME, kind="stable")
    merged = pd.merge_asof(
        wanted,
        listed.rename(columns={"season": "roster_season", "week": "roster_week"}),
        on=TIME,
        by="player_id",
        direction="backward",
    )
    same_season = merged["roster_season"] == merged["season"]
    merged["years_exp"] = merged["years_exp"].where(same_season)
    return merged[[*KEY, "years_exp"]]


def build_features(candidates: pd.DataFrame, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Candidates plus every model feature (see ``FEATURES``).

    Uses ``player_week``, ``team_week``, ``defense_vs_position`` and ``rosters``.
    """
    player_week = tables["player_week"]
    team_week = tables["team_week"]
    frame = _asof(candidates, player_states(player_week), by=["player_id"])
    same_season = frame["state_season"] == frame["season"]
    frame["season_games"] = frame["season_games"].where(same_season, 0).fillna(0)
    frame["season_ppr_mean"] = frame["season_ppr_mean"].where(same_season)
    frame["games_played"] = frame["games_played"].fillna(0)
    frame = frame.merge(season_means(player_week), on=["player_id", "season"], how="left")

    order = team_game_index(team_week)
    frame = frame.merge(
        order[["season", "week", "team", "game_index"]], on=["season", "week", "team"], how="left"
    )
    frame["games_missed_season"] = (frame["game_index"] - frame["season_games"]).clip(lower=0)
    previous = order.assign(game_index=order["game_index"] + 1)[
        ["season", "team", "game_index", "week"]
    ].rename(columns={"week": "previous_team_week"})
    frame = frame.merge(previous, on=["season", "team", "game_index"], how="left")
    played_previous = same_season & (frame["state_week"] == frame["previous_team_week"])
    frame["missed_last_game"] = np.where(
        frame["previous_team_week"].isna(), np.nan, (~played_previous).astype("float64")
    )

    frame = _asof(frame, team_states(team_week), by=["team"])
    frame = frame.merge(
        matchup_table(tables["defense_vs_position"], frame),
        on=["season", "week", "opponent", "position"],
        how="left",
    )

    frame = frame.merge(roster_experience(tables["rosters"], frame), on=KEY, how="left")
    frame["years_exp"] = frame["years_exp"].fillna(frame["season"] - frame["rookie_season"])
    frame["rookie"] = (frame["years_exp"] == 0).astype("int64")
    season_start = pd.to_datetime(frame["season"].astype(str) + "-09-01")
    birth = pd.to_datetime(frame["birth_date"], errors="coerce")
    frame["age"] = (season_start - birth).dt.days / 365.25
    frame["is_home"] = frame["is_home"].astype("float64")
    return frame.sort_values(KEY).reset_index(drop=True)


def attach_actuals(features: pd.DataFrame, player_week: pd.DataFrame) -> pd.DataFrame:
    """Add what actually happened: stat components and points. A candidate without a stat
    row for the game (inactive, or active without recording a stat) scores zero."""
    actual_columns = [
        *COMPONENTS,
        "fantasy_points_ppr",
        "fantasy_points_half",
        "fantasy_points_std",
    ]
    actuals = player_week[[*KEY, *actual_columns]].assign(played=1)
    out = features.merge(actuals, on=KEY, how="left")
    out[[*actual_columns, "played"]] = out[[*actual_columns, "played"]].fillna(0)
    return out
