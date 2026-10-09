"""Leakage tests: a feature for week w may only use information available before kickoff.

Each test perturbs data that must not matter (results from week w onward, and injury
reports or depth charts of other weeks) and checks that week-w candidates and features are
identical; a control test checks the features do react to earlier weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron import features as fx

TARGET = (2026, 3)
TARGET_KEY = TARGET[0] * 100 + TARGET[1]


def build(tables: dict[str, pd.DataFrame], **overrides: pd.DataFrame) -> pd.DataFrame:
    merged = {**tables, **overrides}
    candidates = fx.build_candidates(merged, weeks=[TARGET])
    built = fx.build_features(candidates, merged)
    columns = list(dict.fromkeys([*fx.KEY, "team", "opponent", "position", *fx.FEATURES]))
    return built[columns].reset_index(drop=True)


def time_key(frame: pd.DataFrame) -> pd.Series:
    return frame["season"] * 100 + frame["week"]


def scramble(frame: pd.DataFrame, rows: pd.Series, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """Overwrite every numeric value in ``rows`` (except keys and ``keep``) with noise."""
    out = frame.copy()
    rng = np.random.default_rng(7)
    protected = {"season", "week", *keep}
    for column in out.select_dtypes("number").columns:
        if column not in protected:
            out[column] = out[column].astype("float64")
            out.loc[rows, column] = rng.normal(50, 20, int(rows.sum()))
    return out


@pytest.fixture(scope="module")
def baseline(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = build(tables)
    assert len(frame) > 50
    return frame


def test_results_from_the_target_week_onward_do_not_change_its_features(
    tables: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> None:
    player_week = tables["player_week"]
    team_week = tables["team_week"]
    defense = tables["defense_vs_position"]
    perturbed = build(
        tables,
        player_week=scramble(player_week, time_key(player_week) >= TARGET_KEY),
        # Betting lines for the target week are pre-game information and stay as they are.
        team_week=scramble(
            team_week,
            time_key(team_week) >= TARGET_KEY,
            keep=("implied_points", "team_spread", "total_line"),
        ),
        defense_vs_position=scramble(defense, time_key(defense) >= TARGET_KEY),
    )
    pd.testing.assert_frame_equal(baseline, perturbed)


def test_dropping_results_from_the_target_week_onward_changes_nothing(
    tables: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> None:
    keep = {
        name: tables[name][time_key(tables[name]) < TARGET_KEY]
        for name in ("player_week", "defense_vs_position")
    }
    pd.testing.assert_frame_equal(baseline, build(tables, **keep))


def test_other_weeks_injury_reports_depth_charts_and_rosters_do_not_matter(
    tables: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> None:
    injuries = tables["injuries"].copy()
    other = time_key(injuries) != TARGET_KEY
    injuries.loc[other, "report_status"] = "Out"
    depth = tables["depth_chart_week"].copy()
    depth.loc[time_key(depth) != TARGET_KEY, "depth_rank"] = 9
    rosters = tables["rosters"].copy()
    rosters.loc[time_key(rosters) > TARGET_KEY, "position"] = "TE"
    perturbed = build(tables, injuries=injuries, depth_chart_week=depth, rosters=rosters)
    pd.testing.assert_frame_equal(baseline, perturbed)


def test_earlier_weeks_do_change_the_features(
    tables: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> None:
    """Control: the tests above would notice if history were ignored altogether."""
    player_week = tables["player_week"]
    perturbed = build(tables, player_week=scramble(player_week, time_key(player_week) < TARGET_KEY))
    assert not baseline["ewm_fantasy_points_ppr"].equals(perturbed["ewm_fantasy_points_ppr"])


def test_players_ruled_out_are_not_projected(tables: dict[str, pd.DataFrame]) -> None:
    candidates = fx.build_candidates(tables, weeks=[TARGET])
    starter = candidates.sort_values("depth_rank").iloc[0]
    injuries = tables["injuries"]
    extra = pd.DataFrame(
        [
            {
                "player_id": starter["player_id"],
                "season": TARGET[0],
                "game_type": "REG",
                "week": TARGET[1],
                "team": starter["team"],
                "report_status": "Out",
                "practice_status": None,
                "report_primary_injury": "Ankle",
            }
        ]
    )
    ruled_out = fx.build_candidates(
        {**tables, "injuries": pd.concat([injuries, extra])}, weeks=[TARGET]
    )
    assert starter["player_id"] not in set(ruled_out["player_id"])
    assert len(ruled_out) == len(candidates) - 1


def test_opponent_matchup_matches_the_gold_rolling_average(
    tables: dict[str, pd.DataFrame], baseline: pd.DataFrame
) -> None:
    gold = tables["defense_vs_position"][["season", "week", "team", "position", "ppr_allowed_l6"]]
    joined = baseline.merge(
        gold.rename(columns={"team": "opponent"}),
        on=["season", "week", "opponent", "position"],
    ).dropna(subset=["ppr_allowed_l6"])
    assert not joined.empty
    assert np.allclose(joined["opp_ppr_allowed_l6"], joined["ppr_allowed_l6"])


def test_candidates_cover_the_players_who_scored(tables: dict[str, pd.DataFrame]) -> None:
    weeks = [(2025, 3), (2025, 4), (2026, 3), (2026, 4)]
    candidates = fx.build_candidates(tables, weeks=weeks)
    played = tables["player_week"].merge(pd.DataFrame(weeks, columns=["season", "week"]))
    scorers = played[played["fantasy_points_ppr"] >= 10]
    covered = scorers.merge(candidates[fx.KEY], on=fx.KEY, how="left", indicator=True)
    assert (covered["_merge"] == "both").mean() > 0.95


def test_actuals_are_zero_for_candidates_who_did_not_play(tables: dict[str, pd.DataFrame]) -> None:
    candidates = fx.build_candidates(tables, weeks=[(2026, 2)])
    with_actuals = fx.attach_actuals(candidates, tables["player_week"])
    sat = with_actuals[with_actuals["played"] == 0]
    assert not sat.empty
    assert (sat[["fantasy_points_ppr", *fx.COMPONENTS]] == 0).all().all()
