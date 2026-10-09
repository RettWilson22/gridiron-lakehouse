"""End-to-end model workflow on the fixture: backtest, live week, published tables."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron.projections import PROJECTION_COLUMNS
from gridiron.workflow import (
    Prepared,
    prepare,
    publish,
    run_backtest,
    score_live,
    train_live,
    upcoming_rows,
)

NOW = pd.Timestamp("2026-10-08 12:00", tz="UTC")


@pytest.fixture(scope="module")
def outputs(prepared: Prepared) -> dict[str, pd.DataFrame]:
    backtest, metrics = run_backtest(prepared, [2025], NOW)
    model = train_live(prepared)
    live = score_live(prepared, model.predict, "test", None, NOW)
    projections, risers = publish(prepared, backtest, live)
    assert live is not None
    return {
        "backtest": backtest,
        "metrics": metrics,
        "live": live,
        "projections": projections,
        "risers": risers,
    }


def test_the_upcoming_week_comes_from_the_schedule(prepared: Prepared) -> None:
    assert prepared.upcoming == (2026, 5)


def test_backtest_covers_test_seasons_and_completed_current_weeks(
    outputs: dict[str, pd.DataFrame],
) -> None:
    backtest = outputs["backtest"]
    assert list(backtest.columns) == list(PROJECTION_COLUMNS)
    for season in (2025, 2026):
        weeks = backtest.loc[backtest["season"] == season, "week"]
        assert (int(weeks.min()), int(weeks.max())) == (1, 4)
    assert set(outputs["metrics"]["method"]) == {"model", "last3", "season_avg", "ecr"}


def test_published_projections_join_backtest_and_live_weeks(
    outputs: dict[str, pd.DataFrame],
) -> None:
    projections = outputs["projections"]
    assert not projections.duplicated(["season", "week", "player_id"]).any()
    week5 = projections[(projections["season"] == 2026) & (projections["week"] == 5)]
    assert set(week5["kind"]) == {"live"}
    assert len(week5) == len(outputs["live"])
    assert set(projections[projections["week"] < 5]["kind"]) == {"backtest"}
    for fmt in ("ppr", "half", "std"):
        assert projections[f"pos_rank_{fmt}"].min() == 1
        assert set(projections[f"start_sit_{fmt}"]) <= {"Start", "Flex", "Sit"}


def test_live_rows_are_stable_when_the_job_runs_again(
    prepared: Prepared, outputs: dict[str, pd.DataFrame]
) -> None:
    model = train_live(prepared)
    again = score_live(prepared, model.predict, "test", outputs["live"], NOW)
    assert again is not None

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values("player_id").reset_index(drop=True)
        return frame.astype(object).where(frame.notna(), None)

    pd.testing.assert_frame_equal(normalized(again), normalized(outputs["live"]))


def test_preparing_only_the_upcoming_week_gives_the_same_rows(
    prepared: Prepared, tables: dict[str, pd.DataFrame]
) -> None:
    upcoming = prepare(lambda name: tables[name].copy(), weeks="upcoming")
    assert upcoming.upcoming == prepared.upcoming
    expected = upcoming_rows(prepared).reset_index(drop=True)
    got = upcoming.frame.reset_index(drop=True)
    # An integer column can stay integer when the week has no missing values; the model
    # casts every feature to float64, and the published columns keep their types.
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    published = [c for c in PROJECTION_COLUMNS if c in expected]
    assert list(got[published].dtypes) == list(expected[published].dtypes)


def test_preparing_the_upcoming_week_in_the_offseason_gives_no_rows(
    tables: dict[str, pd.DataFrame],
) -> None:
    team_week = tables["team_week"].assign(is_final=True)
    offseason = prepare(
        lambda name: team_week if name == "team_week" else tables[name].copy(), weeks="upcoming"
    )
    assert offseason.upcoming is None
    assert offseason.frame.empty
