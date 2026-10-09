"""Model-side workflow shared by the Databricks train/score tasks and the local runner.

The job entry points only move tables between Spark (or Parquet files) and pandas; every
step below is plain pandas and scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

from gridiron import backtest as bt
from gridiron import features as fx
from gridiron.model import ProjectionModel
from gridiron.projections import (
    assemble,
    merge_live,
    projection_frame,
    risers,
    upcoming_week,
)

INPUT_TABLES: Final = (
    "player_week",
    "team_week",
    "defense_vs_position",
    "depth_chart_week",
    "injuries",
    "players",
    "rosters",
    "ecr",
)
DEFAULT_TEST_SEASONS: Final = (2023, 2024, 2025)


@dataclass
class Prepared:
    """Feature rows for every completed week plus the upcoming week (or for the upcoming
    week only; see ``prepare``)."""

    frame: pd.DataFrame
    upcoming: tuple[int, int] | None
    tables: dict[str, pd.DataFrame]


def prepare(
    read: Callable[[str], pd.DataFrame], weeks: Literal["all", "upcoming"] = "all"
) -> Prepared:
    """Read the input tables and build feature rows.

    ``weeks="all"`` builds every completed week and the upcoming week (training and the
    backtest need them); ``weeks="upcoming"`` only the upcoming week, which is all the
    score task projects. Features of a row never depend on other rows, so the upcoming
    week's rows are the same either way.
    """
    tables = {name: read(name) for name in INPUT_TABLES}
    team_week = tables["team_week"]
    upcoming = upcoming_week(team_week)
    wanted: set[tuple[int, int]] = set()
    if weeks == "all":
        final = team_week.loc[team_week["is_final"].astype(bool), ["season", "week"]]
        wanted = {(int(s), int(w)) for s, w in final.to_numpy()}
    if upcoming is not None:
        wanted.add(upcoming)
    if not wanted:
        return Prepared(pd.DataFrame(), upcoming, tables)
    candidates = fx.build_candidates(tables, weeks=sorted(wanted))
    built = fx.build_features(candidates, tables)
    return Prepared(fx.attach_actuals(built, tables["player_week"]), upcoming, tables)


def before(frame: pd.DataFrame, week: tuple[int, int]) -> pd.DataFrame:
    key = frame["season"] * 100 + frame["week"]
    return frame[key < week[0] * 100 + week[1]]


def run_backtest(
    prepared: Prepared, test_seasons: Iterable[int], now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward projections for the test seasons and the current season's completed
    weeks, and the accuracy metrics for the test seasons."""
    test_seasons = sorted(set(test_seasons))
    seasons = list(test_seasons)
    limits: dict[int, int] = {}
    if prepared.upcoming is not None:
        current, week = prepared.upcoming
        seasons.append(current)
        limits[current] = week
    projected = bt.walk_forward(prepared.frame, seasons, limits)
    metrics = bt.evaluate(projected, prepared.tables["ecr"], test_seasons)
    published = pd.concat(
        [
            projection_frame(group, "backtest", str(version), now)
            for version, group in projected.groupby("model_version")
        ],
        ignore_index=True,
    )
    return published, metrics


def train_live(prepared: Prepared) -> ProjectionModel:
    """Model for the upcoming week: every completed week before it."""
    if prepared.upcoming is None:
        raise ValueError("no upcoming week to train for")
    history = bt.completed(before(prepared.frame, prepared.upcoming))
    return ProjectionModel.fit(history)


def upcoming_rows(prepared: Prepared) -> pd.DataFrame:
    if prepared.upcoming is None:
        return prepared.frame.iloc[0:0]
    season, week = prepared.upcoming
    frame = prepared.frame
    return frame[(frame["season"] == season) & (frame["week"] == week)]


def score_live(
    prepared: Prepared,
    predict: Callable[[pd.DataFrame], pd.DataFrame],
    model_version: str,
    existing_live: pd.DataFrame | None,
    now: pd.Timestamp,
) -> pd.DataFrame | None:
    """Live projections for the upcoming week merged into the stored live history."""
    rows = upcoming_rows(prepared)
    if rows.empty:
        return existing_live
    predictions = predict(rows)
    predictions.index = rows.index
    fresh = projection_frame(pd.concat([rows, predictions], axis=1), "live", model_version, now)
    return merge_live(existing_live, fresh, now)


def publish(
    prepared: Prepared, backtest_projections: pd.DataFrame, live: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ``projections`` and ``risers`` serving tables."""
    projections = assemble(backtest_projections, live, prepared.tables["ecr"])
    if prepared.upcoming is None:
        return projections, risers(prepared.tables["player_week"], [])
    season, week = prepared.upcoming
    weeks = [(season, w) for w in range(2, week + 1)]
    return projections, risers(prepared.tables["player_week"], weeks)
