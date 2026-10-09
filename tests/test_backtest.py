from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from gridiron import backtest as bt
from gridiron.model import ProjectionModel
from gridiron.tiers import POOL
from gridiron.workflow import Prepared


def synthetic(weeks: int = 3, players: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ecr = [], []
    for week in range(1, weeks + 1):
        for i in range(players):
            actual = float(players - i)
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "player_id": f"p{i}",
                    "position": "WR",
                    "fantasy_points_ppr": actual,
                    "proj_ppr": actual + 1.0,  # perfect order, biased by +1
                    "floor_ppr": actual - 0.5 if i % 2 else actual + 0.1,
                    "ceiling_ppr": actual + 2.0,
                    "baseline_last3": float(i),  # perfectly reversed order
                    "baseline_season_avg": actual,  # exact
                }
            )
            ecr.append(
                {
                    "season": 2024,
                    "week": week,
                    "player_id": f"p{i}",
                    "position": "WR",
                    "ecr": float(i + 1),
                    "ecr_rank": i + 1,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(ecr)


def test_metrics_on_a_known_example() -> None:
    projected, ecr = synthetic()
    metrics = bt.evaluate(projected, ecr, [2024])

    def value(method: str, column: str) -> float:
        row = metrics[(metrics["scope"] == "2024") & (metrics["method"] == method)]
        return float(row[column].iloc[0])

    assert value("model", "mae") == pytest.approx(1.0)
    assert value("model", "rmse") == pytest.approx(1.0)
    assert value("model", "bias") == pytest.approx(1.0)
    assert value("model", "spearman") == pytest.approx(1.0)
    assert value("model", "interval_coverage") == pytest.approx(0.5)
    assert value("last3", "spearman") == pytest.approx(-1.0)
    assert value("season_avg", "mae") == pytest.approx(0.0)
    assert value("ecr", "spearman") == pytest.approx(1.0)
    assert np.isnan(value("ecr", "mae"))
    assert value("model", "weeks") == 3
    assert value("model", "pool_coverage") == pytest.approx(1.0)


def test_pool_uses_the_expert_top_n_and_complete_rows() -> None:
    projected, ecr = synthetic(players=80)
    projected.loc[projected["player_id"] == "p3", "baseline_last3"] = np.nan
    pool = bt.evaluation_pool(bt.attach_ecr(projected, ecr))
    assert pool["ecr_rank"].max() == POOL["WR"]
    assert "p3" not in set(pool["player_id"])


def test_walk_forward_trains_only_on_earlier_seasons(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, int]] = []
    original = ProjectionModel.fit.__func__  # type: ignore[attr-defined]

    def spy(cls: type[ProjectionModel], frame: pd.DataFrame) -> ProjectionModel:
        seen.append((int(frame["season"].max()), int(frame["season"].min())))
        model: ProjectionModel = original(cls, frame)
        return model

    monkeypatch.setattr(ProjectionModel, "fit", classmethod(spy))
    projected = bt.walk_forward(prepared.frame, [2025, 2026], {2026: 3})
    assert seen == [(2024, 2024), (2025, 2024)]
    versions = sorted(projected["model_version"].unique())
    assert [v.rsplit("-", 1)[0] for v in versions] == ["walk-forward-2025", "walk-forward-2026"]
    assert all(re.fullmatch(r"[0-9a-f]{12}", v.rsplit("-", 1)[1]) for v in versions)
    assert projected[projected["season"] == 2026]["week"].max() == 2
    assert projected["is_final"].all()


def test_walk_forward_refits_only_seasons_whose_data_changed(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = bt.walk_forward(prepared.frame, [2025, 2026], {2026: 3})
    fits: list[int] = []
    original = ProjectionModel.fit.__func__  # type: ignore[attr-defined]

    def spy(cls: type[ProjectionModel], frame: pd.DataFrame) -> ProjectionModel:
        fits.append(int(frame["season"].max()) + 1)  # the season the model projects
        model: ProjectionModel = original(cls, frame)
        return model

    monkeypatch.setattr(ProjectionModel, "fit", classmethod(spy))
    # Stored projections come back from a table: other row order, int32 keys.
    stored = first.sample(frac=1.0, random_state=0).astype({"season": "int32", "week": "int32"})
    again = bt.walk_forward(prepared.frame, [2025, 2026], {2026: 3}, stored=stored)
    assert fits == []
    pd.testing.assert_frame_equal(again, first)

    # A newly completed week changes what the 2026 model projects; 2025 is reused.
    later = bt.walk_forward(prepared.frame, [2025, 2026], {2026: 4}, stored=stored)
    assert fits == [2026]
    assert set(later["model_version"]) - set(first["model_version"]) == {
        later.loc[later["season"] == 2026, "model_version"].iloc[0]
    }


def test_pooled_scope_is_the_multi_season_scope() -> None:
    projected, ecr = synthetic()
    later = projected.assign(season=2025)
    metrics = bt.evaluate(
        pd.concat([projected, later]), pd.concat([ecr, ecr.assign(season=2025)]), [2024, 2025]
    )
    assert sorted(metrics["scope"].unique()) == ["2024", "2024-2025", "2025"]
    assert bt.pooled_scope(metrics["scope"]) == "2024-2025"
    assert bt.pooled_scope(["2023", "2023-2025", "2024", "2025"]) == "2023-2025"


def test_candidate_coverage_counts_player_games_in_the_pool() -> None:
    candidates = pd.DataFrame(
        {"season": 2024, "week": 1, "player_id": ["a", "b", "c"], "position": "WR"}
    )
    player_week = pd.DataFrame(
        {
            "season": 2024,
            "week": [1, 1, 1, 1, 2],
            "player_id": ["a", "b", "x", "y", "a"],
            "fantasy_points_ppr": [12.0, 3.0, 15.0, 0.5, 20.0],
        }
    )
    coverage = bt.candidate_coverage(candidates, player_week)
    assert coverage["player_games"] == 4  # week 2 has no candidates and is not counted
    assert coverage["share_all"] == pytest.approx(0.5)
    assert coverage["share_10_plus_ppr"] == pytest.approx(0.5)


def test_a_single_season_has_no_separate_pooled_scope() -> None:
    projected, ecr = synthetic()
    metrics = bt.evaluate(projected, ecr, [2024])
    assert list(metrics["scope"].unique()) == ["2024"]  # not also "2024-2024"
    assert bt.pooled_scope(metrics["scope"]) == "2024"


def bootstrap_pool(weeks: int = 20, players: int = 10, **misses: float) -> pd.DataFrame:
    """One position's pool where each method misses every player by a fixed amount (0 when
    not given) and the experts rank players in the right order."""
    rows = []
    for week in range(1, weeks + 1):
        for i in range(players):
            actual = float(players - i)
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "player_id": f"p{i}",
                    "fantasy_points_ppr": actual,
                    "proj_ppr": actual + misses.get("model", 0.0),
                    "baseline_last3": actual + misses.get("last3", 0.0),
                    "baseline_season_avg": actual + misses.get("season_avg", 0.0),
                    "ecr": float(i + 1),
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_of_identical_methods_is_zero() -> None:
    result = bt.paired_bootstrap(bootstrap_pool(model=1.0, last3=1.0, season_avg=1.0))
    assert set(result.index) == {
        "mae_model_minus_last3",
        "mae_model_minus_season_avg",
        "spearman_model_minus_ecr",
    }
    assert (result[["estimate", "low", "high"]] == 0.0).all().all()
    assert (result["weeks"] == 20).all()


def test_bootstrap_of_a_constant_difference_is_that_difference() -> None:
    pool = bootstrap_pool(model=1.0, last3=3.0, season_avg=-2.0)
    pool["ecr"] = -pool["ecr"]  # experts rank in reverse: -1 against the model's 1
    result = bt.paired_bootstrap(pool)
    expected = {
        "mae_model_minus_last3": -2.0,
        "mae_model_minus_season_avg": -1.0,
        "spearman_model_minus_ecr": 2.0,
    }
    for column in ("estimate", "low", "high"):
        assert result[column].to_dict() == pytest.approx(expected)


def test_bootstrap_resamples_whole_weeks() -> None:
    # The model ties the season average in week 1 and misses by one more point in week 2.
    # Drawing two weeks gives 0 (week 1 twice), 0.5 or 1 (week 2 twice) with probability
    # 1/4, 1/2 and 1/4, so the 95% interval is [0, 1]. Resampling single player-weeks
    # would give a much narrower interval around 0.5.
    pool = bootstrap_pool(weeks=2, model=1.0, season_avg=1.0)
    pool.loc[pool["week"] == 2, "proj_ppr"] += 1.0
    row = bt.paired_bootstrap(pool).loc["mae_model_minus_season_avg"]
    assert row["estimate"] == pytest.approx(0.5)
    assert row["low"] == pytest.approx(0.0)
    assert row["high"] == pytest.approx(1.0)


def test_bootstrap_is_reproducible_and_brackets_a_noisy_difference() -> None:
    rng = np.random.default_rng(1)
    pool = bootstrap_pool(weeks=30, players=20)
    pool["proj_ppr"] += rng.normal(0, 2, len(pool))
    pool["baseline_season_avg"] += rng.normal(0, 2, len(pool)) + np.sign(rng.normal(size=len(pool)))
    first = bt.paired_bootstrap(pool)
    pd.testing.assert_frame_equal(bt.paired_bootstrap(pool), first)
    row = first.loc["mae_model_minus_season_avg"]
    assert row["low"] < row["estimate"] < row["high"] < 0  # the model misses by less


def test_intervals_estimates_match_the_metrics() -> None:
    projected, ecr = synthetic()
    intervals = bt.intervals(projected, ecr, [2024])
    assert set(intervals["scope"]) == {"2024"}
    assert set(intervals["weeks"]) == {3}
    rows = intervals.set_index("comparison")
    # The model misses every player by 1, the season average by 0 and the last-3 average
    # by 4 on average; the model and the experts both rank perfectly.
    assert rows.loc["mae_model_minus_season_avg", "estimate"] == pytest.approx(1.0)
    assert rows.loc["mae_model_minus_season_avg", "low"] == pytest.approx(1.0)
    assert rows.loc["mae_model_minus_season_avg", "high"] == pytest.approx(1.0)
    assert rows.loc["mae_model_minus_last3", "estimate"] == pytest.approx(-3.0)
    assert rows.loc["spearman_model_minus_ecr", "estimate"] == pytest.approx(0.0)
