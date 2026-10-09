from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron import backtest as bt
from gridiron.model import ProjectionModel
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
    assert pool["ecr_rank"].max() == bt.TOP_N["WR"]
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
    assert set(projected["model_version"]) == {"walk-forward-2025", "walk-forward-2026"}
    assert projected[projected["season"] == 2026]["week"].max() == 2
    assert projected["is_final"].all()


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
