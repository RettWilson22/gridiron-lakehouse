from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from gridiron.backtest import completed
from gridiron.features import COMPONENTS
from gridiron.model import POSITION_COMPONENTS, Band, ProjectionModel, _fit_component
from gridiron.workflow import Prepared


@pytest.fixture(scope="module")
def fitted(prepared: Prepared) -> tuple[ProjectionModel, pd.DataFrame]:
    history = completed(prepared.frame)
    model = ProjectionModel.fit(history[history["season"] < 2026])
    test = history[history["season"] == 2026]
    return model, pd.concat([test, model.predict(test)], axis=1)


def test_prediction_columns_and_invariants(fitted: tuple[ProjectionModel, pd.DataFrame]) -> None:
    _, projected = fitted
    for fmt in ("ppr", "half", "std"):
        assert (projected[f"floor_{fmt}"] <= projected[f"proj_{fmt}"] + 1e-9).all()
        assert (projected[f"proj_{fmt}"] <= projected[f"ceiling_{fmt}"] + 1e-9).all()
    for count in ("passing_tds", "receptions", "rushing_tds", "fumbles_lost"):
        assert (projected[f"proj_{count}"] >= 0).all()
    gap = projected["proj_ppr"] - projected["proj_std"]
    assert np.allclose(gap, projected["proj_receptions"])
    assert np.allclose(projected["proj_half"], (projected["proj_ppr"] + projected["proj_std"]) / 2)


def test_each_position_only_projects_its_own_stats(
    fitted: tuple[ProjectionModel, pd.DataFrame],
) -> None:
    _, projected = fitted
    for position, components in POSITION_COMPONENTS.items():
        rows = projected[projected["position"] == position]
        for component in set(COMPONENTS) - set(components):
            assert (rows[f"proj_{component}"] == 0).all(), (position, component)


def test_model_records_what_it_was_trained_on(
    fitted: tuple[ProjectionModel, pd.DataFrame],
) -> None:
    model, _ = fitted
    assert model.trained_through == (2025, 4)
    assert set(model.training_rows) == {"QB", "RB", "WR", "TE"}


def test_every_fitted_estimator_has_early_stopping_off(
    fitted: tuple[ProjectionModel, pd.DataFrame],
) -> None:
    model, _ = fitted
    boosted = [
        estimator
        for estimators in model.components.values()
        for estimator in estimators.values()
        if isinstance(estimator, HistGradientBoostingRegressor)
    ]
    assert boosted
    for estimator in boosted:
        assert estimator.early_stopping is False
        assert estimator.random_state == 0
        assert not estimator.do_early_stopping_


def test_large_training_sets_do_not_switch_early_stopping_on() -> None:
    # scikit-learn's default ("auto") holds out a random 10% and stops early above 10,000
    # rows, so the setting would flip between positions and backtest seasons.
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(10_500, 3)), columns=["a", "b", "c"])
    y = pd.Series(x["a"] * 2 + rng.normal(size=len(x)))
    estimator = _fit_component("rushing_yards", x, y)
    assert not estimator.do_early_stopping_
    assert estimator.n_iter_ == estimator.max_iter


def test_band_is_monotone_and_interpolates() -> None:
    rng = np.random.default_rng(0)
    projected = pd.Series(rng.uniform(0, 25, 2000))
    actual = projected + rng.normal(0, 5, 2000)
    band = Band.fit(projected, actual)
    assert band.floors == sorted(band.floors)
    assert band.ceilings == sorted(band.ceilings)
    grid = pd.Series([0.0, 10.0, 20.0])
    assert (band.floor(grid) < grid.to_numpy() + 1e-9).all()
    assert (band.ceiling(grid) > grid.to_numpy() - 1e-9).all()
    inside = (actual >= band.floor(projected)) & (actual <= band.ceiling(projected))
    assert 0.7 < inside.mean() < 0.9
