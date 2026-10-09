from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from gridiron import model as model_module
from gridiron.backtest import completed
from gridiron.features import COMPONENTS
from gridiron.model import (
    HGB_PARAMS,
    OUTPUT_COLUMNS,
    POSITION_COMPONENTS,
    TARGET,
    Band,
    ProjectionModel,
    _fit_component,
    fingerprint,
    out_of_fold,
)
from gridiron.workflow import Prepared


@pytest.fixture(scope="module")
def fitted(prepared: Prepared) -> tuple[ProjectionModel, pd.DataFrame]:
    history = completed(prepared.frame)
    model = ProjectionModel.fit(history[history["season"] < 2026])
    test = history[history["season"] == 2026]
    return model, pd.concat([test, model.predict(test)], axis=1)


def test_predict_returns_the_output_columns(fitted: tuple[ProjectionModel, pd.DataFrame]) -> None:
    model, projected = fitted
    assert tuple(model.predict(projected.head(3)).columns) == OUTPUT_COLUMNS


def test_fingerprint_follows_what_the_model_learns_from(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = completed(prepared.frame)
    baseline = fingerprint(history)
    assert len(baseline) == 64
    assert fingerprint(history.sample(frac=1.0, random_state=1)) == baseline  # row order
    assert fingerprint(history.assign(player_name="someone else")) == baseline  # not an input

    corrected = history.copy()
    corrected.loc[corrected.index[0], "fantasy_points_ppr"] += 1.0  # a stat correction
    assert fingerprint(corrected) != baseline
    train, test = history[history["season"] < 2026], history[history["season"] == 2026]
    assert fingerprint(train, test) != fingerprint(history)  # which rows train, which test

    monkeypatch.setitem(HGB_PARAMS, "max_iter", 201)
    assert fingerprint(history) != baseline
    monkeypatch.undo()
    monkeypatch.setattr(model_module, "BAND_FOLDS", model_module.BAND_FOLDS + 1)
    assert fingerprint(history) != baseline


def test_out_of_fold_projections_never_see_their_own_season(prepared: Prepared) -> None:
    history = completed(prepared.frame)
    rows = history[(history["season"] < 2026) & (history["position"] == "WR")]
    first = rows["season"] == 2024
    targets = [*POSITION_COMPONENTS["WR"], TARGET]
    changed = rows.copy()
    changed.loc[first, targets] = rows.loc[first, targets] + 3.0  # 2024 results only

    before, after = out_of_fold(rows, "WR"), out_of_fold(changed, "WR")
    assert before.notna().all()
    pd.testing.assert_series_equal(after[first], before[first])  # fitted on 2025 only
    assert not np.allclose(after[~first], before[~first])  # fitted on the changed 2024


def test_bands_are_fit_on_out_of_fold_projections(prepared: Prepared) -> None:
    history = completed(prepared.frame)
    train = history[history["season"] < 2026]
    model = ProjectionModel.fit(train)
    for position in ("QB", "WR"):
        rows = train[train["position"] == position]
        expected = Band.fit(out_of_fold(rows, position), rows[TARGET].astype("float64"))
        assert model.bands[position] == expected


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
