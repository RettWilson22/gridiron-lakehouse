"""Per-position projection model: a projected stat line plus a PPR floor and ceiling.

For each position (QB, RB, WR, TE) the model fits one gradient-boosted regressor per
stat-line component (passing yards, rushing touchdowns, receptions, ...). Point projections
in any scoring format are the scoring rules applied to the projected stat line; because
scoring is linear in the stats, that is the expected score (yardage bonuses aside).

Floor and ceiling are the 10th and 90th percentile of actual PPR points among training
player-weeks with a similar out-of-fold projection: the training seasons are split into
``BAND_FOLDS`` groups, each group is projected by component models fitted on the other
groups, those projections are split into 20 equal-count bins per position, and the
percentiles are interpolated between bin medians (made non-decreasing in the projection).
A model's projections of its own training rows miss by less than its projections of new
data, so bands fitted on them understate the spread; out-of-fold projections are of
seasons the models did not train on.

Design choices were made in exploratory runs on the 2022 season (trained on 2018-2021)
before any 2023-2025 test season was scored. Those runs are not part of the repository;
the reasons, for the record:

* projecting components was about as accurate as projecting PPR points directly, and is
  what custom league scoring needs;
* the shallow, heavily regularized settings below did better than deeper trees;
* quantile gradient boosting for the floor and ceiling collapsed toward zero on this
  zero-inflated target (players who sit score zero), so the bands are empirical.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import gridiron
from gridiron.config import POSITIONS
from gridiron.features import COMPONENTS, FEATURES, KEY
from gridiron.scoring import PRESETS, UNIT_VALUES

SKILL_COMPONENTS: Final = (
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
    "two_point_conversions",
)
POSITION_COMPONENTS: Final[dict[str, tuple[str, ...]]] = {
    "QB": (
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
        "two_point_conversions",
    ),
    **dict.fromkeys(("RB", "WR", "TE"), SKILL_COMPONENTS),
}
# Non-negative counts use a Poisson loss; yardage (which can be negative) squared error.
COUNT_COMPONENTS: Final = frozenset(
    {
        "passing_tds",
        "passing_interceptions",
        "rushing_tds",
        "receptions",
        "receiving_tds",
        "fumbles_lost",
        "two_point_conversions",
    }
)
BAND_BINS: Final = 20
BAND_QUANTILES: Final = (0.1, 0.9)
# Folds (groups of whole seasons) for the out-of-fold projections the bands are fitted on.
# Chosen on the 2022 design season, not the test seasons: three folds covered within half a
# point of one season per fold at every position, with fewer extra fits; two folds covered
# more for QBs and TEs.
BAND_FOLDS: Final = 3
# Early stopping is off on purpose. scikit-learn's default ("auto") turns it on above 10,000
# training rows and then holds out a random 10% of them, so it would be on for some
# positions and backtest seasons and off for others, and the backtest would not test the
# setup that is deployed. Every fit runs all ``max_iter`` iterations on all rows.
HGB_PARAMS: Final[dict[str, Any]] = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}
FORMATS: Final = ("ppr", "half", "std")
FORMAT_PRESETS: Final = {"ppr": "ppr", "half": "half", "std": "standard"}
# Columns ``ProjectionModel.predict`` returns, in order.
OUTPUT_COLUMNS: Final = (
    *(f"proj_{c}" for c in COMPONENTS),
    *(f"proj_{fmt}" for fmt in FORMATS),
    *(f"{kind}_{fmt}" for fmt in FORMATS for kind in ("floor", "ceiling")),
)
TARGET: Final = "fantasy_points_ppr"
# Everything in a frame that a fit or a projection depends on (see ``fingerprint``).
FINGERPRINT_COLUMNS: Final = (*KEY, "position", *FEATURES, *COMPONENTS, TARGET)

# scikit-learn regressors (HistGradientBoostingRegressor or DummyRegressor); untyped.
Estimator: TypeAlias = Any


def fingerprint(*frames: pd.DataFrame) -> str:
    """sha256 of the inputs that decide a model and its projections.

    Covers, for each frame, the key, position, feature, stat-component and target columns
    (rows in key order, so row order does not matter), plus the model settings and the
    ``gridiron`` and scikit-learn versions. Two fits with the same fingerprint give the same
    model, so a stored result can be reused instead of refitting. Bump the package version
    when the model code changes in a way the settings do not show.
    """
    digest = hashlib.sha256()
    for frame in frames:
        rows = frame.sort_values(KEY)[list(FINGERPRINT_COLUMNS)]
        rows = rows.astype({"season": "int64", "week": "int64", "player_id": str, "position": str})
        numeric = [c for c in FINGERPRINT_COLUMNS if c not in (*KEY, "position")]
        rows[numeric] = rows[numeric].astype("float64")
        digest.update(f"{len(rows)} rows\n".encode())
        digest.update(pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes())
    settings = (
        HGB_PARAMS,
        POSITION_COMPONENTS,
        sorted(COUNT_COMPONENTS),
        BAND_BINS,
        BAND_QUANTILES,
        BAND_FOLDS,
        FEATURES,
        gridiron.__version__,
        sklearn.__version__,
    )
    digest.update(repr(settings).encode())
    return digest.hexdigest()


def _fit_component(component: str, x: pd.DataFrame, y: pd.Series) -> Estimator:
    """Gradient boosting for one stat; a constant zero if the stat never happened
    (a Poisson loss needs at least one non-zero target, e.g. no two-point conversions)."""
    if component in COUNT_COMPONENTS:
        if y.sum() <= 0:
            return DummyRegressor(strategy="constant", constant=0.0).fit(x, y)
        return HistGradientBoostingRegressor(loss="poisson", **HGB_PARAMS).fit(x, y)
    return HistGradientBoostingRegressor(loss="squared_error", **HGB_PARAMS).fit(x, y)


def _fit_components(
    rows: pd.DataFrame, position: str, features: tuple[str, ...]
) -> dict[str, Estimator]:
    x = rows[list(features)].astype("float64")
    # A feature with no values at all (e.g. last season's average when training starts
    # with the first season) cannot be binned; as a constant it is unused.
    x = x.fillna({c: 0.0 for c in x.columns if x[c].isna().all()})
    return {
        c: _fit_component(c, x, rows[c].astype("float64")) for c in POSITION_COMPONENTS[position]
    }


def _project_ppr(
    components: dict[str, Estimator], rows: pd.DataFrame, features: tuple[str, ...]
) -> pd.Series:
    x = rows[list(features)].astype("float64")
    line = pd.DataFrame({c: e.predict(x) for c, e in components.items()}, index=rows.index)
    for c in COUNT_COMPONENTS & set(line):
        line[c] = line[c].clip(lower=0)
    return points(line, "ppr")


def out_of_fold(
    rows: pd.DataFrame, position: str, features: tuple[str, ...] = FEATURES
) -> pd.Series:
    """Projected PPR points for one position's training rows, each from component models
    fitted without the row's season (``BAND_FOLDS`` folds of whole seasons)."""
    # A single season (only in the small test fixtures) is split by week instead.
    groups = rows["season"] if rows["season"].nunique() > 1 else rows["week"]
    folds = GroupKFold(n_splits=min(BAND_FOLDS, groups.nunique()))
    projected = pd.Series(np.nan, index=rows.index)
    for fit_on, held_out in folds.split(rows, groups=groups):
        components = _fit_components(rows.iloc[fit_on], position, features)
        projected.iloc[held_out] = _project_ppr(components, rows.iloc[held_out], features)
    return projected


@dataclass
class Band:
    """Floor and ceiling as a function of projected PPR points (piecewise linear)."""

    centers: list[float]
    floors: list[float]
    ceilings: list[float]

    @classmethod
    def fit(cls, projected: pd.Series, actual: pd.Series, bins: int = BAND_BINS) -> Band:
        edges = np.unique(np.quantile(projected, np.linspace(0, 1, bins + 1)))
        index = np.clip(np.searchsorted(edges, projected, side="right") - 1, 0, len(edges) - 2)
        grouped = pd.DataFrame({"bin": index, "projected": projected, "actual": actual}).groupby(
            "bin"
        )
        low, high = BAND_QUANTILES
        centers = grouped["projected"].median().to_numpy()
        floors = np.maximum.accumulate(grouped["actual"].quantile(low).to_numpy())
        ceilings = np.maximum.accumulate(grouped["actual"].quantile(high).to_numpy())
        return cls(centers.tolist(), floors.tolist(), ceilings.tolist())

    def floor(self, projected: pd.Series) -> np.ndarray:
        return np.asarray(np.interp(projected, self.centers, self.floors), dtype="float64")

    def ceiling(self, projected: pd.Series) -> np.ndarray:
        return np.asarray(np.interp(projected, self.centers, self.ceilings), dtype="float64")


def points(frame: pd.DataFrame, preset: str, prefix: str = "") -> pd.Series:
    """Vectorized fantasy points from stat columns (no bonuses; see ``gridiron.scoring``)."""
    settings = PRESETS[preset]
    total = pd.Series(0.0, index=frame.index)
    for stat, key in UNIT_VALUES:
        column = f"{prefix}{stat}"
        if column in frame:
            total = total + frame[column].fillna(0).astype("float64") * settings[key]
    return total


@dataclass
class ProjectionModel:
    """Fitted per-position component regressors and empirical PPR floor/ceiling bands."""

    features: tuple[str, ...] = FEATURES
    components: dict[str, dict[str, Estimator]] = field(default_factory=dict)
    bands: dict[str, Band] = field(default_factory=dict)
    trained_through: tuple[int, int] | None = None
    training_rows: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> ProjectionModel:
        """Fit on candidate rows with actuals attached (``features.attach_actuals``)."""
        model = cls()
        for position in POSITIONS:
            rows = frame[frame["position"] == position]
            if rows.empty:
                raise ValueError(f"no training rows for {position}")
            model.components[position] = _fit_components(rows, position, model.features)
            model.training_rows[position] = len(rows)
            model.bands[position] = Band.fit(
                out_of_fold(rows, position, model.features), rows[TARGET].astype("float64")
            )
        last = frame.sort_values(["season", "week"]).iloc[-1]
        model.trained_through = (int(last["season"]), int(last["week"]))
        return model

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Projected stat line, points in each format, and floor/ceiling per format.

        Floors and ceilings come from the PPR bands; for half PPR and standard they are
        scaled by the ratio of that format's projection to PPR. They are widened if needed
        so that floor <= projection <= ceiling.
        """
        out = pd.DataFrame(index=frame.index)
        for c in COMPONENTS:
            out[f"proj_{c}"] = 0.0
        for position in POSITIONS:
            mask = (frame["position"] == position).to_numpy()
            if not mask.any():
                continue
            x = frame.loc[mask, list(self.features)].astype("float64")
            for c, estimator in self.components[position].items():
                out.loc[mask, f"proj_{c}"] = estimator.predict(x)
        for c in COUNT_COMPONENTS:
            out[f"proj_{c}"] = out[f"proj_{c}"].clip(lower=0)
        for fmt in FORMATS:
            out[f"proj_{fmt}"] = points(out, FORMAT_PRESETS[fmt], prefix="proj_")
        floor_ppr = pd.Series(np.nan, index=frame.index)
        ceiling_ppr = pd.Series(np.nan, index=frame.index)
        for position, band in self.bands.items():
            mask = (frame["position"] == position).to_numpy()
            floor_ppr[mask] = band.floor(out.loc[mask, "proj_ppr"])
            ceiling_ppr[mask] = band.ceiling(out.loc[mask, "proj_ppr"])
        ratio_base = out["proj_ppr"].where(out["proj_ppr"] > 0)
        for fmt in FORMATS:
            ratio = (out[f"proj_{fmt}"] / ratio_base).fillna(1.0)
            out[f"floor_{fmt}"] = np.minimum(floor_ppr * ratio, out[f"proj_{fmt}"])
            out[f"ceiling_{fmt}"] = np.maximum(ceiling_ppr * ratio, out[f"proj_{fmt}"])
        return out
