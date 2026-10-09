"""Per-position projection model: a projected stat line plus a PPR floor and ceiling.

For each position (QB, RB, WR, TE) the model fits one gradient-boosted regressor per
stat-line component (passing yards, rushing touchdowns, receptions, ...). Point projections
in any scoring format are the scoring rules applied to the projected stat line; because
scoring is linear in the stats, that is the expected score (yardage bonuses aside).

Floor and ceiling are the 10th and 90th percentile of actual PPR points among training
player-weeks with a similar projection: training rows are split into 20 equal-count bins
of projected points per position, and the percentiles are interpolated between bin
medians (made non-decreasing in the projection).

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

from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from gridiron.config import POSITIONS
from gridiron.features import COMPONENTS, FEATURES
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

# scikit-learn regressors (HistGradientBoostingRegressor or DummyRegressor); untyped.
Estimator: TypeAlias = Any


def _fit_component(component: str, x: pd.DataFrame, y: pd.Series) -> Estimator:
    """Gradient boosting for one stat; a constant zero if the stat never happened
    (a Poisson loss needs at least one non-zero target, e.g. no two-point conversions)."""
    if component in COUNT_COMPONENTS:
        if y.sum() <= 0:
            return DummyRegressor(strategy="constant", constant=0.0).fit(x, y)
        return HistGradientBoostingRegressor(loss="poisson", **HGB_PARAMS).fit(x, y)
    return HistGradientBoostingRegressor(loss="squared_error", **HGB_PARAMS).fit(x, y)


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
            x = rows[list(model.features)].astype("float64")
            # A feature with no values at all (e.g. last season's average when training
            # starts with the first season) cannot be binned; as a constant it is unused.
            x = x.fillna({c: 0.0 for c in x.columns if x[c].isna().all()})
            model.components[position] = {
                c: _fit_component(c, x, rows[c].astype("float64"))
                for c in POSITION_COMPONENTS[position]
            }
            model.training_rows[position] = len(rows)
            projected = model._project(rows, position)
            model.bands[position] = Band.fit(
                projected, rows["fantasy_points_ppr"].astype("float64")
            )
        last = frame.sort_values(["season", "week"]).iloc[-1]
        model.trained_through = (int(last["season"]), int(last["week"]))
        return model

    def _project(self, rows: pd.DataFrame, position: str) -> pd.Series:
        """Projected PPR points from the component models (for fitting the bands)."""
        x = rows[list(self.features)].astype("float64")
        line = pd.DataFrame(
            {c: e.predict(x) for c, e in self.components[position].items()}, index=rows.index
        )
        for c in COUNT_COMPONENTS & set(line):
            line[c] = line[c].clip(lower=0)
        return points(line, "ppr")

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
