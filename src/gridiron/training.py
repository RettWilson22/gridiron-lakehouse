"""Fit and evaluate the decision model components (pandas + scikit-learn)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridiron.decision import (
    KICK_SNAP_DISTANCE,
    PUNT_TOUCHBACK_OPPONENT_YARDLINE,
    DecisionModel,
    LogisticParams,
)
from gridiron.features import FEATURE_COLUMNS, conversion_training_frame, split_by_season

YARDLINES = np.arange(1, 100)


def logistic_pipeline() -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=1000))]
    )


def fit_conversion(features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    return logistic_pipeline().fit(features[list(FEATURE_COLUMNS)], labels)


def fit_field_goal(decisions: pd.DataFrame) -> LogisticParams:
    kicks = decisions[decisions["decision"] == "field_goal"].dropna(subset=["yardline_100"])
    distance = (kicks["yardline_100"].astype("float64") + KICK_SNAP_DISTANCE).to_frame("distance")
    made = (kicks["outcome"] == "fg_made").astype("int64")
    pipeline = logistic_pipeline().fit(distance, made)
    return LogisticParams.from_pipeline(pipeline, ["distance"])


def _smooth(values: pd.Series, window: int) -> np.ndarray:
    """Reindex onto yardlines 1..99, interpolate gaps and apply a centred rolling mean."""
    full = values.reindex(YARDLINES).interpolate(limit_direction="both")
    return np.asarray(full.rolling(window, center=True, min_periods=1).mean(), dtype="float64")


def fit_ep_curve(first_downs: pd.DataFrame, window: int = 5) -> list[float]:
    """Average nflfastR expected points of first downs by yardline, lightly smoothed."""
    means = first_downs.groupby(first_downs["yardline_100"].astype(int))["ep"].mean()
    return [float(v) for v in _smooth(means, window)]


def punt_opponent_yardline(punts: pd.DataFrame) -> pd.Series:
    """Opponent's yardline_100 after each punt (touchbacks at the 20; blocked punts dropped)."""
    landed = punts.dropna(subset=["yardline_100", "kick_distance"])
    landed = landed[landed["outcome"] == "punt"]
    spot = landed["yardline_100"] - landed["kick_distance"] + landed["return_yards"].fillna(0)
    opponent = (100 - spot).clip(1, 99)
    touchback = landed["touchback"].fillna(0) == 1
    return opponent.where(~touchback, PUNT_TOUCHBACK_OPPONENT_YARDLINE)


def fit_punt_curve(decisions: pd.DataFrame, window: int = 7) -> list[float]:
    punts = decisions[decisions["decision"] == "punt"]
    opponent = punt_opponent_yardline(punts)
    by_spot = opponent.groupby(punts.loc[opponent.index, "yardline_100"].astype(int)).mean()
    return [float(v) for v in _smooth(by_spot, window)]


def classification_metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(labels)),
        "base_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
    }


def calibration_table(labels: pd.Series, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed vs. predicted rate per quantile bin of predicted probability."""
    frame = pd.DataFrame({"label": labels.to_numpy(), "p": probabilities})
    frame["bin"] = pd.qcut(frame["p"], q=bins, labels=False, duplicates="drop")
    table = frame.groupby("bin").agg(
        n=("label", "size"), mean_predicted=("p", "mean"), observed_rate=("label", "mean")
    )
    return table.reset_index()


@dataclass
class TrainingResult:
    model: DecisionModel
    conversion_pipeline: Pipeline
    metrics: dict[str, float] = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)


def train_decision_model(
    decisions: pd.DataFrame,
    first_downs: pd.DataFrame,
    last_train_season: int,
    test_seasons: tuple[int, ...] | None = None,
) -> TrainingResult:
    """Fit every component on seasons <= ``last_train_season`` and evaluate on later ones."""
    train, test = split_by_season(decisions, last_train_season, test_seasons)
    if test.empty:
        raise ValueError("no held-out seasons to evaluate on")

    x_train, y_train = conversion_training_frame(train)
    x_test, y_test = conversion_training_frame(test)
    conversion = fit_conversion(x_train, y_train)
    p_test = conversion.predict_proba(x_test[list(FEATURE_COLUMNS)])[:, 1]

    baseline = logistic_pipeline().fit(x_train[["log_ydstogo"]], y_train)
    p_baseline = baseline.predict_proba(x_test[["log_ydstogo"]])[:, 1]

    fg = fit_field_goal(train)
    test_kicks = test[test["decision"] == "field_goal"]
    fg_p = fg.predict(
        (test_kicks["yardline_100"].to_numpy(dtype="float64") + KICK_SNAP_DISTANCE).reshape(-1, 1)
    )
    fg_labels = (test_kicks["outcome"] == "fg_made").astype("int64")

    train_first_downs = first_downs[first_downs["season"] <= last_train_season]
    model = DecisionModel(
        conversion=LogisticParams.from_pipeline(conversion, list(FEATURE_COLUMNS)),
        fg_make=fg,
        ep_curve=fit_ep_curve(train_first_downs),
        punt_curve=fit_punt_curve(train),
        trained_through_season=last_train_season,
    )

    metrics = {f"conversion_test_{k}": v for k, v in classification_metrics(y_test, p_test).items()}
    metrics["conversion_train_n"] = float(len(y_train))
    metrics.update(
        {
            f"baseline_test_{k}": v
            for k, v in classification_metrics(y_test, p_baseline).items()
            if k in {"auc", "brier", "log_loss"}
        }
    )
    metrics.update(
        {
            f"field_goal_test_{k}": v
            for k, v in classification_metrics(fg_labels, fg_p).items()
            if k in {"n", "auc", "brier"}
        }
    )
    return TrainingResult(model, conversion, metrics, calibration_table(y_test, p_test))
