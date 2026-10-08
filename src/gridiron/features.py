"""Feature preparation for the fourth-down conversion model (pandas, no Spark needed)."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

# Raw situation columns the model consumes; these are what a caller must provide.
SITUATION_COLUMNS: Final = (
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
)

# Model features derived from the situation columns, in model order.
FEATURE_COLUMNS: Final = (
    "log_ydstogo",
    "yardline_100",
    "goal_to_go",
    "score_differential",
    "game_minutes_remaining",
)

TARGET_COLUMN: Final = "converted"


def build_features(situations: pd.DataFrame) -> pd.DataFrame:
    """Turn raw game situations into model features.

    ``log1p(ydstogo)`` captures the strongly diminishing effect of each extra yard to go;
    ``goal_to_go`` marks plays where a conversion is a touchdown (compressed field).
    """
    missing = [c for c in SITUATION_COLUMNS if c not in situations.columns]
    if missing:
        raise KeyError(f"missing situation columns: {missing}")
    ydstogo = situations["ydstogo"].astype("float64").clip(lower=1.0)
    yardline = situations["yardline_100"].astype("float64")
    return pd.DataFrame(
        {
            "log_ydstogo": np.log1p(ydstogo),
            "yardline_100": yardline,
            "goal_to_go": (ydstogo >= yardline).astype("float64"),
            "score_differential": situations["score_differential"].astype("float64"),
            "game_minutes_remaining": situations["game_seconds_remaining"].astype("float64") / 60.0,
        },
        index=situations.index,
    )[list(FEATURE_COLUMNS)]


def conversion_training_frame(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels from fourth-down attempts (``decision == 'go'``) with known context.

    Only attempts are observed, so the model learns conversion odds conditional on a coach
    choosing to go. That selection effect is a known limitation and is documented in the
    README.
    """
    attempts = decisions[decisions["decision"] == "go"].dropna(
        subset=[*SITUATION_COLUMNS, TARGET_COLUMN]
    )
    features = build_features(attempts)
    labels = attempts[TARGET_COLUMN].astype("int64")
    return features, labels


def split_by_season(
    frame: pd.DataFrame, last_train_season: int, test_seasons: tuple[int, ...] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: train on seasons <= ``last_train_season``, test on later seasons.

    ``test_seasons`` optionally restricts the test set (for example to completed seasons).
    """
    train = frame[frame["season"] <= last_train_season]
    test = frame[frame["season"] > last_train_season]
    if test_seasons is not None:
        test = test[test["season"].isin(test_seasons)]
    return train, test
