from __future__ import annotations

import math

import pandas as pd
import pytest

from gridiron.features import (
    FEATURE_COLUMNS,
    build_features,
    conversion_training_frame,
    split_by_season,
)


def situation(**overrides: float) -> pd.DataFrame:
    row: dict[str, float] = {
        "ydstogo": 3,
        "yardline_100": 40,
        "score_differential": -4,
        "game_seconds_remaining": 600,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_build_features_values() -> None:
    features = build_features(situation()).iloc[0]
    assert list(features.index) == list(FEATURE_COLUMNS)
    assert features["log_ydstogo"] == pytest.approx(math.log1p(3))
    assert features["goal_to_go"] == 0.0
    assert features["game_minutes_remaining"] == 10.0


def test_goal_to_go_and_zero_distance_clipping() -> None:
    features = build_features(situation(ydstogo=0, yardline_100=1)).iloc[0]
    assert features["goal_to_go"] == 1.0
    assert features["log_ydstogo"] == pytest.approx(math.log1p(1))


def test_missing_columns_raise() -> None:
    with pytest.raises(KeyError, match="score_differential"):
        build_features(situation().drop(columns="score_differential"))


def test_training_frame_uses_only_attempts(decisions_pdf: pd.DataFrame) -> None:
    features, labels = conversion_training_frame(decisions_pdf)
    attempts = decisions_pdf[decisions_pdf["decision"] == "go"]
    assert len(features) == len(labels) == len(attempts)
    assert labels.sum() == (attempts["outcome"] == "converted").sum()
    assert not features.isna().any().any()


def test_split_by_season_is_temporal(decisions_pdf: pd.DataFrame) -> None:
    train, test = split_by_season(decisions_pdf, last_train_season=2023)
    assert set(train["season"]) == {2023}
    assert set(test["season"]) == {2024}
    _, none = split_by_season(decisions_pdf, 2023, test_seasons=(2025,))
    assert none.empty
