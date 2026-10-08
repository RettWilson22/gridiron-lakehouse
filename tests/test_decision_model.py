from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridiron.decision import (
    DecisionModel,
    LogisticParams,
    coach_aggressiveness,
    is_neutral,
    score_decisions,
)
from gridiron.features import FEATURE_COLUMNS, conversion_training_frame
from gridiron.training import (
    calibration_table,
    fit_conversion,
    fit_ep_curve,
    fit_punt_curve,
    punt_opponent_yardline,
    train_decision_model,
)
from tests.model_helpers import synthetic_decisions, synthetic_first_downs

ARTIFACT = Path(__file__).parent.parent / "artifacts" / "decision_model.json"


@pytest.fixture(scope="module")
def trained() -> DecisionModel:
    result = train_decision_model(synthetic_decisions(), synthetic_first_downs(), 2022)
    return result.model


def test_training_reports_held_out_metrics() -> None:
    result = train_decision_model(synthetic_decisions(), synthetic_first_downs(), 2022)
    m = result.metrics
    assert m["conversion_test_n"] > 0
    assert m["conversion_test_auc"] > 0.6  # synthetic data has a real ydstogo signal
    assert 0 < m["conversion_test_brier"] < 0.25
    assert result.calibration["n"].sum() == m["conversion_test_n"]


def test_training_requires_a_holdout() -> None:
    with pytest.raises(ValueError, match="held-out"):
        train_decision_model(synthetic_decisions(), synthetic_first_downs(), 2030)


def test_exported_params_reproduce_sklearn_predictions() -> None:
    features, labels = conversion_training_frame(synthetic_decisions())
    pipeline = fit_conversion(features, labels)
    params = LogisticParams.from_pipeline(pipeline, list(FEATURE_COLUMNS))
    expected = pipeline.predict_proba(features[list(FEATURE_COLUMNS)])[:, 1]
    np.testing.assert_allclose(params.predict(features.to_numpy()), expected, rtol=1e-10)


def test_json_round_trip(trained: DecisionModel, tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(trained.to_json())
    assert DecisionModel.load(path) == trained


def test_ep_curve_is_complete_and_decreasing() -> None:
    curve = fit_ep_curve(synthetic_first_downs())
    assert len(curve) == 99
    assert curve[0] > curve[49] > curve[98]


def test_punt_opponent_yardline_handles_touchbacks_and_blocks() -> None:
    punts = pd.DataFrame(
        {
            "yardline_100": [60, 45, 70],
            "kick_distance": [45, 40, np.nan],
            "return_yards": [5, 0, 0],
            "touchback": [0, 1, 0],
            "outcome": ["punt", "punt", "punt_blocked"],
        }
    )
    result = punt_opponent_yardline(punts)
    assert result.tolist() == [80.0, 80.0]  # 60 - 45 + 5 = own 20 -> 80; touchback -> 80
    curve = fit_punt_curve(punts.assign(decision="punt"))
    assert len(curve) == 99


@pytest.mark.skipif(not ARTIFACT.exists(), reason="exported model not built")
def test_real_model_makes_football_sense() -> None:
    model = DecisionModel.load(ARTIFACT)
    situations = pd.DataFrame(
        {
            "ydstogo": [1, 15, 2, 20],
            "yardline_100": [40, 80, 1, 30],
            "score_differential": [0, 0, 0, 0],
            "game_seconds_remaining": [1800, 1800, 1800, 1800],
        }
    )
    values = model.option_values(situations)
    best = [str(v) for v in values[["ev_go", "ev_punt", "ev_field_goal"]].idxmax(axis=1)]
    # 4th-and-1 at the 40: go. 4th-and-15 at own 20: punt.
    # 4th-and-goal at the 1: go. 4th-and-20 at the 30: field goal.
    assert best == ["ev_go", "ev_punt", "ev_go", "ev_field_goal"]
    assert (values["p_convert"].diff().iloc[1] < 0) and values["p_field_goal"].iloc[3] > 0.6


def test_score_decisions(trained: DecisionModel) -> None:
    decisions = synthetic_decisions(n_per_season=50)
    scored = score_decisions(decisions, trained)
    assert len(scored) == len(decisions)
    assert (scored["expected_points_lost"] >= -1e-12).all()
    followed = scored[scored["followed_model"]]
    assert np.allclose(followed["expected_points_lost"], 0.0)
    assert set(scored["recommendation"]) <= {"go", "punt", "field_goal"}


def test_neutral_filter() -> None:
    frame = pd.DataFrame(
        {"wp": [0.5, 0.05, 0.5, None], "game_seconds_remaining": [900, 900, 60, 900]}
    )
    assert is_neutral(frame).tolist() == [True, False, False, False]


def test_coach_aggressiveness(trained: DecisionModel) -> None:
    scored = score_decisions(synthetic_decisions(), trained)
    table = coach_aggressiveness(scored)
    assert table[["season", "team"]].duplicated().sum() == 0
    assert table["go_rate_when_recommended"].between(0, 1).all()
    assert (table["went_for_it_when_recommended"] <= table["go_recommendations"]).all()
    neutral = scored[scored["is_neutral_situation"]]
    assert table["neutral_fourth_downs"].sum() == len(neutral)
    assert table.groupby("season")["aggressiveness_rank"].min().eq(1).all()


def test_calibration_table_bins() -> None:
    labels = pd.Series([0, 1] * 50)
    table = calibration_table(labels, np.linspace(0, 1, 100), bins=5)
    assert len(table) == 5 and table["n"].sum() == 100
