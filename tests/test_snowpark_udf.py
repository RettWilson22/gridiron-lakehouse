from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from gridiron.decision import DecisionModel

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "artifacts" / "decision_model.json"


def load_udf() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fourth_down_udf", ROOT / "snowflake" / "snowpark" / "fourth_down_udf.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_udf_matches_python_model() -> None:
    udf = load_udf()
    model = DecisionModel.load(MODEL)
    rng = np.random.default_rng(3)
    situations = pd.DataFrame(
        {
            "ydstogo": rng.integers(1, 20, 300),
            "yardline_100": rng.integers(1, 99, 300),
            "score_differential": rng.integers(-21, 22, 300),
            "game_seconds_remaining": rng.integers(0, 3600, 300),
        }
    )
    expected = model.option_values(situations)
    model_payload = udf.load_model(str(MODEL))
    for i in range(len(situations)):
        result = udf.recommend(*situations.iloc[i].tolist(), model=model_payload)
        want = expected.iloc[i]
        for column in ("p_convert", "p_field_goal"):
            assert result[column] == pytest.approx(want[column], abs=1e-4)
        for column in ("ev_go", "ev_punt", "ev_field_goal"):
            assert result[column] == pytest.approx(want[column], abs=1e-3)
        best = want[["ev_go", "ev_punt", "ev_field_goal"]].astype(float).idxmax()
        assert "ev_" + result["recommendation"] == best


def test_udf_handler_null_in_null_out() -> None:
    udf = load_udf()
    assert udf.udf_handler(None, 40, 0, 1800) is None
    assert udf.udf_handler(1, 40, 0, 1800)["recommendation"] == "go"
