"""Snowflake Python UDF: recommend go / punt / field goal for one fourth-down situation.

Standard library only, so it runs in a Snowflake Python UDF without packages. It reads the
exported ``decision_model.json`` (the same file the Databricks job writes) from the UDF's
import directory and reproduces ``gridiron.decision.DecisionModel.option_values``; a unit
test checks the two agree.
"""

from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache
from typing import Any

MODEL_FILE = "decision_model.json"
TOUCHDOWN_POINTS = 7.0
FIELD_GOAL_POINTS = 3.0
KICK_SNAP_DISTANCE = 17
KICKOFF_OPPONENT_YARDLINE = 70
MISSED_FG_MIN_OPPONENT_YARDLINE = 80


def _default_model_path() -> str:
    import_dir = sys._xoptions.get("snowflake_import_directory")
    if import_dir:
        return os.path.join(str(import_dir), MODEL_FILE)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "artifacts", MODEL_FILE)


@lru_cache(maxsize=4)
def load_model(path: str | None = None) -> dict[str, Any]:
    with open(path or _default_model_path()) as handle:
        model: dict[str, Any] = json.load(handle)
    return model


def _logistic(params: dict[str, Any], values: list[float]) -> float:
    z = params["intercept"]
    for x, mean, scale, coef in zip(
        values, params["means"], params["scales"], params["coefficients"], strict=True
    ):
        z += coef * (x - mean) / scale
    return 1.0 / (1.0 + math.exp(-z))


def _ep(model: dict[str, Any], yardline_100: float) -> float:
    index = min(max(round(yardline_100), 1), 99) - 1
    return float(model["ep_curve"][index])


def recommend(
    ydstogo: float,
    yardline_100: float,
    score_differential: float,
    game_seconds_remaining: float,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    m = model or load_model()
    togo = max(float(ydstogo), 1.0)
    yl = float(yardline_100)
    features = {
        "log_ydstogo": math.log1p(togo),
        "yardline_100": yl,
        "goal_to_go": 1.0 if togo >= yl else 0.0,
        "score_differential": float(score_differential),
        "game_minutes_remaining": float(game_seconds_remaining) / 60.0,
    }
    conversion = m["conversion"]
    p_convert = _logistic(conversion, [features[name] for name in conversion["features"]])
    after_score = -_ep(m, KICKOFF_OPPONENT_YARDLINE)

    # Mirrors DecisionModel.option_values, which uses the raw (unclipped) ydstogo here.
    raw_togo = float(ydstogo)
    if raw_togo >= yl:
        success = TOUCHDOWN_POINTS + after_score
    else:
        success = _ep(m, max(yl - raw_togo, 1.0))
    ev_go = p_convert * success + (1 - p_convert) * -_ep(m, 100 - yl)

    p_fg = _logistic(m["fg_make"], [yl + KICK_SNAP_DISTANCE])
    miss_spot = min(100 - (yl + 7), MISSED_FG_MIN_OPPONENT_YARDLINE)
    ev_fg = p_fg * (FIELD_GOAL_POINTS + after_score) + (1 - p_fg) * -_ep(m, miss_spot)

    punt_index = min(max(round(yl), 1), 99) - 1
    ev_punt = -_ep(m, m["punt_curve"][punt_index])

    values = {"go": ev_go, "punt": ev_punt, "field_goal": ev_fg}
    best = max(values, key=lambda option: values[option])
    return {
        "recommendation": best,
        "p_convert": round(p_convert, 4),
        "p_field_goal": round(p_fg, 4),
        "ev_go": round(ev_go, 3),
        "ev_punt": round(ev_punt, 3),
        "ev_field_goal": round(ev_fg, 3),
        "go_advantage": round(ev_go - max(ev_punt, ev_fg), 3),
    }


def udf_handler(
    ydstogo: float | None,
    yardline_100: float | None,
    score_differential: float | None,
    game_seconds_remaining: float | None,
) -> dict[str, Any] | None:
    """Entry point registered as the UDF handler; NULL in, NULL out."""
    if (
        ydstogo is None
        or yardline_100 is None
        or score_differential is None
        or game_seconds_remaining is None
    ):
        return None
    return recommend(
        float(ydstogo),
        float(yardline_100),
        float(score_differential),
        float(game_seconds_remaining),
    )
