"""Fourth-down decision model: go for it vs. punt vs. field goal, in expected points.

Components, all fitted from the play-by-play itself:

* ``conversion``  - logistic regression, P(convert | situation), the tracked ML model;
* ``fg_make``     - logistic regression on kick distance, P(field goal is good);
* ``ep_curve``    - expected points of a first down at each yardline (1..99), from the
  nflfastR ``ep`` field averaged over first-down snaps and lightly smoothed;
* ``punt_curve``  - expected opponent field position after a punt from each yardline.

Each option is valued as the expected points of the resulting possession state, from the
offense's point of view. This is an expected-points framework, not a win-probability one, so
it ignores clock and score leverage; the aggressiveness metrics therefore only use
"neutral" situations (see ``is_neutral``).

The whole model serialises to a small JSON document so the same arithmetic can run in a
Snowflake Python UDF and in the Streamlit app without scikit-learn.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridiron.features import FEATURE_COLUMNS, build_features

# Simplifying constants. Kept explicit so they are easy to criticise and change.
TOUCHDOWN_POINTS: Final = 7.0  # touchdown plus a near-automatic extra point
FIELD_GOAL_POINTS: Final = 3.0
KICK_SNAP_DISTANCE: Final = 17  # line of scrimmage to kick: 10 (end zone) + 7 (snap/hold)
KICKOFF_OPPONENT_YARDLINE: Final = 70  # opponent starts around its own 30 after a score
MISSED_FG_MIN_OPPONENT_YARDLINE: Final = 80  # missed kick inside the 20 returns to the 20
PUNT_TOUCHBACK_OPPONENT_YARDLINE: Final = 80
NEUTRAL_WP: Final = (0.10, 0.90)
NEUTRAL_MIN_SECONDS: Final = 300
OPTIONS: Final = ("go", "punt", "field_goal")


@dataclass(frozen=True)
class LogisticParams:
    """Standardised logistic regression: p = sigmoid(b + sum(w_i * (x_i - mu_i) / s_i))."""

    features: list[str]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - np.asarray(self.means)) / np.asarray(self.scales)
        logit = self.intercept + z @ np.asarray(self.coefficients)
        return np.asarray(1.0 / (1.0 + np.exp(-logit)))

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, features: list[str]) -> LogisticParams:
        scaler: StandardScaler = pipeline.named_steps["scale"]
        model: LogisticRegression = pipeline.named_steps["model"]
        return cls(
            features=features,
            means=[float(v) for v in scaler.mean_],
            scales=[float(v) for v in scaler.scale_],
            coefficients=[float(v) for v in model.coef_[0]],
            intercept=float(model.intercept_[0]),
        )


@dataclass(frozen=True)
class DecisionModel:
    conversion: LogisticParams
    fg_make: LogisticParams
    ep_curve: list[float]  # index i -> EP of 1st down at yardline_100 = i + 1
    punt_curve: list[float]  # index i -> expected opponent yardline_100 after punt from i + 1
    trained_through_season: int

    # -- serialisation -------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecisionModel:
        return cls(
            conversion=LogisticParams(**payload["conversion"]),
            fg_make=LogisticParams(**payload["fg_make"]),
            ep_curve=[float(v) for v in payload["ep_curve"]],
            punt_curve=[float(v) for v in payload["punt_curve"]],
            trained_through_season=int(payload["trained_through_season"]),
        )

    @classmethod
    def load(cls, path: Path) -> DecisionModel:
        return cls.from_dict(json.loads(path.read_text()))

    # -- valuation -----------------------------------------------------------------------
    def ep_first_down(self, yardline_100: np.ndarray) -> np.ndarray:
        idx = np.clip(np.rint(yardline_100).astype(int), 1, 99) - 1
        return np.asarray(np.asarray(self.ep_curve)[idx], dtype="float64")

    def conversion_probability(self, situations: pd.DataFrame) -> np.ndarray:
        x = build_features(situations)[self.conversion.features].to_numpy(dtype="float64")
        return self.conversion.predict(x)

    def fg_probability(self, yardline_100: np.ndarray) -> np.ndarray:
        distance = (yardline_100 + KICK_SNAP_DISTANCE).astype("float64").reshape(-1, 1)
        return self.fg_make.predict(distance)

    def option_values(self, situations: pd.DataFrame) -> pd.DataFrame:
        """Expected points of each option for every situation row."""
        yl = situations["yardline_100"].to_numpy(dtype="float64")
        togo = situations["ydstogo"].to_numpy(dtype="float64")
        opp_after_score = -self.ep_first_down(np.full_like(yl, KICKOFF_OPPONENT_YARDLINE))

        p_convert = self.conversion_probability(situations)
        is_goal_to_go = togo >= yl
        success = np.where(
            is_goal_to_go,
            TOUCHDOWN_POINTS + opp_after_score,
            self.ep_first_down(np.maximum(yl - togo, 1)),
        )
        failure = -self.ep_first_down(100 - yl)
        go = p_convert * success + (1 - p_convert) * failure

        p_fg = self.fg_probability(yl)
        miss_spot = np.minimum(100 - (yl + 7), MISSED_FG_MIN_OPPONENT_YARDLINE)
        field_goal = p_fg * (FIELD_GOAL_POINTS + opp_after_score) + (1 - p_fg) * (
            -self.ep_first_down(miss_spot)
        )

        punt_idx = np.clip(np.rint(yl).astype(int), 1, 99) - 1
        punt = -self.ep_first_down(np.asarray(self.punt_curve)[punt_idx])

        return pd.DataFrame(
            {
                "p_convert": p_convert,
                "p_field_goal": p_fg,
                "ev_go": go,
                "ev_punt": punt,
                "ev_field_goal": field_goal,
            },
            index=situations.index,
        )


def is_neutral(frame: pd.DataFrame) -> pd.Series:
    """Situations where an expected-points comparison is a fair proxy for winning."""
    low, high = NEUTRAL_WP
    return (
        frame["wp"].between(low, high) & (frame["game_seconds_remaining"] >= NEUTRAL_MIN_SECONDS)
    ).fillna(False)


def score_decisions(decisions: pd.DataFrame, model: DecisionModel) -> pd.DataFrame:
    """Attach option values, the recommendation and the cost of the actual call to each row."""
    values = model.option_values(decisions)
    ev = values[["ev_go", "ev_punt", "ev_field_goal"]].to_numpy()
    best_idx = ev.argmax(axis=1)
    chosen_idx = decisions["decision"].map({o: i for i, o in enumerate(OPTIONS)}).to_numpy()
    best_ev = ev.max(axis=1)
    kick_ev = np.maximum(values["ev_punt"].to_numpy(), values["ev_field_goal"].to_numpy())
    out = decisions.copy()
    for col in values.columns:
        out[col] = values[col].to_numpy()
    out["recommendation"] = np.asarray(OPTIONS)[best_idx]
    out["go_advantage"] = values["ev_go"].to_numpy() - kick_ev
    out["expected_points_lost"] = best_ev - ev[np.arange(len(ev)), chosen_idx]
    out["followed_model"] = out["recommendation"] == out["decision"]
    out["is_neutral_situation"] = is_neutral(out).to_numpy()
    return out


def coach_aggressiveness(scored: pd.DataFrame, min_go_recommendations: int = 1) -> pd.DataFrame:
    """Team-season aggressiveness: how often the team went for it when the model said go.

    Restricted to regular-season, neutral situations.
    """
    df = scored[(scored["season_type"] == "REG") & scored["is_neutral_situation"]].copy()
    df["went_for_it"] = df["decision"] == "go"
    df["model_says_go"] = df["recommendation"] == "go"
    df["go_when_recommended"] = df["went_for_it"] & df["model_says_go"]
    # How much a team left on the table on the plays where it kicked and the model said go.
    df["missed_go_advantage"] = df["go_advantage"].where(df["model_says_go"] & ~df["went_for_it"])
    grouped = df.groupby(["season", "posteam"], as_index=False).agg(
        coach=("coach", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        neutral_fourth_downs=("decision", "size"),
        go_attempts=("went_for_it", "sum"),
        go_recommendations=("model_says_go", "sum"),
        went_for_it_when_recommended=("go_when_recommended", "sum"),
        expected_points_lost=("expected_points_lost", "sum"),
        avg_missed_go_advantage=("missed_go_advantage", "mean"),
    )
    grouped = grouped.rename(columns={"posteam": "team"})
    grouped = grouped[grouped["go_recommendations"] >= min_go_recommendations]
    grouped["go_rate"] = grouped["go_attempts"] / grouped["neutral_fourth_downs"]
    grouped["go_rate_when_recommended"] = (
        grouped["went_for_it_when_recommended"] / grouped["go_recommendations"]
    )
    grouped["expected_points_lost_per_fourth_down"] = (
        grouped["expected_points_lost"] / grouped["neutral_fourth_downs"]
    )
    grouped["aggressiveness_rank"] = (
        grouped.groupby("season")["go_rate_when_recommended"]
        .rank(ascending=False, method="min")
        .astype("int64")
    )
    int_cols = ["neutral_fourth_downs", "go_attempts", "go_recommendations"]
    grouped[[*int_cols, "went_for_it_when_recommended"]] = grouped[
        [*int_cols, "went_for_it_when_recommended"]
    ].astype("int64")
    return grouped.sort_values(["season", "aggressiveness_rank"]).reset_index(drop=True)


__all__ = [
    "FEATURE_COLUMNS",
    "DecisionModel",
    "LogisticParams",
    "coach_aggressiveness",
    "is_neutral",
    "score_decisions",
]
