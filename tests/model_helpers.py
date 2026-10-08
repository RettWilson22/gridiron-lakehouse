"""Synthetic data with a known structure, for testing model fitting deterministically."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_decisions(n_per_season: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2019, 2025):
        ydstogo = rng.integers(1, 15, n_per_season)
        yardline = rng.integers(2, 95, n_per_season)
        decision = rng.choice(["go", "punt", "field_goal"], n_per_season, p=[0.4, 0.4, 0.2])
        p_convert = 1 / (1 + np.exp(-(1.2 - 0.25 * ydstogo)))
        converted = rng.random(n_per_season) < p_convert
        fg_made = rng.random(n_per_season) < 1 / (1 + np.exp(-(5.0 - 0.1 * (yardline + 17))))
        outcome = np.select(
            [decision == "go", decision == "field_goal"],
            [
                np.where(converted, "converted", "failed"),
                np.where(fg_made, "fg_made", "fg_missed"),
            ],
            "punt",
        )
        rows.append(
            pd.DataFrame(
                {
                    "season": season,
                    "season_type": "REG",
                    "posteam": rng.choice(["DET", "GB", "MIN"], n_per_season),
                    "coach": "Coach",
                    "ydstogo": ydstogo,
                    "yardline_100": yardline,
                    "score_differential": rng.integers(-14, 15, n_per_season),
                    "game_seconds_remaining": rng.integers(0, 3600, n_per_season),
                    "wp": rng.uniform(0.05, 0.95, n_per_season),
                    "decision": decision,
                    "outcome": outcome,
                    "converted": pd.Series(converted).where(decision == "go"),
                    "kick_distance": np.where(decision == "punt", 45, np.nan),
                    "return_yards": np.where(decision == "punt", 5, np.nan),
                    "touchback": 0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def synthetic_first_downs(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yardline = rng.integers(1, 100, 5000)
    ep = 6.0 - 0.07 * yardline + rng.normal(0, 0.5, 5000)
    return pd.DataFrame({"season": 2020, "yardline_100": yardline, "ep": ep})
