from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron.projections import (
    PROJECTION_COLUMNS,
    add_rankings,
    assemble,
    merge_live,
    risers,
    upcoming_week,
)

NOW = pd.Timestamp("2026-10-11 16:00", tz="UTC")  # between Thursday's and Sunday's games


def live_rows(values: dict[str, float], generated: str) -> pd.DataFrame:
    kickoffs = {
        "thursday": pd.Timestamp("2026-10-09 00:15", tz="UTC"),
        "sunday": pd.Timestamp("2026-10-11 17:00", tz="UTC"),
    }
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 5,
                "player_id": player,
                "position": "WR",
                "kickoff_at": kickoffs["thursday" if player.startswith("thu") else "sunday"],
                "proj_ppr": value,
                "generated_at": pd.Timestamp(generated, tz="UTC"),
            }
            for player, value in values.items()
        ]
    )


def test_live_projections_freeze_at_kickoff() -> None:
    tuesday = live_rows({"thu1": 10.0, "sun1": 12.0, "sun2": 8.0}, "2026-10-06")
    stored = merge_live(None, tuesday, pd.Timestamp("2026-10-06 12:00", tz="UTC"))
    assert len(stored) == 3

    saturday = live_rows({"thu1": 99.0, "sun1": 13.0, "sun3": 5.0}, "2026-10-10")
    merged = merge_live(stored, saturday, NOW).set_index("player_id")
    assert merged.loc["thu1", "proj_ppr"] == 10.0  # kicked off: Tuesday's value is kept
    assert merged.loc["sun1", "proj_ppr"] == 13.0  # not started: refreshed
    assert "sun3" in merged.index  # newly listed for a game that has not started
    assert "sun2" not in merged.index  # dropped from the pool before kickoff
    assert merged.index.is_unique


def test_a_projection_is_never_created_after_kickoff() -> None:
    late = live_rows({"thu9": 30.0}, "2026-10-10")
    assert merge_live(None, late, NOW).empty


def test_earlier_live_weeks_are_kept() -> None:
    week4 = live_rows({"sun1": 11.0}, "2026-10-01").assign(week=4)
    week5 = live_rows({"sun1": 12.0}, "2026-10-10")
    merged = merge_live(week4, week5, pd.Timestamp("2026-10-10", tz="UTC"))
    assert sorted(merged["week"]) == [4, 5]


def ranked_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for position, count in (("QB", 30), ("WR", 90)):
        for i in range(count):
            ppr = float(rng.uniform(0, 25))
            rows.append(
                {
                    "season": 2026,
                    "week": 5,
                    "player_id": f"{position}{i}",
                    "position": position,
                    "proj_ppr": ppr,
                    "proj_half": ppr * 0.9,
                    "proj_std": ppr * 0.8 if position == "WR" else ppr,
                }
            )
    return pd.DataFrame(rows)


def test_rankings_tiers_and_labels_per_format() -> None:
    ranked = add_rankings(ranked_frame())
    for fmt in ("ppr", "half", "std"):
        for _, group in ranked.groupby("position"):
            ordered = group.sort_values(f"pos_rank_{fmt}")
            assert ordered[f"pos_rank_{fmt}"].tolist() == list(range(1, len(group) + 1))
            assert ordered[f"proj_{fmt}"].is_monotonic_decreasing
            tiers = ordered[f"tier_{fmt}"].dropna()
            assert tiers.is_monotonic_increasing and tiers.iloc[0] == 1
    qbs = ranked[ranked["position"] == "QB"]
    assert qbs["tier_ppr"].isna().sum() == 30 - 24  # tiers only within the top 24 QBs
    assert set(qbs.loc[qbs["pos_rank_ppr"] <= 12, "start_sit_ppr"]) == {"Start"}


def test_assemble_prefers_live_weeks_and_attaches_expert_ranks() -> None:
    frame = ranked_frame()
    for column in PROJECTION_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    backtest = frame.assign(kind="backtest", ecr_rank=np.nan)
    live = frame[frame["position"] == "QB"].assign(kind="live", proj_ppr=1.0, ecr_rank=np.nan)
    ecr = pd.DataFrame(
        {"season": [2026], "week": [5], "player_id": ["QB0"], "position": ["QB"], "ecr_rank": [3]}
    )
    combined = assemble(backtest, live, ecr)
    assert set(combined["kind"]) == {"live"}  # a live week uses only live rows
    assert combined.set_index("player_id").loc["QB0", "ecr_rank"] == 3


def test_upcoming_week_is_the_first_unplayed_week() -> None:
    games = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [4, 5, 5, 6],
            "is_final": [True, True, False, False],
        }
    )
    assert upcoming_week(games) == (2026, 5)
    assert upcoming_week(games.assign(is_final=True)) is None


def test_upcoming_week_ignores_unfinished_games_from_earlier_seasons() -> None:
    # A game that never got a final score (cancelled, say) must not pin the upcoming
    # week to a past season.
    stray = pd.DataFrame({"season": [2024], "week": [17], "is_final": [False]})
    games = pd.DataFrame(
        {"season": [2026, 2026, 2026], "week": [4, 5, 5], "is_final": [True, True, False]}
    )
    assert upcoming_week(pd.concat([stray, games])) == (2026, 5)
    assert upcoming_week(pd.concat([stray, games.assign(is_final=True)])) is None


def test_risers_compare_the_last_three_games_with_earlier_ones() -> None:
    games = []
    for week, xfp in enumerate([4.0, 5.0, 4.0, 12.0, 13.0, 14.0], start=1):
        games.append(
            {
                "season": 2026,
                "week": week,
                "player_id": "p1",
                "player_name": "Riser",
                "position": "WR",
                "team": "DET",
                "snap_share": 0.5 if week < 4 else 0.9,
                "target_share": 0.1,
                "carry_share": 0.0,
                "red_zone_share": 0.0,
                "expected_ppr": xfp,
            }
        )
    frame = risers(pd.DataFrame(games), [(2026, 7)]).iloc[0]
    assert frame["is_riser"]
    assert frame["expected_ppr_change"] == pytest.approx(13.0 - 13.0 / 3)
    assert frame["snap_share_change"] == pytest.approx(0.4)
    assert frame["baseline_basis"] == "earlier games this season"
