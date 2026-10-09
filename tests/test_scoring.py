from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from gridiron import scoring
from gridiron.features import COMPONENTS
from gridiron.model import points
from tests.conftest import LANDING, REPO_ROOT


def load_data_access() -> ModuleType:
    """The app's data module, which is uploaded to Snowflake without the package."""
    path = REPO_ROOT / "snowflake" / "streamlit" / "data_access.py"
    spec = importlib.util.spec_from_file_location("data_access_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoring_is_the_one_list_of_stat_lines() -> None:
    assert tuple(stat for stat, _ in scoring.UNIT_VALUES) == scoring.STATS
    assert set(dict(scoring.UNIT_VALUES).values()) <= set(scoring.SETTINGS)
    # The model projects every scored stat except special teams touchdowns.
    assert tuple(s for s in scoring.STATS if s != "special_teams_tds") == COMPONENTS


def test_app_stat_lists_match_the_package() -> None:
    data_access = load_data_access()
    assert data_access.PROJECTED_STATS == COMPONENTS
    assert data_access.ACTUAL_STATS == scoring.STATS


def test_vectorized_points_score_every_stat_line_column() -> None:
    line = pd.DataFrame({stat: [1.0] for stat in scoring.STATS})
    expected = scoring.fantasy_points(dict.fromkeys(scoring.STATS, 1.0), "ppr")
    assert points(line, "ppr").iloc[0] == pytest.approx(expected)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict("records")]


@pytest.fixture(scope="module")
def stat_lines() -> pd.DataFrame:
    files = sorted((LANDING / "player_stats").rglob("*.parquet"))
    stats = pd.concat(pd.read_parquet(f) for f in files)
    stats = stats[stats["position_group"].isin(["QB", "RB", "WR", "TE"])].copy()
    stats["fumbles_lost"] = (
        stats["sack_fumbles_lost"] + stats["rushing_fumbles_lost"] + stats["receiving_fumbles_lost"]
    )
    stats["two_point_conversions"] = (
        stats["passing_2pt_conversions"]
        + stats["rushing_2pt_conversions"]
        + stats["receiving_2pt_conversions"]
    )
    return stats


def test_defaults_reproduce_nflverse_points(stat_lines: pd.DataFrame) -> None:
    for record in records(stat_lines):
        assert scoring.fantasy_points(record, "standard") == pytest.approx(
            record["fantasy_points"], abs=1e-6
        )
        assert scoring.fantasy_points(record, "ppr") == pytest.approx(
            record["fantasy_points_ppr"], abs=1e-6
        )


def test_vectorized_points_match_the_scalar_scorer(stat_lines: pd.DataFrame) -> None:
    lines = stat_lines.assign(special_teams_tds=0)
    for preset in ("ppr", "half", "standard"):
        expected = [scoring.fantasy_points(r, preset) for r in records(lines)]
        assert points(lines, preset).round(4).tolist() == pytest.approx(expected, abs=1e-6)


def test_presets_and_overrides() -> None:
    line = {"receptions": 6, "receiving_yards": 80, "receiving_tds": 1}
    assert scoring.fantasy_points(line, "standard") == 14.0
    assert scoring.fantasy_points(line, "half") == 17.0
    assert scoring.fantasy_points(line) == 20.0  # PPR is the default
    assert scoring.fantasy_points(line, {"preset": "half", "rec_td": 4}) == 15.0
    assert scoring.fantasy_points(line, {"rec": 1.5}) == 23.0  # overrides on top of PPR
    qb = {"passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1}
    assert scoring.fantasy_points(qb, {"pass_td": 6}) == 22.0


def test_bonuses_and_tight_end_premium() -> None:
    settings = {"bonus_rec_100": 3, "te_rec_premium": 0.5}
    tight_end = {"position": "TE", "receptions": 8, "receiving_yards": 104}
    receiver = {**tight_end, "position": "WR"}
    assert scoring.fantasy_points(tight_end, settings) == pytest.approx(8 + 10.4 + 3 + 4)
    assert scoring.fantasy_points(receiver, settings) == pytest.approx(8 + 10.4 + 3)
    assert scoring.fantasy_points({"receiving_yards": 99}, settings) == pytest.approx(9.9)


def test_bad_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scoring settings"):
        scoring.fantasy_points({}, {"passing_td": 6})
    with pytest.raises(ValueError, match="unknown preset"):
        scoring.fantasy_points({}, "superflex")


def test_udf_handler_null_in_null_out_and_tolerates_missing_values() -> None:
    assert scoring.udf_handler(None, None) is None
    assert scoring.udf_handler({"rushing_yards": None, "rushing_tds": 1}, None) == 6.0
    assert scoring.udf_handler({"receptions": "3"}, {"preset": "half"}) == 1.5
    assert scoring.fantasy_points({"receptions": float("nan"), "rushing_tds": 1}) == 6.0
