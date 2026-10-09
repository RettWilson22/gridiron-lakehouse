"""Fantasy scoring for any league: score a stat line under standard, half PPR, PPR or custom
settings.

Standard library only. The same file is used in three places:

* the Databricks score task (projected stat lines -> points in each format);
* the Snowflake Python UDF ``APP.FANTASY_POINTS(stats OBJECT, scoring OBJECT)``
  (``udf_handler``), uploaded to a stage as-is;
* the Streamlit app's local and public modes, for custom scoring without Snowflake.

The default settings reproduce nflverse ``fantasy_points`` (standard) and
``fantasy_points_ppr`` exactly; a test checks this against real weekly stats.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

# Stat names: the same as the gold ``player_week`` columns.
STATS: Final = (
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
    "two_point_conversions",
    "special_teams_tds",
)

# Points per unit of each stat, plus yardage bonuses and a tight-end reception premium.
STANDARD: Final[dict[str, float]] = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 0.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
    "two_pt": 2.0,
    "st_td": 6.0,
    "te_rec_premium": 0.0,
    "bonus_pass_300": 0.0,
    "bonus_pass_400": 0.0,
    "bonus_rush_100": 0.0,
    "bonus_rush_200": 0.0,
    "bonus_rec_100": 0.0,
    "bonus_rec_200": 0.0,
}
PRESETS: Final[dict[str, dict[str, float]]] = {
    "standard": STANDARD,
    "half": {**STANDARD, "rec": 0.5},
    "ppr": {**STANDARD, "rec": 1.0},
}
SETTINGS: Final = tuple(STANDARD)

_UNIT_VALUES: Final = (
    ("passing_yards", "pass_yd"),
    ("passing_tds", "pass_td"),
    ("passing_interceptions", "pass_int"),
    ("rushing_yards", "rush_yd"),
    ("rushing_tds", "rush_td"),
    ("receptions", "rec"),
    ("receiving_yards", "rec_yd"),
    ("receiving_tds", "rec_td"),
    ("fumbles_lost", "fumble_lost"),
    ("two_point_conversions", "two_pt"),
    ("special_teams_tds", "st_td"),
)
_BONUSES: Final = (
    ("passing_yards", 300, "bonus_pass_300"),
    ("passing_yards", 400, "bonus_pass_400"),
    ("rushing_yards", 100, "bonus_rush_100"),
    ("rushing_yards", 200, "bonus_rush_200"),
    ("receiving_yards", 100, "bonus_rec_100"),
    ("receiving_yards", 200, "bonus_rec_200"),
)


def resolve(scoring: Mapping[str, Any] | str | None = None) -> dict[str, float]:
    """Full settings from a preset name, or from overrides on top of an optional preset.

    ``{"preset": "half", "pass_td": 6}`` is half PPR with six-point passing touchdowns;
    without a preset, overrides apply to full PPR (also the default for no settings).
    Unknown setting names raise ``ValueError`` so a typo cannot silently score zero.
    """
    if scoring is None:
        return dict(PRESETS["ppr"])
    if isinstance(scoring, str):
        scoring = {"preset": scoring}
    preset = str(scoring.get("preset") or "ppr").lower()
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    unknown = set(scoring) - set(SETTINGS) - {"preset"}
    if unknown:
        raise ValueError(f"unknown scoring settings: {sorted(unknown)}")
    settings = dict(PRESETS[preset])
    for name, value in scoring.items():
        if name != "preset" and value is not None:
            settings[name] = float(value)
    return settings


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def fantasy_points(
    stats: Mapping[str, Any], scoring: Mapping[str, Any] | str | None = None
) -> float:
    """Points for one stat line. Missing stats count as zero.

    Yardage bonuses are all-or-nothing thresholds, so applying them to a projected (mean)
    stat line understates their expected value; the app says so where it matters.
    """
    settings = resolve(scoring)
    points = sum(_number(stats.get(stat)) * settings[key] for stat, key in _UNIT_VALUES)
    for stat, threshold, key in _BONUSES:
        if _number(stats.get(stat)) >= threshold:
            points += settings[key]
    if str(stats.get("position") or "").upper() == "TE":
        points += _number(stats.get("receptions")) * settings["te_rec_premium"]
    return round(points, 4)


def udf_handler(stats: Mapping[str, Any] | None, scoring: Mapping[str, Any] | None) -> float | None:
    """Entry point of the Snowflake UDF; a NULL stat line scores NULL."""
    if stats is None:
        return None
    return fantasy_points(stats, scoring)
