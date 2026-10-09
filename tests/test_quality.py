"""Shape checks on third-party names, teams and positions (silver WARN expectations)."""

from __future__ import annotations

import pandas as pd
from pyspark.sql import SparkSession

from gridiron import quality
from gridiron.spark_io import to_spark

GOOD_NAMES: list[str | None] = [
    "Amon-Ra St. Brown",
    "Ja'Marr Chase",
    "Marvin Harrison Jr.",
    "José Núñez",
    "Dražen Petrović",
    "x" * 64,
]
BAD_NAMES: list[str | None] = [
    '<iframe srcdoc="<script>alert(1)</script>">',
    "[click](https://example.test)",
    "Kenneth Murray, Jr.",  # a real box-score name; counted, not dropped
    "Player 2",
    "x" * 65,
    "",
    None,
]


def flags(spark: SparkSession, frame: pd.DataFrame, rule: str) -> list[bool]:
    checked = to_spark(spark, frame).selectExpr(f"{rule} AS ok").toPandas()
    return [bool(v) for v in checked["ok"]]


def test_player_names_must_be_plain_text(spark: SparkSession) -> None:
    names = pd.DataFrame({"player_name": GOOD_NAMES + BAD_NAMES})
    expected = [True] * len(GOOD_NAMES) + [False] * len(BAD_NAMES)
    assert flags(spark, names, quality.PLAIN_PLAYER_NAME) == expected


def test_teams_must_be_nflverse_codes(spark: SparkSession) -> None:
    games = pd.DataFrame(
        {"team": ["DET", "JAC", None, "LV"], "opponent": ["GB", "DET", "DET", "OAK"]}
    )
    rule = quality.known_teams("team", "opponent")
    assert flags(spark, games, rule) == [True, False, False, False]


def test_positions_must_be_fantasy_positions(spark: SparkSession) -> None:
    rows = pd.DataFrame({"position": ["QB", "RB", "WR", "TE", "K", "FB", None]})
    assert flags(spark, rows, quality.FANTASY_POSITION) == [True] * 4 + [False] * 3
