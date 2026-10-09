"""Lakeflow Declarative Pipeline: bronze -> silver -> gold for fantasy football.

This file only declares datasets; all logic lives in the ``gridiron`` package and is unit
tested locally (``gridiron.lakehouse`` builds the same graph with plain Spark). Pipeline
configuration (set in ``resources/pipeline.yml``):

* ``gridiron.landing_root`` - volume directory with one folder per landed dataset;
* ``gridiron.first_season`` - earliest season kept in silver.

Dataset choices:

* Bronze tables are streaming tables fed by Auto Loader, so each landed file is read
  exactly once and the bronze history is append-only.
* Silver tables are materialized views: "keep the newest landed file per season" needs a
  window over all of bronze, which a materialized view recomputes as needed. The largest
  (play-by-play) is about 400k rows.
* Gold tables are materialized views over silver.
"""

from __future__ import annotations

from collections.abc import Callable

from pyspark import pipelines as dp
from pyspark.sql import DataFrame, SparkSession

from gridiron import quality, transforms
from gridiron.config import FIRST_SEASON
from gridiron.lakehouse import BRONZE_TABLES

spark = SparkSession.active()
LANDING_ROOT = spark.conf.get("gridiron.landing_root")
SEASON_FLOOR = int(spark.conf.get("gridiron.first_season") or FIRST_SEASON)


def _declare_bronze(dataset: str, table: str) -> None:
    @dp.table(
        name=table,
        comment=f"Raw {dataset} files landed by the ingest task, one row per row per file.",
        table_properties={"quality": "bronze"},
    )
    def bronze() -> DataFrame:
        raw = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "parquet")
            .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
            .load(f"{LANDING_ROOT}/{dataset}")
        )
        return transforms.with_ingest_metadata(raw)


for _dataset, _table in BRONZE_TABLES.items():
    _declare_bronze(_dataset, _table)


def _rules(name: str) -> Callable[[Callable[[], DataFrame]], Callable[[], DataFrame]]:
    """Apply a dataset's drop and warn expectations (when it has any)."""

    def decorate(fn: Callable[[], DataFrame]) -> Callable[[], DataFrame]:
        if quality.WARN.get(name):
            fn = dp.expect_all(quality.WARN[name])(fn)
        if quality.DROP.get(name):
            fn = dp.expect_all_or_drop(quality.DROP[name])(fn)
        return fn

    return decorate


def _read(name: str) -> DataFrame:
    return spark.read.table(name)


# Silver --------------------------------------------------------------------------------


@dp.materialized_view(
    name="plays",
    comment="Play-by-play columns used for red-zone and team volume; newest file per season.",
    table_properties={"quality": "silver"},
    cluster_by=["season", "game_id"],
)
@_rules("plays")
def plays() -> DataFrame:
    return transforms.silver_plays(_read("bronze_plays"), SEASON_FLOOR)


@dp.materialized_view(
    name="player_stats",
    comment="Weekly player box scores (nflverse stats_player_week).",
    table_properties={"quality": "silver"},
)
@_rules("player_stats")
def player_stats() -> DataFrame:
    return transforms.silver_player_stats(_read("bronze_player_stats"))


@dp.materialized_view(
    name="players",
    comment="nflverse player table: identity and crosswalk ids.",
    table_properties={"quality": "silver"},
)
@_rules("players")
def players() -> DataFrame:
    return transforms.silver_players(_read("bronze_players"))


@dp.materialized_view(
    name="snap_counts",
    comment="Offensive snap counts mapped from Pro Football Reference ids to GSIS ids.",
    table_properties={"quality": "silver"},
)
@_rules("snap_counts")
def snap_counts() -> DataFrame:
    return transforms.silver_snap_counts(_read("bronze_snap_counts"), _read("players"))


@dp.materialized_view(
    name="rosters",
    comment="Weekly team rosters.",
    table_properties={"quality": "silver"},
)
@_rules("rosters")
def rosters() -> DataFrame:
    return transforms.silver_rosters(_read("bronze_rosters"))


@dp.materialized_view(
    name="injuries",
    comment="Official weekly injury reports.",
    table_properties={"quality": "silver"},
)
@_rules("injuries")
def injuries() -> DataFrame:
    return transforms.silver_injuries(_read("bronze_injuries"))


@dp.materialized_view(
    name="depth_charts",
    comment="Skill-position depth charts in one shape across the 2024 and 2025 formats.",
    table_properties={"quality": "silver"},
)
@_rules("depth_charts")
def depth_charts() -> DataFrame:
    return transforms.silver_depth_charts(_read("bronze_depth_charts"))


@dp.materialized_view(
    name="ff_opportunity",
    comment="Expected PPR points per player-game (ffverse ffopportunity).",
    table_properties={"quality": "silver"},
)
@_rules("ff_opportunity")
def ff_opportunity() -> DataFrame:
    return transforms.silver_ff_opportunity(_read("bronze_ff_opportunity"))


@dp.materialized_view(
    name="schedules",
    comment="Every game, played and upcoming, with betting lines and kickoff time.",
    table_properties={"quality": "silver"},
)
@_rules("schedules")
def schedules() -> DataFrame:
    return transforms.silver_schedules(_read("bronze_schedules"), SEASON_FLOOR)


@dp.materialized_view(
    name="player_ids",
    comment="FantasyPros id to GSIS id crosswalk (DynastyProcess).",
    table_properties={"quality": "silver"},
)
@_rules("player_ids")
def player_ids() -> DataFrame:
    return transforms.silver_player_ids(_read("bronze_player_ids"))


@dp.materialized_view(
    name="ecr",
    comment="FantasyPros weekly positional expert consensus rankings, mapped to NFL weeks.",
    table_properties={"quality": "silver"},
)
@_rules("ecr")
def ecr() -> DataFrame:
    return transforms.silver_ecr(_read("bronze_ecr"), _read("schedules"), _read("player_ids"))


# Gold ----------------------------------------------------------------------------------


@dp.materialized_view(
    name="team_week",
    comment="One row per team and regular-season game: betting lines and offensive volume.",
    table_properties={"quality": "gold"},
)
@_rules("team_week")
def team_week() -> DataFrame:
    return transforms.team_week(_read("schedules"), _read("plays"), _read("player_stats"))


@dp.materialized_view(
    name="player_week",
    comment="One row per QB/RB/WR/TE per game: fantasy points in three formats and usage.",
    table_properties={"quality": "gold"},
    cluster_by=["season", "week"],
)
@_rules("player_week")
def player_week() -> DataFrame:
    return transforms.player_week(
        _read("player_stats"),
        _read("snap_counts"),
        _read("plays"),
        _read("team_week"),
        _read("ff_opportunity"),
    )


@dp.materialized_view(
    name="defense_vs_position",
    comment="Fantasy points allowed per defense, game and position, with a rolling average.",
    table_properties={"quality": "gold"},
)
def defense_vs_position() -> DataFrame:
    return transforms.defense_vs_position(_read("player_week"), _read("team_week"))


@dp.materialized_view(
    name="depth_chart_week",
    comment="Each team's skill-position depth chart as published before each game.",
    table_properties={"quality": "gold"},
)
def depth_chart_week() -> DataFrame:
    return transforms.depth_chart_week(_read("depth_charts"), _read("schedules"))
