"""Lakeflow Declarative Pipeline: bronze -> silver -> gold.

This file only declares datasets; all logic lives in the ``gridiron`` package and is unit
tested locally. Configuration (set in ``resources/pipeline.yml``):

* ``gridiron.landing_path`` - volume directory the ingest task writes season files to.

Dataset choices:

* ``bronze_plays`` is a streaming table fed by Auto Loader, so each landed file is read
  exactly once and the bronze history is append-only.
* ``plays`` (silver) is a materialized view: de-duplicating "latest version per play" needs a
  window over all of bronze, which a materialized view recomputes incrementally where it can
  and fully where it must. At ~550k rows that is cheap.
* Gold tables are materialized views over silver.
"""

from __future__ import annotations

from pyspark import pipelines as dp
from pyspark.sql import DataFrame, SparkSession

from gridiron import quality, transforms

spark = SparkSession.active()
LANDING_PATH = spark.conf.get("gridiron.landing_path")


@dp.table(
    name="bronze_plays",
    comment="Raw nflverse play-by-play, one row per play per landed file version.",
    table_properties={"quality": "bronze"},
)
def bronze_plays() -> DataFrame:
    raw = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .load(LANDING_PATH)
    )
    return transforms.with_ingest_metadata(raw)


@dp.materialized_view(
    name="plays",
    comment="Typed, column-pruned plays; latest ingested version of each (game_id, play_id).",
    table_properties={"quality": "silver"},
    cluster_by=["season", "game_id"],
)
@dp.expect_all_or_drop(quality.SILVER_DROP_RULES)
@dp.expect_all(quality.SILVER_WARN_RULES)
def plays() -> DataFrame:
    return transforms.silver_plays(spark.read.table("bronze_plays"))


@dp.materialized_view(
    name="fourth_down_decisions",
    comment="Every fourth-down go / punt / field goal decision with context and outcome.",
    table_properties={"quality": "gold"},
)
@dp.expect_all_or_drop(quality.DECISION_DROP_RULES)
def fourth_down_decisions() -> DataFrame:
    return transforms.fourth_down_decisions(spark.read.table("plays"))


@dp.materialized_view(
    name="team_season_summary",
    comment="Regular-season offensive efficiency and fourth-down tendencies per team.",
    table_properties={"quality": "gold"},
)
def team_season_summary() -> DataFrame:
    return transforms.team_season_summary(
        spark.read.table("plays"), spark.read.table("fourth_down_decisions")
    )


@dp.materialized_view(
    name="game_summary",
    comment="One row per game: final score, volume and fourth-down choices per side.",
    table_properties={"quality": "gold"},
)
def game_summary() -> DataFrame:
    return transforms.game_summary(
        spark.read.table("plays"), spark.read.table("fourth_down_decisions")
    )
