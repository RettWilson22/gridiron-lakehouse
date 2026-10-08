"""Bronze -> silver -> gold transformations as plain PySpark functions.

The Lakeflow pipeline in ``databricks/pipelines`` only wires these functions to datasets,
so every transformation here is unit tested locally against a checked-in fixture.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from gridiron.config import FIELD_GOAL_PLAY_TYPE, GO_PLAY_TYPES, PUNT_PLAY_TYPE, SILVER_COLUMNS

INGESTED_AT = "_ingested_at"
SOURCE_FILE = "_source_file"


def with_ingest_metadata(raw: DataFrame) -> DataFrame:
    """Attach file-level lineage from the hidden ``_metadata`` column of file sources.

    Works for both ``spark.read.parquet`` and Auto Loader (``cloudFiles``) readers.
    """
    return raw.select(
        "*",
        F.col("_metadata.file_modification_time").alias(INGESTED_AT),
        F.col("_metadata.file_path").alias(SOURCE_FILE),
    )


def _cast(name: str, spark_type: str) -> Column:
    if spark_type == "date":
        return F.to_date(F.col(name)).alias(name)
    return F.col(name).cast(spark_type).alias(name)


def silver_plays(bronze: DataFrame) -> DataFrame:
    """Typed, column-pruned plays, de-duplicated on (game_id, play_id).

    Bronze is append-only and can hold several downloads of the in-progress season, so the
    most recently ingested version of each play wins.
    """
    typed = bronze.select(
        *(_cast(name, spark_type) for name, spark_type in SILVER_COLUMNS.items()),
        F.col(INGESTED_AT),
    )
    latest_first = Window.partitionBy("game_id", "play_id").orderBy(
        F.col(INGESTED_AT).desc_nulls_last()
    )
    return (
        typed.withColumn("_rank", F.row_number().over(latest_first))
        .where(F.col("_rank") == 1)
        .drop("_rank")
    )


def _decision() -> Column:
    return (
        F.when(F.col("play_type").isin(*GO_PLAY_TYPES), F.lit("go"))
        .when(F.col("play_type") == PUNT_PLAY_TYPE, F.lit("punt"))
        .when(F.col("play_type") == FIELD_GOAL_PLAY_TYPE, F.lit("field_goal"))
    )


def _outcome() -> Column:
    return (
        F.when(
            F.col("decision") == "go",
            F.when(F.col("fourth_down_converted") == 1, F.lit("converted")).otherwise(
                F.lit("failed")
            ),
        )
        .when(
            F.col("decision") == "field_goal",
            F.when(F.col("field_goal_result") == "made", F.lit("fg_made"))
            .when(F.col("field_goal_result") == "blocked", F.lit("fg_blocked"))
            .otherwise(F.lit("fg_missed")),
        )
        .when(
            F.col("decision") == "punt",
            F.when(F.col("punt_blocked") == 1, F.lit("punt_blocked")).otherwise(F.lit("punt")),
        )
    )


def fourth_down_decisions(plays: DataFrame) -> DataFrame:
    """One row per fourth-down snap where the offense went for it, punted or kicked.

    Plays nullified by penalty (``play_type = 'no_play'``), kneels and spikes are excluded:
    they are not a go/punt/kick decision that can be evaluated.
    """
    is_home = F.col("posteam_type") == "home"
    return (
        plays.where((F.col("down") == 4) & F.col("posteam").isNotNull())
        .withColumn("decision", _decision())
        .where(F.col("decision").isNotNull())
        .withColumn("outcome", _outcome())
        .select(
            "game_id",
            "play_id",
            "season",
            "season_type",
            "week",
            "game_date",
            "posteam",
            "defteam",
            F.when(is_home, F.col("home_coach")).otherwise(F.col("away_coach")).alias("coach"),
            "posteam_type",
            "qtr",
            "ydstogo",
            "yardline_100",
            "game_seconds_remaining",
            "score_differential",
            "posteam_timeouts_remaining",
            "wp",
            "ep",
            "epa",
            "wpa",
            "kick_distance",
            "return_yards",
            "touchback",
            "decision",
            "outcome",
            F.when(F.col("decision") == "go", F.col("outcome") == "converted").alias("converted"),
        )
    )


def first_down_expected_points(plays: DataFrame) -> DataFrame:
    """First-down snaps with their nflfastR expected points, used to fit the EP curve."""
    return plays.where(
        (F.col("down") == 1) & F.col("ep").isNotNull() & F.col("yardline_100").isNotNull()
    ).select("season", "yardline_100", "ep")


def team_season_summary(plays: DataFrame, decisions: DataFrame) -> DataFrame:
    """Regular-season offensive efficiency and fourth-down tendencies per team and season."""
    offense = (
        plays.where(
            (F.col("season_type") == "REG")
            & F.col("play_type").isin(*GO_PLAY_TYPES)
            & F.col("epa").isNotNull()
        )
        .groupBy("season", F.col("posteam").alias("team"))
        .agg(
            F.count("*").alias("offensive_plays"),
            F.avg("epa").alias("epa_per_play"),
            F.avg(F.when(F.col("epa") > 0, 1.0).otherwise(0.0)).alias("success_rate"),
        )
    )
    fourth = (
        decisions.where(F.col("season_type") == "REG")
        .groupBy("season", F.col("posteam").alias("team"))
        .agg(
            F.count("*").alias("fourth_downs"),
            F.sum(F.when(F.col("decision") == "go", 1).otherwise(0)).alias("go_attempts"),
            F.sum(F.when(F.col("outcome") == "converted", 1).otherwise(0)).alias("go_conversions"),
            F.sum(F.when(F.col("decision") == "punt", 1).otherwise(0)).alias("punts"),
            F.sum(F.when(F.col("decision") == "field_goal", 1).otherwise(0)).alias(
                "field_goal_attempts"
            ),
        )
    )
    return (
        offense.join(fourth, ["season", "team"], "left")
        .fillna(
            0, ["fourth_downs", "go_attempts", "go_conversions", "punts", "field_goal_attempts"]
        )
        .withColumn(
            "go_rate",
            F.when(F.col("fourth_downs") > 0, F.col("go_attempts") / F.col("fourth_downs")),
        )
        .withColumn(
            "conversion_rate",
            F.when(F.col("go_attempts") > 0, F.col("go_conversions") / F.col("go_attempts")),
        )
    )


def game_summary(plays: DataFrame, decisions: DataFrame) -> DataFrame:
    """One row per game: final score, play volume and each side's fourth-down choices."""
    games = plays.groupBy("game_id").agg(
        F.first("season").alias("season"),
        F.first("season_type").alias("season_type"),
        F.first("week").alias("week"),
        F.first("game_date").alias("game_date"),
        F.first("home_team").alias("home_team"),
        F.first("away_team").alias("away_team"),
        F.first("home_coach").alias("home_coach"),
        F.first("away_coach").alias("away_coach"),
        F.max("home_score").alias("home_score"),
        F.max("away_score").alias("away_score"),
        F.sum(F.when(F.col("play_type").isin(*GO_PLAY_TYPES), 1).otherwise(0)).alias(
            "scrimmage_plays"
        ),
    )
    go = F.col("decision") == "go"
    side = decisions.groupBy("game_id").agg(
        F.sum(F.when(F.col("posteam_type") == "home", 1).otherwise(0)).alias("home_fourth_downs"),
        F.sum(F.when(go & (F.col("posteam_type") == "home"), 1).otherwise(0)).alias(
            "home_go_attempts"
        ),
        F.sum(F.when(F.col("posteam_type") == "away", 1).otherwise(0)).alias("away_fourth_downs"),
        F.sum(F.when(go & (F.col("posteam_type") == "away"), 1).otherwise(0)).alias(
            "away_go_attempts"
        ),
    )
    counts = ["home_fourth_downs", "home_go_attempts", "away_fourth_downs", "away_go_attempts"]
    return (
        games.join(side, "game_id", "left")
        .fillna(0, counts)
        .withColumn(
            "winner",
            F.when(F.col("home_score") > F.col("away_score"), F.col("home_team"))
            .when(F.col("away_score") > F.col("home_score"), F.col("away_team"))
            .otherwise(F.lit("TIE")),
        )
    )


def cast_like(df: DataFrame, reference: StructType) -> DataFrame:
    """Cast columns that also exist in ``reference`` back to the reference types.

    A round trip through pandas turns nullable integer columns into floats; this restores
    the gold table types after scoring in pandas.
    """
    types = {field.name: field.dataType for field in reference.fields}
    return df.select(
        *(F.col(c).cast(types[c]).alias(c) if c in types else F.col(c) for c in df.columns)
    )
