"""The bronze -> silver -> gold dataset graph, for local runs and tests.

``databricks/pipelines/gridiron_pipeline.py`` declares exactly this graph as Lakeflow
datasets (with the same expectation rules); this module builds it with plain Spark so the
whole flow runs on a laptop and in CI. A test checks that both declare the same datasets.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from gridiron import quality, transforms

# Landed dataset -> bronze table name. Play-by-play keeps the table name it had before
# the other datasets were added, so its Auto Loader checkpoint carries over.
BRONZE_TABLES: dict[str, str] = {
    "pbp": "bronze_plays",
    "player_stats": "bronze_player_stats",
    "snap_counts": "bronze_snap_counts",
    "rosters": "bronze_rosters",
    "injuries": "bronze_injuries",
    "depth_charts": "bronze_depth_charts",
    "ff_opportunity": "bronze_ff_opportunity",
    "schedules": "bronze_schedules",
    "players": "bronze_players",
    "player_ids": "bronze_player_ids",
    "ecr": "bronze_ecr",
}

SILVER_TABLES = (
    "plays",
    "player_stats",
    "snap_counts",
    "rosters",
    "injuries",
    "depth_charts",
    "ff_opportunity",
    "schedules",
    "players",
    "player_ids",
    "ecr",
)
GOLD_TABLES = ("team_week", "player_week", "defense_vs_position", "depth_chart_week")


def read_landing(spark: SparkSession, landing_root: Path) -> dict[str, DataFrame]:
    """Bronze DataFrames straight from a landing folder (what Auto Loader reads in the cloud)."""
    bronze = {}
    for name in BRONZE_TABLES:
        files = sorted(str(p) for p in (landing_root / name).rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"nothing landed for {name} under {landing_root}")
        # Play-by-play files have one column whose type drifts between seasons; silver
        # never reads it, and merging schemas would fail on it. Every other dataset is
        # type-normalized at landing, so its schemas merge cleanly.
        reader = spark.read.option("mergeSchema", str(name != "pbp").lower())
        bronze[name] = transforms.with_ingest_metadata(reader.parquet(*files))
    return bronze


def _drop(name: str, df: DataFrame) -> DataFrame:
    return quality.apply_drop_rules(df, quality.DROP.get(name, {}))


def build_silver(bronze: Mapping[str, DataFrame], first_season: int) -> dict[str, DataFrame]:
    """Silver datasets from bronze DataFrames keyed by landed dataset name."""
    s: dict[str, DataFrame] = {}
    s["players"] = _drop("players", transforms.silver_players(bronze["players"]))
    s["player_ids"] = _drop("player_ids", transforms.silver_player_ids(bronze["player_ids"]))
    s["schedules"] = _drop(
        "schedules", transforms.silver_schedules(bronze["schedules"], first_season)
    )
    s["plays"] = _drop("plays", transforms.silver_plays(bronze["pbp"], first_season))
    s["player_stats"] = _drop(
        "player_stats", transforms.silver_player_stats(bronze["player_stats"])
    )
    s["snap_counts"] = _drop(
        "snap_counts", transforms.silver_snap_counts(bronze["snap_counts"], s["players"])
    )
    s["rosters"] = _drop("rosters", transforms.silver_rosters(bronze["rosters"]))
    s["injuries"] = _drop("injuries", transforms.silver_injuries(bronze["injuries"]))
    s["depth_charts"] = _drop(
        "depth_charts", transforms.silver_depth_charts(bronze["depth_charts"])
    )
    s["ff_opportunity"] = _drop(
        "ff_opportunity", transforms.silver_ff_opportunity(bronze["ff_opportunity"])
    )
    s["ecr"] = _drop("ecr", transforms.silver_ecr(bronze["ecr"], s["schedules"], s["player_ids"]))
    return s


def build_gold(silver: Mapping[str, DataFrame]) -> dict[str, DataFrame]:
    g: dict[str, DataFrame] = {}
    g["team_week"] = transforms.team_week(
        silver["schedules"], silver["plays"], silver["player_stats"]
    )
    g["player_week"] = transforms.player_week(
        silver["player_stats"],
        silver["snap_counts"],
        silver["plays"],
        g["team_week"],
        silver["ff_opportunity"],
    )
    g["defense_vs_position"] = transforms.defense_vs_position(g["player_week"], g["team_week"])
    g["depth_chart_week"] = transforms.depth_chart_week(silver["depth_charts"], silver["schedules"])
    return g
