"""Local SparkSession used by tests and the local pipeline runner."""

from __future__ import annotations

from pyspark.sql import SparkSession


def local_session(app_name: str = "gridiron-local", shuffle_partitions: int = 4) -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
