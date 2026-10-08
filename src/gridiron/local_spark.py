"""Local SparkSession used by tests and the local runner scripts."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession


def local_session(
    app_name: str = "gridiron-local",
    shuffle_partitions: int = 4,
    warehouse_dir: Path | None = None,
) -> SparkSession:
    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "4g")
        # Databricks enables Arrow for pandas <-> Spark conversion by default; match it.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    if warehouse_dir is not None:
        builder = builder.config("spark.sql.warehouse.dir", str(warehouse_dir))
    return builder.getOrCreate()
