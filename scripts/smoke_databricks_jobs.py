"""Smoke-test the Databricks job entry points against a local Spark catalog.

Loads the local lakehouse parquet output into a local Spark database, then runs the real
``train_and_score`` and ``publish_serving`` entry points against it with a local MLflow
tracking store and registration disabled. This checks the wiring (table names, pandas <->
Spark conversion, MLflow logging, publish SQL) without a Databricks workspace.

    python scripts/smoke_databricks_jobs.py --lakehouse-dir data/lakehouse
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from gridiron.local_spark import local_session

JOBS = Path(__file__).resolve().parent.parent / "databricks" / "jobs"


def load_job(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"job_{name}", JOBS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lakehouse-dir", type=Path, default=Path("data/lakehouse"))
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="gridiron-smoke-"))
    # A throwaway file store keeps the smoke run dependency-free (mlflow-skinny has no
    # SQL backend); MLflow requires an explicit opt-in for it.
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{work / 'mlruns'}"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    spark = local_session("gridiron-smoke", warehouse_dir=work / "warehouse")
    spark.sparkContext.setLogLevel("ERROR")
    # Fail loudly instead of silently falling back to the slow non-Arrow conversion.
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
    spark.sql("CREATE DATABASE IF NOT EXISTS gridiron")
    spark.sql("CREATE DATABASE IF NOT EXISTS gridiron_serving")
    for name in ("plays", "fourth_down_decisions", "team_season_summary", "game_summary"):
        spark.read.parquet(str(args.lakehouse_dir / name)).write.mode("overwrite").saveAsTable(
            f"gridiron.{name}"
        )

    common = ["--catalog", "spark_catalog", "--schema", "gridiron"]
    load_job("train_and_score").main(
        [
            *common,
            "--experiment",
            "gridiron-smoke",
            "--model-dir",
            str(work / "models"),
            "--skip-registration",
        ]
    )
    load_job("publish_serving").main([*common, "--serving-schema", "gridiron_serving"])
    # Second publish exercises the refresh path (table already exists).
    load_job("publish_serving").main([*common, "--serving-schema", "gridiron_serving"])
    for row in spark.sql("SHOW TABLES IN gridiron_serving").collect():
        count = spark.table(f"gridiron_serving.{row.tableName}").count()
        print(f"gridiron_serving.{row.tableName}: {count} rows")
    print(f"smoke run artifacts in {work}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
