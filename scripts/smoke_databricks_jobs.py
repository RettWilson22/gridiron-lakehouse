"""Smoke-test the Databricks job entry points against a local Spark catalog.

Loads the local lakehouse Parquet output into a local Spark database, then runs the real
``train``, ``score`` and ``publish_serving`` entry points against it with a local MLflow
tracking store and Unity Catalog registration disabled. This checks the wiring (table
names, pandas <-> Spark conversion, MLflow logging and loading, publish SQL) without a
Databricks workspace.

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
from gridiron.workflow import INPUT_TABLES

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
    parser.add_argument("--test-seasons", default="2025", help="kept short for a smoke run")
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
    for name in INPUT_TABLES:
        spark.read.parquet(str(args.lakehouse_dir / f"{name}.parquet")).write.mode(
            "overwrite"
        ).saveAsTable(f"gridiron.{name}")

    common = ["--catalog", "spark_catalog", "--schema", "gridiron"]
    train = load_job("train")
    train.main(
        [
            *common,
            "--experiment",
            "gridiron-smoke",
            "--test-seasons",
            args.test_seasons,
            "--skip-registration",
        ]
    )
    score = load_job("score")
    score.main([*common, "--model-uri", str(train.LAST_MODEL_URI)])
    # A second score run must keep the stored live projections stable.
    score.main([*common, "--model-uri", str(train.LAST_MODEL_URI)])
    publish = load_job("publish_serving")
    publish.main([*common, "--serving-schema", "gridiron_serving"])
    # Second publish exercises the refresh path (table already exists).
    publish.main([*common, "--serving-schema", "gridiron_serving"])
    for row in spark.sql("SHOW TABLES IN gridiron_serving").collect():
        count = spark.table(f"gridiron_serving.{row.tableName}").count()
        print(f"gridiron_serving.{row.tableName}: {count} rows")
    print(f"smoke run artifacts in {work}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
