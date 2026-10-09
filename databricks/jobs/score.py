"""Job task: project the upcoming week and publish projections and risers.

* loads the champion model from Unity Catalog (or ``--model-uri``);
* projects every candidate player for the upcoming week; projections for games that have
  already kicked off are never replaced, so the stored record is what was shown pre-game;
* writes ``live_projections`` (the stored live history), ``projections`` (backtest weeks
  plus live weeks, with ranks, tiers and start/sit labels in every format) and ``risers``.
"""

from __future__ import annotations

import argparse
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from gridiron.registry import CHAMPION_ALIAS, model_input, registered_name
from gridiron.spark_io import read_table, write_table
from gridiron.workflow import prepare, publish, score_live


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--model-uri",
        default=None,
        help="defaults to models:/<catalog>.<schema>.fantasy_projection@champion",
    )
    parser.add_argument(
        "--registry-uri",
        default="databricks-uc",
        help="MLflow model registry (a local file store in smoke runs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    now = pd.Timestamp.now(tz="UTC")

    def table(name: str) -> str:
        return f"{args.catalog}.{args.schema}.{name}"

    prepared = prepare(lambda name: read_table(spark, table(name)), weeks="upcoming")
    existing = (
        read_table(spark, table("live_projections"))
        if spark.catalog.tableExists(table("live_projections"))
        else None
    )
    live = existing
    if prepared.upcoming is not None:
        if args.model_uri:
            uri, version = args.model_uri, args.model_uri
        else:
            mlflow.set_registry_uri(args.registry_uri)
            name = registered_name(args.catalog, args.schema)
            champion = MlflowClient().get_model_version_by_alias(name, CHAMPION_ALIAS)
            uri, version = f"models:/{name}@{CHAMPION_ALIAS}", f"{name} v{champion.version}"
        model = mlflow.pyfunc.load_model(uri)
        live = score_live(
            prepared, lambda rows: model.predict(model_input(rows)), version, existing, now
        )
    backtest_projections = read_table(spark, table("backtest_projections"))
    projections, risers = publish(prepared, backtest_projections, live)
    outputs = {"projections": projections, "risers": risers}
    if live is not None:
        outputs["live_projections"] = live
    for name, frame in outputs.items():
        write_table(spark, frame, table(name))
        print(f"wrote {table(name)}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    # Databricks reports any SystemExit as a failed task, even exit code 0, so only exit
    # explicitly on failure.
    if (code := main()) != 0:
        sys.exit(code)
