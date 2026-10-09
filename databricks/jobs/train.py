"""Job task: walk-forward backtest, then train and register the model for the upcoming week.

* backtests the configured seasons (each season projected by a model trained only on the
  seasons before it) and backfills the in-progress season's completed weeks the same way;
* writes ``backtest_projections`` and ``backtest_metrics`` to the gold schema;
* trains the live model on every completed week before the upcoming week, logs it to MLflow
  with the backtest metrics, and registers it in Unity Catalog
  (``<catalog>.<schema>.fantasy_projection``) with the ``champion`` alias.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from gridiron import backtest as bt
from gridiron import registry
from gridiron.features import FEATURES
from gridiron.model import HGB_PARAMS
from gridiron.spark_io import read_table, write_table
from gridiron.workflow import DEFAULT_TEST_SEASONS, prepare, run_backtest, train_live, upcoming_rows

MODEL_NAME = "fantasy_projection"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--experiment", required=True, help="MLflow experiment path")
    parser.add_argument(
        "--test-seasons",
        default=",".join(str(s) for s in DEFAULT_TEST_SEASONS),
        help="comma separated",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="log the model without registering it (local smoke runs without Unity Catalog)",
    )
    return parser.parse_args(argv)


def run_task(argv: list[str] | None = None) -> str | None:
    """Run the task. Returns the logged model's URI, or None if there is no upcoming week."""
    args = parse_args(argv)
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    now = pd.Timestamp.now(tz="UTC")

    def table(name: str) -> str:
        return f"{args.catalog}.{args.schema}.{name}"

    prepared = prepare(lambda name: read_table(spark, table(name)))
    test_seasons = [int(s) for s in args.test_seasons.split(",")]
    backtest_projections, metrics = run_backtest(prepared, test_seasons, now)
    write_table(spark, backtest_projections, table("backtest_projections"))
    write_table(spark, metrics, table("backtest_metrics"))
    print(f"backtest: {len(backtest_projections)} projections; upcoming {prepared.upcoming}")

    if prepared.upcoming is None:
        print("no upcoming regular-season week; nothing to train for")
        return None
    model = train_live(prepared)

    if not args.skip_registration:
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)
    registered_name = f"{args.catalog}.{args.schema}.{MODEL_NAME}"
    season, week = prepared.upcoming
    with mlflow.start_run(run_name=f"fantasy-projection-{season}-w{week:02d}") as run:
        mlflow.log_params(
            {
                "upcoming_week": f"{season}-{week:02d}",
                "trained_through": str(model.trained_through),
                "test_seasons": args.test_seasons,
                "features": len(FEATURES),
                **{f"hgb_{k}": v for k, v in HGB_PARAMS.items()},
                **{f"training_rows_{p}": n for p, n in model.training_rows.items()},
            }
        )
        pooled = metrics[metrics["scope"] == bt.pooled_scope(metrics["scope"])]
        for row in pooled.itertuples():
            for metric in ("mae", "rmse", "spearman", "interval_coverage"):
                value = getattr(row, metric)
                if pd.notna(value):
                    mlflow.log_metric(f"{row.position}_{row.method}_{metric}", float(value))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_metrics.csv"
            metrics.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        info = registry.log_model(
            model,
            upcoming_rows(prepared),
            None if args.skip_registration else registered_name,
        )
        if not args.skip_registration:
            version = info.registered_model_version
            MlflowClient().set_registered_model_alias(registered_name, "champion", version)
            print(f"run {run.info.run_id}: registered {registered_name} v{version} as champion")
        else:
            print(f"run {run.info.run_id}: logged {info.model_uri}")
    return str(info.model_uri)


def main(argv: list[str] | None = None) -> int:
    run_task(argv)
    return 0


if __name__ == "__main__":
    # Databricks reports any SystemExit as a failed task, even exit code 0, so only exit
    # explicitly on failure.
    if (code := main()) != 0:
        sys.exit(code)
