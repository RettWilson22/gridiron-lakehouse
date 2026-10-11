"""Job task: walk-forward backtest, then train and register the model for the upcoming week.

* backtests the configured seasons (each season projected by a model trained only on the
  seasons before it) and backfills the in-progress season's completed weeks the same way,
  reusing the stored projections of any season whose data did not change;
* writes ``backtest_projections`` and ``backtest_metrics`` to the gold schema;
* trains the live model on every completed week before the upcoming week, logs it to MLflow
  with the backtest metrics, and registers it in Unity Catalog
  (``<catalog>.<schema>.fantasy_projection``) with the ``champion`` alias. The run and the
  model version are tagged with the training data's fingerprint; when the champion already
  has the fingerprint of today's training data (the Thursday and Saturday runs, usually),
  fitting and registering are skipped.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from gridiron import backtest as bt
from gridiron import joblog, registry
from gridiron.features import FEATURES
from gridiron.model import HGB_PARAMS, ProjectionModel, fingerprint
from gridiron.spark_io import read_table, write_table
from gridiron.workflow import (
    DEFAULT_TEST_SEASONS,
    Prepared,
    live_history,
    prepare,
    reused_versions,
    run_backtest,
    upcoming_rows,
)


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
        "--registry-uri",
        default="databricks-uc",
        help="MLflow model registry (a local file store in smoke runs)",
    )
    return parser.parse_args(argv)


def run_task(argv: list[str] | None = None) -> str | None:
    """Run the task. Returns the champion model's URI, or None if there is no upcoming week."""
    args = parse_args(argv)
    joblog.setup()
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    now = pd.Timestamp.now(tz="UTC")

    def table(name: str) -> str:
        return f"{args.catalog}.{args.schema}.{name}"

    prepared = prepare(lambda name: read_table(spark, table(name)))
    stored = (
        read_table(spark, table("backtest_projections"))
        if spark.catalog.tableExists(table("backtest_projections"))
        else None
    )
    test_seasons = [int(s) for s in args.test_seasons.split(",")]
    backtest_projections, metrics = run_backtest(prepared, test_seasons, now, stored)
    write_table(spark, backtest_projections, table("backtest_projections"))
    write_table(spark, metrics, table("backtest_metrics"))
    summary: dict[str, object] = {
        "backtest_projections": len(backtest_projections),
        "backtest_seasons": backtest_projections["model_version"].nunique(),
        "reused_seasons": len(reused_versions(stored, backtest_projections)),
        "upcoming": prepared.upcoming,
    }
    if prepared.upcoming is None:
        joblog.summary("train", **summary)
        return None
    champion = train_champion(args, prepared, prepared.upcoming, metrics)
    joblog.summary("train", **summary, **champion)
    return f"models:/{champion['model']}@{registry.CHAMPION_ALIAS}"


def train_champion(
    args: argparse.Namespace,
    prepared: Prepared,
    upcoming: tuple[int, int],
    metrics: pd.DataFrame,
) -> dict[str, str]:
    """Fit, log and register the live model as champion, unless the champion is already
    trained on the same data. Returns what happened, for the task summary."""
    history = live_history(prepared)
    trained_on = fingerprint(history)
    mlflow.set_registry_uri(args.registry_uri)
    client = MlflowClient()
    name = registry.registered_name(args.catalog, args.schema)
    champion = registry.champion_trained_on(client, name, trained_on)
    if champion is not None:
        return {"model": name, "version": str(champion.version), "action": "kept"}

    model = ProjectionModel.fit(history)
    mlflow.set_experiment(args.experiment)
    season, week = upcoming
    with mlflow.start_run(run_name=f"fantasy-projection-{season}-w{week:02d}") as run:
        mlflow.set_tag(registry.FINGERPRINT_TAG, trained_on)
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
        mlflow.log_metrics(bt.pooled_metrics(metrics))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_metrics.csv"
            metrics.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        info = registry.log_model(model, upcoming_rows(prepared), name)
        version = str(info.registered_model_version)
        registry.promote(client, name, version, trained_on)
    return {"model": name, "version": version, "action": "registered", "run_id": run.info.run_id}


if __name__ == "__main__":
    # No sys.exit: Databricks reports any SystemExit as a failed task, even exit code 0.
    run_task()
