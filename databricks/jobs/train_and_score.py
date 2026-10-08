"""Job task: train the fourth-down decision model, register it, and score every fourth down.

* trains on seasons <= ``--last-train-season`` and evaluates on ``--test-seasons``;
* logs parameters, held-out metrics, the calibration table and the exported JSON model to
  MLflow, and registers the scikit-learn conversion model in Unity Catalog
  (``models:/<catalog>.<schema>.fourth_down_conversion``) with the ``champion`` alias;
* writes ``fourth_down_scored`` and ``coach_aggressiveness`` gold tables;
* copies the JSON model to a volume so Snowflake can load it into a stage for the UDF.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import pyarrow as pa
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from gridiron import transforms
from gridiron.decision import coach_aggressiveness, score_decisions
from gridiron.features import FEATURE_COLUMNS, conversion_training_frame
from gridiron.training import train_decision_model

MODEL_NAME = "fourth_down_conversion"
BOOLEAN_COLUMNS = ("converted", "followed_model", "is_neutral_situation")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--experiment", required=True, help="MLflow experiment path")
    parser.add_argument("--model-dir", required=True, help="volume directory for the JSON model")
    parser.add_argument("--last-train-season", type=int, default=2023)
    parser.add_argument("--test-seasons", default="2024,2025", help="comma separated")
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="log the model without registering it (local smoke runs without Unity Catalog)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spark = SparkSession.builder.getOrCreate()

    def table(name: str) -> str:
        return f"{args.catalog}.{args.schema}.{name}"

    decisions_sdf = spark.read.table(table("fourth_down_decisions"))
    decisions: pd.DataFrame = decisions_sdf.toPandas()
    first_downs: pd.DataFrame = transforms.first_down_expected_points(
        spark.read.table(table("plays"))
    ).toPandas()
    test_seasons = tuple(int(s) for s in args.test_seasons.split(","))
    result = train_decision_model(decisions, first_downs, args.last_train_season, test_seasons)

    if not args.skip_registration:
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)
    registered_name = f"{args.catalog}.{args.schema}.{MODEL_NAME}"
    with mlflow.start_run(run_name="fourth-down-decision-model") as run:
        mlflow.log_params(
            {
                "last_train_season": args.last_train_season,
                "test_seasons": args.test_seasons,
                "features": ",".join(FEATURE_COLUMNS),
                "model": "StandardScaler + LogisticRegression(C=1.0)",
            }
        )
        mlflow.log_metrics(result.metrics)
        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = Path(tmp) / "calibration.csv"
            result.calibration.to_csv(calibration_path, index=False)
            mlflow.log_artifact(str(calibration_path))
            model_path = Path(tmp) / "decision_model.json"
            model_path.write_text(result.model.to_json())
            mlflow.log_artifact(str(model_path))

        x_train, _ = conversion_training_frame(
            decisions[decisions["season"] <= args.last_train_season]
        )
        sample = x_train[list(FEATURE_COLUMNS)].head(20)
        info = mlflow.sklearn.log_model(
            result.conversion_pipeline,
            name="conversion_model",
            signature=infer_signature(sample, result.conversion_pipeline.predict_proba(sample)),
            input_example=sample.head(3),
            registered_model_name=None if args.skip_registration else registered_name,
        )
        if not args.skip_registration:
            version = info.registered_model_version
            MlflowClient().set_registered_model_alias(registered_name, "champion", version)
            print(f"run {run.info.run_id}: registered {registered_name} v{version} as champion")
        print(json.dumps(result.metrics, indent=2))

    scored = score_decisions(decisions, result.model)
    scored = scored.astype(dict.fromkeys(BOOLEAN_COLUMNS, "boolean"))
    aggressiveness = coach_aggressiveness(scored)
    for name, frame in {
        "fourth_down_scored": scored,
        "coach_aggressiveness": aggressiveness,
    }.items():
        # Going through a pyarrow Table (Spark >= 4.0) avoids pandas-version-specific
        # conversion paths for nullable and Arrow-backed string columns.
        (
            transforms.cast_like(
                spark.createDataFrame(pa.Table.from_pandas(frame, preserve_index=False)),
                decisions_sdf.schema,
            )
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table(name))
        )
        print(f"wrote {table(name)}: {len(frame)} rows")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "decision_model.json").write_text(result.model.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
