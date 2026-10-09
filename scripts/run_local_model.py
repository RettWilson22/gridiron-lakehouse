"""Run the model side locally: walk-forward backtest, live model, projections and risers.

Reads the Parquet output of ``run_local_pipeline.py`` and does what the Databricks
``train`` and ``score`` tasks do, without MLflow:

    python scripts/run_local_model.py --lakehouse-dir data/lakehouse

Writes ``backtest_projections``, ``projections``, ``live_projections``, ``risers`` and
``backtest_metrics`` Parquet files next to the inputs, the metrics to
``artifacts/backtest_metrics.json`` and bootstrap intervals for the model's margins over
the baselines and the experts to ``artifacts/backtest_intervals.json``. Re-running keeps
live projections for games that have already kicked off, and reuses the stored backtest
projections of every season whose data did not change instead of refitting it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from gridiron import backtest as bt
from gridiron.workflow import (
    DEFAULT_TEST_SEASONS,
    backtest_intervals,
    prepare,
    publish,
    reused_versions,
    run_backtest,
    score_live,
    train_live,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lakehouse-dir", type=Path, default=Path("data/lakehouse"))
    parser.add_argument("--test-seasons", type=int, nargs="+", default=list(DEFAULT_TEST_SEASONS))
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/backtest_metrics.json"))
    parser.add_argument(
        "--intervals-out", type=Path, default=Path("artifacts/backtest_intervals.json")
    )
    parser.add_argument(
        "--now",
        type=pd.Timestamp,
        default=None,
        help="override the current time (UTC), e.g. to reproduce a run",
    )
    args = parser.parse_args(argv)
    now = pd.Timestamp.now(tz="UTC") if args.now is None else args.now
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    started = time.monotonic()

    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(args.lakehouse_dir / f"{name}.parquet")

    prepared = prepare(read)
    print(f"feature rows: {len(prepared.frame)}; upcoming week: {prepared.upcoming}")
    stored_path = args.lakehouse_dir / "backtest_projections.parquet"
    stored = pd.read_parquet(stored_path) if stored_path.exists() else None
    backtest_projections, metrics = run_backtest(prepared, args.test_seasons, now, stored)
    reused = reused_versions(stored, backtest_projections)
    seasons = backtest_projections["model_version"].nunique()
    print(
        f"backtest projections: {len(backtest_projections)} rows; reused {len(reused)} of "
        f"{seasons} seasons"
    )

    live_path = args.lakehouse_dir / "live_projections.parquet"
    existing = pd.read_parquet(live_path) if live_path.exists() else None
    live = existing
    if prepared.upcoming is not None:
        model = train_live(prepared)
        season, week = prepared.upcoming
        live = score_live(prepared, model.predict, f"local-{season}-w{week:02d}", existing, now)
        rows = 0 if live is None else len(live)
        print(f"live model trained through {model.trained_through}; live rows: {rows}")
    projections, risers = publish(prepared, backtest_projections, live)

    outputs = {
        "backtest_projections": backtest_projections,
        "projections": projections,
        "risers": risers,
        "backtest_metrics": metrics,
    }
    if live is not None:
        outputs["live_projections"] = live
    for name, frame in outputs.items():
        frame.to_parquet(args.lakehouse_dir / f"{name}.parquet", index=False)
        print(f"{name}: {len(frame)} rows")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    coverage = bt.candidate_coverage(prepared.frame, prepared.tables["player_week"])
    report = {
        "generated_at": now.isoformat(),
        "test_seasons": args.test_seasons,
        "upcoming_week": prepared.upcoming,
        "candidate_coverage": {k: round(v, 4) for k, v in coverage.items()},
        "metrics": json.loads(metrics.round(4).to_json(orient="records")),
    }
    args.metrics_out.write_text(json.dumps(report, indent=2) + "\n")
    intervals = backtest_intervals(prepared, backtest_projections, args.test_seasons)
    args.intervals_out.parent.mkdir(parents=True, exist_ok=True)
    intervals_report = {
        "generated_at": now.isoformat(),
        "test_seasons": args.test_seasons,
        "method": "paired bootstrap over weeks, 95% percentile intervals",
        "resamples": bt.RESAMPLES,
        "seed": bt.SEED,
        "intervals": json.loads(intervals.round(4).to_json(orient="records")),
    }
    args.intervals_out.write_text(json.dumps(intervals_report, indent=2) + "\n")
    print(f"candidate pool coverage: {coverage}")
    pooled = metrics[metrics["scope"] == bt.pooled_scope(metrics["scope"])]
    print(pooled.round(3).to_string(index=False))
    print(
        intervals[intervals["scope"] == bt.pooled_scope(metrics["scope"])]
        .round(3)
        .to_string(index=False)
    )
    print(f"finished in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
