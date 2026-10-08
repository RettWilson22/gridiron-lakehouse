"""Train and evaluate the decision model on locally produced gold tables.

Reads the parquet output of ``run_local_pipeline.py``, prints held-out metrics, writes the
exported model JSON and the scored / aggregated gold tables:

    python scripts/train_local.py --lakehouse-dir data/lakehouse \\
        --last-train-season 2023 --test-seasons 2024 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from gridiron.decision import coach_aggressiveness, score_decisions
from gridiron.training import train_decision_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lakehouse-dir", type=Path, default=Path("data/lakehouse"))
    parser.add_argument("--last-train-season", type=int, default=2023)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/decision_model.json"))
    parser.add_argument(
        "--metrics-out", type=Path, default=Path("artifacts/local_training_metrics.json")
    )
    args = parser.parse_args(argv)

    decisions = pd.read_parquet(args.lakehouse_dir / "fourth_down_decisions")
    first_downs = pd.read_parquet(args.lakehouse_dir / "first_down_expected_points")
    result = train_decision_model(
        decisions, first_downs, args.last_train_season, tuple(args.test_seasons)
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(result.model.to_json() + "\n")
    report = {
        "train_seasons": f"{int(decisions['season'].min())}-{args.last_train_season}",
        "test_seasons": args.test_seasons,
        "metrics": {k: round(v, 4) for k, v in result.metrics.items()},
        "calibration": result.calibration.round(4).to_dict(orient="records"),
    }
    args.metrics_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    scored = score_decisions(decisions, result.model)
    scored.to_parquet(args.lakehouse_dir / "fourth_down_scored.parquet", index=False)
    aggressiveness = coach_aggressiveness(scored)
    aggressiveness.to_parquet(args.lakehouse_dir / "coach_aggressiveness.parquet", index=False)
    neutral = scored[scored["is_neutral_situation"]]
    print(
        f"scored {len(scored)} fourth downs; neutral: {len(neutral)}; "
        f"model says go on {neutral['recommendation'].eq('go').mean():.1%} of neutral plays; "
        f"coaches went for it on {neutral['decision'].eq('go').mean():.1%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
