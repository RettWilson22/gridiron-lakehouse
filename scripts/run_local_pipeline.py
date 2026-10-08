"""Run the bronze -> silver -> gold transformations locally with plain Spark.

This mirrors the Lakeflow pipeline (same functions, same expectation rules) but writes
parquet instead of Unity Catalog tables, so the whole flow can be exercised on a laptop:

    python scripts/run_local_pipeline.py --landing-dir data/landing --out-dir data/lakehouse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gridiron import quality, transforms
from gridiron.local_spark import local_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-dir", type=Path, default=Path("data/landing"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/lakehouse"))
    args = parser.parse_args(argv)

    spark = local_session(shuffle_partitions=16)
    spark.sparkContext.setLogLevel("WARN")
    files = sorted(str(p) for p in args.landing_dir.glob("*/*.parquet"))
    if not files:
        parser.error(f"no parquet files under {args.landing_dir}; run the ingest job first")
    raw = spark.read.parquet(*files)
    bronze = transforms.with_ingest_metadata(raw)

    silver_all = transforms.silver_plays(bronze).cache()
    warn_counts = quality.count_failures(silver_all, quality.SILVER_WARN_RULES)
    plays = quality.apply_drop_rules(silver_all, quality.SILVER_DROP_RULES).cache()
    decisions = quality.apply_drop_rules(
        transforms.fourth_down_decisions(plays), quality.DECISION_DROP_RULES
    ).cache()

    outputs = {
        "plays": plays,
        "fourth_down_decisions": decisions,
        "first_down_expected_points": transforms.first_down_expected_points(plays),
        "team_season_summary": transforms.team_season_summary(plays, decisions),
        "game_summary": transforms.game_summary(plays, decisions),
    }
    for name, frame in outputs.items():
        frame.write.mode("overwrite").parquet(str(args.out_dir / name))
        print(f"{name}: {spark.read.parquet(str(args.out_dir / name)).count()} rows")

    print(f"bronze rows: {raw.count()}, silver rows before drop rules: {silver_all.count()}")
    print("warn-rule failures:", warn_counts)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
