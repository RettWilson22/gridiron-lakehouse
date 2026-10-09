"""Run the bronze -> silver -> gold transformations locally with plain Spark.

This mirrors the Lakeflow pipeline (same functions, same expectation rules) but reads the
landing folder directly and writes one Parquet file per silver and gold dataset:

    python scripts/run_local_pipeline.py --landing-root data/landing --out-dir data/lakehouse
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from gridiron import quality
from gridiron.config import FIRST_SEASON
from gridiron.lakehouse import build_gold, build_silver, read_landing
from gridiron.local_spark import local_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-root", type=Path, default=Path("data/landing"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/lakehouse"))
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    args = parser.parse_args(argv)

    started = time.monotonic()
    spark = local_session(shuffle_partitions=16)
    spark.sparkContext.setLogLevel("WARN")
    silver = build_silver(read_landing(spark, args.landing_root), args.first_season)
    silver = {name: df.cache() for name, df in silver.items()}
    gold = build_gold(silver)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {**silver, **gold}.items():
        # Single-file outputs keep the layout identical to the checked-in gold fixtures,
        # so dbt and the model scripts can point at either directory.
        table = frame.toArrow()
        pq.write_table(table, args.out_dir / f"{name}.parquet")
        warnings = quality.count_failures(frame, quality.WARN.get(name, {}))
        failing = {rule: n for rule, n in warnings.items() if n}
        note = f"; warn-rule failures {failing}" if failing else ""
        print(f"{name}: {table.num_rows} rows{note}")
    print(f"local pipeline finished in {time.monotonic() - started:.0f}s")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
