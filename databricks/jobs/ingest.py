"""Job task: land the public fantasy football datasets in the bronze volume.

Runs unchanged on Databricks serverless (``--landing-root /Volumes/...``) and locally
(``--landing-root data/landing``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridiron.config import FIRST_SEASON, current_season
from gridiron.datasets import DATASETS
from gridiron.ingest import ingest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-root", required=True, type=Path)
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument(
        "--last-season",
        type=int,
        default=None,
        help="defaults to the current season, which is always refreshed",
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    current = current_season()
    last = args.last_season or current
    results = ingest(args.datasets, range(args.first_season, last + 1), args.landing_root, current)
    for result in results:
        print(result.describe())
    return 0


if __name__ == "__main__":
    # Databricks reports any SystemExit as a failed task, even exit code 0, so only exit
    # explicitly on failure.
    if (code := main()) != 0:
        sys.exit(code)
