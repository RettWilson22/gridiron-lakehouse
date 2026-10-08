"""Job task: land nflverse play-by-play season files in the bronze volume.

Runs unchanged on Databricks serverless (``--landing-dir /Volumes/...``) and locally
(``--landing-dir data/landing``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridiron.config import FIRST_SEASON, current_season
from gridiron.ingest import ingest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-dir", required=True, type=Path)
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument(
        "--last-season",
        type=int,
        default=None,
        help="defaults to the current season, which is always refreshed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    current = current_season()
    last = args.last_season or current
    results = ingest(range(args.first_season, last + 1), args.landing_dir, current)
    for result in results:
        print(f"{result.season}: {result.action}" + (f" -> {result.path}" if result.path else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
