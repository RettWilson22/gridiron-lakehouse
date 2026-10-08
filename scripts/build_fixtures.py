"""Rebuild the checked-in test fixtures from locally downloaded nflverse data.

1. ``tests/fixtures/pbp_sample.parquet``: three full games (raw nflverse columns that silver
   uses), small enough to commit and rich enough to cover every fourth-down outcome.
2. ``tests/fixtures/gold/*.parquet``: gold tables derived from that sample with the real
   transformations and the exported decision model. dbt's ``ci`` target reads these.

    python scripts/build_fixtures.py --landing-dir data/landing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gridiron import quality, transforms
from gridiron.config import SILVER_COLUMNS
from gridiron.decision import DecisionModel, coach_aggressiveness, score_decisions
from gridiron.local_spark import local_session

FIXTURE_GAMES = {
    2023: ["2023_08_LV_DET"],
    2024: ["2024_09_DET_GB", "2024_18_MIN_DET"],
}
FIXTURES = Path("tests/fixtures")


def build_sample(landing_dir: Path) -> pd.DataFrame:
    frames = []
    for season, games in FIXTURE_GAMES.items():
        files = sorted((landing_dir / str(season)).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"season {season} not landed under {landing_dir}")
        table = pq.read_table(files[-1], columns=list(SILVER_COLUMNS))
        frame = table.to_pandas()
        frames.append(frame[frame["game_id"].isin(games)])
    sample: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return sample.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def build_gold(sample_path: Path, model_path: Path, out_dir: Path) -> None:
    spark = local_session()
    spark.sparkContext.setLogLevel("ERROR")
    bronze = transforms.with_ingest_metadata(spark.read.parquet(str(sample_path)))
    plays = quality.apply_drop_rules(transforms.silver_plays(bronze), quality.SILVER_DROP_RULES)
    decisions = quality.apply_drop_rules(
        transforms.fourth_down_decisions(plays), quality.DECISION_DROP_RULES
    )
    gold = {
        "fourth_down_decisions": decisions,
        "team_season_summary": transforms.team_season_summary(plays, decisions),
        "game_summary": transforms.game_summary(plays, decisions),
    }
    model = DecisionModel.load(model_path)
    scored = score_decisions(gold["fourth_down_decisions"].toPandas(), model)
    # Same pandas -> Spark -> typed path as the Databricks train_and_score job.
    for name, pdf in {
        "fourth_down_scored": scored,
        "coach_aggressiveness": coach_aggressiveness(scored),
    }.items():
        gold[name] = transforms.cast_like(
            spark.createDataFrame(pa.Table.from_pandas(pdf, preserve_index=False)),
            gold["fourth_down_decisions"].schema,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sdf in gold.items():
        pq.write_table(sdf.toArrow(), out_dir / f"{name}.parquet")
    spark.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-dir", type=Path, default=Path("data/landing"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/decision_model.json"))
    args = parser.parse_args(argv)

    sample = build_sample(args.landing_dir)
    sample_path = FIXTURES / "pbp_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    print(f"wrote {len(sample)} rows x {sample.shape[1]} columns to {sample_path}")
    build_gold(sample_path, args.model, FIXTURES / "gold")
    for path in sorted((FIXTURES / "gold").glob("*.parquet")):
        print(f"{path}: {len(pd.read_parquet(path))} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
