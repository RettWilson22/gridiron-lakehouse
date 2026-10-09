"""Export the public snapshot: the app's dbt marts as small Parquet files.

The public copy of the app (``streamlit_public/``, for Streamlit Community Cloud) reads
these files instead of Snowflake. They are produced by the same dbt models from the same
serving tables, so the public copy is a static snapshot of the Snowflake app as of the
export. Steps:

1. collect the serving tables, either from a local run (``--from-dir``, default
   ``data/lakehouse``) or straight from Databricks (``--from-databricks``, using the
   same ``.env`` settings and CLI-profile auth as ``make sync``);
2. build the dbt project over them in a throwaway DuckDB database;
3. write each mart the app reads to ``streamlit_public/snapshot/<mart>.parquet``, with
   only the columns listed in ``EXPORTED_COLUMNS`` and the per-player FantasyPros ranks
   blanked (``BLANKED_COLUMNS``).

    python scripts/export_public_snapshot.py
    set -a && . ./.env && set +a && python scripts/export_public_snapshot.py --from-databricks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

from gridiron.serving import SERVING_TABLES

ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = ROOT / "snowflake" / "dbt"
SNAPSHOT_DIR = ROOT / "streamlit_public" / "snapshot"
# The columns copied into the public snapshot, per mart the app queries. This is an
# allowlist: a column a dbt change adds stays out of the public copy until it is listed
# here, after checking it is fine to publish.
EXPORTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "mart_cheat_sheet": (
        "projection_key",
        "season",
        "week",
        "is_upcoming",
        "game_final",
        "kind",
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent",
        "is_home",
        "matchup",
        "kickoff_at",
        "implied_points",
        "team_spread",
        "depth_rank",
        "injury_status",
        "injury",
        "opp_matchup_rank",
        "ecr_rank",
        *(
            f"{kind}_{fmt}"
            for fmt in ("ppr", "half", "std")
            for kind in ("proj", "floor", "ceiling", "pos_rank", "tier", "start_sit")
        ),
        "proj_passing_yards",
        "proj_passing_tds",
        "proj_passing_interceptions",
        "proj_rushing_yards",
        "proj_rushing_tds",
        "proj_receptions",
        "proj_receiving_yards",
        "proj_receiving_tds",
        "proj_fumbles_lost",
        "proj_two_point_conversions",
        "baseline_last3",
        "baseline_season_avg",
        "actual_ppr",
        "actual_half",
        "actual_std",
        "model_version",
        "generated_at",
    ),
    "mart_risers": (
        "riser_key",
        "season",
        "week",
        "player_id",
        "player_name",
        "position",
        "team",
        "is_riser",
        "riser_rank",
        "baseline_basis",
        "games_recent",
        "games_earlier",
        *(
            f"{metric}_{part}"
            for metric in (
                "expected_ppr",
                "snap_share",
                "target_share",
                "carry_share",
                "red_zone_share",
            )
            for part in ("recent", "before", "change")
        ),
        "proj_ppr",
        "pos_rank_ppr",
        "matchup",
    ),
    "mart_projection_scorecard": (
        "scorecard_key",
        "season",
        "week",
        "position",
        "method",
        "kind",
        "n",
        "abs_error_sum",
        "squared_error_sum",
        "error_sum",
        "mae",
        "inside_count",
        "spearman",
    ),
    "mart_player_game_log": (
        "player_week_key",
        "season",
        "week",
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent",
        "fantasy_points_ppr",
        "fantasy_points_half",
        "fantasy_points_std",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "fumbles_lost",
        "two_point_conversions",
        "special_teams_tds",
        "targets",
        "carries",
        "snap_share",
        "target_share",
        "carry_share",
        "expected_ppr",
        "proj_ppr",
        "floor_ppr",
        "ceiling_ppr",
        "projection_kind",
    ),
    "mart_backtest_summary": (
        "metric_key",
        "scope",
        "position",
        "method",
        "method_label",
        "n",
        "weeks",
        "mae",
        "rmse",
        "bias",
        "spearman",
        "interval_coverage",
        "pool_coverage",
    ),
}
MARTS = tuple(EXPORTED_COLUMNS)
# Exported as typed NULLs: the app expects the column, but per-player FantasyPros ranks
# are third-party content, so the public copy keeps only the accuracy comparison against
# them (mart_backtest_summary), not the ranks themselves.
BLANKED_COLUMNS: dict[str, dict[str, str]] = {"mart_cheat_sheet": {"ecr_rank": "INTEGER"}}


def download_from_databricks(out_dir: Path) -> None:
    """Fetch the serving tables through a Databricks SQL warehouse (see ``make sync``)."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    from databricks import sql as databricks_sql  # noqa: PLC0415 - "sync" extra

    from gridiron.serving import qualified  # noqa: PLC0415
    from gridiron.sync_config import SyncConfig  # noqa: PLC0415

    config = SyncConfig.from_env(os.environ)
    with databricks_sql.connect(**config.databricks_connect_kwargs()) as connection:
        for table in SERVING_TABLES:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {qualified(config.source_catalog, config.source_schema, table)}"
                )
                pq.write_table(cursor.fetchall_arrow(), out_dir / f"{table}.parquet")


def build_marts(serving_dir: Path, database: Path) -> None:
    """dbt build (DuckDB) over a folder of serving-table Parquet files."""
    dbt = Path(sys.executable).parent / "dbt"
    env = {**os.environ, "DBT_DUCKDB_PATH": str(database)}
    command = [
        str(dbt),
        "build",
        "--target",
        "ci",
        "--profiles-dir",
        ".",
        "--vars",
        json.dumps({"gold_dir": str(serving_dir)}),
    ]
    subprocess.run(command, cwd=DBT_DIR, env=env, check=True)


def export_marts(database: Path, out_dir: Path) -> dict[str, int]:
    """Copy each mart the app reads from a DuckDB build into Parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    with duckdb.connect(str(database), read_only=True) as con:
        for mart, columns in EXPORTED_COLUMNS.items():
            target = (out_dir / f"{mart}.parquet").as_posix()
            blanked = BLANKED_COLUMNS.get(mart, {})
            select = ", ".join(
                f"CAST(NULL AS {blanked[c]}) AS {c}" if c in blanked else c for c in columns
            )
            con.execute(f"COPY (SELECT {select} FROM marts.{mart}) TO '{target}' (FORMAT PARQUET)")
            result = con.execute(f"SELECT count(*) FROM marts.{mart}").fetchone()
            rows[mart] = int(result[0]) if result else 0
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-dir", type=Path, default=ROOT / "data" / "lakehouse")
    source.add_argument("--from-databricks", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=SNAPSHOT_DIR)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        serving = work / "serving"
        serving.mkdir()
        if args.from_databricks:
            download_from_databricks(serving)
        else:
            for table in SERVING_TABLES:
                shutil.copyfile(args.from_dir / f"{table}.parquet", serving / f"{table}.parquet")
        database = work / "snapshot.duckdb"
        build_marts(serving, database)
        shutil.rmtree(args.out_dir, ignore_errors=True)
        for mart, count in export_marts(database, args.out_dir).items():
            print(f"{mart}: {count} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
