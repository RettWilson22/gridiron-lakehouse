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
   per-player FantasyPros ranks left out (see ``REDACTED``).

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
# The marts the app queries (mirrors snowflake/streamlit/data_access.MARTS).
MARTS = (
    "mart_cheat_sheet",
    "mart_risers",
    "mart_projection_scorecard",
    "mart_player_game_log",
    "mart_backtest_summary",
)


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


# Columns blanked in the public copy: FantasyPros rankings are third-party content, so the
# public snapshot keeps only the accuracy comparison against them, not the ranks themselves.
REDACTED: dict[str, tuple[str, ...]] = {"mart_cheat_sheet": ("ecr_rank",)}


def export_marts(database: Path, out_dir: Path) -> dict[str, int]:
    """Copy each mart the app reads from a DuckDB build into Parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    with duckdb.connect(str(database), read_only=True) as con:
        for mart in MARTS:
            target = (out_dir / f"{mart}.parquet").as_posix()
            blanked = ", ".join(f"NULL AS {column}" for column in REDACTED.get(mart, ()))
            select = f"* REPLACE ({blanked})" if blanked else "*"
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
