"""Sync: copy the Databricks serving tables into native Snowflake tables (SYNCED).

This is the production path with Databricks Free Edition: its Unity Catalog Iceberg REST
endpoint returns table metadata but no storage credentials for tables on Databricks default
storage, so Snowflake cannot read the files in place (details in docs/deploy.md). For each
serving table the script:

1. reads the table from a Databricks SQL warehouse as Arrow;
2. loads it into a temporary staging table shaped exactly like the target;
3. MERGEs changed and new rows into ``SYNCED.<TABLE>`` and deletes rows that disappeared,
   in one transaction, so the streams on ``SYNCED.PLAYER_WEEK`` and ``SYNCED.PROJECTIONS``
   only see real changes (a new ``generated_at`` alone is not a change);
4. rolls the transaction back instead when the source came back empty or the delete would
   remove more than a fifth of the table (see ``shrink_problem``), unless ``--allow-shrink``
   is given.

Configuration comes from environment variables (see ``.env.example``); nothing is
hard-coded. Databricks auth uses ``DATABRICKS_TOKEN`` if set, otherwise a Databricks CLI
profile. Run ``snowflake/sync/01_synced_tables.sql`` once before the first sync.

    python scripts/sync_to_snowflake.py                       # all serving tables
    python scripts/sync_to_snowflake.py --tables projections
    python scripts/sync_to_snowflake.py --tables backtest_metrics --allow-shrink  # new test seasons
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
from databricks import sql as databricks_sql
from snowflake.connector import connect as snowflake_connect
from snowflake.connector.pandas_tools import write_pandas

from gridiron.serving import SERVING_TABLES, qualified
from gridiron.snowflake_sql import PRIMARY_KEYS, delete_missing_sql, merge_sql, sf_ident
from gridiron.sync_config import SyncConfig

log = logging.getLogger("sync_to_snowflake")
# A column that moves on every Databricks run even when nothing else does. It is not a
# change on its own, so the streams on SYNCED tables only see real changes.
CHANGE_IGNORED = ("generated_at",)
# Mirroring an empty or partial read would wipe SYNCED rows, and the streams on them would
# then delete the matching APP.PROJECTION_RESULTS rows. Risers is empty in the offseason.
MAX_DELETED_SHARE = 0.2
MAY_SHRINK = frozenset({"risers"})


def shrink_problem(table: str, staged: int, existing: int, deleted: int) -> str | None:
    """Why a sync of ``table`` looks like a broken source rather than a real change, or None."""
    if table in MAY_SHRINK:
        return None
    if staged == 0:
        return f"{table}: the source returned no rows"
    if deleted > MAX_DELETED_SHARE * existing:
        return f"{table}: the sync would delete {deleted} of {existing} rows"
    return None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", choices=SERVING_TABLES, default=SERVING_TABLES)
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="mirror the source even when it is empty or drops more than a fifth of a table",
    )
    args = parser.parse_args(argv)

    config = SyncConfig.from_env(os.environ)
    log.info("sync settings: %s", config.describe())

    with (
        databricks_sql.connect(**config.databricks_connect_kwargs()) as source,
        snowflake_connect(**config.snowflake_connect_kwargs()) as target,
    ):
        for table in args.tables:
            with source.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {qualified(config.source_catalog, config.source_schema, table)}"
                )
                arrow = cursor.fetchall_arrow()
            frame: pd.DataFrame = arrow.to_pandas(types_mapper=pd.ArrowDtype)
            frame.columns = [sf_ident(c) for c in frame.columns]

            schema = sf_ident(config.snowflake_schema)
            target_name = f"{schema}.{sf_ident(table)}"
            stage_name = f"{sf_ident(table)}__STAGE"
            cur = target.cursor()
            cur.execute(
                f"CREATE OR REPLACE TEMPORARY TABLE {schema}.{stage_name} LIKE {target_name}"
            )
            ok, _, rows, _ = write_pandas(
                target, frame, stage_name, schema=schema, quote_identifiers=False
            )
            if not ok:
                raise RuntimeError(f"staging load failed for {table}")
            keys = PRIMARY_KEYS[table]
            cur.execute(f"SELECT COUNT(*) FROM {target_name}")
            count = cur.fetchone()
            existing = count[0] if count else 0
            cur.execute("BEGIN")
            cur.execute(
                merge_sql(
                    target_name,
                    f"{schema}.{stage_name}",
                    list(arrow.column_names),
                    keys,
                    ignore=CHANGE_IGNORED,
                )
            )
            merged = cur.rowcount
            cur.execute(delete_missing_sql(target_name, f"{schema}.{stage_name}", keys))
            deleted = cur.rowcount or 0
            log.info(
                "%s: staged %s rows, had %s, merged %s, deleting %s",
                table,
                rows,
                existing,
                merged,
                deleted,
            )
            problem = shrink_problem(table, rows, existing, deleted)
            if problem and not args.allow_shrink:
                cur.execute("ROLLBACK")
                raise RuntimeError(
                    f"{problem}; rolled back (rerun with --allow-shrink if expected)"
                )
            cur.execute("COMMIT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
