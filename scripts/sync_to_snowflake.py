"""Sync: copy the Databricks serving tables into native Snowflake tables (SYNCED).

This is the production path with Databricks Free Edition: its Unity Catalog Iceberg REST
endpoint returns table metadata but no storage credentials for tables on Databricks default
storage, so Snowflake cannot read the files in place (details in docs/deploy.md). For each
serving table the script:

1. reads the table from a Databricks SQL warehouse as Arrow;
2. loads it into a temporary staging table shaped exactly like the target;
3. MERGEs changed and new rows into ``SYNCED.<TABLE>`` and deletes rows that disappeared,
   in one transaction, so the stream on ``SYNCED.PLAYER_WEEK`` only sees real changes.

Configuration comes from environment variables (see ``.env.example``); nothing is
hard-coded. Databricks auth uses ``DATABRICKS_TOKEN`` if set, otherwise a Databricks CLI
profile. Run ``snowflake/sync/01_synced_tables.sql`` once before the first sync.

    python scripts/sync_to_snowflake.py                       # all serving tables
    python scripts/sync_to_snowflake.py --tables projections
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", choices=SERVING_TABLES, default=SERVING_TABLES)
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
            cur.execute("BEGIN")
            cur.execute(
                merge_sql(target_name, f"{schema}.{stage_name}", list(arrow.column_names), keys)
            )
            merged = cur.rowcount
            cur.execute(delete_missing_sql(target_name, f"{schema}.{stage_name}", keys))
            deleted = cur.rowcount
            cur.execute("COMMIT")
            log.info("%s: staged %s rows, merged %s, deleted %s", table, rows, merged, deleted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
