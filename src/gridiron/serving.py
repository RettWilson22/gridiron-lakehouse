"""Serving layer: UniForm (Iceberg-readable) copies of gold tables for Snowflake.

Lakeflow materialized views and streaming tables cannot carry the ``IcebergCompatV2``
UniForm properties, so the job publishes plain Unity Catalog managed Delta tables into a
dedicated serving schema. That schema is also the only one the Snowflake principal is
granted ``EXTERNAL USE SCHEMA`` on.

Tables are created once and then refreshed with ``INSERT OVERWRITE`` (not
``CREATE OR REPLACE``) so the table keeps its identity in the Iceberg REST catalog and
Snowflake's Iceberg tables keep pointing at it.
"""

from __future__ import annotations

import re
from typing import Final

# Gold datasets published for Snowflake, in dependency-free order.
SERVING_TABLES: Final = (
    "fourth_down_decisions",
    "fourth_down_scored",
    "team_season_summary",
    "game_summary",
    "coach_aggressiveness",
)

UNIFORM_PROPERTIES: Final[dict[str, str]] = {
    "delta.columnMapping.mode": "name",
    "delta.enableIcebergCompatV2": "true",
    "delta.universalFormat.enabledFormats": "iceberg",
    # Iceberg reads cannot be enabled on tables with deletion vectors.
    "delta.enableDeletionVectors": "false",
}

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"unsafe identifier: {name!r}")
    return f"`{name}`"


def qualified(catalog: str, schema: str, table: str) -> str:
    return ".".join(_ident(part) for part in (catalog, schema, table))


def table_properties_sql(properties: dict[str, str] = UNIFORM_PROPERTIES) -> str:
    pairs = ",\n  ".join(f"'{key}' = '{value}'" for key, value in properties.items())
    return f"TBLPROPERTIES (\n  {pairs}\n)"


def publish_statements(
    catalog: str, source_schema: str, serving_schema: str, table: str
) -> list[str]:
    """SQL to create (first run only) and refresh one UniForm serving table."""
    source = qualified(catalog, source_schema, table)
    target = qualified(catalog, serving_schema, table)
    return [
        f"CREATE TABLE IF NOT EXISTS {target}\n{table_properties_sql()}\n"
        f"AS SELECT * FROM {source} WHERE 1 = 0",
        f"INSERT OVERWRITE {target} SELECT * FROM {source}",
    ]
