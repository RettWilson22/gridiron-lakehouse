"""Serving layer: UniForm (Iceberg-readable) copies of the gold tables Snowflake reads.

Lakeflow materialized views cannot carry the ``IcebergCompatV2`` UniForm properties, so
the publish task copies each serving dataset into a plain Unity Catalog managed Delta table
in a dedicated serving schema. That schema is also the only one an external reader would be
granted access to.

Tables are created once and then refreshed with ``INSERT OVERWRITE`` (not
``CREATE OR REPLACE``) so each keeps its identity in the Iceberg REST catalog. Only when a
table's columns change is it replaced, because ``INSERT OVERWRITE`` cannot change a schema.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

# Datasets published for Snowflake. The first three are pipeline outputs; the rest are
# written by the train and score job tasks.
SERVING_TABLES: Final = (
    "player_week",
    "team_week",
    "defense_vs_position",
    "projections",
    "risers",
    "backtest_metrics",
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
    catalog: str,
    source_schema: str,
    serving_schema: str,
    table: str,
    *,
    existing_columns: Sequence[str] | None = None,
    source_columns: Sequence[str] | None = None,
) -> list[str]:
    """SQL to publish one serving table.

    ``existing_columns`` are the serving table's current columns (``None`` if it does not
    exist yet) and ``source_columns`` the gold table's. When both are given and differ, the
    table is replaced; otherwise it is created if missing and overwritten in place.
    """
    source = qualified(catalog, source_schema, table)
    target = qualified(catalog, serving_schema, table)
    if (
        existing_columns is not None
        and source_columns is not None
        and list(existing_columns) != list(source_columns)
    ):
        return [
            f"CREATE OR REPLACE TABLE {target}\n{table_properties_sql()}\nAS SELECT * FROM {source}"
        ]
    return [
        f"CREATE TABLE IF NOT EXISTS {target}\n{table_properties_sql()}\n"
        f"AS SELECT * FROM {source} WHERE 1 = 0",
        f"INSERT OVERWRITE {target} SELECT * FROM {source}",
    ]
