"""Pure helpers that generate Snowflake SQL for the serving tables.

Used by ``scripts/generate_snowflake_sql.py`` (checked-in DDL and views) and by the sync
script (MERGE statements). Nothing here opens a connection, so it is unit tested.

Naming convention on the Snowflake side: every serving table is exposed with unquoted,
upper-case identifiers in both the ``ICEBERG`` (views over Iceberg tables) and ``SYNCED``
(native tables) schemas, so dbt can switch between them with a single variable.
"""

from __future__ import annotations

import re
from typing import Final

import pyarrow as pa

PRIMARY_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "player_week": ("season", "week", "player_id"),
    "team_week": ("season", "week", "team"),
    "defense_vs_position": ("season", "week", "team", "position"),
    "projections": ("season", "week", "player_id"),
    "risers": ("season", "week", "player_id"),
    "backtest_metrics": ("scope", "position", "method"),
}

_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sf_ident(name: str) -> str:
    """Unquoted Snowflake identifier (resolves to upper case); rejects anything unusual."""
    if not _SAFE.match(name):
        raise ValueError(f"unsafe identifier: {name!r}")
    return name.upper()


def snowflake_type(arrow_type: pa.DataType) -> str:
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    if pa.types.is_integer(arrow_type):
        return "NUMBER(19,0)" if arrow_type.bit_width == 64 else "NUMBER(10,0)"
    if pa.types.is_floating(arrow_type):
        return "FLOAT"
    if pa.types.is_date(arrow_type):
        return "DATE"
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP_TZ" if arrow_type.tz else "TIMESTAMP_NTZ"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "VARCHAR"
    raise TypeError(f"no Snowflake mapping for {arrow_type}")


def create_table_sql(schema: str, table: str, arrow_schema: pa.Schema) -> str:
    columns = ",\n".join(f"    {sf_ident(f.name)} {snowflake_type(f.type)}" for f in arrow_schema)
    keys = ", ".join(sf_ident(k) for k in PRIMARY_KEYS[table])
    return (
        f"CREATE TABLE IF NOT EXISTS {sf_ident(schema)}.{sf_ident(table)} (\n"
        f"{columns},\n    PRIMARY KEY ({keys})\n);"
    )


def iceberg_table_sql(schema: str, table: str, catalog_integration: str) -> str:
    """Externally managed Iceberg table over a Unity Catalog serving table (lower case)."""
    return (
        f"CREATE ICEBERG TABLE IF NOT EXISTS {sf_ident(schema)}.{sf_ident(table)}\n"
        f"  CATALOG = '{sf_ident(catalog_integration)}'\n"
        f"  CATALOG_TABLE_NAME = '{table}'\n"
        f"  AUTO_REFRESH = TRUE;"
    )


def iceberg_view_sql(
    view_schema: str, iceberg_schema: str, table: str, arrow_schema: pa.Schema
) -> str:
    """Upper-case view over an Iceberg table whose Unity Catalog names are lower case."""
    columns = ",\n".join(f'    "{f.name}" AS {sf_ident(f.name)}' for f in arrow_schema)
    return (
        f"CREATE OR REPLACE VIEW {sf_ident(view_schema)}.{sf_ident(table)} AS\nSELECT\n"
        f"{columns}\nFROM {sf_ident(iceberg_schema)}.{sf_ident(table)};"
    )


def merge_sql(target: str, staging: str, columns: list[str], keys: tuple[str, ...]) -> str:
    """Upsert staging into target, touching only rows whose values actually changed.

    Skipping unchanged rows keeps the stream on the target table free of no-op updates.
    """
    cols = [sf_ident(c) for c in columns]
    key_cols = [sf_ident(k) for k in keys]
    missing = set(key_cols) - set(cols)
    if missing:
        raise ValueError(f"key columns not in table: {sorted(missing)}")
    non_keys = [c for c in cols if c not in key_cols]
    on = " AND ".join(f"t.{k} = s.{k}" for k in key_cols)
    changed = " OR ".join(f"NOT EQUAL_NULL(t.{c}, s.{c})" for c in non_keys)
    update = ", ".join(f"{c} = s.{c}" for c in non_keys)
    insert_cols = ", ".join(cols)
    insert_vals = ", ".join(f"s.{c}" for c in cols)
    return (
        f"MERGE INTO {target} t USING {staging} s ON {on}\n"
        f"WHEN MATCHED AND ({changed}) THEN UPDATE SET {update}\n"
        f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    )


def delete_missing_sql(target: str, staging: str, keys: tuple[str, ...]) -> str:
    """Remove target rows that no longer exist upstream, so the copy mirrors the source."""
    on = " AND ".join(f"s.{sf_ident(k)} = t.{sf_ident(k)}" for k in keys)
    return f"DELETE FROM {target} t WHERE NOT EXISTS (SELECT 1 FROM {staging} s WHERE {on})"
