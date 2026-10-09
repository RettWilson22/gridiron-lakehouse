from __future__ import annotations

import pyarrow as pa
import pytest

from gridiron.serving import SERVING_TABLES
from gridiron.snowflake_sql import (
    PRIMARY_KEYS,
    create_table_sql,
    delete_missing_sql,
    iceberg_table_sql,
    iceberg_view_sql,
    merge_sql,
    sf_ident,
    snowflake_type,
)
from scripts.generate_snowflake_sql import main as generate_main


def test_every_serving_table_has_a_primary_key() -> None:
    assert set(PRIMARY_KEYS) == set(SERVING_TABLES)


@pytest.mark.parametrize(
    ("arrow_type", "expected"),
    [
        (pa.int32(), "NUMBER(10,0)"),
        (pa.int64(), "NUMBER(19,0)"),
        (pa.float64(), "FLOAT"),
        (pa.bool_(), "BOOLEAN"),
        (pa.string(), "VARCHAR"),
        (pa.large_string(), "VARCHAR"),
        (pa.date32(), "DATE"),
        (pa.timestamp("us", tz="UTC"), "TIMESTAMP_TZ"),
    ],
)
def test_snowflake_type_mapping(arrow_type: pa.DataType, expected: str) -> None:
    assert snowflake_type(arrow_type) == expected


def test_unmapped_type_raises() -> None:
    with pytest.raises(TypeError):
        snowflake_type(pa.list_(pa.int32()))


def test_identifiers_are_upper_cased_and_validated() -> None:
    assert sf_ident("game_id") == "GAME_ID"
    with pytest.raises(ValueError):
        sf_ident('x"; DROP TABLE y; --')


SCHEMA = pa.schema(
    [
        ("season", pa.int32()),
        ("week", pa.int32()),
        ("team", pa.string()),
        ("implied_points", pa.float64()),
    ]
)


def test_create_table_sql() -> None:
    sql = create_table_sql("synced", "team_week", SCHEMA)
    assert sql.startswith("CREATE TABLE IF NOT EXISTS SYNCED.TEAM_WEEK (")
    assert "    IMPLIED_POINTS FLOAT," in sql
    assert "PRIMARY KEY (SEASON, WEEK, TEAM)" in sql


def test_iceberg_table_points_at_the_lower_case_catalog_name() -> None:
    sql = iceberg_table_sql("ICEBERG_RAW", "team_week", "GRIDIRON_UNITY_CATALOG")
    assert sql.startswith("CREATE ICEBERG TABLE IF NOT EXISTS ICEBERG_RAW.TEAM_WEEK")
    assert "CATALOG_TABLE_NAME = 'team_week'" in sql


def test_iceberg_view_maps_lower_case_columns() -> None:
    sql = iceberg_view_sql("ICEBERG", "ICEBERG_RAW", "team_week", SCHEMA)
    assert '"implied_points" AS IMPLIED_POINTS' in sql
    assert sql.endswith("FROM ICEBERG_RAW.TEAM_WEEK;")


def test_merge_only_updates_changed_rows() -> None:
    sql = merge_sql(
        "SYNCED.T", "SYNCED.T__STAGE", ["season", "team", "proj_ppr"], ("season", "team")
    )
    assert "ON t.SEASON = s.SEASON AND t.TEAM = s.TEAM" in sql
    assert "WHEN MATCHED AND (NOT EQUAL_NULL(t.PROJ_PPR, s.PROJ_PPR))" in sql
    assert "UPDATE SET PROJ_PPR = s.PROJ_PPR" in sql
    assert "INSERT (SEASON, TEAM, PROJ_PPR) VALUES (s.SEASON, s.TEAM, s.PROJ_PPR)" in sql


def test_merge_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="key columns"):
        merge_sql("T", "S", ["a"], ("b",))


def test_delete_missing_sql() -> None:
    assert delete_missing_sql("SYNCED.T", "SYNCED.S", ("player_id",)) == (
        "DELETE FROM SYNCED.T t WHERE NOT EXISTS "
        "(SELECT 1 FROM SYNCED.S s WHERE s.PLAYER_ID = t.PLAYER_ID)"
    )


def test_generated_sql_is_up_to_date() -> None:
    assert generate_main(["--check"]) == 0, "run scripts/generate_snowflake_sql.py"
