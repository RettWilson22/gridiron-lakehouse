from __future__ import annotations

import pytest

from gridiron.serving import (
    SERVING_TABLES,
    UNIFORM_PROPERTIES,
    publish_statements,
    qualified,
    table_properties_sql,
)


def test_uniform_properties_enable_iceberg_without_deletion_vectors() -> None:
    assert UNIFORM_PROPERTIES["delta.enableIcebergCompatV2"] == "true"
    assert UNIFORM_PROPERTIES["delta.universalFormat.enabledFormats"] == "iceberg"
    assert UNIFORM_PROPERTIES["delta.enableDeletionVectors"] == "false"
    assert UNIFORM_PROPERTIES["delta.columnMapping.mode"] in {"name", "id"}


def test_publish_creates_once_then_overwrites_in_place() -> None:
    create, refresh = publish_statements("workspace", "gridiron", "gridiron_serving", "projections")
    assert create.startswith(
        "CREATE TABLE IF NOT EXISTS `workspace`.`gridiron_serving`.`projections`"
    )
    assert table_properties_sql() in create
    assert create.endswith("FROM `workspace`.`gridiron`.`projections` WHERE 1 = 0")
    assert refresh == (
        "INSERT OVERWRITE `workspace`.`gridiron_serving`.`projections` "
        "SELECT * FROM `workspace`.`gridiron`.`projections`"
    )
    assert "REPLACE" not in create


def test_unchanged_columns_keep_the_table_identity() -> None:
    statements = publish_statements(
        "workspace",
        "gridiron",
        "gridiron_serving",
        "risers",
        existing_columns=["season", "week"],
        source_columns=["season", "week"],
    )
    assert len(statements) == 2 and "REPLACE" not in " ".join(statements)


def test_changed_columns_replace_the_table() -> None:
    (statement,) = publish_statements(
        "workspace",
        "gridiron",
        "gridiron_serving",
        "risers",
        existing_columns=["season", "week"],
        source_columns=["season", "week", "is_riser"],
    )
    assert statement.startswith("CREATE OR REPLACE TABLE `workspace`.`gridiron_serving`.`risers`")
    assert table_properties_sql() in statement


@pytest.mark.parametrize("bad", ["gridiron; DROP TABLE x", "a.b", "", "1abc"])
def test_identifiers_are_validated(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe identifier"):
        qualified("workspace", bad, "t")


def test_serving_tables_include_model_outputs() -> None:
    assert {"projections", "risers", "backtest_metrics", "player_week"} <= set(SERVING_TABLES)
