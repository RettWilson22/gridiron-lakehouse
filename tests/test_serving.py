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
    create, refresh = publish_statements(
        "workspace", "gridiron", "gridiron_serving", "game_summary"
    )
    assert create.startswith(
        "CREATE TABLE IF NOT EXISTS `workspace`.`gridiron_serving`.`game_summary`"
    )
    assert table_properties_sql() in create
    assert create.endswith("FROM `workspace`.`gridiron`.`game_summary` WHERE 1 = 0")
    assert refresh == (
        "INSERT OVERWRITE `workspace`.`gridiron_serving`.`game_summary` "
        "SELECT * FROM `workspace`.`gridiron`.`game_summary`"
    )
    assert "REPLACE" not in create


@pytest.mark.parametrize("bad", ["gridiron; DROP TABLE x", "a.b", "", "1abc"])
def test_identifiers_are_validated(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe identifier"):
        qualified("workspace", bad, "t")


def test_serving_tables_include_model_outputs() -> None:
    assert {"fourth_down_scored", "coach_aggressiveness"} <= set(SERVING_TABLES)
