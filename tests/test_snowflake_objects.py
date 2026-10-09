"""Static checks of the hand-written Snowflake SQL against the generated table contracts."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gridiron import scoring
from gridiron.serving import SERVING_TABLES
from tests.conftest import GOLD, REPO_ROOT

SNOWFLAKE = REPO_ROOT / "snowflake"
ALIASES = {"p": "projections", "w": "player_week", "g": "team_week"}


def columns(table: str) -> set[str]:
    return {name.upper() for name in pq.read_schema(GOLD / f"{table}.parquet").names}


@pytest.mark.parametrize(
    "path",
    [
        SNOWFLAKE / "streams_tasks" / "01_projection_results_stream_task.sql",
        SNOWFLAKE / "streams_tasks" / "02_projection_results_dynamic_table.sql",
    ],
)
def test_results_sql_only_uses_existing_tables_and_columns(path: Path) -> None:
    sql = path.read_text()
    for schema, table in re.findall(r"\b(SYNCED|ICEBERG)\.([A-Z_]+)\b", sql):
        if not table.endswith("_CHANGES"):
            assert table.lower() in SERVING_TABLES, f"{schema}.{table}"
    for alias, column in re.findall(r"\b([pwg])\.([A-Z_]+)\b", sql):
        assert column in columns(ALIASES[alias]), f"{alias}.{column} in {path.name}"


def test_udf_registration_points_at_the_scoring_handler() -> None:
    sql = (SNOWFLAKE / "snowpark" / "01_create_udf.sql").read_text()
    handler = re.search(r"HANDLER = '(\w+)\.(\w+)'", sql)
    assert handler and handler.groups() == ("scoring", "udf_handler")
    assert "PUT file://src/gridiron/scoring.py @APP.CODE" in sql
    assert callable(scoring.udf_handler)
    for column in re.findall(r"'([a-z_]+)', [A-Z_]+", sql):
        assert column in scoring.STATS or column == "position", column


def test_udf_handler_needs_only_the_standard_library() -> None:
    source = (REPO_ROOT / "src" / "gridiron" / "scoring.py").read_text()
    imports = re.findall(r"^(?:from|import) ([\w.]+)", source, flags=re.MULTILINE)
    assert set(imports) <= {"__future__", "collections.abc", "math", "typing"}
