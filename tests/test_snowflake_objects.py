"""Static checks of the hand-written Snowflake SQL against the generated table contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gridiron import scoring
from gridiron.serving import SERVING_TABLES
from gridiron.tiers import POOL
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


def test_sql_pool_sizes_match_the_python_pool() -> None:
    """The evaluation pool (positional top N) is hard-coded in dbt and in the task SQL."""
    dbt_project = (SNOWFLAKE / "dbt" / "dbt_project.yml").read_text()
    pool_var = re.search(r"^\s*pool_size: (\{.*\})\s*$", dbt_project, flags=re.MULTILINE)
    assert pool_var, "pool_size var not found in dbt_project.yml"
    assert json.loads(pool_var.group(1)) == POOL

    task_sql = (SNOWFLAKE / "streams_tasks" / "01_projection_results_stream_task.sql").read_text()
    decode = re.search(r"DECODE\(POSITION, ([^)]*)\)", task_sql)
    assert decode, "DECODE(POSITION, ...) not found in the stream task SQL"
    pairs = re.findall(r"'(\w+)', (\d+)", decode.group(1))
    assert {position: int(size) for position, size in pairs} == POOL


def test_every_results_stream_wakes_the_task_and_feeds_the_changed_weeks() -> None:
    sql = (SNOWFLAKE / "streams_tasks" / "01_projection_results_stream_task.sql").read_text()
    streams = re.findall(r"CREATE STREAM IF NOT EXISTS (SYNCED\.\w+)\s+ON TABLE (SYNCED\.\w+)", sql)
    assert {table for _, table in streams} == {"SYNCED.PLAYER_WEEK", "SYNCED.PROJECTIONS"}
    when = re.search(r"\bWHEN (.*?)\nAS\n", sql, flags=re.DOTALL)
    assert when, "task WHEN clause not found"
    changed = re.search(r"INSERT INTO APP\.RESULTS_CHANGED_WEEKS.*?;", sql, flags=re.DOTALL)
    assert changed, "changed-weeks INSERT not found"
    for stream, _ in streams:
        assert f"SYSTEM$STREAM_HAS_DATA('GRIDIRON.{stream}')" in when.group(1), stream
        assert f"FROM {stream}" in changed.group(0), stream


def grants_to(role: str, sql: str) -> list[str]:
    """Every GRANT statement to ``role`` in a script, comments removed, on one line."""
    statements = " ".join(re.sub(r"--[^\n]*", "", sql).split()).split(";")
    return [
        s.strip() for s in statements if s.strip().startswith("GRANT ") and f"TO ROLE {role}" in s
    ]


def test_the_app_owner_role_only_reaches_marts_and_the_scoring_function() -> None:
    roles = (SNOWFLAKE / "setup" / "02_database_and_roles.sql").read_text()
    udf = (SNOWFLAKE / "snowpark" / "01_create_udf.sql").read_text()
    assert "CREATE ROLE IF NOT EXISTS GRIDIRON_APP_OWNER;" in roles
    grants = grants_to("GRIDIRON_APP_OWNER", roles + udf)
    assert grants
    for grant in grants:
        assert not re.search(r"SYNCED|STAGING|ICEBERG|ALL FUNCTIONS|FUTURE FUNCTIONS", grant), grant
    assert any("CREATE STREAMLIT ON SCHEMA GRIDIRON.APP" in g for g in grants)
    assert any("ON FUNCTION APP.FANTASY_POINTS(OBJECT, OBJECT)" in g for g in grants)


def test_the_app_is_created_as_the_app_owner() -> None:
    sql = (SNOWFLAKE / "streamlit" / "01_create_streamlit.sql").read_text()
    owner = sql.index("USE ROLE GRIDIRON_APP_OWNER;")
    assert owner < sql.index("CREATE STREAMLIT")
    assert "USE ROLE" not in sql[owner + 1 : sql.index("CREATE STREAMLIT")]


def test_reader_and_loader_grants() -> None:
    roles = (SNOWFLAKE / "setup" / "02_database_and_roles.sql").read_text()
    reader = grants_to("GRIDIRON_READER", roles)
    assert reader and not [g for g in reader if "STAGING" in g]
    loader = grants_to("GRIDIRON_LOADER", roles)
    assert any("CREATE STAGE ON SCHEMA GRIDIRON.SYNCED" in g for g in loader)  # write_pandas
