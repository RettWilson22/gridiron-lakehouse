"""The checked-in gold fixtures are the column contract for the generated Snowflake DDL;
check that the pipeline and model code still produce exactly those columns and types."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import DataFrame

from gridiron.serving import SERVING_TABLES
from gridiron.spark_io import arrow_table
from gridiron.workflow import Prepared, publish, run_backtest, score_live, train_live
from tests.conftest import GOLD

NOW = pd.Timestamp("2026-10-08 12:00", tz="UTC")
PIPELINE_TABLES = ("player_week", "team_week", "defense_vs_position")


def fixture_schema(name: str) -> pa.Schema:
    return pq.read_schema(GOLD / f"{name}.parquet")


def compatible(produced: pa.Schema, expected: pa.Schema) -> None:
    assert produced.names == expected.names
    for field in produced:
        want = expected.field(field.name).type
        if pa.types.is_null(field.type) or pa.types.is_null(want):
            continue  # an all-null column in a small sample has no type of its own
        if pa.types.is_timestamp(field.type) and pa.types.is_timestamp(want):
            continue  # unit (us/ns) differs between Spark and pandas writers
        assert field.type == want, field.name


def test_every_serving_table_has_a_fixture() -> None:
    assert {p.stem for p in GOLD.glob("*.parquet")} == set(SERVING_TABLES)


@pytest.mark.parametrize("name", PIPELINE_TABLES)
def test_pipeline_tables_match_the_contract(name: str, gold: dict[str, DataFrame]) -> None:
    compatible(gold[name].toArrow().schema, fixture_schema(name))


def test_model_tables_match_the_contract(prepared: Prepared) -> None:
    backtest, metrics = run_backtest(prepared, [2025], NOW)
    live = score_live(prepared, train_live(prepared).predict, "test", None, NOW)
    projections, risers = publish(prepared, backtest, live)
    for name, frame in {
        "projections": projections,
        "risers": risers,
        "backtest_metrics": metrics,
    }.items():
        compatible(arrow_table(frame).schema, fixture_schema(name))


def test_offseason_tables_keep_the_contract(prepared: Prepared) -> None:
    """With no upcoming week there are no risers, but the table keeps its columns and
    types: a changed schema makes publish_serving replace the serving table (losing its
    Iceberg identity) and an untyped empty Parquet file breaks the DuckDB dbt build."""
    offseason = Prepared(prepared.frame, None, prepared.tables)
    backtest, _ = run_backtest(offseason, [2025], NOW)
    projections, risers = publish(offseason, backtest, None)
    assert risers.empty
    compatible(arrow_table(projections).schema, fixture_schema("projections"))
    written = pa.Table.from_pandas(risers, preserve_index=False).schema  # as to_parquet sees it
    expected = fixture_schema("risers")
    assert written.names == expected.names
    for field in written:
        assert field.type == expected.field(field.name).type, field.name
