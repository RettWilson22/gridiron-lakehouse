"""Check the Lakeflow pipeline file wires datasets correctly, without a Databricks runtime.

``pyspark.pipelines`` on Databricks provides expectation decorators that open-source Spark
does not, so a recording stand-in replaces the module. Each dataset function is then
called in dependency order against temp views built from the landing fixture.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pyspark
import pytest
from pyspark.sql import DataFrame, SparkSession

from gridiron import quality
from gridiron.lakehouse import BRONZE_TABLES, GOLD_TABLES, SILVER_TABLES

PIPELINE_FILE = Path(__file__).parent.parent / "databricks" / "pipelines" / "gridiron_pipeline.py"


class RecordingPipelines(ModuleType):
    def __init__(self) -> None:
        super().__init__("pyspark.pipelines")
        self.datasets: dict[str, dict[str, Any]] = {}

    def _register(self, kind: str, **options: Any) -> Callable[[Callable[[], DataFrame]], Any]:
        def decorator(fn: Callable[[], DataFrame]) -> Callable[[], DataFrame]:
            entry = self.datasets.setdefault(options.get("name", fn.__name__), {})
            entry.update(kind=kind, fn=fn, options=options)
            return fn

        return decorator

    def table(self, **options: Any) -> Callable[[Callable[[], DataFrame]], Any]:
        return self._register("streaming_table", **options)

    def materialized_view(self, **options: Any) -> Callable[[Callable[[], DataFrame]], Any]:
        return self._register("materialized_view", **options)

    def _expect(self, key: str, rules: dict[str, str]) -> Callable[[Any], Any]:
        def decorator(fn: Any) -> Any:
            self.datasets.setdefault(fn.__name__, {}).setdefault(key, {}).update(rules)
            return fn

        return decorator

    def expect_all(self, rules: dict[str, str]) -> Callable[[Any], Any]:
        return self._expect("expect_all", rules)

    def expect_all_or_drop(self, rules: dict[str, str]) -> Callable[[Any], Any]:
        return self._expect("expect_all_or_drop", rules)


@pytest.fixture
def pipeline(
    spark: SparkSession, bronze: dict[str, DataFrame], monkeypatch: pytest.MonkeyPatch
) -> RecordingPipelines:
    fake = RecordingPipelines()
    monkeypatch.setitem(sys.modules, "pyspark.pipelines", fake)
    monkeypatch.setattr(pyspark, "pipelines", fake, raising=False)
    spark.conf.set("gridiron.landing_root", "/Volumes/test/gridiron/landing")
    spark.conf.set("gridiron.first_season", "2018")
    for dataset, table in BRONZE_TABLES.items():
        bronze[dataset].createOrReplaceTempView(table)

    spec = importlib.util.spec_from_file_location("gridiron_pipeline", PIPELINE_FILE)
    assert spec and spec.loader
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return fake


def test_pipeline_declares_the_same_graph_as_the_local_build(pipeline: RecordingPipelines) -> None:
    kinds = {name: d["kind"] for name, d in pipeline.datasets.items()}
    assert kinds == {
        **dict.fromkeys(BRONZE_TABLES.values(), "streaming_table"),
        **dict.fromkeys((*SILVER_TABLES, *GOLD_TABLES), "materialized_view"),
    }


def test_expectations_match_the_quality_rules(pipeline: RecordingPipelines) -> None:
    for name in (*SILVER_TABLES, *GOLD_TABLES):
        entry = pipeline.datasets[name]
        assert entry.get("expect_all_or_drop", {}) == quality.DROP.get(name, {}), name
        assert entry.get("expect_all", {}) == quality.WARN.get(name, {}), name


def test_bronze_tables_read_their_landing_folders(pipeline: RecordingPipelines) -> None:
    for dataset, table in BRONZE_TABLES.items():
        assert dataset in pipeline.datasets[table]["options"]["comment"]


def test_pipeline_datasets_build_from_upstream_views(pipeline: RecordingPipelines) -> None:
    order = (  # dependencies first
        "players",
        "player_ids",
        "schedules",
        "plays",
        "player_stats",
        "snap_counts",
        "rosters",
        "injuries",
        "depth_charts",
        "ff_opportunity",
        "ecr",
        "team_week",
        "player_week",
        "defense_vs_position",
        "depth_chart_week",
    )
    assert set(order) == set(SILVER_TABLES) | set(GOLD_TABLES)
    for name in order:
        frame: DataFrame = pipeline.datasets[name]["fn"]()
        frame.createOrReplaceTempView(name)
        assert frame.count() > 0, name
