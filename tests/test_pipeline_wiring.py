"""Check the Lakeflow pipeline file wires datasets correctly, without a Databricks runtime.

``pyspark.pipelines`` on Databricks provides expectation decorators that open-source Spark
does not, so a recording stand-in replaces the module. Each dataset function is then
called in dependency order against temp views built from the fixture.
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
    spark: SparkSession, bronze: DataFrame, monkeypatch: pytest.MonkeyPatch
) -> RecordingPipelines:
    fake = RecordingPipelines()
    monkeypatch.setitem(sys.modules, "pyspark.pipelines", fake)
    monkeypatch.setattr(pyspark, "pipelines", fake, raising=False)
    spark.conf.set("gridiron.landing_path", "/Volumes/test/gridiron/landing/pbp")
    bronze.createOrReplaceTempView("bronze_plays")

    spec = importlib.util.spec_from_file_location("gridiron_pipeline", PIPELINE_FILE)
    assert spec and spec.loader
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return fake


def test_pipeline_declares_expected_datasets(pipeline: RecordingPipelines) -> None:
    kinds = {name: d["kind"] for name, d in pipeline.datasets.items()}
    assert kinds == {
        "bronze_plays": "streaming_table",
        "plays": "materialized_view",
        "fourth_down_decisions": "materialized_view",
        "team_season_summary": "materialized_view",
        "game_summary": "materialized_view",
    }
    assert pipeline.datasets["plays"]["expect_all_or_drop"] == quality.SILVER_DROP_RULES
    assert pipeline.datasets["plays"]["expect_all"] == quality.SILVER_WARN_RULES
    assert (
        pipeline.datasets["fourth_down_decisions"]["expect_all_or_drop"]
        == quality.DECISION_DROP_RULES
    )


def test_pipeline_datasets_build_from_upstream_views(
    pipeline: RecordingPipelines, sample_pdf: Any
) -> None:
    for name in ("plays", "fourth_down_decisions", "team_season_summary", "game_summary"):
        frame: DataFrame = pipeline.datasets[name]["fn"]()
        frame.createOrReplaceTempView(name)
        assert frame.count() > 0, name
    assert pipeline.datasets["plays"]["fn"]().count() == len(sample_pdf)
