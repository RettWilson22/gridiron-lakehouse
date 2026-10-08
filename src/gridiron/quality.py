"""Data-quality expectations for the silver layer.

The same rule dictionaries drive Lakeflow expectations on Databricks
(``@dp.expect_all_or_drop`` / ``@dp.expect_all``) and the local test helpers below, so the
rules are written once as SQL boolean expressions. Every rule is written to be null-safe
(it evaluates to TRUE or FALSE, never NULL) so local and pipeline behaviour cannot diverge.
"""

from __future__ import annotations

from typing import Final

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Rows failing any of these are dropped from silver.
SILVER_DROP_RULES: Final[dict[str, str]] = {
    "has_game_id": "game_id IS NOT NULL",
    "has_play_id": "play_id IS NOT NULL",
    "has_season": "season IS NOT NULL",
    "valid_season_type": "season_type IS NOT NULL AND season_type IN ('REG', 'POST')",
}

# Rows failing these are kept but counted in the pipeline event log.
SILVER_WARN_RULES: Final[dict[str, str]] = {
    "down_in_range": "down IS NULL OR down BETWEEN 1 AND 4",
    "yardline_in_range": "yardline_100 IS NULL OR yardline_100 BETWEEN 1 AND 99",
    "ydstogo_positive": "ydstogo IS NULL OR ydstogo >= 0",
    "wp_is_probability": "wp IS NULL OR wp BETWEEN 0 AND 1",
    "fourth_down_outcome_exclusive": (
        "NOT (coalesce(fourth_down_converted, 0) = 1 AND coalesce(fourth_down_failed, 0) = 1)"
    ),
}

# Gold fourth-down rows must have the context the model needs.
DECISION_DROP_RULES: Final[dict[str, str]] = {
    "has_context": (
        "ydstogo IS NOT NULL AND yardline_100 IS NOT NULL "
        "AND game_seconds_remaining IS NOT NULL AND score_differential IS NOT NULL"
    ),
    "known_decision": "decision IS NOT NULL AND decision IN ('go', 'punt', 'field_goal')",
}


def combined(rules: dict[str, str]) -> str:
    return " AND ".join(f"({expr})" for expr in rules.values())


def apply_drop_rules(df: DataFrame, rules: dict[str, str]) -> DataFrame:
    """Local equivalent of ``expect_all_or_drop``."""
    return df.where(F.coalesce(F.expr(combined(rules)), F.lit(False)))


def count_failures(df: DataFrame, rules: dict[str, str]) -> dict[str, int]:
    """Local equivalent of the event-log metrics for ``expect_all``."""
    aggs = [
        F.sum(F.when(F.coalesce(F.expr(expr), F.lit(False)), 0).otherwise(1)).alias(name)
        for name, expr in rules.items()
    ]
    row = df.agg(*aggs).first()
    assert row is not None
    return {name: int(row[name] or 0) for name in rules}
