from __future__ import annotations

import datetime as dt

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from gridiron import quality, transforms
from gridiron.config import SILVER_COLUMNS


def test_silver_has_contract_columns_and_types(plays: DataFrame) -> None:
    types = dict(plays.dtypes)
    for name, spark_type in SILVER_COLUMNS.items():
        assert types[name] == spark_type, name
    assert transforms.INGESTED_AT in types


def test_silver_is_unique_on_game_and_play(plays: DataFrame, sample_pdf: pd.DataFrame) -> None:
    assert plays.count() == len(sample_pdf)
    assert plays.select("game_id", "play_id").distinct().count() == plays.count()


def test_silver_keeps_latest_ingested_version(bronze: DataFrame) -> None:
    newer = bronze.limit(5).select(
        *(F.col(c) for c in bronze.columns if c not in {"yards_gained", transforms.INGESTED_AT}),
        F.lit(99.0).alias("yards_gained"),
        F.lit(dt.datetime(2100, 1, 1)).alias(transforms.INGESTED_AT),
    )
    combined = bronze.unionByName(newer)
    silver = transforms.silver_plays(combined)

    assert silver.count() == bronze.count()
    assert silver.where(F.col("yards_gained") == 99).count() == 5


def test_drop_rules_remove_rows_without_keys(plays: DataFrame) -> None:
    broken = plays.limit(3).withColumn("game_id", F.lit(None).cast("string"))
    kept = quality.apply_drop_rules(plays.unionByName(broken), quality.SILVER_DROP_RULES)
    assert kept.count() == plays.count()


def test_warn_rules_count_violations(plays: DataFrame) -> None:
    assert set(quality.count_failures(plays, quality.SILVER_WARN_RULES).values()) == {0}
    bad = plays.limit(2).withColumn("down", F.lit(7))
    counts = quality.count_failures(plays.unionByName(bad), quality.SILVER_WARN_RULES)
    assert counts["down_in_range"] == 2


def test_fourth_down_decisions_match_raw_counts(
    decisions_pdf: pd.DataFrame, sample_pdf: pd.DataFrame
) -> None:
    raw = sample_pdf[(sample_pdf["down"] == 4) & sample_pdf["posteam"].notna()]
    expected = (
        raw["play_type"]
        .map({"run": "go", "pass": "go", "punt": "punt", "field_goal": "field_goal"})
        .value_counts()
    )
    actual = decisions_pdf["decision"].value_counts()
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False)


def test_fourth_down_outcomes_are_consistent(decisions_pdf: pd.DataFrame) -> None:
    go = decisions_pdf[decisions_pdf["decision"] == "go"]
    kicks = decisions_pdf[decisions_pdf["decision"] != "go"]
    assert set(go["outcome"]) <= {"converted", "failed"}
    assert (go["converted"] == (go["outcome"] == "converted")).all()
    assert kicks["converted"].isna().all()
    assert {"converted", "failed", "fg_made", "fg_missed", "punt"} <= set(decisions_pdf["outcome"])


def test_decision_coach_is_the_offense_coach(decisions_pdf: pd.DataFrame) -> None:
    det = decisions_pdf[decisions_pdf["posteam"] == "DET"]
    assert set(det["coach"]) == {"Dan Campbell"}


def test_team_season_summary(plays: DataFrame, decisions: DataFrame) -> None:
    summary = transforms.team_season_summary(plays, decisions).toPandas()
    assert summary[["season", "team"]].duplicated().sum() == 0
    det_2024 = summary[(summary["season"] == 2024) & (summary["team"] == "DET")].iloc[0]
    assert det_2024["go_rate"] == det_2024["go_attempts"] / det_2024["fourth_downs"]
    assert 0 < det_2024["success_rate"] < 1


def test_game_summary(plays: DataFrame, decisions: DataFrame) -> None:
    games = transforms.game_summary(plays, decisions).toPandas().set_index("game_id")
    assert len(games) == 3
    gb = games.loc["2024_09_DET_GB"]
    assert (gb["home_team"], gb["away_team"]) == ("GB", "DET")
    assert gb["winner"] == ("GB" if gb["home_score"] > gb["away_score"] else "DET")
    totals = games["home_fourth_downs"] + games["away_fourth_downs"]
    assert int(totals.sum()) == decisions.count()


def test_first_down_expected_points(plays: DataFrame) -> None:
    first = transforms.first_down_expected_points(plays)
    assert first.columns == ["season", "yardline_100", "ep"]
    assert first.where(F.col("ep").isNull()).count() == 0
