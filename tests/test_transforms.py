from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from gridiron import quality, transforms
from gridiron.config import POSITIONS


def test_latest_landed_keeps_only_the_newest_file_per_season(bronze: dict[str, DataFrame]) -> None:
    stats = bronze["player_stats"]
    newer = (
        stats.where(F.col("season") == 2026)
        .limit(5)
        .select(
            *(
                c
                for c in stats.columns
                if c not in {transforms.INGESTED_AT, transforms.SOURCE_FILE}
            ),
            F.lit(dt.datetime(2100, 1, 1)).alias(transforms.INGESTED_AT),
            F.lit("/landing/player_stats/2026/stats_player_week_2026_ffffffffffff.parquet").alias(
                transforms.SOURCE_FILE
            ),
        )
    )
    latest = transforms.latest_landed(stats.unionByName(newer))
    seasons = {
        r["season"]: r["n"] for r in latest.groupBy("season").agg(F.count("*").alias("n")).collect()
    }
    assert seasons[2026] == 5  # the newer 2026 file replaces the older one entirely
    assert seasons[2025] == stats.where(F.col("season") == 2025).count()


def test_silver_player_stats_contract(silver: dict[str, DataFrame]) -> None:
    types = dict(silver["player_stats"].dtypes)
    for name, spark_type in transforms.PLAYER_STATS_COLUMNS.items():
        target = "player_name" if name == "player_display_name" else name
        assert types[target] == spark_type, name


def test_relocated_teams_use_current_codes() -> None:
    frame = pd.DataFrame({"team": ["OAK", "DET", None]})
    assert set(transforms.TEAM_ALIASES) == {"OAK", "SD", "STL"}
    # Applied through Spark in silver; the mapping itself is a plain dict.
    assert [transforms.TEAM_ALIASES.get(t, t) for t in frame["team"]] == ["LV", "DET", None]


def test_depth_charts_cover_both_formats_with_the_same_rank_meaning(
    tables: dict[str, pd.DataFrame],
) -> None:
    depth = tables["depth_charts"]
    weekly = depth[depth["week"].notna()]
    snapshots = depth[depth["published_at"].notna()]
    assert set(weekly["season"]) == {2024}
    assert set(snapshots["season"]) == {2025, 2026}
    assert set(depth["position"]) <= set(POSITIONS)
    assert depth["depth_rank"].min() == 1
    # Each team's starting QB is rank 1 in both formats.
    for frame in (weekly, snapshots):
        starters = frame[(frame["position"] == "QB") & (frame["depth_rank"] == 1)]
        assert starters.groupby(["team", "week", "published_at"], dropna=False).size().max() == 1


def test_depth_chart_week_never_uses_a_snapshot_from_after_kickoff(
    tables: dict[str, pd.DataFrame],
) -> None:
    chart = tables["depth_chart_week"].merge(
        tables["team_week"][["game_id", "team", "kickoff_at"]], on=["game_id", "team"]
    )
    snapshots = chart[chart["published_at"].notna()]
    assert not snapshots.empty
    assert (
        pd.to_datetime(snapshots["published_at"]) < pd.to_datetime(snapshots["kickoff_at"])
    ).all()
    # The fixture also contains the first snapshot after every kickoff; none were used.
    landed = tables["depth_charts"]
    later = landed[landed["published_at"].notna()]["published_at"].max()
    assert pd.to_datetime(snapshots["published_at"]).max() <= later
    assert chart.groupby(["season", "week", "team", "player_id"]).size().max() == 1


def test_schedules_kickoff_is_eastern_wall_clock_as_utc(tables: dict[str, pd.DataFrame]) -> None:
    games = tables["schedules"]
    game = games[games["gametime"] == "13:00"].iloc[0]
    kickoff = pd.Timestamp(game["kickoff_at"])
    kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
    assert kickoff.hour in {17, 18}  # 1 pm Eastern is 17:00 UTC in DST, 18:00 otherwise


def test_team_week_lines_and_volume(tables: dict[str, pd.DataFrame]) -> None:
    teams = tables["team_week"]
    assert not teams.duplicated(["season", "week", "team"]).any()
    home = teams[teams["is_home"]].iloc[0]
    assert home["implied_points"] == pytest.approx((home["total_line"] + home["team_spread"]) / 2)
    pairs = teams.groupby("game_id")["team_spread"].sum()
    assert (pairs.abs() < 1e-9).all()  # the two sides' spreads mirror each other
    final = teams[teams["is_final"]]
    assert final["offensive_plays"].notna().all()
    assert teams[~teams["is_final"]]["points_for"].isna().all()


def test_player_week_points_and_usage(tables: dict[str, pd.DataFrame]) -> None:
    week = tables["player_week"]
    assert not week.duplicated(["season", "week", "player_id"]).any()
    assert set(week["position"]) == set(POSITIONS)
    assert (
        (week["fantasy_points_ppr"] - week["fantasy_points_std"])
        .round(6)
        .equals(week["receptions"].astype("float64").round(6))
    )
    half = (week["fantasy_points_ppr"] + week["fantasy_points_std"]) / 2
    assert (week["fantasy_points_half"] - half).abs().max() < 1e-9
    for share in ("target_share", "carry_share", "snap_share"):
        assert week[share].dropna().between(0, 1).all(), share
    team_targets = week.groupby(["game_id", "team"])["target_share"].sum()
    assert (team_targets <= 1 + 1e-9).all()


def test_red_zone_counts_match_play_by_play(tables: dict[str, pd.DataFrame]) -> None:
    plays = tables["plays"]
    week = tables["player_week"]
    leader = week.sort_values("red_zone_carries").iloc[-1]
    raw = plays[
        (plays["game_id"] == leader["game_id"])
        & (plays["rusher_player_id"] == leader["player_id"])
        & (plays["rush_attempt"] == 1)
        & (plays["two_point_attempt"].fillna(0) == 0)
        & (plays["yardline_100"] <= 20)
    ]
    assert len(raw) == leader["red_zone_carries"] > 0


def test_defense_vs_position_rolling_average_excludes_the_current_game(
    tables: dict[str, pd.DataFrame],
) -> None:
    allowed = tables["defense_vs_position"].sort_values(["team", "position", "season", "week"])
    assert not allowed.duplicated(["season", "week", "team", "position"]).any()
    for _, group in allowed.groupby(["team", "position"]):
        values = group["ppr_allowed"].tolist()
        expected = [
            sum(values[max(0, i - 6) : i]) / len(values[max(0, i - 6) : i]) if i else None
            for i in range(len(values))
        ]
        actual = group["ppr_allowed_l6"].tolist()
        for want, got in zip(expected, actual, strict=True):
            if want is None:
                assert pd.isna(got)
            else:
                assert got == pytest.approx(want)


def test_ecr_maps_friday_rankings_to_the_week_being_played(tables: dict[str, pd.DataFrame]) -> None:
    ecr = tables["ecr"]
    weeks = tables["schedules"].query("game_type == 'REG'").groupby(["season", "week"])
    bounds = weeks["gameday"].agg(["min", "max"]).reset_index()
    joined = ecr.merge(bounds, on=["season", "week"])
    scrape = pd.to_datetime(joined["scrape_date"])
    assert (scrape <= pd.to_datetime(joined["max"])).all()
    assert (scrape >= pd.to_datetime(joined["min"]) - np.timedelta64(7, "D")).all()
    assert not ecr.duplicated(["season", "week", "position", "ecr_rank"]).any()
    assert ecr["player_id"].notna().mean() > 0.95


def test_drop_rules_remove_rows_without_keys(silver: dict[str, DataFrame]) -> None:
    stats = silver["player_stats"]
    broken = stats.limit(3).withColumn("player_id", F.lit(None).cast("string"))
    kept = quality.apply_drop_rules(stats.unionByName(broken), quality.DROP["player_stats"])
    assert kept.count() == stats.count()


def test_warn_rules_hold_on_the_fixture(
    silver: dict[str, DataFrame], gold: dict[str, DataFrame]
) -> None:
    for name, frame in {**silver, **gold}.items():
        failures = quality.count_failures(frame, quality.WARN.get(name, {}))
        # A few players have no id crosswalk entry, so no snap count either.
        allowed = {"mapped_to_gsis_id", "has_snap_count"}
        assert {k: v for k, v in failures.items() if v and k not in allowed} == {}, name
