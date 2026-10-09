"""Bronze -> silver -> gold transformations as plain PySpark functions.

The Lakeflow pipeline in ``databricks/pipelines`` only wires these functions to datasets,
so every transformation here is unit tested locally against checked-in fixtures.

Conventions:

* Bronze tables are the landed files plus ``_ingested_at`` and ``_source_file``.
* Silver keeps the rows of the most recently landed file per season folder (or per snapshot
  for single-file datasets), selects and types the columns this project uses, and renames
  identifiers consistently (``player_id`` is always the NFL GSIS id).
* Gold tables are keyed for the fantasy use case: player-week, team-week and
  defense-week-position, regular season only.
"""

from __future__ import annotations

from functools import reduce

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from gridiron.config import GOAL_LINE_YARDLINE, POSITIONS, RED_ZONE_YARDLINE, REGULAR_SEASON

INGESTED_AT = "_ingested_at"
SOURCE_FILE = "_source_file"
PARTITION = "_partition"

INTEGER_TYPES = frozenset({"int", "bigint", "smallint"})

# nflverse depth chart labels (2024 and earlier) -> fantasy position.
OLD_DEPTH_POSITIONS = {"QB": "QB", "RB": "RB", "HB": "RB", "WR": "WR", "TE": "TE"}

# Relocated franchises: schedules use the abbreviation of the time (OAK in 2018-2019) while
# player stats and play-by-play use the current one (LV). Silver uses current codes.
# Game ids are left untouched (``2018_01_LA_OAK``), so joins use game_id plus team.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def with_ingest_metadata(raw: DataFrame) -> DataFrame:
    """Attach file-level lineage from the hidden ``_metadata`` column of file sources.

    Works for both ``spark.read.parquet`` and Auto Loader (``cloudFiles``) readers.
    """
    return raw.select(
        "*",
        F.col("_metadata.file_modification_time").alias(INGESTED_AT),
        F.col("_metadata.file_path").alias(SOURCE_FILE),
    )


def latest_landed(bronze: DataFrame) -> DataFrame:
    """Rows of the most recently landed file per season folder (or per snapshot dataset).

    Every landed file is a complete copy of its season (or of the whole dataset), so the
    newest file supersedes older ones, including rows that were deleted upstream. Adds
    ``_partition``: the season folder name, or an empty string for snapshot datasets.
    """
    partition = F.regexp_extract(F.col(SOURCE_FILE), r"/(\d{4})/[^/]*$", 1)
    latest = F.max(F.struct(F.col(INGESTED_AT), F.col(SOURCE_FILE))).over(
        Window.partitionBy(PARTITION)
    )
    return (
        bronze.withColumn(PARTITION, partition)
        .withColumn("_latest", latest)
        .where(F.col("_latest")[SOURCE_FILE] == F.col(SOURCE_FILE))
        .drop("_latest")
    )


def typed(name: str, spark_type: str, source: str | None = None) -> Column:
    """Cast a source column, turning unparseable values into NULL (caught by expectations).

    Integers go through DOUBLE first because landed numeric columns are float64 and some
    string sources hold values such as ``"28013.0"``.
    """
    src = f"`{source or name}`"
    if spark_type in INTEGER_TYPES:
        expr = f"try_cast(try_cast({src} AS DOUBLE) AS {spark_type})"
    else:
        expr = f"try_cast({src} AS {spark_type})"
    return F.expr(expr).alias(name)


def clean_id(name: str, source: str | None = None) -> Column:
    """Identifier column with blanks and the literal ``NA`` turned into NULL."""
    value = F.trim(F.col(source or name).cast("string"))
    return F.when((value == "") | (value == "NA"), None).otherwise(value).alias(name)


def select_typed(df: DataFrame, contract: dict[str, str]) -> DataFrame:
    return df.select(*(typed(n, t) for n, t in contract.items()))


def team_code(column: Column) -> Column:
    """Current franchise abbreviation (see ``TEAM_ALIASES``)."""
    mapping = F.create_map(*(F.lit(x) for pair in TEAM_ALIASES.items() for x in pair))
    return F.coalesce(mapping[column], column)


def normalize_teams(df: DataFrame, *columns: str) -> DataFrame:
    for name in columns:
        df = df.withColumn(name, team_code(F.col(name)))
    return df


def _sum(condition: Column) -> Column:
    return F.sum(F.when(condition, 1).otherwise(0))


# ---------------------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------------------

PLAY_COLUMNS: dict[str, str] = {
    "game_id": "string",
    "play_id": "bigint",
    "season": "int",
    "season_type": "string",
    "week": "int",
    "posteam": "string",
    "defteam": "string",
    "play_type": "string",
    "down": "int",
    "yardline_100": "int",
    "rush_attempt": "int",
    "pass_attempt": "int",
    "sack": "int",
    "two_point_attempt": "int",
    "rusher_player_id": "string",
    "receiver_player_id": "string",
    "passer_player_id": "string",
    "air_yards": "double",
    "yards_gained": "int",
    "touchdown": "int",
}


def silver_plays(bronze: DataFrame, first_season: int) -> DataFrame:
    """Typed, column-pruned plays from the latest landed file of each season."""
    pruned = bronze.select(*PLAY_COLUMNS, INGESTED_AT, SOURCE_FILE)
    plays = select_typed(latest_landed(pruned), PLAY_COLUMNS).where(F.col("season") >= first_season)
    return normalize_teams(plays, "posteam", "defteam")


PLAYER_STATS_COLUMNS: dict[str, str] = {
    "player_id": "string",
    "player_display_name": "string",
    "position": "string",
    "position_group": "string",
    "season": "int",
    "week": "int",
    "season_type": "string",
    "game_id": "string",
    "team": "string",
    "opponent_team": "string",
    "completions": "int",
    "attempts": "int",
    "passing_yards": "int",
    "passing_tds": "int",
    "passing_interceptions": "int",
    "sacks_suffered": "int",
    "sack_fumbles_lost": "int",
    "passing_air_yards": "int",
    "passing_2pt_conversions": "int",
    "carries": "int",
    "rushing_yards": "int",
    "rushing_tds": "int",
    "rushing_fumbles_lost": "int",
    "rushing_2pt_conversions": "int",
    "receptions": "int",
    "targets": "int",
    "receiving_yards": "int",
    "receiving_tds": "int",
    "receiving_fumbles_lost": "int",
    "receiving_air_yards": "int",
    "receiving_2pt_conversions": "int",
    "special_teams_tds": "int",
    "fantasy_points": "double",
    "fantasy_points_ppr": "double",
}


def silver_player_stats(bronze: DataFrame) -> DataFrame:
    """Weekly player box scores (nflverse ``stats_player_week``)."""
    stats = select_typed(latest_landed(bronze), PLAYER_STATS_COLUMNS)
    stats = normalize_teams(stats, "team", "opponent_team")
    return stats.withColumnRenamed("player_display_name", "player_name")


def silver_players(bronze: DataFrame) -> DataFrame:
    """nflverse player table: identity, crosswalk ids and draft details."""
    latest = latest_landed(bronze)
    return latest.select(
        clean_id("player_id", "gsis_id"),
        F.col("display_name").cast("string").alias("player_name"),
        F.col("position").cast("string").alias("position"),
        F.col("position_group").cast("string").alias("position_group"),
        clean_id("pfr_id"),
        typed("birth_date", "date"),
        typed("rookie_season", "int"),
        typed("draft_round", "int"),
        typed("draft_pick", "int"),
        F.col("latest_team").cast("string").alias("latest_team"),
    )


SNAP_COLUMNS: dict[str, str] = {
    "game_id": "string",
    "season": "int",
    "game_type": "string",
    "week": "int",
    "pfr_player_id": "string",
    "player": "string",
    "position": "string",
    "team": "string",
    "opponent": "string",
    "offense_snaps": "int",
    "offense_pct": "double",
}


def silver_snap_counts(bronze: DataFrame, players: DataFrame) -> DataFrame:
    """Offensive snap counts (Pro Football Reference ids) mapped to GSIS player ids."""
    snaps = normalize_teams(select_typed(latest_landed(bronze), SNAP_COLUMNS), "team", "opponent")
    crosswalk = players.where(F.col("pfr_id").isNotNull()).select(
        F.col("pfr_id").alias("pfr_player_id"), "player_id"
    )
    return snaps.join(crosswalk, "pfr_player_id", "left")


ROSTER_COLUMNS: dict[str, str] = {
    "season": "int",
    "week": "int",
    "game_type": "string",
    "team": "string",
    "full_name": "string",
    "position": "string",
    "status": "string",
    "years_exp": "int",
    "rookie_year": "int",
}


def silver_rosters(bronze: DataFrame) -> DataFrame:
    latest = latest_landed(bronze)
    rosters = latest.select(
        clean_id("player_id", "gsis_id"), *(typed(n, t) for n, t in ROSTER_COLUMNS.items())
    )
    return normalize_teams(rosters, "team")


INJURY_COLUMNS: dict[str, str] = {
    "season": "int",
    "game_type": "string",
    "week": "int",
    "team": "string",
    "full_name": "string",
    "position": "string",
    "report_primary_injury": "string",
    "report_status": "string",
    "practice_status": "string",
}


def silver_injuries(bronze: DataFrame) -> DataFrame:
    """Official injury reports: one row per player per week (final report of the week)."""
    latest = latest_landed(bronze)
    injuries = latest.select(
        clean_id("player_id", "gsis_id"), *(typed(n, t) for n, t in INJURY_COLUMNS.items())
    )
    return normalize_teams(injuries, "team")


def _old_depth_charts(latest: DataFrame) -> DataFrame:
    """2024 and earlier: one weekly depth chart per team, rank per position slot."""
    position = F.trim(F.col("depth_position"))
    mapping = F.create_map(*(F.lit(x) for pair in OLD_DEPTH_POSITIONS.items() for x in pair))
    return latest.where(F.col("club_code").isNotNull() & (F.col("formation") == "Offense")).select(
        typed("season", "int"),
        typed("week", "int"),
        F.col("game_type").cast("string").alias("game_type"),
        F.lit(None).cast("timestamp").alias("published_at"),
        F.col("club_code").cast("string").alias("team"),
        clean_id("player_id", "gsis_id"),
        F.col("full_name").cast("string").alias("player_name"),
        mapping[position].alias("position"),
        typed("depth_rank", "int", "depth_team"),
    )


def _new_depth_charts(latest: DataFrame) -> DataFrame:
    """2025 onward: timestamped snapshots; ``pos_rank`` ranks a position across slots.

    The rank within each slot (1 = starter at that slot) matches the older format's
    ``depth_team``, so both formats yield the same ``depth_rank`` semantics.
    """
    snapshots = latest.where(F.col("dt").isNotNull() & F.col("pos_abb").isin(*POSITIONS))
    slot = Window.partitionBy("team", "dt", "pos_abb", "pos_slot").orderBy(
        F.col("pos_rank").cast("int")
    )
    return snapshots.select(
        F.col(PARTITION).cast("int").alias("season"),
        F.lit(None).cast("int").alias("week"),
        F.lit(None).cast("string").alias("game_type"),
        F.to_timestamp(F.col("dt"), "yyyy-MM-dd'T'HH:mm:ssX").alias("published_at"),
        F.col("team").cast("string").alias("team"),
        clean_id("player_id", "gsis_id"),
        F.col("player_name").cast("string").alias("player_name"),
        F.col("pos_abb").cast("string").alias("position"),
        F.dense_rank().over(slot).cast("int").alias("depth_rank"),
    )


def silver_depth_charts(bronze: DataFrame) -> DataFrame:
    """Offensive skill-position depth charts in one shape across both nflverse formats.

    One row per player and chart (a week for the old format, a snapshot time for the new
    one), keeping the player's best rank if listed in several slots.
    """
    latest = latest_landed(bronze)
    parts = []
    if "club_code" in latest.columns:
        parts.append(_old_depth_charts(latest))
    if "dt" in latest.columns:
        parts.append(_new_depth_charts(latest))
    if not parts:
        raise ValueError("bronze depth charts have neither the weekly nor the snapshot format")
    union = normalize_teams(
        reduce(DataFrame.unionByName, parts).where(F.col("position").isNotNull()), "team"
    )
    best = Window.partitionBy("season", "week", "published_at", "team", "player_id").orderBy(
        F.col("depth_rank").asc_nulls_last(), "position"
    )
    return (
        union.withColumn("_rank", F.row_number().over(best))
        .where(F.col("_rank") == 1)
        .drop("_rank")
    )


FF_OPPORTUNITY_COLUMNS: dict[str, str] = {
    "season": "int",
    "week": "int",
    "game_id": "string",
    "player_id": "string",
    "position": "string",
    "total_fantasy_points_exp": "double",
    "total_touchdown_exp": "double",
    "total_yards_gained_exp": "double",
}


def silver_ff_opportunity(bronze: DataFrame) -> DataFrame:
    """Expected PPR points per player-game from ffverse ``ffopportunity``."""
    return select_typed(latest_landed(bronze), FF_OPPORTUNITY_COLUMNS)


SCHEDULE_COLUMNS: dict[str, str] = {
    "game_id": "string",
    "season": "int",
    "game_type": "string",
    "week": "int",
    "gameday": "date",
    "gametime": "string",
    "away_team": "string",
    "home_team": "string",
    "away_score": "int",
    "home_score": "int",
    "spread_line": "double",
    "total_line": "double",
    "roof": "string",
}


def silver_schedules(bronze: DataFrame, first_season: int) -> DataFrame:
    """Every game (played and upcoming) with closing or current betting lines.

    ``spread_line`` is the expected home margin (positive = home favored). ``kickoff_at``
    is the scheduled kickoff as an instant; nflverse publishes Eastern wall-clock times.
    """
    games = normalize_teams(
        select_typed(latest_landed(bronze), SCHEDULE_COLUMNS), "home_team", "away_team"
    ).where(F.col("season") >= first_season)
    hour_minute = F.split(F.col("gametime"), ":")
    kickoff = F.make_timestamp(
        F.year("gameday"),
        F.month("gameday"),
        F.dayofmonth("gameday"),
        hour_minute.getItem(0).cast("int"),
        hour_minute.getItem(1).cast("int"),
        F.lit(0),
        F.lit("America/New_York"),
    )
    return games.withColumn("kickoff_at", kickoff)


def silver_player_ids(bronze: DataFrame) -> DataFrame:
    """FantasyPros id -> GSIS id crosswalk (DynastyProcess), one row per FantasyPros id."""
    latest = latest_landed(bronze).select(
        typed("fantasypros_id", "bigint"), clean_id("player_id", "gsis_id")
    )
    best = Window.partitionBy("fantasypros_id").orderBy(F.col("player_id").asc_nulls_last())
    return (
        latest.withColumn("_rank", F.row_number().over(best))
        .where(F.col("_rank") == 1)
        .drop("_rank")
    )


ECR_COLUMNS: dict[str, str] = {
    "fantasypros_id": "bigint",
    "player_name": "string",
    "position": "string",
    "team": "string",
    "ecr": "double",
    "sd": "double",
    "best": "double",
    "worst": "double",
    "scrape_date": "date",
}


def regular_season_weeks(schedules: DataFrame) -> DataFrame:
    """First and last game day of every regular-season week."""
    return (
        schedules.where(F.col("game_type") == REGULAR_SEASON)
        .groupBy("season", "week")
        .agg(F.min("gameday").alias("first_gameday"), F.max("gameday").alias("last_gameday"))
    )


def silver_ecr(bronze: DataFrame, schedules: DataFrame, player_ids: DataFrame) -> DataFrame:
    """FantasyPros weekly positional expert consensus rankings, mapped to NFL weeks.

    A ranking scraped on a given date belongs to the first regular-season week whose last
    game is on or after that date (rankings are scraped on Fridays, between Thursday and
    Sunday games). When a week has several scrapes, the latest one is kept.
    """
    latest = latest_landed(bronze).select(
        typed("fantasypros_id", "bigint", "id"),
        F.col("player").cast("string").alias("player_name"),
        F.col("pos").cast("string").alias("position"),
        F.col("team").cast("string").alias("team"),
        *(typed(n, ECR_COLUMNS[n]) for n in ("ecr", "sd", "best", "worst", "scrape_date")),
        F.col(PARTITION).cast("int").alias("season"),
    )
    weeks = regular_season_weeks(schedules)
    dates = latest.select("season", "scrape_date").distinct()
    week_of = (
        dates.join(weeks, "season")
        .where(F.col("last_gameday") >= F.col("scrape_date"))
        .groupBy("season", "scrape_date")
        .agg(F.min("week").alias("week"))
    )
    mapped = latest.join(week_of, ["season", "scrape_date"], "left")
    latest_scrape = Window.partitionBy("season", "week", "position")
    mapped = mapped.withColumn("_last", F.max("scrape_date").over(latest_scrape)).where(
        F.col("scrape_date") == F.col("_last")
    )
    positional = Window.partitionBy("season", "week", "position").orderBy("ecr", "fantasypros_id")
    return (
        mapped.drop("_last")
        .withColumn("ecr_rank", F.row_number().over(positional))
        .join(player_ids, "fantasypros_id", "left")
    )


# ---------------------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------------------


def team_games(schedules: DataFrame) -> DataFrame:
    """Regular-season games from each team's point of view, with pre-game lines.

    ``team_spread`` is the expected margin for the team; ``implied_points`` is the
    betting-market team total, ``(total_line + team_spread) / 2``.
    """
    games = schedules.where(F.col("game_type") == REGULAR_SEASON)

    def side(is_home: bool) -> DataFrame:
        team, opponent = ("home", "away") if is_home else ("away", "home")
        sign = 1.0 if is_home else -1.0
        return games.select(
            "season",
            "week",
            "game_id",
            F.col(f"{team}_team").alias("team"),
            F.col(f"{opponent}_team").alias("opponent"),
            F.lit(is_home).alias("is_home"),
            "kickoff_at",
            (F.col("spread_line") * sign).alias("team_spread"),
            "total_line",
            ((F.col("total_line") + F.col("spread_line") * sign) / 2).alias("implied_points"),
            F.col(f"{team}_score").alias("points_for"),
            F.col(f"{opponent}_score").alias("points_against"),
            (F.col("home_score").isNotNull() & F.col("away_score").isNotNull()).alias("is_final"),
        )

    return side(True).unionByName(side(False))


def team_week(schedules: DataFrame, plays: DataFrame, player_stats: DataFrame) -> DataFrame:
    """One row per team and regular-season game (played and upcoming): lines and volume."""
    scrimmage = plays.where(
        F.col("play_type").isin("pass", "run") & (F.coalesce("two_point_attempt", F.lit(0)) == 0)
    )
    volume = scrimmage.groupBy("game_id", F.col("posteam").alias("team")).agg(
        F.count("*").alias("offensive_plays"),
        F.sum("pass_attempt").alias("dropbacks"),
        F.sum("rush_attempt").alias("rush_attempts"),
        _sum(F.col("yardline_100") <= RED_ZONE_YARDLINE).alias("red_zone_plays"),
    )
    regular = player_stats.where(F.col("season_type") == REGULAR_SEASON)
    usage = regular.groupBy("game_id", "team").agg(
        F.sum("targets").alias("team_targets"),
        F.sum("carries").alias("team_carries"),
        F.sum("receiving_air_yards").alias("team_air_yards"),
        *(
            F.sum(F.when(F.col("position_group") == p, F.col("fantasy_points_ppr"))).alias(
                f"ppr_points_{p.lower()}"
            )
            for p in POSITIONS
        ),
    )
    return (
        team_games(schedules)
        .join(volume, ["game_id", "team"], "left")
        .join(usage, ["game_id", "team"], "left")
    )


def player_red_zone(plays: DataFrame) -> DataFrame:
    """Red-zone targets and carries and goal-line carries per player-game."""
    real = F.coalesce(F.col("two_point_attempt"), F.lit(0)) == 0
    red_zone = F.col("yardline_100") <= RED_ZONE_YARDLINE
    targets = plays.where(
        real
        & (F.col("pass_attempt") == 1)
        & (F.coalesce("sack", F.lit(0)) == 0)
        & F.col("receiver_player_id").isNotNull()
    ).groupBy("game_id", F.col("receiver_player_id").alias("player_id"))
    carries = plays.where(
        real & (F.col("rush_attempt") == 1) & F.col("rusher_player_id").isNotNull()
    ).groupBy("game_id", F.col("rusher_player_id").alias("player_id"))
    return targets.agg(_sum(red_zone).alias("red_zone_targets")).join(
        carries.agg(
            _sum(red_zone).alias("red_zone_carries"),
            _sum(F.col("yardline_100") <= GOAL_LINE_YARDLINE).alias("goal_line_carries"),
        ),
        ["game_id", "player_id"],
        "full",
    )


def _share(numerator: str, denominator: str) -> Column:
    return F.when(F.col(denominator) > 0, F.col(numerator) / F.col(denominator))


def player_week(
    player_stats: DataFrame,
    snap_counts: DataFrame,
    plays: DataFrame,
    team_week_df: DataFrame,
    ff_opportunity: DataFrame,
) -> DataFrame:
    """One row per QB/RB/WR/TE per regular-season game played: points and usage.

    Fantasy points: nflverse ``fantasy_points`` (standard) and ``fantasy_points_ppr``;
    half PPR is their average. Shares are of the team's totals in that game.
    """
    stats = player_stats.where(
        (F.col("season_type") == REGULAR_SEASON) & F.col("position_group").isin(*POSITIONS)
    )
    snaps = (
        snap_counts.where(F.col("player_id").isNotNull())
        .groupBy("game_id", "player_id")
        .agg(
            F.max("offense_snaps").alias("offense_snaps"), F.max("offense_pct").alias("snap_share")
        )
    )
    expected = ff_opportunity.groupBy("game_id", "player_id").agg(
        F.max("total_fantasy_points_exp").alias("expected_ppr")
    )
    teams = team_week_df.select(
        "game_id",
        "team",
        "opponent",
        "is_home",
        "team_targets",
        "team_carries",
        "team_air_yards",
        "red_zone_plays",
    )
    zero = ["red_zone_targets", "red_zone_carries", "goal_line_carries"]
    joined = (
        stats.join(teams, ["game_id", "team"], "left")
        .join(snaps, ["game_id", "player_id"], "left")
        .join(player_red_zone(plays), ["game_id", "player_id"], "left")
        .join(expected, ["game_id", "player_id"], "left")
        .fillna(0, zero)
    )
    return joined.select(
        "season",
        "week",
        "game_id",
        "player_id",
        "player_name",
        F.col("position_group").alias("position"),
        "team",
        F.coalesce("opponent", "opponent_team").alias("opponent"),
        "is_home",
        F.col("fantasy_points_ppr").alias("fantasy_points_ppr"),
        ((F.col("fantasy_points_ppr") + F.col("fantasy_points")) / 2).alias("fantasy_points_half"),
        F.col("fantasy_points").alias("fantasy_points_std"),
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        (
            F.coalesce("sack_fumbles_lost", F.lit(0))
            + F.coalesce("rushing_fumbles_lost", F.lit(0))
            + F.coalesce("receiving_fumbles_lost", F.lit(0))
        ).alias("fumbles_lost"),
        (
            F.coalesce("passing_2pt_conversions", F.lit(0))
            + F.coalesce("rushing_2pt_conversions", F.lit(0))
            + F.coalesce("receiving_2pt_conversions", F.lit(0))
        ).alias("two_point_conversions"),
        "special_teams_tds",
        "offense_snaps",
        "snap_share",
        _share("targets", "team_targets").alias("target_share"),
        _share("carries", "team_carries").alias("carry_share"),
        F.col("receiving_air_yards").alias("air_yards"),
        _share("receiving_air_yards", "team_air_yards").alias("air_yards_share"),
        "red_zone_targets",
        "red_zone_carries",
        "goal_line_carries",
        F.when(
            F.col("red_zone_plays") > 0,
            (F.col("red_zone_targets") + F.col("red_zone_carries")) / F.col("red_zone_plays"),
        ).alias("red_zone_share"),
        "expected_ppr",
    )


def defense_vs_position(player_week_df: DataFrame, team_week_df: DataFrame) -> DataFrame:
    """Fantasy points each defense allowed to each position, per game and rolling.

    ``ppr_allowed_l6`` averages the defense's previous six games (it may reach into the
    previous season) and never includes the game on the row, so it is the matchup strength
    known before that game. ``matchup_rank`` ranks defenses by it within the week
    (1 = allows the most points).
    """
    played = team_week_df.where(F.col("is_final")).select(
        "season", "week", "game_id", F.col("team").alias("defense"), F.col("opponent")
    )
    allowed = player_week_df.groupBy("game_id", F.col("opponent").alias("defense"), "position").agg(
        F.sum("fantasy_points_ppr").alias("ppr_allowed"),
        F.sum("fantasy_points_half").alias("half_allowed"),
        F.sum("fantasy_points_std").alias("std_allowed"),
        F.count("*").alias("players"),
    )
    base = (
        played.withColumn("position", F.explode(F.array(*(F.lit(p) for p in POSITIONS))))
        .join(allowed, ["game_id", "defense", "position"], "left")
        .fillna(0, ["ppr_allowed", "half_allowed", "std_allowed", "players"])
    )
    prior = Window.partitionBy("defense", "position").orderBy("season", "week").rowsBetween(-6, -1)
    rolled = base.withColumn("ppr_allowed_l6", F.avg("ppr_allowed").over(prior)).withColumn(
        "games_l6", F.count("ppr_allowed").over(prior)
    )
    rank = Window.partitionBy("season", "week", "position").orderBy(
        F.col("ppr_allowed_l6").desc_nulls_last(), "defense"
    )
    return rolled.withColumn("matchup_rank", F.rank().over(rank)).withColumnRenamed(
        "defense", "team"
    )


def depth_chart_week(depth_charts: DataFrame, schedules: DataFrame) -> DataFrame:
    """Each team's skill-position depth chart as published before each regular-season game.

    Weekly charts (2024 and earlier) are used as published for the week. For snapshot
    charts (2025 onward) the latest snapshot strictly before the team's kickoff is used, so
    a chart never reflects anything that happened in or after the game. For upcoming games
    that is simply the latest snapshot.
    """
    games = team_games(schedules).select("season", "week", "game_id", "team", "kickoff_at")
    columns = [
        "season",
        "week",
        "game_id",
        "team",
        "player_id",
        "player_name",
        "position",
        "depth_rank",
        "published_at",
    ]
    weekly = (
        depth_charts.where(F.col("week").isNotNull() & (F.col("game_type") == REGULAR_SEASON))
        .drop("published_at")
        .join(games.drop("kickoff_at"), ["season", "week", "team"])
        .withColumn("published_at", F.lit(None).cast("timestamp"))
        .select(*columns)
    )
    snapshots = depth_charts.where(F.col("published_at").isNotNull()).drop("week")
    as_of = (
        snapshots.select("season", "team", "published_at")
        .distinct()
        .join(games, ["season", "team"])
        .where(F.col("published_at") < F.col("kickoff_at"))
        .groupBy("season", "week", "game_id", "team")
        .agg(F.max("published_at").alias("published_at"))
    )
    daily = snapshots.join(as_of, ["season", "team", "published_at"]).select(*columns)
    return weekly.unionByName(daily)
