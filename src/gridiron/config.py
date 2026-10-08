"""Shared constants: source location, season range and the silver column contract."""

from __future__ import annotations

import datetime as dt
from typing import Final

NFLVERSE_PBP_URL: Final = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
)

# nflverse play-by-play with full nflfastR fields starts in 1999; this project uses the
# modern era by default.
FIRST_SEASON: Final = 2015

# Columns kept in silver. Everything else in the ~370 column source is dropped.
# Types are the Spark SQL types the silver layer casts to.
SILVER_COLUMNS: Final[dict[str, str]] = {
    "game_id": "string",
    "play_id": "bigint",
    "season": "int",
    "season_type": "string",
    "week": "int",
    "game_date": "date",
    "home_team": "string",
    "away_team": "string",
    "home_coach": "string",
    "away_coach": "string",
    "posteam": "string",
    "defteam": "string",
    "posteam_type": "string",
    "qtr": "int",
    "down": "int",
    "ydstogo": "int",
    "yardline_100": "int",
    "game_seconds_remaining": "int",
    "score_differential": "int",
    "posteam_timeouts_remaining": "int",
    "play_type": "string",
    "yards_gained": "int",
    "first_down": "int",
    "touchdown": "int",
    "fourth_down_converted": "int",
    "fourth_down_failed": "int",
    "field_goal_result": "string",
    "kick_distance": "int",
    "return_yards": "int",
    "touchback": "int",
    "punt_blocked": "int",
    "fixed_drive": "int",
    "fixed_drive_result": "string",
    "ep": "double",
    "epa": "double",
    "wp": "double",
    "wpa": "double",
    "home_score": "int",
    "away_score": "int",
    "total_home_score": "int",
    "total_away_score": "int",
}

# Plays that count as "going for it" on fourth down. Fake punts and fake field goals are
# recorded by nflverse as run or pass plays, so they count as going for it.
GO_PLAY_TYPES: Final = ("run", "pass")
PUNT_PLAY_TYPE: Final = "punt"
FIELD_GOAL_PLAY_TYPE: Final = "field_goal"


def current_season(today: dt.date | None = None) -> int:
    """Return the NFL season that is in progress (or most recently finished) on ``today``.

    The regular season starts in September; January and February games belong to the
    previous calendar year's season.
    """
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1
