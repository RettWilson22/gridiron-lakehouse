"""Shared constants: season range, positions and the in-progress season."""

from __future__ import annotations

import datetime as dt
from typing import Final

# Every source used here covers 2018 onward; five seasons of history before the first
# backtest season (2023) is enough for the walk-forward evaluation.
FIRST_SEASON: Final = 2018

# Fantasy positions modelled. Fullbacks are folded into RB (nflverse position group).
POSITIONS: Final = ("QB", "RB", "WR", "TE")

# Regular-season weeks only: fantasy seasons end before the playoffs.
REGULAR_SEASON: Final = "REG"

# Red-zone and goal-line opportunity thresholds (yards from the opponent end zone).
RED_ZONE_YARDLINE: Final = 20
GOAL_LINE_YARDLINE: Final = 5


def current_season(today: dt.date | None = None) -> int:
    """Return the NFL season that is in progress (or most recently finished) on ``today``.

    The regular season starts in September; January and February games belong to the
    previous calendar year's season.
    """
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1
