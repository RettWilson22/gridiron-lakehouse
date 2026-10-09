"""Registry of the public datasets landed in bronze.

Each dataset lands under ``<landing_root>/<name>/`` and becomes one bronze streaming table
(``bronze_<name>``) in the pipeline. Seasonal datasets are published as one file per
season; snapshot datasets are a single file that is replaced upstream.

Sources (all verified on 2026-10-08):

* nflverse-data GitHub releases (https://github.com/nflverse/nflverse-data/releases):
  play-by-play, weekly player stats, snap counts, weekly rosters, injuries, depth charts,
  schedules (with betting lines) and the player ID table.
* ffverse/ffopportunity releases: expected fantasy points per player-game.
* DynastyProcess data repository: FantasyPros expert consensus rankings (weekly history)
  and the FantasyPros-to-GSIS player ID crosswalk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

NFLVERSE: Final = "https://github.com/nflverse/nflverse-data/releases/download"
FFOPPORTUNITY: Final = "https://github.com/ffverse/ffopportunity/releases/download/latest-data"
DYNASTYPROCESS: Final = "https://github.com/dynastyprocess/data/raw/master/files"

Kind = Literal["seasonal", "snapshot", "ecr"]


@dataclass(frozen=True)
class Dataset:
    """One landed dataset.

    ``url`` contains ``{season}`` for seasonal datasets. ``stem`` names the landed files:
    ``<stem>_<season>_<sha12>.parquet`` (seasonal) or ``<stem>_<sha12>.parquet`` (snapshot).
    ``required`` datasets fail the ingest task when a download fails; optional ones (the
    benchmark rankings) only log a warning so the weekly refresh still runs.

    ``normalize_types`` widens integer columns to float64 at landing. Upstream files are not
    typed consistently across seasons (for example injuries ``season`` and ``week`` are
    doubles before 2021 and int32 after), and one Auto Loader bronze table needs one schema.
    Silver casts every column it keeps to its real type.
    """

    name: str
    url: str
    kind: Kind
    stem: str
    source_format: Literal["parquet", "csv"] = "parquet"
    required: bool = True
    normalize_types: bool = True

    @property
    def seasonal(self) -> bool:
        return self.kind != "snapshot"

    def season_url(self, season: int) -> str:
        if self.kind != "seasonal":
            raise ValueError(f"{self.name} is not published per season")
        return self.url.format(season=season)


DATASETS: Final[dict[str, Dataset]] = {
    d.name: d
    for d in (
        # Play-by-play is landed as published, without type widening: silver reads only
        # 20 of its columns (``transforms.PLAY_COLUMNS``), and none of those change type
        # between seasons.
        Dataset(
            "pbp",
            f"{NFLVERSE}/pbp/play_by_play_{{season}}.parquet",
            "seasonal",
            "play_by_play",
            normalize_types=False,
        ),
        Dataset(
            "player_stats",
            f"{NFLVERSE}/stats_player/stats_player_week_{{season}}.parquet",
            "seasonal",
            "stats_player_week",
        ),
        Dataset(
            "snap_counts",
            f"{NFLVERSE}/snap_counts/snap_counts_{{season}}.parquet",
            "seasonal",
            "snap_counts",
        ),
        Dataset(
            "rosters",
            f"{NFLVERSE}/weekly_rosters/roster_weekly_{{season}}.parquet",
            "seasonal",
            "roster_weekly",
        ),
        Dataset(
            "injuries", f"{NFLVERSE}/injuries/injuries_{{season}}.parquet", "seasonal", "injuries"
        ),
        Dataset(
            "depth_charts",
            f"{NFLVERSE}/depth_charts/depth_charts_{{season}}.parquet",
            "seasonal",
            "depth_charts",
        ),
        Dataset(
            "ff_opportunity",
            f"{FFOPPORTUNITY}/ep_weekly_{{season}}.parquet",
            "seasonal",
            "ep_weekly",
        ),
        Dataset("schedules", f"{NFLVERSE}/schedules/games.parquet", "snapshot", "games"),
        Dataset("players", f"{NFLVERSE}/players/players.parquet", "snapshot", "players"),
        Dataset(
            "player_ids",
            f"{DYNASTYPROCESS}/db_playerids.csv",
            "snapshot",
            "db_playerids",
            source_format="csv",
            required=False,
        ),
        # One ~40 MB file holding every FantasyPros ranking page since 2019. The ingest
        # keeps only weekly positional rankings and splits them into one file per season,
        # so completed seasons are written once and only the current season changes.
        Dataset(
            "ecr",
            f"{DYNASTYPROCESS}/db_fpecr.parquet",
            "ecr",
            "fp_ecr_weekly",
            required=False,
        ),
    )
}

# Snapshot datasets that are landed once per run regardless of the season range.
SNAPSHOT_DATASETS: Final = tuple(n for n, d in DATASETS.items() if d.kind == "snapshot")
SEASONAL_DATASETS: Final = tuple(n for n, d in DATASETS.items() if d.kind == "seasonal")

# FantasyPros weekly positional rankings (ecr_type "wp") for the modelled positions.
ECR_TYPE_WEEKLY_POSITIONAL: Final = "wp"
ECR_COLUMNS: Final = (
    "fp_page",
    "page_type",
    "ecr_type",
    "player",
    "id",
    "pos",
    "team",
    "ecr",
    "sd",
    "best",
    "worst",
    "scrape_date",
)
