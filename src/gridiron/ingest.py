"""Bronze landing: download nflverse season files into a directory (a Unity Catalog volume).

Layout written under ``landing_dir``::

    <landing_dir>/<season>/play_by_play_<season>_<sha12>.parquet

The content hash in the file name makes the landing zone append-only and idempotent:

* a completed season that already has a file is skipped without a network call;
* the current season is always re-downloaded, but a new file is only written when the
  content changed, so Auto Loader only sees genuinely new data;
* silver de-duplicates on (game_id, play_id) keeping the most recently ingested row, so
  several versions of the current season can coexist in bronze.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from gridiron.config import NFLVERSE_PBP_URL

log = logging.getLogger(__name__)

Fetcher = Callable[[str], BinaryIO]
_CHUNK = 1 << 20


@dataclass(frozen=True)
class IngestResult:
    season: int
    action: str  # "skipped", "unchanged" or "written"
    path: Path | None = None


def season_url(season: int) -> str:
    return NFLVERSE_PBP_URL.format(season=season)


def seasons_to_fetch(requested: Iterable[int], present: Iterable[int], current: int) -> list[int]:
    """Seasons that need a download: anything missing, plus the current season always."""
    present_set = set(present)
    return sorted({s for s in requested if s not in present_set or s == current})


def landed_hashes(landing_dir: Path, season: int) -> set[str]:
    """Content hashes already landed for a season (parsed from file names)."""
    season_dir = landing_dir / str(season)
    if not season_dir.is_dir():
        return set()
    prefix = f"play_by_play_{season}_"
    return {
        p.stem.removeprefix(prefix)
        for p in season_dir.glob(f"{prefix}*.parquet")
        if p.stem.startswith(prefix)
    }


def present_seasons(landing_dir: Path) -> list[int]:
    if not landing_dir.is_dir():
        return []
    return sorted(
        int(p.name)
        for p in landing_dir.iterdir()
        if p.name.isdigit() and landed_hashes(landing_dir, int(p.name))
    )


def _default_fetch(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": "gridiron-lakehouse"})
    return urllib.request.urlopen(request, timeout=120)  # type: ignore[no-any-return]


def download_season(
    season: int, landing_dir: Path, fetch: Fetcher = _default_fetch
) -> IngestResult:
    """Download one season; write it only if its content hash is new."""
    season_dir = landing_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    # Download to a local temp file first: volumes do not support partial/append writes well,
    # and a failed download must never leave a truncated parquet file in the landing zone.
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with fetch(season_url(season)) as response:
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                tmp.write(chunk)
    try:
        sha12 = digest.hexdigest()[:12]
        if sha12 in landed_hashes(landing_dir, season):
            log.info("season %s unchanged (%s)", season, sha12)
            return IngestResult(season, "unchanged")
        target = season_dir / f"play_by_play_{season}_{sha12}.parquet"
        shutil.copyfile(tmp_path, target)
        log.info("season %s written to %s", season, target)
        return IngestResult(season, "written", target)
    finally:
        os.unlink(tmp_path)


def ingest(
    seasons: Iterable[int], landing_dir: Path, current: int, fetch: Fetcher = _default_fetch
) -> list[IngestResult]:
    requested = sorted(set(seasons))
    todo = seasons_to_fetch(requested, present_seasons(landing_dir), current)
    results = [IngestResult(s, "skipped") for s in requested if s not in todo]
    results += [download_season(s, landing_dir, fetch) for s in todo]
    return sorted(results, key=lambda r: r.season)
