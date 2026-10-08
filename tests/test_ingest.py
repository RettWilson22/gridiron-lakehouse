from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pytest

from gridiron.config import current_season
from gridiron.ingest import (
    download_season,
    ingest,
    landed_hashes,
    present_seasons,
    season_url,
    seasons_to_fetch,
)


class FakeRemote:
    """Stands in for GitHub: serves bytes per season and records requests."""

    def __init__(self, payloads: dict[int, bytes]) -> None:
        self.payloads = payloads
        self.requests: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.requests.append(url)
        season = int(url.rsplit("_", 1)[-1].removesuffix(".parquet"))
        return io.BytesIO(self.payloads[season])


def test_season_url_matches_nflverse_release_layout() -> None:
    assert season_url(2024) == (
        "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet"
    )


@pytest.mark.parametrize(
    ("today", "expected"),
    [(dt.date(2026, 10, 8), 2026), (dt.date(2027, 1, 20), 2026), (dt.date(2026, 8, 31), 2025)],
)
def test_current_season(today: dt.date, expected: int) -> None:
    assert current_season(today) == expected


def test_seasons_to_fetch_skips_present_but_always_refreshes_current() -> None:
    assert seasons_to_fetch([2022, 2023, 2024], present=[2022, 2024], current=2024) == [
        2023,
        2024,
    ]


def test_download_writes_once_per_distinct_content(tmp_path: Path) -> None:
    remote = FakeRemote({2026: b"week-1"})
    first = download_season(2026, tmp_path, remote)
    second = download_season(2026, tmp_path, remote)
    remote.payloads[2026] = b"week-2"
    third = download_season(2026, tmp_path, remote)

    assert (first.action, second.action, third.action) == ("written", "unchanged", "written")
    assert first.path is not None and first.path.read_bytes() == b"week-1"
    assert len(landed_hashes(tmp_path, 2026)) == 2
    assert len(list((tmp_path / "2026").iterdir())) == 2


def test_ingest_is_idempotent_for_completed_seasons(tmp_path: Path) -> None:
    remote = FakeRemote({2024: b"a", 2025: b"b", 2026: b"c"})
    first = ingest([2024, 2025, 2026], tmp_path, current=2026, fetch=remote)
    assert [r.action for r in first] == ["written", "written", "written"]

    remote.requests.clear()
    second = ingest([2024, 2025, 2026], tmp_path, current=2026, fetch=remote)
    assert [r.action for r in second] == ["skipped", "skipped", "unchanged"]
    assert remote.requests == [season_url(2026)]
    assert present_seasons(tmp_path) == [2024, 2025, 2026]


def test_failed_download_leaves_no_partial_file(tmp_path: Path) -> None:
    class Broken(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise ConnectionError("network dropped")

    with pytest.raises(ConnectionError):
        download_season(2024, tmp_path, lambda url: Broken())
    assert landed_hashes(tmp_path, 2024) == set()
