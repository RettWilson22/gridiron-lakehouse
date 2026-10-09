from __future__ import annotations

import datetime as dt
import io
import urllib.error
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gridiron.config import current_season
from gridiron.datasets import DATASETS
from gridiron.ingest import (
    content_hash,
    download_and_land,
    ingest,
    landed_hashes,
    present_seasons,
    season_of_date,
    seasons_to_fetch,
    split_ecr,
)

PBP = DATASETS["pbp"]


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


class FakeRemote:
    """Stands in for GitHub: serves bytes per URL and records requests."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requests: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.requests.append(url)
        if url not in self.payloads:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]
        return io.BytesIO(self.payloads[url])


def test_urls_match_the_verified_release_layout() -> None:
    base = "https://github.com/nflverse/nflverse-data/releases/download"
    assert PBP.season_url(2024) == f"{base}/pbp/play_by_play_2024.parquet"
    assert DATASETS["player_stats"].season_url(2024) == (
        f"{base}/stats_player/stats_player_week_2024.parquet"
    )
    assert DATASETS["schedules"].url == f"{base}/schedules/games.parquet"
    assert DATASETS["ecr"].url.endswith("dynastyprocess/data/raw/master/files/db_fpecr.parquet")
    with pytest.raises(ValueError, match="not published per season"):
        DATASETS["schedules"].season_url(2024)


@pytest.mark.parametrize(
    ("today", "expected"),
    [(dt.date(2026, 10, 8), 2026), (dt.date(2027, 1, 20), 2026), (dt.date(2026, 8, 31), 2025)],
)
def test_current_season(today: dt.date, expected: int) -> None:
    assert current_season(today) == expected


def test_season_of_date_puts_january_in_the_previous_season() -> None:
    assert season_of_date(dt.date(2026, 1, 3)) == 2025
    assert season_of_date(dt.date(2025, 9, 12)) == 2025


def test_seasons_to_fetch_skips_present_but_always_refreshes_current() -> None:
    assert seasons_to_fetch([2022, 2023, 2024], present=[2022, 2024], current=2024) == [
        2023,
        2024,
    ]


def test_download_writes_once_per_distinct_content(tmp_path: Path) -> None:
    url = PBP.season_url(2026)
    remote = FakeRemote({url: b"week-1"})
    first = download_and_land(PBP, 2026, tmp_path, remote)
    second = download_and_land(PBP, 2026, tmp_path, remote)
    remote.payloads[url] = b"week-2"
    third = download_and_land(PBP, 2026, tmp_path, remote)

    assert (first.action, second.action, third.action) == ("written", "unchanged", "written")
    assert first.path is not None and first.path.read_bytes() == b"week-1"
    assert first.path.parent == tmp_path / "pbp" / "2026"
    assert first.path.name.startswith("play_by_play_2026_")
    assert len(landed_hashes(tmp_path, PBP, 2026)) == 2


def test_ingest_is_idempotent_for_completed_seasons(tmp_path: Path) -> None:
    remote = FakeRemote({PBP.season_url(s): str(s).encode() for s in (2024, 2025, 2026)})
    first = ingest(["pbp"], [2024, 2025, 2026], tmp_path, current=2026, fetch=remote)
    assert [r.action for r in first] == ["written", "written", "written"]

    remote.requests.clear()
    second = ingest(["pbp"], [2024, 2025, 2026], tmp_path, current=2026, fetch=remote)
    assert [r.action for r in second] == ["skipped", "skipped", "unchanged"]
    assert remote.requests == [PBP.season_url(2026)]
    assert present_seasons(tmp_path, PBP) == [2024, 2025, 2026]


def test_current_season_not_published_yet_is_not_an_error(tmp_path: Path) -> None:
    remote = FakeRemote({PBP.season_url(2025): b"x"})
    results = ingest(["pbp"], [2025, 2026], tmp_path, current=2026, fetch=remote)
    assert [r.action for r in results] == ["written", "not_published"]
    with pytest.raises(urllib.error.HTTPError):
        ingest(["pbp"], [2024], tmp_path, current=2026, fetch=remote)


def test_failed_download_leaves_no_partial_file(tmp_path: Path) -> None:
    class Broken(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise ConnectionError("network dropped")

    with pytest.raises(ConnectionError):
        download_and_land(PBP, 2024, tmp_path, lambda url: Broken())
    assert landed_hashes(tmp_path, PBP, 2024) == set()
    assert not list(tmp_path.rglob("*.parquet"))


def test_integer_columns_are_widened_so_seasons_share_a_schema(tmp_path: Path) -> None:
    dataset = DATASETS["injuries"]
    source = pd.DataFrame({"season": pd.array([2021], "int32"), "team": ["DET"]})
    remote = FakeRemote({dataset.season_url(2021): parquet_bytes(source)})
    result = download_and_land(dataset, 2021, tmp_path, remote)
    assert result.path is not None
    schema = pq.read_schema(result.path)
    assert schema.field("season").type == pa.float64()
    assert schema.field("team").type == pa.string()


def test_csv_crosswalk_is_landed_as_parquet_strings(tmp_path: Path) -> None:
    dataset = DATASETS["player_ids"]
    csv = b"fantasypros_id,gsis_id,name\n28013,00-0041562,A\nNA,NA,B\n"
    result = download_and_land(dataset, None, tmp_path, FakeRemote({dataset.url: csv}))
    assert result.path is not None and result.path.parent == tmp_path / "player_ids"
    frame = pd.read_parquet(result.path)
    assert frame["fantasypros_id"].tolist()[0] == "28013"
    assert frame["gsis_id"].isna().tolist() == [False, True]


def ecr_frame() -> pd.DataFrame:
    rows = [
        ("wp", "RB", "2024-12-27", "1"),
        ("wp", "WR", "2025-01-03", "2"),  # January belongs to the 2024 season
        ("wp", "QB", "2025-09-12", "3"),
        ("wp", "DST", "2025-09-12", "4"),  # not a modelled position
        ("rp", "RB", "2025-09-12", "5"),  # season-long ranking, not weekly
    ]
    frame = pd.DataFrame(rows, columns=["ecr_type", "pos", "scrape_date", "id"])
    for column in ("fp_page", "page_type", "player", "team", "ecr", "sd", "best", "worst"):
        frame[column] = "x"
    return frame


def test_ecr_history_is_filtered_to_weekly_positional_and_split_by_season() -> None:
    parts = split_ecr(pa.Table.from_pandas(ecr_frame(), preserve_index=False))
    assert sorted(parts) == [2024, 2025]
    assert parts[2024]["id"].to_pylist() == ["1", "2"]
    assert parts[2025]["id"].to_pylist() == ["3"]
    assert content_hash(parts[2024]) == content_hash(parts[2024])
    assert content_hash(parts[2024]) != content_hash(parts[2025])


def test_ecr_completed_seasons_are_unchanged_on_rerun(tmp_path: Path) -> None:
    remote = FakeRemote({DATASETS["ecr"].url: parquet_bytes(ecr_frame())})
    first = ingest(["ecr"], [2024, 2025], tmp_path, current=2025, fetch=remote)
    second = ingest(["ecr"], [2024, 2025], tmp_path, current=2025, fetch=remote)
    assert [r.action for r in first] == ["written", "written"]
    assert [r.action for r in second] == ["unchanged", "unchanged"]


def test_optional_dataset_failure_is_tolerated_only_after_a_first_landing(
    tmp_path: Path,
) -> None:
    remote = FakeRemote({})
    with pytest.raises(urllib.error.HTTPError):
        ingest(["ecr"], [2025], tmp_path, current=2025, fetch=remote)
    remote.payloads[DATASETS["ecr"].url] = parquet_bytes(ecr_frame())
    ingest(["ecr"], [2025], tmp_path, current=2025, fetch=remote)
    remote.payloads.clear()
    results = ingest(["ecr"], [2025], tmp_path, current=2025, fetch=remote)
    assert [r.action for r in results] == ["failed"]
