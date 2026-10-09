"""Bronze landing: download the public datasets into a directory (a Unity Catalog volume).

Layout written under ``landing_root``::

    <landing_root>/<dataset>/<season>/<stem>_<season>_<sha12>.parquet   seasonal datasets
    <landing_root>/<dataset>/<stem>_<sha12>.parquet                     snapshot datasets

The content hash in the file name makes the landing zone append-only and idempotent:

* a completed season that already has a file is skipped without a network call;
* the current season (and every snapshot dataset) is always re-downloaded, but a new file
  is only written when the content changed, so Auto Loader only sees genuinely new data;
* silver keeps the rows of the most recently landed file per season (or per snapshot), so
  several versions of the same season can coexist in bronze.

Two sources need light handling at landing time, both documented in ``datasets.py``: the
player ID crosswalk is published as CSV and is converted to Parquet (all columns as
strings), and the expert-consensus-ranking history is filtered to weekly positional
rankings and split into one file per season.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gridiron.datasets import (
    DATASETS,
    ECR_COLUMNS,
    ECR_TYPE_WEEKLY_POSITIONAL,
    Dataset,
)

log = logging.getLogger(__name__)

Fetcher = Callable[[str], BinaryIO]
_CHUNK = 1 << 20
NOT_FOUND = 404


@dataclass(frozen=True)
class IngestResult:
    dataset: str
    season: int | None
    action: str  # "skipped", "unchanged", "written", "not_published" or "failed"
    path: Path | None = None

    def describe(self) -> str:
        where = f" {self.season}" if self.season is not None else ""
        target = f" -> {self.path}" if self.path else ""
        return f"{self.dataset}{where}: {self.action}{target}"


def _default_fetch(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": "gridiron-lakehouse"})
    return urllib.request.urlopen(request, timeout=300)  # type: ignore[no-any-return]


def season_of_date(day: dt.date) -> int:
    """NFL season a calendar date belongs to (January-February games are the prior season)."""
    return day.year if day.month >= 3 else day.year - 1


def file_prefix(dataset: Dataset, season: int | None) -> str:
    return f"{dataset.stem}_{season}_" if season is not None else f"{dataset.stem}_"


def target_dir(landing_root: Path, dataset: Dataset, season: int | None) -> Path:
    base = landing_root / dataset.name
    return base / str(season) if season is not None else base


def landed_hashes(landing_root: Path, dataset: Dataset, season: int | None) -> set[str]:
    """Content hashes already landed for a dataset season (parsed from file names)."""
    folder = target_dir(landing_root, dataset, season)
    if not folder.is_dir():
        return set()
    prefix = file_prefix(dataset, season)
    hashes = set()
    for path in folder.glob(f"{prefix}*.parquet"):
        suffix = path.stem.removeprefix(prefix)
        if len(suffix) == 12 and all(c in "0123456789abcdef" for c in suffix):
            hashes.add(suffix)
    return hashes


def present_seasons(landing_root: Path, dataset: Dataset) -> list[int]:
    folder = landing_root / dataset.name
    if not folder.is_dir():
        return []
    return sorted(
        int(p.name)
        for p in folder.iterdir()
        if p.name.isdigit() and landed_hashes(landing_root, dataset, int(p.name))
    )


def has_landed(landing_root: Path, dataset: Dataset) -> bool:
    folder = landing_root / dataset.name
    return folder.is_dir() and any(folder.rglob("*.parquet"))


def seasons_to_fetch(requested: Iterable[int], present: Iterable[int], current: int) -> list[int]:
    """Seasons that need a download: anything missing, plus the current season always."""
    present_set = set(present)
    return sorted({s for s in requested if s not in present_set or s == current})


def _download(url: str, fetch: Fetcher) -> tuple[Path, str]:
    """Stream ``url`` to a local temp file; return its path and 12-character content hash.

    Volumes do not handle partial writes well, and a failed download must never leave a
    truncated file in the landing zone, so files are only copied once complete.
    """
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".download", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with fetch(url) as response:
                while chunk := response.read(_CHUNK):
                    digest.update(chunk)
                    tmp.write(chunk)
        except BaseException:
            tmp.close()
            os.unlink(tmp_path)
            raise
    return tmp_path, digest.hexdigest()[:12]


def _write_if_new(
    source: Path, sha12: str, landing_root: Path, dataset: Dataset, season: int | None
) -> IngestResult:
    if sha12 in landed_hashes(landing_root, dataset, season):
        log.info("%s %s unchanged (%s)", dataset.name, season, sha12)
        return IngestResult(dataset.name, season, "unchanged")
    folder = target_dir(landing_root, dataset, season)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{file_prefix(dataset, season)}{sha12}.parquet"
    shutil.copyfile(source, target)
    log.info("%s %s written to %s", dataset.name, season, target)
    return IngestResult(dataset.name, season, "written", target)


def normalize_types(table: pa.Table) -> pa.Table:
    """Widen integer columns to float64 so every season of a dataset shares one schema."""
    fields = [
        pa.field(f.name, pa.float64()) if pa.types.is_integer(f.type) else f for f in table.schema
    ]
    return table.cast(pa.schema(fields))


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    """Convert a CSV file to Parquet with every column as a nullable string."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=["", "NA"])
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet_path)


def download_and_land(
    dataset: Dataset,
    season: int | None,
    landing_root: Path,
    fetch: Fetcher = _default_fetch,
) -> IngestResult:
    """Download one seasonal or snapshot file and land it if its content is new."""
    url = dataset.season_url(season) if season is not None else dataset.url
    tmp_path, sha12 = _download(url, fetch)
    try:
        if dataset.source_format == "parquet" and not dataset.normalize_types:
            return _write_if_new(tmp_path, sha12, landing_root, dataset, season)
        converted = tmp_path.with_suffix(".parquet")
        try:
            if dataset.source_format == "csv":
                csv_to_parquet(tmp_path, converted)
            else:
                pq.write_table(normalize_types(pq.read_table(tmp_path)), converted)
            return _write_if_new(converted, sha12, landing_root, dataset, season)
        finally:
            converted.unlink(missing_ok=True)
    finally:
        os.unlink(tmp_path)


def split_ecr(table: pa.Table) -> dict[int, pa.Table]:
    """Weekly positional rankings for QB/RB/WR/TE, split by the season of the scrape date."""
    weekly = table.filter(
        pc.and_(
            pc.equal(table["ecr_type"], ECR_TYPE_WEEKLY_POSITIONAL),
            pc.is_in(table["pos"], pa.array(["QB", "RB", "WR", "TE"])),
        )
    ).select(list(ECR_COLUMNS))
    weekly = weekly.cast(pa.schema([(c, pa.string()) for c in ECR_COLUMNS]))
    seasons = [season_of_date(dt.date.fromisoformat(d)) for d in weekly["scrape_date"].to_pylist()]
    weekly = weekly.append_column("_season", pa.array(seasons, pa.int32()))
    out = {}
    for season in sorted(set(seasons)):
        part = weekly.filter(pc.equal(weekly["_season"], season)).drop_columns(["_season"])
        out[season] = part.sort_by(
            [("scrape_date", "ascending"), ("fp_page", "ascending"), ("id", "ascending")]
        )
    return out


def content_hash(table: pa.Table) -> str:
    """Hash of a table's values (not its Parquet bytes, which vary by writer version)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(table.column_names)
    writer.writerows(zip(*(table[c].to_pylist() for c in table.column_names), strict=True))
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()[:12]


def land_ecr(
    seasons: Iterable[int],
    landing_root: Path,
    fetch: Fetcher = _default_fetch,
) -> list[IngestResult]:
    dataset = DATASETS["ecr"]
    tmp_path, _ = _download(dataset.url, fetch)
    try:
        parts = split_ecr(pq.read_table(tmp_path))
    finally:
        os.unlink(tmp_path)
    results = []
    for season in sorted(set(seasons)):
        if season not in parts:
            results.append(IngestResult(dataset.name, season, "not_published"))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.parquet"
            pq.write_table(parts[season], path)
            results.append(
                _write_if_new(path, content_hash(parts[season]), landing_root, dataset, season)
            )
    return results


def _is_not_found(error: BaseException) -> bool:
    return isinstance(error, urllib.error.HTTPError) and error.code == NOT_FOUND


def ingest_dataset(
    dataset: Dataset,
    seasons: Iterable[int],
    landing_root: Path,
    current: int,
    fetch: Fetcher = _default_fetch,
) -> list[IngestResult]:
    if dataset.kind == "ecr":
        return land_ecr(seasons, landing_root, fetch)
    if dataset.kind == "snapshot":
        return [download_and_land(dataset, None, landing_root, fetch)]
    requested = sorted(set(seasons))
    todo = seasons_to_fetch(requested, present_seasons(landing_root, dataset), current)
    results = [IngestResult(dataset.name, s, "skipped") for s in requested if s not in todo]
    for season in todo:
        try:
            results.append(download_and_land(dataset, season, landing_root, fetch))
        except urllib.error.HTTPError as error:
            # Early in a new season some files are not published yet; that is expected.
            if season == current and _is_not_found(error):
                results.append(IngestResult(dataset.name, season, "not_published"))
            else:
                raise
    return sorted(results, key=lambda r: r.season or 0)


def ingest(
    datasets: Iterable[str],
    seasons: Iterable[int],
    landing_root: Path,
    current: int,
    fetch: Fetcher = _default_fetch,
) -> list[IngestResult]:
    """Land every requested dataset.

    An optional dataset that fails only logs a warning, provided it has landed before: the
    pipeline then keeps using the last landed files. On a first run there is nothing to
    fall back on (and Auto Loader cannot infer a schema from an empty folder), so the
    failure is raised.
    """
    seasons = sorted(set(seasons))
    results: list[IngestResult] = []
    for name in datasets:
        dataset = DATASETS[name]
        try:
            results += ingest_dataset(dataset, seasons, landing_root, current, fetch)
        except Exception:
            if dataset.required or not has_landed(landing_root, dataset):
                raise
            log.warning("optional dataset %s failed; continuing", name, exc_info=True)
            results.append(IngestResult(name, None, "failed"))
    return results
