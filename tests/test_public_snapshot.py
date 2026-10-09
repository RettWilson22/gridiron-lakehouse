"""The committed public snapshot (streamlit_public/snapshot) carries no FantasyPros ranks
and only the columns the export allows."""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pandas as pd
import pyarrow.parquet as pq

from tests.conftest import REPO_ROOT

SNAPSHOT = REPO_ROOT / "streamlit_public" / "snapshot"


def load_exporter() -> ModuleType:
    path = REPO_ROOT / "scripts" / "export_public_snapshot.py"
    spec = importlib.util.spec_from_file_location("export_public_snapshot_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_snapshot_has_no_expert_ranks() -> None:
    files = sorted(SNAPSHOT.glob("*.parquet"))
    assert files
    for path in files:
        frame = pd.read_parquet(path)
        for column in [c for c in frame.columns if "ecr" in c.lower()]:
            assert frame[column].isna().all(), f"{path.name}: {column} has values"


def test_the_committed_snapshot_only_has_allowed_columns() -> None:
    exporter = load_exporter()
    files = {path.stem: path for path in SNAPSHOT.glob("*.parquet")}
    assert set(files) == set(exporter.EXPORTED_COLUMNS) == set(exporter.MARTS)
    for mart, path in files.items():
        assert pq.read_schema(path).names == list(exporter.EXPORTED_COLUMNS[mart]), mart


def test_blanked_columns_are_allowed_columns() -> None:
    exporter = load_exporter()
    for mart, columns in exporter.BLANKED_COLUMNS.items():
        assert set(columns) <= set(exporter.EXPORTED_COLUMNS[mart]), mart
