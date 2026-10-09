"""Run the Streamlit app headlessly in local (DuckDB) and public (snapshot) modes.

Uses the dbt CI build of the fixture marts (``make dbt-ci`` runs before pytest in
``make check`` and in CI). The same file runs under Streamlit 1.52.2, the version pinned
for Streamlit in Snowflake.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
APP = ROOT / "snowflake" / "streamlit" / "streamlit_app.py"
PUBLIC_APP = ROOT / "streamlit_public" / "streamlit_app.py"
CI_DB = ROOT / "snowflake" / "dbt" / "ci.duckdb"

pytestmark = pytest.mark.skipif(
    not CI_DB.exists(), reason="run `make dbt-ci` first to build the DuckDB marts"
)


def load_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_public_snapshot", ROOT / "scripts" / "export_public_snapshot.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    """Streamlit caches live for the whole process; each test picks its own data source."""
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("GRIDIRON_APP_MODE", "local")
    monkeypatch.setenv("GRIDIRON_DUCKDB", str(CI_DB))
    monkeypatch.syspath_prepend(str(APP.parent))
    test = AppTest.from_file(str(APP), default_timeout=120)
    test.run()
    assert not test.exception
    return test


def sheet(app: AppTest) -> pd.DataFrame:
    frame: pd.DataFrame = app.dataframe[0].value
    return frame


def test_cheat_sheet_renders_the_upcoming_week(app: AppTest) -> None:
    assert app.header[0].value.endswith("cheat sheet")
    assert list(app.radio(key="section").options) == [
        "Cheat sheet",
        "Player index",
        "Start / Sit",
        "Risers",
        "Track record",
    ]
    assert "(upcoming)" in str(app.sidebar.selectbox[0].format_func(app.sidebar.selectbox[0].value))
    table = sheet(app)
    assert not table.empty
    assert {"Tier", "Rank", "Player", "Matchup", "Proj", "Range (10th-90th)"} <= set(table.columns)
    assert table["Rank"].is_monotonic_increasing
    assert "Actual" not in table.columns  # the upcoming week has not been played


def test_positions_formats_and_past_weeks(app: AppTest) -> None:
    app.radio(key="position").set_value("WR").run()
    assert not app.exception
    ppr = sheet(app)
    app.sidebar.radio(key="scoring_format").set_value("Standard").run()
    assert not app.exception
    standard = sheet(app)
    assert standard["Proj"].sum() < ppr["Proj"].sum()  # receptions are worth nothing
    week = app.sidebar.selectbox[0]
    first = min(int(label.split()[1]) for label in week.options)  # labels read "Week N"
    week.set_value(first).run()
    assert not app.exception
    assert app.sidebar.selectbox[0].value == 1
    assert "Actual" in sheet(app).columns


def test_custom_scoring_reranks_with_the_scoring_handler(app: AppTest) -> None:
    app.radio(key="position").set_value("WR").run()
    ppr = sheet(app).set_index("Player")["Proj"]
    app.sidebar.radio(key="scoring_format").set_value("Custom").run()
    app.sidebar.number_input(key="rec").set_value(0.0).run()
    assert not app.exception
    custom = sheet(app).set_index("Player")["Proj"]
    assert (custom.reindex(ppr.index) < ppr + 1e-9).all()
    assert custom.is_monotonic_decreasing


def open_section(app: AppTest, name: str) -> None:
    app.radio(key="section").set_value(name).run()
    assert not app.exception


def test_start_sit_and_other_sections(app: AppTest) -> None:
    open_section(app, "Start / Sit")
    compare = app.multiselect(key="compare")
    options = compare.options[:2]
    compare.set_value(options).run()
    assert not app.exception
    assert any(md.value.startswith("**Start ") for md in app.markdown)
    open_section(app, "Risers")
    app.toggle(key="all_changes").set_value(True).run()
    open_section(app, "Track record")
    app.selectbox(key="scope").set_value(app.selectbox(key="scope").options[-1]).run()
    app.radio(key="record_position").set_value("TE").run()
    assert not app.exception


def card_names(app: AppTest) -> list[str]:
    return [md.value for md in app.markdown if "gl-card-name" in md.value]


def index_table(app: AppTest) -> pd.DataFrame:
    frame: pd.DataFrame = next(df.value for df in app.dataframe if "Found by" in df.value.columns)
    return frame


def test_cheat_sheet_search_understands_initials(app: AppTest) -> None:
    app.radio(key="position").set_value("WR").run()
    app.text_input(key="search").set_value("arsb").run()
    assert not app.exception
    assert list(sheet(app)["Player"]) == ["Amon-Ra St. Brown"]


def test_player_index_finds_players_and_opens_a_card(app: AppTest) -> None:
    open_section(app, "Player index")
    app.text_input(key="index_query").set_value("jeferson").run()  # misspelled
    assert not app.exception
    assert index_table(app)["Player"].iloc[0] == "Justin Jefferson"
    assert any("Justin Jefferson" in name for name in card_names(app))

    app.text_input(key="index_query").set_value("lions wr").run()
    assert not app.exception
    found = index_table(app)
    assert set(found["Team"]) == {"DET"} and set(found["Pos"]) == {"WR"}
    assert app.radio(key="section").value == "Player index"  # stays put across reruns


def test_player_index_suggests_a_spelling_and_browses_by_letter(app: AppTest) -> None:
    open_section(app, "Player index")
    app.text_input(key="index_query").set_value("jared gofff qq").run()
    assert not app.exception
    app.text_input(key="index_query").set_value("").run()
    letters = app.radio(key="index_letter")
    assert "S" in letters.options
    letters.set_value("S").run()
    assert not app.exception


def test_public_mode_reads_an_exported_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = load_exporter().export_marts(CI_DB, tmp_path)
    assert rows["mart_cheat_sheet"] > 0
    exported = pd.read_parquet(tmp_path / "mart_cheat_sheet.parquet")
    assert exported["ecr_rank"].isna().all()  # third-party ranks stay out of the public copy
    monkeypatch.setenv("GRIDIRON_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.delenv("GRIDIRON_APP_MODE", raising=False)
    test = AppTest.from_file(str(PUBLIC_APP), default_timeout=120)
    test.run()
    assert not test.exception
    assert "static snapshot" in " ".join(c.value for c in test.sidebar.caption)
    assert not sheet(test).empty
