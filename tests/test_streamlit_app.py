"""Run the Streamlit app headlessly against the dbt CI build (fixture data in DuckDB)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
APP = ROOT / "snowflake" / "streamlit" / "streamlit_app.py"
CI_DB = ROOT / "snowflake" / "dbt" / "ci.duckdb"

pytestmark = pytest.mark.skipif(
    not CI_DB.exists(), reason="run `make dbt-ci` first to build the DuckDB marts"
)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("GRIDIRON_DUCKDB", str(CI_DB))
    monkeypatch.syspath_prepend(str(APP.parent))
    return AppTest.from_file(str(APP), default_timeout=60)


def test_app_renders_every_tab_without_errors(app: AppTest) -> None:
    app.run()
    assert not app.exception
    assert app.title[0].value == "Fourth Down Explorer"
    assert [m.label for m in app.metric][:2] == ["Neutral fourth downs", "Model said go"]
    assert len(app.tabs) == 4
    assert "Recommendation:" in " ".join(md.value for md in app.markdown)


def test_switching_team_and_calculator_inputs(app: AppTest) -> None:
    app.run()
    team_box = app.sidebar.selectbox[1]
    team_box.select_index(len(team_box.options) - 1).run()
    assert not app.exception
    app.number_input[0].set_value(15).run()  # 4th and 15
    app.slider[0].set_value(80).run()  # own 20
    assert not app.exception
    assert "Recommendation: Punt" in " ".join(md.value for md in app.markdown)
