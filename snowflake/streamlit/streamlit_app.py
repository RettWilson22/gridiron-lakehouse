"""Fantasy Football Cheat Sheet: weekly projections, tiers, a player index, start/sit and a
track record.

Runs as Streamlit in Snowflake (dbt marts in GRIDIRON), locally against a DuckDB build of
the same marts, or as a public copy that reads a static Parquet snapshot:

    make dbt-ci && make app      # fixture data
    make local && make app       # full local run
    make app-public              # the public snapshot
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from data_access import ACTUAL_STATS, PROJECTED_STATS, Source, connect, load_helper

tiers = load_helper("tiers")
player_search = load_helper("player_search")

POSITIONS = ("QB", "RB", "WR", "TE")
FORMATS = {"PPR": "ppr", "Half PPR": "half", "Standard": "std", "Custom": "custom"}
# Players shown by default and tiered: two starters' worth per team in a 12-team league
# (the same pool the backtest scores), e.g. "24 QBs, 48 RBs, 72 WRs and 24 TEs".
POOL: dict[str, int] = dict(tiers.POOL)
_POOL_PARTS = [f"{size} {position}s" for position, size in POOL.items()]
POOL_TEXT = ", ".join(_POOL_PARTS[:-1]) + " and " + _POOL_PARTS[-1]
CHART_PLAYERS = 24
ROW_HEIGHT = 30  # compact table rows, like box-score agate
REPO_URL = "https://github.com/RettWilson22/gridiron-lakehouse"
# A newspaper sports page: ink on newsprint, field green for the model and the active
# section, brick red only for warnings (Sit, injury designations).
INK = "#1b1b1b"
GREEN = "#2f5d3a"
BRICK = "#a33a2c"
GRAY = "#8a8272"
# Text and figures in a serif with lining numerals (Georgia's old-style figures don't line
# up in a table); Georgia for the nameplate and headlines.
TEXT_FONT = "Charter, 'Bitstream Charter', Cambria, 'Times New Roman', serif"
# Comparison lines (color, dash): the model stands out, the baselines recede.
METHOD_STYLE = {
    "Model": (GREEN, [1, 0]),
    "Last 3 average": (GRAY, [1, 0]),
    "Season average": (GRAY, [5, 3]),
    "Expert rankings": (BRICK, [1, 0]),
}
# Chart keys sit under the plot, two entries to a row so they fit on a phone.
KEY = alt.Legend(orient="bottom", columns=2)

# System fonts only (Streamlit in Snowflake blocks web fonts). Streamlit in Snowflake does
# not read .streamlit/config.toml, so the page colors and type are set here as well.
STYLE = """
<style>
:root { --gl-paper: #f7f4ec; --gl-paper2: #efeadf; --gl-ink: #1b1b1b; --gl-muted: #5c574d;
  --gl-green: #2f5d3a; --gl-brick: #a33a2c; --gl-rule: #c9c1b0;
  --gl-serif: Georgia, "Times New Roman", serif;
  --gl-text: Charter, "Bitstream Charter", Cambria, "Times New Roman", serif; }
header[data-testid="stHeader"] { background: transparent; }
footer { display: none; }
.stApp { background: var(--gl-paper); color: var(--gl-ink); }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea { font-family: var(--gl-text); }
[data-testid="stSidebar"] { background: var(--gl-paper2); border-right: 1px solid var(--gl-rule); }
[data-testid="stSidebar"] h2 { font-size: 1.05rem; font-variant-caps: small-caps;
  letter-spacing: 0.08em; border-bottom: 1px solid var(--gl-ink); padding: 0 0 0.2rem;
  margin-bottom: 0.4rem; }
/* Narrow margins, like a page set edge to edge: the wide stat tables fit at 1366px. */
.block-container { padding: 2.6rem 2rem 4rem; max-width: 1180px; }
h1, h2, h3, h4 { font-family: var(--gl-serif) !important; color: var(--gl-ink); }
[data-testid="stCaptionContainer"] { opacity: 1; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--gl-muted); }
.stApp a { color: var(--gl-green); }
.stMarkdownColoredText { color: var(--gl-brick) !important; }

.gl-masthead { text-align: center; margin-bottom: 0.4rem; }
.gl-nameplate { font-family: var(--gl-serif); font-weight: 700; color: var(--gl-ink);
  font-size: clamp(1.9rem, 7.4vw, 3.9rem); line-height: 1.05; letter-spacing: -0.01em;
  padding-bottom: 0.45rem; white-space: nowrap; }
.gl-rules { border-top: 1px solid var(--gl-ink); border-bottom: 3px solid var(--gl-ink);
  height: 6px; }
.gl-dateline { display: flex; justify-content: space-between; gap: 0 1rem; flex-wrap: wrap;
  font-family: var(--gl-text); font-variant-caps: small-caps; letter-spacing: 0.06em;
  font-size: 0.95rem; color: var(--gl-ink); padding: 0.3rem 0;
  border-bottom: 1px solid var(--gl-ink); }

/* Section links, like a newspaper's index line: small caps between thin rules. The
   leftmost rule on each line is clipped, so the links wrap cleanly on a phone. */
.st-key-section { width: 100%; overflow: hidden; border-bottom: 1px solid var(--gl-ink);
  padding: 0.1rem 0 0.45rem; margin-bottom: 0.6rem; }
.st-key-section [role="radiogroup"] { gap: 0.3rem 0; flex-wrap: wrap;
  margin-left: calc(-1rem - 1px); }
.st-key-section [role="radiogroup"] label { border-left: 1px solid var(--gl-rule);
  padding: 0.05rem 1rem; margin: 0; cursor: pointer; }
.st-key-section [role="radiogroup"] label > div:first-child { display: none; }
.st-key-section [role="radiogroup"] label > div:last-child { padding-left: 0; }
.st-key-section [role="radiogroup"] label p { font-size: 1.08rem; font-variant-caps: small-caps;
  letter-spacing: 0.05em; color: var(--gl-ink); text-underline-offset: 5px; }
.st-key-section [role="radiogroup"] label:hover p { text-decoration: underline 1px var(--gl-rule); }
.st-key-section [role="radiogroup"] label:has(input:checked) p { color: var(--gl-green);
  font-weight: 700; text-decoration: underline 2px var(--gl-green); }
.st-key-section [role="radiogroup"] label:has(input:focus-visible) p,
.st-key-index_letter [role="radiogroup"] label:has(input:focus-visible) p {
  outline: 2px solid var(--gl-green); outline-offset: 2px; }

/* The A-Z player index reads like the index of a book: plain letters, current one underlined. */
.st-key-index_letter [role="radiogroup"] { gap: 2px; flex-wrap: wrap; }
.st-key-index_letter [role="radiogroup"] label { padding: 0.1rem 0.3rem; margin: 0;
  cursor: pointer; }
.st-key-index_letter [role="radiogroup"] label > div:first-child { display: none; }
.st-key-index_letter [role="radiogroup"] label > div:last-child { padding-left: 0; }
.st-key-index_letter [role="radiogroup"] label p { font-size: 1.1rem; color: var(--gl-ink);
  text-underline-offset: 4px; }
.st-key-index_letter [role="radiogroup"] label:has(input:checked) p { color: var(--gl-green);
  font-weight: 700; text-decoration: underline 2px var(--gl-green); }

/* Headline numbers as a box-score line: small-caps labels, serif figures, thin rules. */
.stHorizontalBlock:has(.stMetric) { gap: 0;
  border-top: 1px solid var(--gl-ink); border-bottom: 1px solid var(--gl-ink); }
.stHorizontalBlock:has(.stMetric) > .stColumn {
  padding: 0.45rem 1rem 0.5rem; }
.stHorizontalBlock:has(.stMetric) > .stColumn:first-child {
  padding-left: 0; }
.stHorizontalBlock:has(.stMetric) > .stColumn + .stColumn {
  border-left: 1px solid var(--gl-rule); }
/* Figures line up when a label wraps: column heads sit on the figures, as in print. */
.stVerticalBlock:has(> .stElementContainer:only-child > .stMetric) { justify-content: flex-end; }
[data-testid="stMetricLabel"] p { font-variant-caps: small-caps; letter-spacing: 0.05em;
  color: var(--gl-muted); white-space: normal; }
[data-testid="stMetricLabel"] > div:first-child { overflow: visible; }
[data-testid="stMetricValue"] { color: var(--gl-ink); font-size: 2.1rem; line-height: 1.2; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { font-family: var(--gl-text); }

/* Notes read as an editor's note between rules, not a colored box. */
.stAlert [data-testid="stAlertContainer"] { background: transparent; border-radius: 0;
  border-top: 1px solid var(--gl-rule); border-bottom: 1px solid var(--gl-rule);
  padding: 0.55rem 0; }
.stAlert [data-testid="stAlertContainer"] p { color: var(--gl-ink); font-style: italic; }

/* Tables open with a heavy rule, like a box score. */
[data-testid="stDataFrame"] { border-top: 2px solid var(--gl-ink); }

/* Toggles and chosen players as plain ink boxes, not app pills. */
.stCheckbox label[data-baseweb="checkbox"] > div:first-child { background: transparent;
  border: 1px solid var(--gl-ink); border-radius: 0; }
.stCheckbox label[data-baseweb="checkbox"] > div:first-child > div { background: var(--gl-ink);
  border-radius: 0; box-shadow: none; }
.stCheckbox label:has(input:checked) > div:first-child { background: var(--gl-green);
  border-color: var(--gl-green); }
.stCheckbox label:has(input:checked) > div:first-child > div { background: var(--gl-paper); }
[data-testid="stMultiSelect"] [data-baseweb="tag"] { background: transparent;
  border: 1px solid var(--gl-ink); border-radius: 0; color: var(--gl-ink); max-width: none; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: var(--gl-ink); max-width: none; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg { color: var(--gl-brick); }
.stHorizontalBlock:has(.stMetric) [data-testid="stCaptionContainer"] p:not(:last-child) {
  margin-bottom: 0.2rem; }

/* Player profile box. */
.st-key-player_card { border: 1px solid var(--gl-ink); border-top-width: 3px;
  padding: 0.2rem 1rem 0.6rem; margin-top: 0.4rem; }
.st-key-player_card [data-testid="stHorizontalBlock"] { border-bottom-color: var(--gl-rule); }
.stApp .gl-card-name { font-family: var(--gl-serif); font-size: 1.75rem; font-weight: 700;
  color: var(--gl-ink); margin: 0.4rem 0 0; line-height: 1.15; }
.stApp .gl-card-meta { font-variant-caps: small-caps; letter-spacing: 0.06em;
  color: var(--gl-muted); font-size: 1rem; margin-bottom: 0.2rem; }
.gl-footer { margin-top: 3rem; padding: 0.7rem 0; border-top: 3px double var(--gl-ink);
  font-size: 0.85rem; color: var(--gl-muted); }
html, body, .stApp, [data-testid="stMain"] { overflow-x: hidden; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 2.8rem; }
  [data-testid="stMain"] h2 { font-size: 1.8rem; }
  .gl-dateline { justify-content: center; font-size: 0.85rem; gap: 0 0.5rem; }
  .gl-dateline span:nth-child(2)::before { content: "\\00b7"; margin-right: 0.5rem; }
  .gl-dateline span:last-child { flex-basis: 100%; }
  .st-key-section [role="radiogroup"] { margin-left: calc(-0.7rem - 1px); }
  .st-key-section [role="radiogroup"] label { padding: 0.05rem 0.7rem; }
  .st-key-section [role="radiogroup"] label p { font-size: 1rem; }
  /* Stacked on a phone: one ledger line per figure, label left and figure right. */
  .stHorizontalBlock:has(.stMetric) > .stColumn {
    padding: 0.3rem 0; }
  .stHorizontalBlock:has(.stMetric) > .stColumn + .stColumn {
    border-left: none; border-top: 1px solid var(--gl-rule); }
  [data-testid="stMetric"] > div { display: flex; justify-content: space-between;
    align-items: baseline; gap: 1rem; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; flex: none; }
}
</style>
"""
MASTHEAD = """
<div class="gl-masthead">
  <div class="gl-nameplate">Gridiron Lakehouse</div>
  <div class="gl-rules"></div>
  <div class="gl-dateline"><span>{season} season</span><span>Week {week} projections</span>
    <span>Fantasy football cheat sheet</span></div>
</div>
"""

st.set_page_config(page_title="Gridiron Lakehouse | Fantasy Cheat Sheet", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
masthead = st.empty()  # filled in once the week is known, for the dateline


@st.cache_resource
def source() -> Source:
    return connect()


@st.cache_data(ttl=600)
def run(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    frame: pd.DataFrame = source().query(sql, list(params))
    return frame


@st.cache_data(ttl=600)
def custom_points(
    table: str, *, stats: tuple[str, ...], prefix: str, season: int, week: int, scoring: str
) -> pd.DataFrame:
    """Points under custom settings (JSON) for every player in one season and week."""
    frame: pd.DataFrame = source().custom_points(
        table,
        stats=stats,
        prefix=prefix,
        where="season = ? and week = ?",
        params=[season, week],
        scoring=json.loads(scoring),
    )
    return frame


@st.cache_data(ttl=600)
def week_sheets(season: int, week: int, fmt: str, settings_json: str) -> pd.DataFrame:
    """Every position's sheet for one week in one scoring format, best first within each
    position. ``settings_json`` holds the custom league settings when ``fmt`` is "custom";
    a change of settings costs two queries (projected and actual points for the week)."""
    rows = run("select * from marts.mart_cheat_sheet where season = ? and week = ?", (season, week))
    if rows.empty:
        return rows
    if fmt != "custom":
        out = rows.assign(
            proj=rows[f"proj_{fmt}"],
            floor=rows[f"floor_{fmt}"],
            ceiling=rows[f"ceiling_{fmt}"],
            rank=rows[f"pos_rank_{fmt}"],
            tier=rows[f"tier_{fmt}"],
            advice=rows[f"start_sit_{fmt}"],
            actual=rows[f"actual_{fmt}"],
        )
        return out.sort_values(["position", "rank"]).reset_index(drop=True)

    projected = custom_points(
        "marts.mart_cheat_sheet",
        stats=PROJECTED_STATS,
        prefix="proj_",
        season=season,
        week=week,
        scoring=settings_json,
    )
    out = rows.merge(projected, on="player_id")
    ratio = (out["custom_points"] / out["proj_ppr"].where(out["proj_ppr"] > 0)).fillna(1.0)
    out = out.assign(
        proj=out["custom_points"],
        floor=(out["floor_ppr"] * ratio).clip(upper=out["custom_points"]),
        ceiling=(out["ceiling_ppr"] * ratio).clip(lower=out["custom_points"]),
    )
    out = out.sort_values(["position", "proj"], ascending=[True, False]).reset_index(drop=True)
    out["rank"] = out.groupby("position").cumcount() + 1
    out["tier"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for position, group in out.groupby("position"):
        pool = group[group["rank"] <= POOL.get(str(position), 0)]
        out.loc[pool.index, "tier"] = tiers.assign_tiers(pool["proj"].tolist())
    out["advice"] = [
        tiers.start_sit(str(p), int(r)) for p, r in zip(out["position"], out["rank"], strict=True)
    ]
    actual = custom_points(
        "marts.mart_player_game_log",
        stats=ACTUAL_STATS,
        prefix="",
        season=season,
        week=week,
        scoring=settings_json,
    ).rename(columns={"custom_points": "actual_custom"})
    out = out.merge(actual, on="player_id", how="left")
    out["actual"] = out["actual_custom"].where(out["actual_ppr"].notna())
    out.loc[out["actual_ppr"].notna() & out["actual"].isna(), "actual"] = 0.0
    return out


def fmt_range(low: float, high: float) -> str:
    return f"{low:.1f}\u2013{high:.1f}"  # an en dash, as in print


# Markdown, HTML and Streamlit syntax characters (links, images, emphasis, raw HTML,
# :color[...] directives, $math$).
_MARKDOWN_SYNTAX = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|~$:&])")


def plain(value: object) -> str:
    """Third-party or typed text for anything Streamlit renders as Markdown (markdown,
    captions, info boxes, metric and button labels): every syntax character is
    backslash-escaped, so the text shows exactly as written and never as a link, an image
    or HTML. Player names, teams and injury notes come from public data feeds."""
    return _MARKDOWN_SYNTAX.sub(r"\\\1", str(value))


def show_chart(chart: Any) -> None:
    """An Altair chart set like a newspaper graphic: serif type, ink axes, no frame, and
    faint rules only on the value axis. Titles go in a heading above the chart: Streamlit's
    fit-to-height layout can push a Vega title out of view."""
    st.altair_chart(
        chart.configure(font=TEXT_FONT)
        .configure_axis(
            grid=False,
            domain=True,
            labelLimit=220,
            domainColor=INK,
            tickColor=INK,
            labelColor=INK,
            titleColor=INK,
            labelFont=TEXT_FONT,
            titleFont=TEXT_FONT,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_axisX(labelAngle=0)
        # The unit reads across the top of the value axis, as in a printed chart.
        .configure_axisY(
            titleAngle=0, titleAlign="left", titleBaseline="bottom", titleX=0, titleY=-8
        )
        .configure_axisQuantitative(grid=True, gridColor="#e2dccf")
        .configure_view(stroke=None)
        .configure_legend(
            labelFont=TEXT_FONT,
            labelColor=INK,
            labelFontSize=12,
            titleFont=TEXT_FONT,
            titleColor=INK,
            symbolType="stroke",
            symbolStrokeWidth=2.5,
            symbolSize=180,
            offset=6,
            padding=0,
        ),
        width="stretch",
    )


def box_score(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    flagged: tuple[str, ...] = (),
    **options: Any,
) -> Any:
    """A table set like a box score: compact rows and capitalized column heads (labels
    only; the data keeps its column names). Cells in ``flagged`` columns that call for
    action are printed in brick red."""
    config = config or {}
    heads: dict[str, Any] = {
        str(c): {**config.get(str(c), {}), "label": str(c).upper()} for c in frame.columns
    }
    return st.dataframe(
        frame.style.map(flag, subset=list(flagged)) if flagged else frame,
        hide_index=True,
        width="stretch",
        # Up to 13 rows, then scroll; the frame ends on a row boundary, not mid-row (the
        # header row is 36px whatever the row height, plus 2px of border).
        height=min(len(frame), 13) * ROW_HEIGHT + 38,
        row_height=ROW_HEIGHT,
        column_config=heads,
        **options,
    )


def flag(value: object) -> str:
    """Brick red for the cells that call for action: Sit and any injury designation."""
    return f"color: {BRICK}" if value not in ("", "Start", "Flex") else ""


# Sidebar ---------------------------------------------------------------------------------

weeks = run(
    "select season, week, max(cast(is_upcoming as int)) as upcoming, max(kind) as kind "
    "from marts.mart_cheat_sheet group by season, week order by week desc"
)
if weeks.empty:
    st.error("No projections found.")
    st.stop()
season = int(weeks["season"].max())
upcoming = weeks.loc[weeks["upcoming"] == 1, "week"]
upcoming_week = int(upcoming.iloc[0]) if not upcoming.empty else None
default_week = upcoming_week if upcoming_week is not None else int(weeks["week"].max())
week_options = [int(w) for w in weeks["week"]]

st.sidebar.header("Settings")
week = st.sidebar.selectbox(
    "Week",
    week_options,
    index=week_options.index(default_week),
    format_func=lambda w: f"Week {w}" + (" (upcoming)" if w == upcoming_week else ""),
)
format_label = st.sidebar.radio("Scoring", list(FORMATS), key="scoring_format")
fmt = FORMATS[str(format_label)]

scoring: dict[str, Any] = {}
if fmt == "custom":
    # A form, so changing several settings reruns the sheet once, on Apply.
    with st.sidebar.form("league_scoring"):
        st.markdown("**League scoring**")
        reception = st.number_input("Points per reception", 0.0, 2.0, 1.0, 0.25, key="rec")
        pass_td = st.number_input("Passing touchdown", 2.0, 8.0, 4.0, 1.0, key="pass_td")
        pass_yards = st.number_input("Passing yards per point", 10, 50, 25, 5, key="pass_yd")
        interception = st.number_input("Interception", -6.0, 0.0, -2.0, 1.0, key="int")
        yards = st.number_input("Rushing / receiving yards per point", 5, 20, 10, 1, key="yd")
        touchdown = st.number_input("Rushing / receiving touchdown", 4.0, 8.0, 6.0, 1.0, key="td")
        fumble = st.number_input("Fumble lost", -4.0, 0.0, -2.0, 1.0, key="fumble")
        premium = st.number_input("Tight end bonus per reception", 0.0, 1.5, 0.0, 0.25, key="te")
        bonus_100 = st.number_input(
            "100-yard rushing or receiving bonus", 0.0, 6.0, 0.0, 1.0, key="bonus_100"
        )
        bonus_300 = st.number_input("300-yard passing bonus", 0.0, 6.0, 0.0, 1.0, key="bonus_300")
        st.form_submit_button("Apply", type="primary")
    scoring = {
        "preset": "ppr",
        "rec": reception,
        "pass_td": pass_td,
        "pass_yd": 1 / pass_yards,
        "pass_int": interception,
        "rush_yd": 1 / yards,
        "rec_yd": 1 / yards,
        "rush_td": touchdown,
        "rec_td": touchdown,
        "fumble_lost": fumble,
        "te_rec_premium": premium,
        "bonus_rush_100": bonus_100,
        "bonus_rec_100": bonus_100,
        "bonus_pass_300": bonus_300,
    }
    st.sidebar.caption(
        "Custom points are computed from each player's projected stat line"
        + (" by the FANTASY_POINTS UDF in Snowflake." if source().name == "snowflake" else ".")
        + " Floors and ceilings are scaled from PPR. Yardage bonuses applied to an average "
        "stat line understate their real value."
    )
settings_json = json.dumps(scoring, sort_keys=True) if fmt == "custom" else ""
st.sidebar.caption(f"Data: {source().description}")
st.sidebar.markdown(
    f"Built on Databricks and Snowflake. [How it works]({REPO_URL}#readme) · "
    f"[Source code]({REPO_URL})"
)


def scope_order(scope: str) -> tuple[int, str]:
    """Sort key for backtest scopes: the pooled scope ("2023-2025") has the longest name,
    so it sorts last. Same rule as ``gridiron.backtest.pooled_scope``."""
    return len(scope), scope


def headline_numbers() -> dict[str, str]:
    """The backtest in three numbers, from the pooled backtest scope."""
    summary = run("select * from marts.mart_backtest_summary")
    if summary.empty:
        return {}
    scope = max(summary["scope"].unique(), key=scope_order)
    rows = summary[summary["scope"] == scope].pivot_table(
        index="position", columns="method", values=["mae", "n"]
    )
    weights = rows[("n", "model")]
    model = (rows[("mae", "model")] * weights).sum() / weights.sum()
    last3 = (rows[("mae", "last3")] * weights).sum() / weights.sum()
    return {
        "scope": scope,
        "player_weeks": f"{int(weights.sum()):,}",
        "error_cut": f"{(last3 - model) / last3:.0%}",
    }


week_info = weeks[weeks["week"] == week].iloc[0]
masthead.markdown(MASTHEAD.format(season=season, week=week), unsafe_allow_html=True)
st.header(f"Week {week} cheat sheet")
if week_info["kind"] == "live":
    st.caption(
        f"{season} week {week}: live projections from the weekly Databricks job. A "
        "projection is frozen once its game kicks off, so past weeks show exactly what was "
        "published beforehand."
    )
else:
    st.caption(
        f"{season} week {week}: projected by the walk-forward backtest (a model trained on "
        "earlier seasons, using only information from before kickoff), because live "
        "projections started later in the season."
    )

# Every position for the selected week and format, best projection first. Sections slice
# it by position; it is computed (and cached) once per week, format and settings.
everyone = week_sheets(season, week, fmt, settings_json).sort_values(
    "proj", ascending=False, kind="stable"
)

headline = headline_numbers()
metric_columns = st.columns(3)
metric_columns[0].metric(f"Players projected, week {week}", f"{len(everyone):,}")
if headline:
    metric_columns[1].metric(
        "Less error than a last-3-games average",
        headline["error_cut"],
        help="Mean absolute error in PPR points, weighted across positions, in the "
        f"{headline['scope']} walk-forward backtest. See Track record.",
    )
    metric_columns[2].metric(
        "Player-weeks backtested",
        headline["player_weeks"],
        help=f"Seasons {headline['scope']}, each projected only with information from "
        "before kickoff.",
    )


def position_sheet(position: str) -> pd.DataFrame:
    """The selected week's sheet for one position, in rank order."""
    return everyone[everyone["position"] == position].sort_values("rank").reset_index(drop=True)


@st.cache_data(ttl=600)
def season_players(season: int) -> pd.DataFrame:
    """Everyone with a game or a projection this season, with their latest team."""
    seen = pd.concat(
        [
            run(
                "select player_id, player_name, position, team, week "
                "from marts.mart_player_game_log where season = ?",
                (season,),
            ),
            run(
                "select player_id, player_name, position, team, week "
                "from marts.mart_cheat_sheet where season = ?",
                (season,),
            ),
        ],
        ignore_index=True,
    )
    latest = seen.sort_values("week").drop_duplicates("player_id", keep="last")
    return latest.drop(columns="week").reset_index(drop=True)


def as_players(frame: pd.DataFrame) -> list[Any]:
    return [
        player_search.Player(str(r.player_id), str(r.player_name), str(r.position), str(r.team))
        for r in frame.itertuples()
    ]


# Cheat sheet -------------------------------------------------------------------------------


def cheat_sheet_section() -> None:
    position = str(st.radio("Position", POSITIONS, horizontal=True, key="position"))
    sheet = position_sheet(position)
    search = st.text_input(
        "Find a player",
        key="search",
        placeholder="Name, initials or a close spelling",
        max_chars=player_search.MAX_QUERY_CHARS,
    )
    show_all = st.toggle("Show every projected player", key="show_all")
    view = sheet if show_all else sheet[sheet["rank"] <= POOL[position]]
    if search.strip():
        found = [m.player.player_id for m in player_search.search(as_players(sheet), search)]
        view = sheet.set_index("player_id").loc[found].reset_index() if found else sheet.head(0)
    if bool(week_info["upcoming"]) and (sheet["injury_status"] == "None").all():
        st.info(
            "No game-status designations yet for this week: they come with the final "
            "injury report (Friday for Sunday games), and the Saturday run picks them up."
        )
    table = pd.DataFrame(
        {
            "Tier": view["tier"],
            "Rank": view["rank"],
            "Player": view["player_name"],
            "Team": view["team"],
            "Matchup": view["matchup"],
            "Proj": view["proj"].round(1),
            "Range (10th-90th)": [
                fmt_range(lo, hi) for lo, hi in zip(view["floor"], view["ceiling"], strict=True)
            ],
            "Injury": view["injury_status"].replace({"None": ""}),
            "Start / Sit": view["advice"],
        }
    )
    if view["ecr_rank"].notna().any():
        table["Expert rank"] = view["ecr_rank"]
    if view["actual"].notna().any():
        table["Actual"] = view["actual"].round(1)
    box_score(
        table,
        {
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Actual": st.column_config.NumberColumn(format="%.1f"),
        },
        flagged=("Injury", "Start / Sit"),
    )

    chart_rows = sheet.head(CHART_PLAYERS).assign(
        label=lambda d: d["rank"].astype(str) + ". " + d["player_name"]
    )
    if not chart_rows.empty:
        base = alt.Chart(chart_rows).encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
            # Tiers alternate green and gray, like shaded bands in a printed table.
            color=alt.Color("tier:O", title="Tier", scale=alt.Scale(range=[GREEN, GRAY])),
            tooltip=["player_name", "matchup", alt.Tooltip("proj:Q", format=".1f"), "tier"],
        )
        ranges = base.mark_rule(strokeWidth=3, opacity=0.5).encode(
            x=alt.X("floor:Q", title="Fantasy points (10th to 90th percentile range)"),
            x2="ceiling:Q",
        )
        points = base.mark_circle(size=70, opacity=1).encode(x="proj:Q")
        show_chart((ranges + points).properties(height=22 * len(chart_rows)))


# Player index ------------------------------------------------------------------------------


def render_player_card(player: pd.Series) -> None:
    """One player: this week's projection and the season so far."""
    name, position, team = (
        html.escape(str(player[c])) for c in ("player_name", "position", "team")
    )
    this_week = everyone[everyone["player_id"] == player["player_id"]]
    with st.container(key="player_card"):  # the profile box
        st.markdown(  # raw HTML for the card's styling, so every value is HTML-escaped
            f'<p class="gl-card-name">{name}</p><p class="gl-card-meta">{position}, {team}</p>',
            unsafe_allow_html=True,
        )
        if this_week.empty:
            st.caption(f"Not projected for week {week} (bye week or not expected to play).")
        else:
            row = this_week.iloc[0]
            cells = st.columns(4)
            cells[0].metric(f"Week {week} projection", f"{row['proj']:.1f}")
            cells[1].metric("Range (10th-90th)", fmt_range(row["floor"], row["ceiling"]))
            tier = "" if pd.isna(row["tier"]) else f", tier {int(row['tier'])}"
            cells[2].metric("Position rank", f"{row['position']}{int(row['rank'])}{tier}")
            cells[3].metric("Start / Sit", str(row["advice"]))
            injury = row["injury_status"]
            st.caption(
                f"{plain(row['matchup'])}, team total {row['implied_points']:.1f}"
                + (f". Injury report: :red[{plain(injury)}]" if injury != "None" else "")
            )

    games = run(
        "select week, opponent, fantasy_points_ppr, proj_ppr, snap_share, targets, carries, "
        "receptions, passing_yards, rushing_yards, receiving_yards, passing_tds, rushing_tds, "
        "receiving_tds from marts.mart_player_game_log where season = ? and player_id = ? "
        "order by week",
        (season, player["player_id"]),
    )
    if games.empty:
        st.caption(f"No games yet in {season}.")
        return
    st.markdown(
        f"**{season} so far:** {len(games)} games, {games['fantasy_points_ppr'].mean():.1f} PPR "
        f"points per game, best {games['fantasy_points_ppr'].max():.1f}."
    )
    log = pd.DataFrame(
        {
            "Week": games["week"],
            "Opponent": games["opponent"],
            "PPR points": games["fantasy_points_ppr"].round(1),
            "Projected": games["proj_ppr"].round(1),
            "Snap %": (games["snap_share"] * 100).round(0),
            "Targets": games["targets"],
            "Carries": games["carries"],
            "Yards": (
                games["passing_yards"] + games["rushing_yards"] + games["receiving_yards"]
            ).round(0),
            "TDs": games["passing_tds"] + games["rushing_tds"] + games["receiving_tds"],
        }
    )
    one_place = st.column_config.NumberColumn(format="%.1f")
    box_score(log, {"PPR points": one_place, "Projected": one_place})
    trend = games.melt(
        id_vars="week",
        value_vars=["fantasy_points_ppr", "proj_ppr"],
        var_name="series",
        value_name="points",
    ).dropna()
    trend["series"] = trend["series"].map(
        {"fantasy_points_ppr": "Actual", "proj_ppr": "Projected before kickoff"}
    )
    show_chart(
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("points:Q", title="PPR points"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(range=[GREEN, GRAY]),
                legend=KEY,
            ),
        )
        .properties(height=260)
    )


def use_suggestion(name: str) -> None:
    st.session_state.index_query = name


def player_index_section() -> None:
    roster = season_players(season)
    universe = as_players(roster)
    relevance = dict(zip(everyone["player_id"].astype(str), everyone["proj"], strict=True))
    query = st.text_input(
        "Search the player index",
        key="index_query",
        placeholder="Try: cmc, jsn, mccaffery, jefferson justin, lions wr, chiefs rb",
        max_chars=player_search.MAX_QUERY_CHARS,
    )
    st.caption(
        "Finds players by full or partial name, last name first, initials, common nicknames "
        "and close spellings. Team and position words narrow the list."
    )
    if query.strip():
        matches = player_search.search(universe, query, relevance)
        if not matches:
            hint = player_search.did_you_mean(universe, query)
            st.info(f'No players match "{plain(query)}".')
            if hint:
                st.button(f"Search for {plain(hint)}", on_click=use_suggestion, args=(hint,))
        ids = [m.player.player_id for m in matches]
        how = {m.player.player_id: m.reason for m in matches}
        listing = roster.set_index("player_id").loc[ids].reset_index() if ids else roster.head(0)
        listing["Found by"] = listing["player_id"].map(how)
        strong = bool(matches) and matches[0].score >= player_search.PREFIX
        clear_winner = len(matches) == 1 or (strong and matches[0].score > matches[1].score)
    else:
        letters = sorted({player_search.index_letter(p.name) for p in universe})
        left, right = st.columns([3, 1])
        letter = left.radio("Index", letters, horizontal=True, key="index_letter")
        position_filter = right.selectbox("Position", ["All", *POSITIONS], key="index_position")
        in_letter = roster[roster["player_name"].map(player_search.index_letter) == letter]
        if position_filter != "All":
            in_letter = in_letter[in_letter["position"] == position_filter]
        listing = in_letter.assign(
            surname=in_letter["player_name"].map(player_search.last_name)
        ).sort_values(["surname", "player_name"])
        clear_winner = False

    projections = everyone.set_index("player_id")
    table = pd.DataFrame(
        {
            "Player": listing["player_name"],
            "Pos": listing["position"],
            "Team": listing["team"],
            f"Week {week}": listing["player_id"]
            .map(projections["proj"])
            .map(lambda v: "" if pd.isna(v) else f"{v:.1f}"),
            "Rank": listing["player_id"]
            .map(projections["rank"])
            .map(lambda v: "" if pd.isna(v) else str(int(v))),
        }
    )
    if "Found by" in listing:
        table["Found by"] = listing["Found by"]
    if not table.empty:
        st.caption(f"{len(table)} player{'s' if len(table) != 1 else ''}. Select a row to open it.")
        index_event = box_score(
            table,
            on_select="rerun",
            selection_mode="single-row",
            key=f"index_table_{query.strip().lower()}_{len(table)}",
        )
        rows = index_event.selection.rows if index_event is not None else []
        if rows:
            render_player_card(listing.iloc[rows[0]])
        elif clear_winner:
            render_player_card(listing.iloc[0])


# Start / Sit -----------------------------------------------------------------------------


def start_sit_section() -> None:
    labels = {
        row.player_id: f"{row.player_name} ({row.position}, {row.team})"
        for row in everyone.itertuples()
    }
    by_label = {label: pid for pid, label in labels.items()}
    # A typical flex decision as the opening example: two running backs ranked back to back.
    backs = everyone[everyone["position"] == "RB"].sort_values("proj", ascending=False)
    example = [labels[pid] for pid in backs["player_id"].iloc[11:13]]
    chosen_labels = st.multiselect(
        "Compare up to three players",
        list(by_label),
        default=example,
        max_selections=3,
        key="compare",
        help="Starts with two running backs ranked back to back; pick your own players.",
    )
    chosen = [by_label[label] for label in chosen_labels]
    if not chosen:
        st.caption("Pick two or three players to compare their projections side by side.")
        return
    picked = everyone.set_index("player_id").loc[chosen].reset_index()
    picked = picked.sort_values("proj", ascending=False).reset_index(drop=True)
    columns = st.columns(len(picked))
    for column, row in zip(columns, picked.to_dict("records"), strict=True):
        injury = row["injury_status"]
        with column:
            st.metric(plain(row["player_name"]), f"{row['proj']:.1f}", help="Projected points")
            advice = plain(row["advice"])
            if row["advice"] == "Sit":
                advice = f":red[{advice}]"
            st.caption(
                f"{plain(row['position'])}{row['rank']} | tier {row['tier']} | {advice}\n\n"
                f"{plain(row['matchup'])}, team total {row['implied_points']:.1f}\n\n"
                f"Range {fmt_range(row['floor'], row['ceiling'])}"
                + (f"\n\nInjury: :red[{plain(injury)}]" if injury != "None" else "")
            )
    if len(picked) > 1:
        best, second = picked.iloc[0], picked.iloc[1]
        margin = best["proj"] - second["proj"]
        overlap = min(best["ceiling"], second["ceiling"]) - max(best["floor"], second["floor"])
        width = best["ceiling"] - best["floor"]
        close = margin < 1.0 or overlap > 0.8 * width
        st.markdown(
            f"**Start {plain(best['player_name'])}**: projected {margin:.1f} points ahead"
            + (", a close call given how much the ranges overlap." if close else ".")
        )
    history = run(
        "select player_id, week, fantasy_points_ppr, proj_ppr from marts.mart_player_game_log "
        "where season = ? and week < ? order by week",
        (season, week),
    )
    recent = history[history["player_id"].isin(chosen)]
    if not recent.empty:
        recent = recent.assign(player=recent["player_id"].map(labels))
        st.markdown("**Recent games (PPR)**")
        show_chart(
            alt.Chart(recent)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("fantasy_points_ppr:Q", title="PPR points"),
                color=alt.Color(
                    "player:N",
                    title=None,
                    scale=alt.Scale(range=[GREEN, INK, GRAY]),
                    legend=KEY,
                ),
            )
            .properties(height=280)
        )


# Risers ----------------------------------------------------------------------------------


def risers_section() -> None:
    st.caption(
        "Usage over each player's last three games against his earlier games this season "
        "(or last season, early on). Expected PPR points (ffverse expected fantasy points) "
        "turn targets, carries and their field position into points; a riser gained at "
        "least 2.5 expected points per game and now averages 5 or more."
    )
    show_every_change = st.toggle("Show every usage change, not just risers", key="all_changes")
    movers = run(
        "select * from marts.mart_risers where season = ? and week = ? order by riser_rank",
        (season, week),
    )
    if not show_every_change:
        movers = movers[movers["is_riser"].astype(bool)]
    if movers.empty:
        st.caption("No usage trends for this week yet (each player needs three games).")
        return
    box_score(
        pd.DataFrame(
            {
                "Player": movers["player_name"],
                "Pos": movers["position"],
                "Team": movers["team"],
                "xPPR last 3": movers["expected_ppr_recent"].round(1),
                "xPPR before": movers["expected_ppr_before"].round(1),
                "Change": movers["expected_ppr_change"].round(1),
                "Snap share": (movers["snap_share_recent"] * 100).round(0),
                "Snap change": (movers["snap_share_change"] * 100).round(0),
                "Target share change": (movers["target_share_change"] * 100).round(1),
                "This week": movers["proj_ppr"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}"),
                "Rank": movers["pos_rank_ppr"].map(lambda v: "" if pd.isna(v) else str(int(v))),
            }
        ),
        {
            "xPPR last 3": st.column_config.NumberColumn(
                format="%.1f", help="Expected PPR points per game over the last three games"
            ),
            "xPPR before": st.column_config.NumberColumn(
                format="%.1f", help="Expected PPR points per game before that"
            ),
            "Change": st.column_config.NumberColumn(format="%.1f"),
            "Target share change": st.column_config.NumberColumn(format="%.1f"),
            "This week": st.column_config.TextColumn(
                help="This week's PPR projection; blank if not projected"
            ),
            "Rank": st.column_config.TextColumn(help="Position rank this week"),
        },
    )
    if movers["proj_ppr"].isna().any():
        st.caption("Blank projection: not projected this week (bye week or not expected to play).")


# Track record ----------------------------------------------------------------------------


def track_record_section() -> None:
    summary = run("select * from marts.mart_backtest_summary order by scope, position, method")
    scopes = sorted(summary["scope"].unique(), key=scope_order, reverse=True)
    scope = st.selectbox("Backtest seasons", scopes, key="scope")
    chosen_scope = summary[summary["scope"] == scope]
    pivot = chosen_scope.pivot_table(index="position", columns="method", values=["mae", "spearman"])
    record = pd.DataFrame(
        {
            "Model MAE": pivot[("mae", "model")],
            "Last 3 MAE": pivot[("mae", "last3")],
            "Season avg MAE": pivot[("mae", "season_avg")],
            "Model rank corr.": pivot[("spearman", "model")],
            "Expert (ECR) rank corr.": pivot[("spearman", "ecr")],
        }
    ).reindex(list(POSITIONS))
    coverage = chosen_scope[chosen_scope["method"] == "model"].set_index("position")
    record["Inside 10th-90th"] = coverage["interval_coverage"] * 100
    record["Player-weeks"] = coverage["n"]
    two_places = st.column_config.NumberColumn(format="%.2f")
    box_score(
        record.reset_index().rename(columns={"position": "Position"}),
        {
            **{name: two_places for name in record.columns if "MAE" in name or "corr" in name},
            "Inside 10th-90th": st.column_config.NumberColumn(format="%.0f%%"),
            "Player-weeks": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "Walk-forward backtest: each season is projected by a model trained only on earlier "
        "seasons, with features from before each game. Scored on the players FantasyPros "
        f"ranked in the top {POOL_TEXT} that week. MAE is in PPR "
        "points (lower is better); rank correlation is Spearman's, within each week "
        "(higher is better). Expert rankings have no point values, so they are only "
        "compared on ranking. A calibrated 80% range should contain about 80% of outcomes."
    )

    weekly = run(
        "select season, week, position, method, kind, n, abs_error_sum, spearman "
        "from marts.mart_projection_scorecard order by season, week"
    )
    seasons = sorted(weekly["season"].unique(), reverse=True)
    if not seasons:
        return
    left, right = st.columns(2)
    record_season = left.selectbox("Season", seasons, key="record_season")
    record_position = right.radio("Position", POSITIONS, horizontal=True, key="record_position")
    detail = weekly[(weekly["season"] == record_season) & (weekly["position"] == record_position)]
    detail = detail.assign(
        mae=detail["abs_error_sum"] / detail["n"],
        method=detail["method"].map(
            {
                "model": "Model",
                "last3": "Last 3 average",
                "season_avg": "Season average",
                "ecr": "Expert rankings",
            }
        ),
    )

    def method_chart(methods: list[str], value: str, title: str) -> Any:
        """One line per method; the key lists only the methods drawn."""
        colors, dashes = zip(*(METHOD_STYLE[m] for m in methods), strict=True)
        return (
            alt.Chart(detail[detail["method"].isin(methods)])
            .mark_line(point=True)
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y(f"{value}:Q", title=title),
                color=alt.Color(
                    "method:N",
                    title=None,
                    scale=alt.Scale(domain=methods, range=colors),
                    legend=KEY,
                ),
                strokeDash=alt.StrokeDash(
                    "method:N",
                    title=None,
                    scale=alt.Scale(domain=methods, range=dashes),
                    legend=KEY,
                ),
            )
            .properties(height=280)
        )

    st.markdown("**Points: lower is better**")
    show_chart(
        method_chart(
            ["Model", "Last 3 average", "Season average"], "mae", "Mean absolute error (PPR)"
        )
    )
    st.markdown("**Ranking: higher is better**")
    show_chart(method_chart(["Model", "Expert rankings"], "spearman", "Rank correlation"))
    live_weeks = detail[detail["kind"] == "live"]["week"].unique()
    st.caption(
        f"Weeks scored with live (pre-game) projections in {record_season}: "
        + (", ".join(str(w) for w in sorted(live_weeks)) if len(live_weeks) else "none yet")
        + ". Other weeks come from the walk-forward backtest."
    )


# Section navigation that keeps its place when a widget reruns the script (built-in tabs
# jump back to the first tab on a rerun), set as a newspaper's section links. Only the
# selected section runs.
SECTIONS: dict[str, Callable[[], None]] = {
    "Cheat sheet": cheat_sheet_section,
    "Player index": player_index_section,
    "Start / Sit": start_sit_section,
    "Risers": risers_section,
    "Track record": track_record_section,
}
section = st.radio(
    "Section", list(SECTIONS), horizontal=True, key="section", label_visibility="collapsed"
)
SECTIONS[str(section)]()

st.markdown(
    f'<div class="gl-footer">Gridiron Lakehouse, built by Rett Wilson. '
    f'<a href="{REPO_URL}">Source code and method</a>. Data: nflverse (CC BY 4.0) and ffverse '
    "expected fantasy points; FantasyPros expert ranks are used for evaluation only.</div>",
    unsafe_allow_html=True,
)
