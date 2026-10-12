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
from collections.abc import Callable, Hashable
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
ROW_HEIGHT = 34
REPO_URL = "https://github.com/RettWilson22/gridiron-lakehouse"
# Color has a job: field green for Start, the model and the active section; red for Sit and
# Out or Doubtful; amber for Questionable. Every text color clears WCAG AA on the page and
# on white.
INK = "#16181c"
MUTED = "#5c6168"
GREEN = "#1e6b3c"
RED = "#b42318"
AMBER = "#9a5b00"
GRAY = "#7b8087"
BAND = "#f3f0e8"  # every other tier in the cheat sheet
# Barlow is served with the public copy (streamlit_public/.streamlit/config.toml). Streamlit
# in Snowflake blocks web fonts, so both stacks fall back to system faces.
TEXT_FONT = "Barlow, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif"
DISPLAY_FONT = (
    "'Barlow Semi Condensed', 'Avenir Next Condensed', 'Bahnschrift SemiCondensed', "
    "'Roboto Condensed', 'Arial Narrow', sans-serif"
)
# Comparison lines (color, dash, width): the model stands out, the baselines recede.
METHOD_STYLE = {
    "Model": (GREEN, [1, 0], 3.0),
    "Last 3 average": (GRAY, [1, 0], 1.5),
    "Season average": (GRAY, [5, 3], 1.5),
    "Expert rankings": (GRAY, [1, 0], 1.5),
}
# Chart keys sit under the plot, two entries to a row so they fit on a phone.
KEY = alt.Legend(orient="bottom", columns=2)

# Streamlit in Snowflake does not read .streamlit/config.toml, so the page colors and type
# are set here as well. Rules that differ between Streamlit 1.52 (Snowflake) and 1.65 (the
# public copy) name the 1.52 element first.
STYLE = (
    "<style>:root { "
    f"--gl-ink: {INK}; --gl-muted: {MUTED}; --gl-green: {GREEN}; --gl-red: {RED}; "
    f"--gl-amber: {AMBER}; --gl-text: {TEXT_FONT}; --gl-display: {DISPLAY_FONT}; "
    "--gl-page: #f6f4ef; --gl-side: #eeebe4; --gl-surface: #ffffff; --gl-rule: #dcd8cf; "
    "--gl-edge: #857f74; }"
    """
header[data-testid="stHeader"] { background: transparent; }
footer { display: none; }
.stApp { background: var(--gl-page); color: var(--gl-ink); font-family: var(--gl-text); }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-baseweb="select"] div { font-family: var(--gl-text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--gl-display) !important;
  color: var(--gl-ink); letter-spacing: 0; }
.stApp a { color: var(--gl-green); }
.block-container, [data-testid="stMainBlockContainer"] { padding: 1.5rem 2rem 3rem;
  max-width: 1200px; }
html, body, .stApp, [data-testid="stMain"] { overflow-x: hidden; }
.stApp [data-baseweb="input"], .stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="textarea"], .stApp button { border-radius: 3px; }
/* Streamlit in Snowflake has no theme, and its default red accent would read as a warning. */
.stApp label[data-baseweb="radio"]:has(input:checked) > div:first-child {
  background-color: var(--gl-green); }
.stApp [data-testid^="stBaseButton-primary"] { background-color: var(--gl-green);
  border-color: var(--gl-green); color: #ffffff; }
.stApp [data-testid^="stBaseButton-primary"]:hover { background-color: #17552f; }
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="select"] > div:focus-within {
  border-color: var(--gl-green); }

[data-testid="stSidebar"] { background: var(--gl-side); border-right: 1px solid var(--gl-rule); }
[data-testid="stSidebar"] h3 { font-size: 1.1rem; font-weight: 700; padding: 0.6rem 0 0; }
[data-testid="stCaptionContainer"] { opacity: 1; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--gl-muted);
  font-size: 0.95rem; }

/* Header: wordmark, one muted line, then the section tabs. */
.gl-wordmark { font-family: var(--gl-display); font-weight: 700; font-size: 1.75rem;
  line-height: 1.1; text-transform: uppercase; letter-spacing: 0.02em; color: var(--gl-ink); }
.gl-tagline { font-size: 0.95rem; color: var(--gl-muted); margin-top: 0.15rem; }

/* Section tabs: the radio keeps its place across reruns, which st.tabs does not. */
.st-key-section { width: 100%; border-bottom: 1px solid var(--gl-rule); margin-top: -0.6rem; }
.st-key-section [role="radiogroup"] { gap: 0 1.75rem; flex-wrap: wrap; }
.st-key-section [role="radiogroup"] label { margin: 0 0 -1px; padding: 0.6rem 0 0.45rem;
  border-bottom: 3px solid transparent; cursor: pointer; }
.st-key-section [role="radiogroup"] label p { font-family: var(--gl-display); font-weight: 600;
  font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gl-muted); }
.st-key-section [role="radiogroup"] label:hover { border-bottom-color: var(--gl-rule); }
.st-key-section [role="radiogroup"] label:hover p { color: var(--gl-ink); }
.st-key-section [role="radiogroup"] label:has(input:checked) {
  border-bottom-color: var(--gl-green); }
.st-key-section [role="radiogroup"] label:has(input:checked) p { color: var(--gl-ink); }

/* Position pickers as a segmented control. */
.st-key-position [role="radiogroup"], .st-key-record_position [role="radiogroup"] { gap: 0;
  flex-wrap: nowrap; }
.st-key-position [role="radiogroup"] label, .st-key-record_position [role="radiogroup"] label {
  margin: 0 0 0 -1px; padding: 0.35rem 0.95rem; background: var(--gl-surface);
  border: 1px solid var(--gl-edge); cursor: pointer; }
.st-key-position label:first-child, .st-key-record_position label:first-child {
  border-radius: 3px 0 0 3px; }
.st-key-position label:last-child, .st-key-record_position label:last-child {
  border-radius: 0 3px 3px 0; }
.st-key-position [role="radiogroup"] label p, .st-key-record_position [role="radiogroup"] label p {
  font-family: var(--gl-display); font-weight: 600; font-size: 0.95rem; letter-spacing: 0.04em;
  color: var(--gl-ink); }
.st-key-position label:has(input:checked), .st-key-record_position label:has(input:checked) {
  background: var(--gl-ink); border-color: var(--gl-ink); }
.st-key-position label:has(input:checked) p, .st-key-record_position label:has(input:checked) p {
  color: #ffffff; }

/* The A-Z player index: plain letters, the current one underlined. */
.st-key-index_letter [role="radiogroup"] { gap: 2px; flex-wrap: wrap; }
.st-key-index_letter [role="radiogroup"] label { padding: 0.1rem 0.35rem; margin: 0;
  cursor: pointer; }
.st-key-index_letter [role="radiogroup"] label p { font-family: var(--gl-display);
  font-weight: 600; font-size: 1.05rem; color: var(--gl-muted); text-underline-offset: 5px; }
.st-key-index_letter [role="radiogroup"] label:has(input:checked) p { color: var(--gl-ink);
  text-decoration: underline 3px var(--gl-green); }

/* The radio dot is the label's first child in 1.52 and sits beside the text in 1.65. */
.st-key-section [role="radiogroup"] label > div:first-child,
.st-key-section [role="radiogroup"] label div:has(+ [data-testid="stMarkdownContainer"]),
.st-key-position [role="radiogroup"] label > div:first-child,
.st-key-position [role="radiogroup"] label div:has(+ [data-testid="stMarkdownContainer"]),
.st-key-record_position [role="radiogroup"] label > div:first-child,
.st-key-record_position [role="radiogroup"] label div:has(+ [data-testid="stMarkdownContainer"]),
.st-key-index_letter [role="radiogroup"] label > div:first-child,
.st-key-index_letter [role="radiogroup"] label div:has(+ [data-testid="stMarkdownContainer"]) {
  display: none; }
.st-key-section [role="radiogroup"] label > div:last-child,
.st-key-position [role="radiogroup"] label > div:last-child,
.st-key-record_position [role="radiogroup"] label > div:last-child,
.st-key-index_letter [role="radiogroup"] label > div:last-child { padding-left: 0; }
/* Keyboard focus ring. In 1.65 the browser's :focus-visible also matches after a click, so
   use the data-focus-visible flag react-aria sets for keyboard focus only. */
.stApp [role="radiogroup"] label:not([data-rac]):has(input:focus-visible),
.stApp [role="radiogroup"] label[data-focus-visible] {
  outline: 2px solid var(--gl-green); outline-offset: 2px; }

/* Section titles: one heading and at most one muted line. */
[data-testid="stMain"] h2 { font-size: 2rem; font-weight: 700; line-height: 1.1;
  padding: 0.5rem 0 0; }
[data-testid="stMain"] h3 { font-size: 1.2rem; font-weight: 700; padding: 0.75rem 0 0; }

/* Headline numbers: small muted labels, figures that line up. */
[data-testid="stMetricLabel"] p { font-family: var(--gl-display); font-weight: 600;
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--gl-muted);
  white-space: normal; line-height: 1.25; }
[data-testid="stMetricLabel"] > div:first-child { overflow: visible; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { font-family: var(--gl-text);
  font-weight: 600; font-variant-numeric: tabular-nums; }
[data-testid="stMetricValue"] { color: var(--gl-ink); font-size: 1.85rem; line-height: 1.15; }
.stVerticalBlock:has(> .stElementContainer:only-child > .stMetric) { justify-content: flex-end; }
/* Metric rows (the headline numbers and the player card) are divided by thin rules. */
.st-key-kpis .stHorizontalBlock, .st-key-player_card .stHorizontalBlock { gap: 0; }
.st-key-kpis .stColumn, .st-key-player_card .stColumn { padding: 0 1rem; }
.st-key-kpis .stColumn:first-child, .st-key-player_card .stColumn:first-child { padding-left: 0; }
.st-key-kpis .stColumn + .stColumn, .st-key-player_card .stColumn + .stColumn {
  border-left: 1px solid var(--gl-rule); }
.st-key-kpis .stHorizontalBlock { flex-wrap: nowrap; }
.st-key-kpis .stColumn { min-width: 0; }
.st-key-kpis [data-testid="stMetricValue"] { font-size: 1.6rem; }

/* Notes, not alarms. */
.stAlert [data-testid="stAlertContainer"] { background: var(--gl-surface); color: var(--gl-ink);
  border: 1px solid var(--gl-rule); border-left: 3px solid var(--gl-muted); border-radius: 3px;
  padding: 0.5rem 0.85rem; }
.stAlert [data-testid="stAlertContainer"] p { color: var(--gl-ink); font-size: 0.95rem; }

/* Toggles and chosen players: square, quiet, green when on. */
.stCheckbox label[data-baseweb="checkbox"] > div:first-child,
.stCheckbox label > span + div { background: var(--gl-surface);
  border: 1px solid var(--gl-edge); border-radius: 3px; }
.stCheckbox label[data-baseweb="checkbox"] > div:first-child > div,
.stCheckbox label > span + div > div { background: var(--gl-muted);
  border-radius: 2px; box-shadow: none; }
.stCheckbox label:has(input:checked) > div:first-child,
.stCheckbox label:has(input:checked) > span + div { background: var(--gl-green);
  border-color: var(--gl-green); }
.stCheckbox label:has(input:checked) > div:first-child > div,
.stCheckbox label:has(input:checked) > span + div > div { background: var(--gl-surface); }
[data-testid="stMultiSelect"] [data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-tag] { background: var(--gl-page);
  border: 1px solid var(--gl-edge); border-radius: 2px; color: var(--gl-ink); max-width: none; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-tag] span { color: var(--gl-ink); max-width: none; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg,
[data-testid="stMultiSelect"] [data-tag] svg { color: var(--gl-muted); }

/* Tags: position rank, tier, start/sit and injury in one row. */
.gl-tags { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.4rem 0 0.2rem; }
.gl-tag { font-family: var(--gl-display); font-weight: 600; font-size: 0.8rem; line-height: 1;
  text-transform: uppercase; letter-spacing: 0.05em; padding: 0.3rem 0.45rem;
  border: 1px solid var(--gl-rule); border-radius: 2px; color: var(--gl-ink);
  background: var(--gl-surface); }
.gl-tag-start { background: var(--gl-green); border-color: var(--gl-green); color: #ffffff; }
.gl-tag-sit, .gl-tag-out { color: var(--gl-red); border-color: var(--gl-red); }
.gl-tag-questionable { color: var(--gl-amber); border-color: var(--gl-amber); }

/* Start / Sit: the verdict first, then one card per player. */
.gl-verdict { background: var(--gl-surface); border: 1px solid var(--gl-rule);
  border-left: 4px solid var(--gl-green); border-radius: 3px; padding: 0.8rem 1.1rem;
  margin-bottom: 1rem; }
.stApp .gl-verdict-call { font-family: var(--gl-display); font-weight: 700; font-size: 1.6rem;
  line-height: 1.15; margin: 0; color: var(--gl-ink); }
.gl-verdict-call span { color: var(--gl-green); }
.stApp .gl-verdict-why { margin: 0.25rem 0 0; color: var(--gl-ink); }
.gl-compare { background: var(--gl-surface); border: 1px solid var(--gl-rule); border-radius: 4px;
  padding: 0.85rem 1.1rem 0.9rem; margin-bottom: 1rem; }
.gl-compare-pick { border: 2px solid var(--gl-green); }
.stApp .gl-compare-name { font-family: var(--gl-display); font-weight: 700; font-size: 1.45rem;
  line-height: 1.15; margin: 0.55rem 0 0; color: var(--gl-ink); }
.stApp .gl-compare-proj { font-weight: 600; font-size: 2.6rem; line-height: 1.1; margin: 0.2rem 0 0;
  font-variant-numeric: tabular-nums; color: var(--gl-ink); }
.gl-compare:not(.gl-compare-pick) .gl-compare-proj { color: var(--gl-muted); }
.gl-compare-proj span { font-size: 0.9rem; font-weight: 500; color: var(--gl-muted);
  margin-left: 0.4rem; }
.stApp .gl-compare-line { margin: 0.15rem 0 0; color: var(--gl-muted);
  font-variant-numeric: tabular-nums; }

/* Player card. */
.st-key-player_card { background: var(--gl-surface); border: 1px solid var(--gl-rule);
  border-top: 3px solid var(--gl-ink); border-radius: 3px; padding: 0.9rem 1.25rem 1rem;
  margin-top: 0.5rem; }
.stApp .gl-card-name { font-family: var(--gl-display); font-size: 2rem; font-weight: 700;
  color: var(--gl-ink); margin: 0; line-height: 1.1; }
.gl-footer { margin-top: 3rem; padding-top: 0.75rem; border-top: 1px solid var(--gl-rule);
  font-size: 0.85rem; color: var(--gl-muted); }
.stApp .gl-footer a { color: var(--gl-muted); }

@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 3.25rem 1rem 2rem; }
  .gl-wordmark { font-size: 1.4rem; }
  .gl-tagline { font-size: 0.85rem; }
  .st-key-section [role="radiogroup"] { gap: 0 1.1rem; }
  .st-key-section [role="radiogroup"] label { padding: 0.5rem 0 0.35rem; }
  .st-key-section [role="radiogroup"] label p { font-size: 0.9rem; }
  [data-testid="stMain"] h2 { font-size: 1.6rem; }
  .st-key-kpis .stColumn { flex: 1 1 0; width: auto; padding: 0 0.6rem; }
  .st-key-kpis .stColumn:first-child { padding-left: 0; }
  .st-key-kpis [data-testid="stMetricValue"] { font-size: 1.3rem; }
  .st-key-kpis [data-testid="stMetricLabel"] p { font-size: 0.68rem; letter-spacing: 0.04em; }
  /* The player card's stacked metrics read as a ledger: label left, figure right. */
  .st-key-player_card .stColumn { padding: 0.3rem 0; }
  .st-key-player_card .stColumn + .stColumn { border-left: none;
    border-top: 1px solid var(--gl-rule); }
  .st-key-player_card [data-testid="stMetric"] > div { display: flex;
    justify-content: space-between; align-items: baseline; gap: 1rem; }
  .st-key-player_card [data-testid="stMetricValue"] { font-size: 1.4rem; flex: none; }
  .stApp .gl-compare-proj { font-size: 2.2rem; }
}
</style>
"""
)
HEADER = """
<div class="gl-wordmark">Gridiron Lakehouse</div>
<div class="gl-tagline">Fantasy football cheat sheet · {season} season · Week {week} ·
  {scoring}</div>
"""

st.set_page_config(page_title="Gridiron Lakehouse | Fantasy Cheat Sheet", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)


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
_LINK_SYNTAX = str.maketrans({"[": "&#91;", "]": "&#93;", "(": "&#40;", ")": "&#41;"})


def plain(value: object) -> str:
    """Third-party or typed text for anything Streamlit renders as Markdown (markdown,
    captions, info boxes, metric and button labels): every syntax character is
    backslash-escaped, so the text shows exactly as written and never as a link, an image
    or HTML. Player names, teams and injury notes come from public data feeds."""
    return _MARKDOWN_SYNTAX.sub(r"\\\1", str(value))


def html_text(value: object) -> str:
    """Third-party text for the raw HTML blocks (cards and tags): HTML-escaped, on one line
    (a blank line would end the HTML block and let Markdown parse the rest), and with link
    brackets as entities."""
    return html.escape(" ".join(str(value).split())).translate(_LINK_SYNTAX)


def show_chart(chart: Any) -> None:
    """An Altair chart in the app's type and colors: muted axes, no frame, and faint rules
    only on the value axis. Titles go in a heading above the chart: Streamlit's
    fit-to-height layout can push a Vega title out of view."""
    st.altair_chart(
        chart.configure(font=TEXT_FONT, background="transparent")
        .configure_axis(
            grid=False,
            domain=True,
            labelLimit=220,
            domainColor="#b9b4a9",
            tickColor="#b9b4a9",
            labelColor=MUTED,
            titleColor=MUTED,
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
        .configure_axisQuantitative(grid=True, gridColor="#e6e2d9")
        .configure_view(stroke=None)
        .configure_legend(
            labelFont=TEXT_FONT,
            labelColor=INK,
            labelFontSize=13,
            titleFont=TEXT_FONT,
            titleColor=INK,
            symbolType="stroke",
            symbolStrokeWidth=3,
            symbolSize=180,
            offset=6,
            padding=0,
        ),
        width="stretch",
    )


def box_score(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    styles: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    **options: Any,
) -> Any:
    """A table with uppercase column heads (labels only; the data keeps its column names)
    and blank missing values. ``styles`` returns per-cell CSS for pandas ``Styler.apply``;
    Streamlit honors its color, background-color and font-weight."""
    config = config or {}
    heads: dict[str, Any] = {
        str(c): {**config.get(str(c), {}), "label": str(c).upper()} for c in frame.columns
    }
    return st.dataframe(
        frame.style.apply(styles, axis=None) if styles else frame,
        hide_index=True,
        width="stretch",
        # Up to 13 rows, then scroll; the frame ends on a row boundary, not mid-row (the
        # header row is 36px whatever the row height, plus 2px of border).
        height=min(len(frame), 13) * ROW_HEIGHT + 38,
        row_height=ROW_HEIGHT,
        column_config=heads,
        placeholder="",
        **options,
    )


def advice_css(advice: object) -> str:
    return {
        "Start": f"color: {GREEN}; font-weight: 600",
        "Sit": f"color: {RED}; font-weight: 600",
    }.get(str(advice), "font-weight: 600")


def injury_css(status: object) -> str:
    if status in ("", "None"):
        return ""
    return f"color: {AMBER if status == 'Questionable' else RED}; font-weight: 600"


def sheet_styles(table: pd.DataFrame) -> pd.DataFrame:
    """Cheat sheet emphasis: every other tier shaded, projection and rank bold, the
    range muted, Start green, Sit red, Questionable amber and other designations red."""
    css = pd.DataFrame("", index=table.index, columns=table.columns)
    shaded = table["Tier"].fillna(0).astype(int) % 2 == 0
    css.loc[shaded & table["Tier"].notna(), :] = f"background-color: {BAND}; "
    css["Player"] += "font-weight: 500"
    css["Rank"] += "font-weight: 600"
    css["Proj"] += f"font-weight: 600; color: {INK}"
    css["Range (10th-90th)"] += f"color: {MUTED}"
    css["Team"] += f"color: {MUTED}"
    css["Matchup"] += f"color: {MUTED}"
    css["Tier"] += f"color: {MUTED}"
    css["Start / Sit"] += table["Start / Sit"].map(advice_css)
    css["Injury"] += table["Injury"].map(injury_css)
    return css


def tag(text: str, kind: str = "") -> str:
    classes = f"gl-tag gl-tag-{kind}" if kind else "gl-tag"
    return f'<span class="{classes}">{html_text(text)}</span>'


def advice_tag(advice: str) -> str:
    return tag(advice, {"Start": "start", "Sit": "sit"}.get(advice, ""))


def player_tags(row: Any, *, advice: bool = True) -> str:
    """Position rank, tier, start/sit and any injury designation as one row of tags."""
    tags = [tag(f"{row['position']}{int(row['rank'])}")]
    if not pd.isna(row["tier"]):
        tags.append(tag(f"Tier {int(row['tier'])}"))
    if advice:
        tags.append(advice_tag(str(row["advice"])))
    injury = str(row["injury_status"])
    if injury != "None":
        tags.append(tag(injury, "questionable" if injury == "Questionable" else "out"))
    return '<div class="gl-tags">' + "".join(tags) + "</div>"


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

st.sidebar.subheader("Week", anchor=False)
week = st.sidebar.selectbox(
    "Week",
    week_options,
    index=week_options.index(default_week),
    format_func=lambda w: f"Week {w}" + (" (upcoming)" if w == upcoming_week else ""),
    label_visibility="collapsed",
)
st.sidebar.subheader("Scoring", anchor=False)
format_label = st.sidebar.radio(
    "Scoring", list(FORMATS), key="scoring_format", label_visibility="collapsed"
)
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
        "Points are recomputed from each player's projected stat line. Ranges are scaled "
        "from PPR, and yardage bonuses come out low because they are applied to an average "
        "stat line."
    )
settings_json = json.dumps(scoring, sort_keys=True) if fmt == "custom" else ""
# Naive timestamps (Snowflake TIMESTAMP_NTZ) are UTC.
as_of = pd.to_datetime(
    run("select max(generated_at) as generated_at from marts.mart_cheat_sheet")["generated_at"],
    utc=True,
).iloc[0]
clock = f"{as_of:%I:%M %p}".lstrip("0")
st.sidebar.caption(
    f"Updated {as_of:%b} {as_of.day}, {as_of:%Y}, {clock} UTC",
    help=f"Projections generated {as_of:%Y-%m-%d %H:%M} UTC, read from {source().description}.",
)


def scope_order(scope: str) -> tuple[int, str]:
    """Sort key for backtest scopes: the pooled scope ("2023-2025") has the longest name,
    so it sorts last. Same rule as ``gridiron.backtest.pooled_scope``."""
    return len(scope), scope


def headline_numbers() -> dict[str, str]:
    """Pooled backtest headline: player-weeks, and how much the model cuts last-3's error."""
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
st.markdown(
    HEADER.format(
        season=season,
        week=week,
        scoring="Custom scoring" if fmt == "custom" else format_label,
    ),
    unsafe_allow_html=True,
)
nav = st.container()  # the section tabs, filled in once the sections are defined

# Every position for the selected week and format, best projection first. Sections slice
# it by position; it is computed (and cached) once per week, format and settings.
everyone = week_sheets(season, week, fmt, settings_json).sort_values(
    "proj", ascending=False, kind="stable"
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


def section_title(title: str, note: str) -> None:
    st.header(title, anchor=False)
    st.caption(note)


def cheat_sheet_section() -> None:
    intro, numbers = st.columns([5, 4], vertical_alignment="bottom")
    with intro:
        section_title(
            f"Week {week} cheat sheet",
            "Each projection locks at kickoff, so past weeks show exactly what was published "
            "before the games."
            if week_info["kind"] == "live"
            else "Live projections started later in the season, so this week comes from the "
            "backtest: a model trained on earlier seasons, using only what was known before "
            "kickoff.",
        )
    headline = headline_numbers()
    with numbers.container(key="kpis"):
        cells = st.columns(3)
        cells[0].metric("Players projected", f"{len(everyone):,}")
        if headline:
            cells[1].metric(
                "Less error than a 3-game average",
                headline["error_cut"],
                help="Mean absolute error in PPR points, weighted across positions, in the "
                f"{headline['scope']} backtest, against each player's average over his last "
                "three games. See Track record.",
            )
            cells[2].metric(
                "Player-weeks backtested",
                headline["player_weeks"],
                help=f"Seasons {headline['scope']}, each projected only with information from "
                "before kickoff.",
            )

    picker, finder, toggle = st.columns([3, 4, 3], vertical_alignment="center")
    position = str(
        picker.radio(
            "Position", POSITIONS, horizontal=True, key="position", label_visibility="collapsed"
        )
    )
    sheet = position_sheet(position)
    search = finder.text_input(
        "Find a player",
        key="search",
        placeholder="Find a player: name, initials or a close spelling",
        max_chars=player_search.MAX_QUERY_CHARS,
        label_visibility="collapsed",
    )
    show_all = toggle.toggle("Show every projected player", key="show_all")
    view = sheet if show_all else sheet[sheet["rank"] <= POOL[position]]
    if search.strip():
        found = [m.player.player_id for m in player_search.search(as_players(sheet), search)]
        view = sheet.set_index("player_id").loc[found].reset_index() if found else sheet.head(0)
    if bool(week_info["upcoming"]) and (sheet["injury_status"] == "None").all():
        st.info(
            "No injury designations yet this week. They come with the final injury report "
            "(Friday for Sunday games) and show up after the Saturday update."
        )
    # Name, projection and call first, so they fit on a phone without scrolling the table.
    table = pd.DataFrame(
        {
            "Tier": view["tier"],
            "Rank": view["rank"],
            "Player": view["player_name"],
            "Proj": view["proj"].round(1),
            "Start / Sit": view["advice"],
            "Range (10th-90th)": [
                fmt_range(lo, hi) for lo, hi in zip(view["floor"], view["ceiling"], strict=True)
            ],
            "Injury": view["injury_status"].replace({"None": ""}),
            "Team": view["team"],
            "Matchup": view["matchup"],
        }
    )
    if view["ecr_rank"].notna().any():
        table["Expert rank"] = view["ecr_rank"]
    if view["actual"].notna().any():
        table["Actual"] = view["actual"].round(1)
    box_score(
        table,
        {
            "Tier": st.column_config.NumberColumn(format="%d", width=52),
            "Rank": st.column_config.NumberColumn(format="%d", width=56),
            "Player": st.column_config.TextColumn(width=150),
            "Proj": st.column_config.NumberColumn(format="%.1f", width=64),
            "Expert rank": st.column_config.NumberColumn(format="%d"),
            "Actual": st.column_config.NumberColumn(format="%.1f"),
        },
        styles=sheet_styles,
    )

    chart_rows = sheet.head(CHART_PLAYERS).assign(
        label=lambda d: d["rank"].astype(str) + ". " + d["player_name"]
    )
    if not chart_rows.empty:
        st.subheader(f"Top {len(chart_rows)} {position}s: projection and range", anchor=False)
        base = alt.Chart(chart_rows).encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
            # Tiers alternate green and gray, like the shaded bands in the table.
            color=alt.Color("tier:O", title="Tier", scale=alt.Scale(range=[GREEN, GRAY])),
            tooltip=["player_name", "matchup", alt.Tooltip("proj:Q", format=".1f"), "tier"],
        )
        ranges = base.mark_rule(strokeWidth=3, opacity=0.45).encode(
            x=alt.X("floor:Q", title="Fantasy points (10th to 90th percentile range)"),
            x2="ceiling:Q",
        )
        points = base.mark_circle(size=70, opacity=1).encode(x="proj:Q")
        show_chart((ranges + points).properties(height=22 * len(chart_rows)))


def render_player_card(player: pd.Series) -> None:
    """One player: this week's projection and the season so far."""
    this_week = everyone[everyone["player_id"] == player["player_id"]]
    with st.container(key="player_card"):
        meta = f"{html_text(player['position'])} · {html_text(player['team'])}"
        tags = "" if this_week.empty else player_tags(this_week.iloc[0])
        st.markdown(  # raw HTML for the card's styling, so every value is HTML-escaped
            f'<p class="gl-card-name">{html_text(player["player_name"])}</p>'
            f'<div class="gl-tagline">{meta}</div>{tags}',
            unsafe_allow_html=True,
        )
        if this_week.empty:
            st.caption(f"Not projected for week {week} (bye week or not expected to play).")
        else:
            row = this_week.iloc[0]
            cells = st.columns(4)
            cells[0].metric(f"Week {week} projection", f"{row['proj']:.1f}")
            cells[1].metric("Range (10th-90th)", fmt_range(row["floor"], row["ceiling"]))
            cells[2].metric("Matchup", plain(row["matchup"]))
            cells[3].metric(
                "Team total",
                f"{row['implied_points']:.1f}",
                help="Points his team is expected to score, from the betting lines",
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
    st.subheader(f"{season} so far", anchor=False)
    st.caption(
        f"{len(games)} games, {games['fantasy_points_ppr'].mean():.1f} PPR points per game, "
        f"best {games['fantasy_points_ppr'].max():.1f}."
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
    series = ["Actual", "Projected before kickoff"]
    show_chart(
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("points:Q", title="PPR points"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=series, range=[GREEN, GRAY]),
                legend=KEY,
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                title=None,
                scale=alt.Scale(domain=series, range=[[1, 0], [5, 3]]),
                legend=KEY,
            ),
        )
        .properties(height=260)
    )


def use_suggestion(name: str) -> None:
    st.session_state.index_query = name


def player_index_section() -> None:
    section_title(
        "Player index",
        "Search by name, last name first, initials, nickname or a close spelling, or browse by "
        "last name. Team and position words narrow the list.",
    )
    roster = season_players(season)
    universe = as_players(roster)
    relevance = dict(zip(everyone["player_id"].astype(str), everyone["proj"], strict=True))
    query = st.text_input(
        "Search the player index",
        key="index_query",
        placeholder="Try: cmc, jsn, mccaffery, jefferson justin, lions wr, chiefs rb",
        max_chars=player_search.MAX_QUERY_CHARS,
        label_visibility="collapsed",
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
        left, right = st.columns([3, 1], vertical_alignment="bottom")
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
            f"Week {week}": listing["player_id"].map(projections["proj"]).round(1),
            "Rank": listing["player_id"].map(projections["rank"]).astype("Int64"),
        }
    )
    if "Found by" in listing:
        table["Found by"] = listing["Found by"]
    if not table.empty:
        st.caption(f"{len(table)} player{'s' if len(table) != 1 else ''}. Select a row to open it.")
        index_event = box_score(
            table,
            {
                f"Week {week}": st.column_config.NumberColumn(
                    format="%.1f", help=f"Week {week} projection; blank if not projected"
                ),
                "Rank": st.column_config.NumberColumn(format="%d", help="Position rank"),
            },
            on_select="rerun",
            selection_mode="single-row",
            key=f"index_table_{query.strip().lower()}_{len(table)}",
        )
        rows = index_event.selection.rows
        if rows:
            render_player_card(listing.iloc[rows[0]])
        elif clear_winner:
            render_player_card(listing.iloc[0])


def compare_card(row: dict[Hashable, Any], verdict: str) -> str:
    """One player in the comparison, as raw HTML; ``verdict`` is "Start", "Sit" or ""."""
    pick = " gl-compare-pick" if verdict == "Start" else ""
    return (
        f'<div class="gl-compare{pick}">{advice_tag(verdict) if verdict else ""}'
        f'<p class="gl-compare-name">{html_text(row["player_name"])}</p>'
        f'<p class="gl-compare-proj">{row["proj"]:.1f}<span>projected points</span></p>'
        f"{player_tags(row, advice=False)}"
        f'<p class="gl-compare-line">{html_text(row["matchup"])}, team total '
        f"{row['implied_points']:.1f}</p>"
        f'<p class="gl-compare-line">Range {fmt_range(row["floor"], row["ceiling"])}</p></div>'
    )


def start_sit_section() -> None:
    section_title("Start / Sit", "Compare up to three players for one lineup spot.")
    labels = {
        row.player_id: f"{row.player_name} ({row.position}, {row.team})"
        for row in everyone.itertuples()
    }
    by_label = {label: pid for pid, label in labels.items()}
    # A typical flex decision as the opening example: two running backs ranked back to back.
    backs = everyone[everyone["position"] == "RB"].sort_values("proj", ascending=False)
    example = [labels[pid] for pid in backs["player_id"].iloc[11:13]]
    chosen_labels = st.multiselect(
        "Players to compare",
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
    if len(picked) > 1:
        best, second = picked.iloc[0], picked.iloc[1]
        margin = best["proj"] - second["proj"]
        overlap = min(best["ceiling"], second["ceiling"]) - max(best["floor"], second["floor"])
        width = best["ceiling"] - best["floor"]
        if overlap > 0.8 * width:
            why = ", but their ranges overlap almost entirely, so it is a close call."
        else:
            why = ", a close call." if margin < 1.0 else "."
        st.markdown(
            f'<div class="gl-verdict"><p class="gl-verdict-call"><span>Start</span> '
            f"{html_text(best['player_name'])}</p>"
            f'<p class="gl-verdict-why">Projected {margin:.1f} points ahead of '
            f"{html_text(second['player_name'])}{why}</p></div>",
            unsafe_allow_html=True,
        )
    columns = st.columns(len(picked))
    for index, (column, row) in enumerate(zip(columns, picked.to_dict("records"), strict=True)):
        verdict = "" if len(picked) == 1 else ("Start" if index == 0 else "Sit")
        column.markdown(compare_card(row, verdict), unsafe_allow_html=True)
    history = run(
        "select player_id, week, fantasy_points_ppr, proj_ppr from marts.mart_player_game_log "
        "where season = ? and week < ? order by week",
        (season, week),
    )
    recent = history[history["player_id"].isin(chosen)]
    if not recent.empty:
        recent = recent.assign(player=recent["player_id"].map(labels))
        st.subheader("Recent games, PPR points", anchor=False)
        show_chart(
            alt.Chart(recent)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("fantasy_points_ppr:Q", title="PPR points"),
                color=alt.Color(
                    "player:N",
                    title=None,
                    # The recommended player in green, the others in ink and gray.
                    scale=alt.Scale(
                        domain=[labels[pid] for pid in picked["player_id"]],
                        range=[GREEN, INK, GRAY],
                    ),
                    legend=KEY,
                ),
            )
            .properties(height=280)
        )


def risers_section() -> None:
    section_title(
        "Risers",
        "Players whose role is growing: at least 2.5 more expected PPR points per game over "
        "their last three games than before, and 5 or more now.",
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

    def change_styles(table: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=table.index, columns=table.columns)
        css["Player"] = "font-weight: 500"
        css["Change"] = [
            f"font-weight: 600; color: {GREEN if v > 0 else RED if v < 0 else INK}"
            for v in table["Change"]
        ]
        return css

    box_score(
        pd.DataFrame(
            {
                "Player": movers["player_name"],
                "Pos": movers["position"],
                "Team": movers["team"],
                "Change": movers["expected_ppr_change"].round(1),
                "xPPR last 3": movers["expected_ppr_recent"].round(1),
                "xPPR before": movers["expected_ppr_before"].round(1),
                "Snap share": (movers["snap_share_recent"] * 100).round(0),
                "Snap change": (movers["snap_share_change"] * 100).round(0),
                "Target share change": (movers["target_share_change"] * 100).round(1),
                "This week": movers["proj_ppr"].round(1),
                "Rank": movers["pos_rank_ppr"].astype("Int64"),
            }
        ),
        {
            "Change": st.column_config.NumberColumn(
                format="%+.1f", help="Change in expected PPR points per game"
            ),
            "xPPR last 3": st.column_config.NumberColumn(
                format="%.1f",
                help="Expected PPR points per game over the last three games: targets, "
                "carries and where on the field they came, valued in points (ffverse)",
            ),
            "xPPR before": st.column_config.NumberColumn(
                format="%.1f",
                help="Expected PPR points per game before that: earlier this season, or last "
                "season early on",
            ),
            "Snap share": st.column_config.NumberColumn(format="%d%%"),
            "Snap change": st.column_config.NumberColumn(format="%+d"),
            "Target share change": st.column_config.NumberColumn(format="%+.1f"),
            "This week": st.column_config.NumberColumn(
                format="%.1f", help="This week's PPR projection; blank if not projected"
            ),
            "Rank": st.column_config.NumberColumn(format="%d", help="Position rank this week"),
        },
        styles=change_styles,
    )
    if movers["proj_ppr"].isna().any():
        st.caption("Blank projection: not projected this week (bye week or not expected to play).")


def track_record_section() -> None:
    section_title(
        "Track record",
        "How the projections did against simple averages and expert rankings, scored only "
        "on games played after each projection was made.",
    )
    summary = run("select * from marts.mart_backtest_summary order by scope, position, method")
    scopes = sorted(summary["scope"].unique(), key=scope_order, reverse=True)
    scope = st.selectbox("Seasons", scopes, key="scope")
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
    record = record.reset_index().rename(columns={"position": "Position"})
    errors = ["Model MAE", "Last 3 MAE", "Season avg MAE"]
    rankings = ["Model rank corr.", "Expert (ECR) rank corr."]

    def best_styles(table: pd.DataFrame) -> pd.DataFrame:
        """The best figure in each row green: the lowest error and the highest correlation."""
        css = pd.DataFrame(f"color: {MUTED}", index=table.index, columns=table.columns)
        css[["Position", *errors, *rankings]] = f"color: {INK}"
        best = f"color: {GREEN}; font-weight: 600"
        css[errors] = css[errors].mask(table[errors].eq(table[errors].min(axis=1), axis=0), best)
        css[rankings] = css[rankings].mask(
            table[rankings].eq(table[rankings].max(axis=1), axis=0), best
        )
        return css

    error_help = "Average miss in PPR points per player-week; lower is better"
    rank_help = "Spearman rank correlation within each week, averaged; higher is better"
    box_score(
        record,
        {
            **{
                name: st.column_config.NumberColumn(format="%.2f", help=error_help)
                for name in errors
            },
            **{
                name: st.column_config.NumberColumn(format="%.2f", help=rank_help)
                for name in rankings
            },
            "Inside 10th-90th": st.column_config.NumberColumn(
                format="%.0f%%",
                help="Share of results inside the 10th-90th percentile range; about 80% is "
                "well calibrated",
            ),
            "Player-weeks": st.column_config.NumberColumn(format="%d"),
        },
        styles=best_styles,
    )
    st.caption(
        "Green marks the best figure in each row. Each season is projected by a model "
        "trained only on earlier seasons and scored on the players FantasyPros ranked in the "
        f"top {POOL_TEXT} that week. Expert rankings have no point values, so they are "
        "compared on ranking only."
    )

    weekly = run(
        "select season, week, position, method, kind, n, abs_error_sum, spearman "
        "from marts.mart_projection_scorecard order by season, week"
    )
    seasons = sorted(weekly["season"].unique(), reverse=True)
    if not seasons:
        return
    left, right = st.columns(2, vertical_alignment="bottom")
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
        colors, dashes, widths = zip(*(METHOD_STYLE[m] for m in methods), strict=True)
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
                strokeWidth=alt.StrokeWidth(
                    "method:N", scale=alt.Scale(domain=methods, range=widths), legend=None
                ),
            )
            .properties(height=280)
        )

    st.subheader("Average miss per player: lower is better", anchor=False)
    show_chart(method_chart(["Model", "Last 3 average", "Season average"], "mae", "PPR points"))
    st.subheader("Ranking accuracy: higher is better", anchor=False)
    show_chart(method_chart(["Model", "Expert rankings"], "spearman", "Rank correlation"))
    live_weeks = detail[detail["kind"] == "live"]["week"].unique()
    st.caption(
        f"Weeks with live projections in {record_season}: "
        + (", ".join(str(w) for w in sorted(live_weeks)) if len(live_weeks) else "none yet")
        + ". The other weeks come from the backtest."
    )


# Only the selected section runs.
SECTIONS: dict[str, Callable[[], None]] = {
    "Cheat sheet": cheat_sheet_section,
    "Player index": player_index_section,
    "Start / Sit": start_sit_section,
    "Risers": risers_section,
    "Track record": track_record_section,
}
section = nav.radio(
    "Section", list(SECTIONS), horizontal=True, key="section", label_visibility="collapsed"
)
SECTIONS[str(section)]()

st.markdown(
    f'<div class="gl-footer">Built by Rett Wilson · <a href="{REPO_URL}">Source code</a> · '
    "Data: nflverse (CC BY 4.0) and ffverse expected fantasy points. FantasyPros expert "
    "ranks are used only to grade the model.</div>",
    unsafe_allow_html=True,
)
