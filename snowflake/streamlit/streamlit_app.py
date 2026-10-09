"""Fantasy Football Cheat Sheet: weekly projections, tiers, a player index, start/sit and a
track record.

Runs as Streamlit in Snowflake (dbt marts in GRIDIRON), locally against a DuckDB build of
the same marts, or as a public copy that reads a static Parquet snapshot:

    make dbt-ci && make app      # fixture data
    make local && make app       # full local run
    make app-public              # the public snapshot
"""

from __future__ import annotations

import json
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
LABEL_COLOR = "#1f4e79"
REPO_URL = "https://github.com/RettWilson22/gridiron-lakehouse"
# Comparison lines: the model stands out, the baselines recede.
METHOD_COLORS = alt.Scale(
    domain=["Model", "Last 3 average", "Season average", "Expert rankings"],
    range=[LABEL_COLOR, "#9aa5b1", "#c8a24a", "#c0504d"],
)

# A plain, classic website look: paper background, serif headings, square corners, a navy
# masthead with a gold rule. System fonts only (Streamlit in Snowflake blocks web fonts).
STYLE = """
<style>
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg { color: #fdfaf2; }
footer { display: none; }
.stApp { background: #f6f3ec; }
[data-testid="stSidebar"] { background: #ebe5d8; border-right: 1px solid #cfc6b4; }
.block-container { padding-top: 0; max-width: 1180px; }
h1, h2, h3, h4 { font-family: Georgia, "Times New Roman", serif !important; color: #1d2a36; }

.gl-masthead { background: #1d2a36; color: #fdfaf2; margin: 0 -100vw 1.4rem;
  padding: 1.2rem 100vw 1.05rem; border-bottom: 5px solid #c9a24a; }
.gl-logo { display: flex; align-items: center; gap: 16px; }
.gl-icon { width: 50px; height: 50px; flex: none; }
.gl-brand { font-family: Georgia, "Times New Roman", serif; font-size: 2.5rem; font-weight: 700;
  line-height: 1; }
.gl-brand em { font-weight: 400; font-style: italic; color: #c9a24a; }
.gl-tagline { font-size: 0.76rem; color: #b9c4cf; margin-top: 0.45rem; text-transform: uppercase;
  letter-spacing: 1.6px; }

.st-key-section [role="radiogroup"] { gap: 4px; flex-wrap: wrap;
  border-bottom: 1px solid #cfc6b4; margin-bottom: 0.8rem; }
.st-key-section [role="radiogroup"] label { background: #ebe5d8; border: 1px solid #cfc6b4;
  border-bottom: none; padding: 0.45rem 1.05rem; margin: 0 0 -1px 0; cursor: pointer; }
.st-key-section [role="radiogroup"] label > div:first-child { display: none; }
.st-key-section [role="radiogroup"] label p { font-weight: 600; }
.st-key-section [role="radiogroup"] label:has(input:checked) { background: #f6f3ec;
  box-shadow: inset 0 3px 0 #1f4e79; }
.st-key-section [role="radiogroup"] label:has(input:checked) p { color: #1f4e79; }

/* The A-Z player index reads like the index of a book: plain letters, current one boxed. */
.st-key-index_letter [role="radiogroup"] { gap: 2px 6px; flex-wrap: wrap; }
.st-key-index_letter [role="radiogroup"] label { padding: 0.1rem 0.45rem; margin: 0;
  border: 1px solid transparent; cursor: pointer; }
.st-key-index_letter [role="radiogroup"] label > div:first-child { display: none; }
.st-key-index_letter [role="radiogroup"] label p { font-family: Georgia, "Times New Roman", serif;
  font-size: 1.05rem; color: #1a5ea8; }
.st-key-index_letter [role="radiogroup"] label:has(input:checked) { border-color: #1f4e79;
  background: #fdfaf2; }
.st-key-index_letter [role="radiogroup"] label:has(input:checked) p { color: #1d2a36;
  font-weight: 700; }

[data-testid="stMetric"] { background: #fdfaf2; border: 1px solid #cfc6b4; padding: 0.6rem 0.9rem; }
[data-testid="stMetricValue"] { font-family: Georgia, "Times New Roman", serif; }

.gl-card-name { font-family: Georgia, "Times New Roman", serif; font-size: 1.6rem;
  font-weight: 700; color: #1d2a36; margin: 0.6rem 0 0; }
.gl-card-meta { color: #5c5a52; font-size: 0.9rem; margin-bottom: 0.6rem; }
.gl-footer { margin-top: 3rem; padding: 1rem 0; border-top: 1px solid #cfc6b4;
  font-size: 0.82rem; color: #6b6458; }
html, body, .stApp, [data-testid="stMain"] { overflow-x: hidden; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .gl-masthead { padding-top: 0.85rem; padding-bottom: 0.8rem; margin-bottom: 1rem; }
  .gl-icon { width: 34px; height: 34px; }
  .gl-logo { gap: 10px; }
  .gl-brand { font-size: 1.6rem; white-space: nowrap; }
  .gl-tagline { font-size: 0.64rem; letter-spacing: 1px; }
  [data-testid="stMetric"] { padding: 0.35rem 0.7rem; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .st-key-section [role="radiogroup"] label { padding: 0.4rem 0.55rem; }
  .st-key-section [role="radiogroup"] label p { font-size: 0.82rem; }
}
</style>
"""
MASTHEAD = """
<div class="gl-masthead">
  <div class="gl-logo">
    <svg class="gl-icon" viewBox="0 0 48 48" fill="none" stroke="#c9a24a" stroke-width="2.4"
         stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <ellipse cx="24" cy="24" rx="19" ry="11" transform="rotate(-35 24 24)"/>
      <path d="M17.5 30.5l13-13"/>
      <path d="M20.5 24.5l3 3M23.5 21.5l3 3M26.5 18.5l3 3"/>
    </svg>
    <div>
      <div class="gl-brand">Gridiron <em>Lakehouse</em></div>
      <div class="gl-tagline">Weekly fantasy football projections</div>
    </div>
  </div>
</div>
"""

st.set_page_config(page_title="Gridiron Lakehouse | Fantasy Cheat Sheet", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(MASTHEAD, unsafe_allow_html=True)


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
    return f"{low:.1f} - {high:.1f}"


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
        "Find a player", key="search", placeholder="Name, initials or a close spelling"
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
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Proj": st.column_config.NumberColumn(format="%.1f"),
            "Actual": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    chart_rows = sheet.head(CHART_PLAYERS).assign(
        label=lambda d: d["rank"].astype(str) + ". " + d["player_name"]
    )
    if not chart_rows.empty:
        base = alt.Chart(chart_rows).encode(
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("tier:O", title="Tier", scale=alt.Scale(scheme="tableau10")),
            tooltip=["player_name", "matchup", alt.Tooltip("proj:Q", format=".1f"), "tier"],
        )
        ranges = base.mark_rule(strokeWidth=3, opacity=0.5).encode(
            x=alt.X("floor:Q", title="Fantasy points (10th to 90th percentile range)"),
            x2="ceiling:Q",
        )
        points = base.mark_circle(size=70).encode(x="proj:Q")
        st.altair_chart((ranges + points).properties(height=18 * len(chart_rows)), width="stretch")


# Player index ------------------------------------------------------------------------------


def render_player_card(player: pd.Series) -> None:
    """One player: this week's projection and the season so far."""
    st.markdown(
        f'<p class="gl-card-name">{player["player_name"]}</p>'
        f'<p class="gl-card-meta">{player["position"]}, {player["team"]}</p>',
        unsafe_allow_html=True,
    )
    this_week = everyone[everyone["player_id"] == player["player_id"]]
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
            f"{row['matchup']}, team total {row['implied_points']:.1f}"
            + (f". Injury report: {injury}" if injury != "None" else "")
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
    st.dataframe(
        log,
        hide_index=True,
        width="stretch",
        column_config={"Projected": st.column_config.NumberColumn(format="%.1f")},
    )
    trend = games.melt(
        id_vars="week",
        value_vars=["fantasy_points_ppr", "proj_ppr"],
        var_name="series",
        value_name="points",
    ).dropna()
    trend["series"] = trend["series"].map(
        {"fantasy_points_ppr": "Actual", "proj_ppr": "Projected before kickoff"}
    )
    st.altair_chart(
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("points:Q", title="PPR points"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(range=[LABEL_COLOR, "#c9a24a"]),
                legend=alt.Legend(orient="bottom"),
            ),
        )
        .properties(height=220),
        width="stretch",
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
    )
    st.caption(
        "Finds players by full or partial name, last name first, initials, common nicknames "
        "and close spellings. Team and position words narrow the list."
    )
    if query.strip():
        matches = player_search.search(universe, query, relevance)
        if not matches:
            hint = player_search.did_you_mean(universe, query)
            st.info(f'No players match "{query}".')
            if hint:
                st.button(f"Search for {hint}", on_click=use_suggestion, args=(hint,))
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
        index_event = st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=min(36 * len(table) + 38, 420),
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
            st.metric(str(row["player_name"]), f"{row['proj']:.1f}", help="Projected points")
            st.caption(
                f"{row['position']}{row['rank']} | tier {row['tier']} | {row['advice']}\n\n"
                f"{row['matchup']}, team total {row['implied_points']:.1f}\n\n"
                f"Range {fmt_range(row['floor'], row['ceiling'])}"
                + (f"\n\nInjury: {injury}" if injury != "None" else "")
            )
    if len(picked) > 1:
        best, second = picked.iloc[0], picked.iloc[1]
        margin = best["proj"] - second["proj"]
        overlap = min(best["ceiling"], second["ceiling"]) - max(best["floor"], second["floor"])
        width = best["ceiling"] - best["floor"]
        close = margin < 1.0 or overlap > 0.8 * width
        st.markdown(
            f"**Start {best['player_name']}**: projected {margin:.1f} points ahead"
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
        st.altair_chart(
            alt.Chart(recent)
            .mark_line(point=True)
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("fantasy_points_ppr:Q", title="PPR points"),
                color=alt.Color("player:N", title=None, legend=alt.Legend(orient="bottom")),
            )
            .properties(height=240),
            width="stretch",
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
    st.dataframe(
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
        hide_index=True,
        width="stretch",
        column_config={
            "xPPR last 3": st.column_config.NumberColumn(
                help="Expected PPR points per game over the last three games"
            ),
            "xPPR before": st.column_config.NumberColumn(
                help="Expected PPR points per game before that"
            ),
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
    st.dataframe(
        record.reset_index().rename(columns={"position": "Position"}),
        hide_index=True,
        width="stretch",
        column_config={
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
    mae_chart = (
        alt.Chart(detail[detail["method"] != "Expert rankings"])
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("mae:Q", title="Mean absolute error (PPR)"),
            color=alt.Color(
                "method:N", title=None, scale=METHOD_COLORS, legend=alt.Legend(orient="bottom")
            ),
        )
        .properties(height=260, title="Points: lower is better")
    )
    rank_chart = (
        alt.Chart(detail[detail["method"].isin(["Model", "Expert rankings"])])
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Week"),
            y=alt.Y("spearman:Q", title="Rank correlation"),
            color=alt.Color(
                "method:N", title=None, scale=METHOD_COLORS, legend=alt.Legend(orient="bottom")
            ),
        )
        .properties(height=260, title="Ranking: higher is better")
    )
    st.altair_chart(mae_chart, width="stretch")
    st.altair_chart(rank_chart, width="stretch")
    live_weeks = detail[detail["kind"] == "live"]["week"].unique()
    st.caption(
        f"Weeks scored with live (pre-game) projections in {record_season}: "
        + (", ".join(str(w) for w in sorted(live_weeks)) if len(live_weeks) else "none yet")
        + ". Other weeks come from the walk-forward backtest."
    )


# Section navigation that keeps its place when a widget reruns the script (built-in tabs
# jump back to the first tab on a rerun), drawn to look like classic tabs. Only the
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
