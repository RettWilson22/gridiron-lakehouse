"""Fantasy Football Cheat Sheet: weekly projections, tiers, start/sit and a track record.

Runs as Streamlit in Snowflake (dbt marts in GRIDIRON), locally against a DuckDB build of
the same marts, or as a public copy that reads a static Parquet snapshot:

    make dbt-ci && make app      # fixture data
    make local && make app       # full local run
    make app-public              # the public snapshot
"""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from data_access import ACTUAL_STATS, PROJECTED_STATS, Source, connect, load_helper

tiers = load_helper("tiers")

POSITIONS = ("QB", "RB", "WR", "TE")
FORMATS = {"PPR": "ppr", "Half PPR": "half", "Standard": "std", "Custom": "custom"}
# Players shown by default and tiered: two starters' worth per team in a 12-team league
# (the same pool the backtest scores).
POOL = {"QB": 24, "RB": 48, "WR": 72, "TE": 24}
CHART_PLAYERS = 24
LABEL_COLOR = "#1f4e79"
REPO_URL = "https://github.com/RettWilson22/gridiron-lakehouse"
# Comparison lines: the model stands out, the baselines recede.
METHOD_COLORS = alt.Scale(
    domain=["Model", "Last 3 average", "Season average", "Expert rankings"],
    range=[LABEL_COLOR, "#9aa5b1", "#c8a24a", "#c0504d"],
)

st.set_page_config(page_title="Gridiron Lakehouse | Fantasy Cheat Sheet", layout="wide")


@st.cache_resource
def source() -> Source:
    return connect()


@st.cache_data(ttl=600)
def run(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    frame: pd.DataFrame = source().query(sql, list(params))
    return frame


@st.cache_data(ttl=600)
def custom_points(
    table: str,
    *,
    stats: tuple[str, ...],
    prefix: str,
    week_key: tuple[int, int, str],
    scoring: str,
) -> pd.DataFrame:
    """Points under custom settings for one season, week and position (JSON settings)."""
    frame: pd.DataFrame = source().custom_points(
        table,
        stats=stats,
        prefix=prefix,
        where="season = ? and week = ? and position = ?",
        params=list(week_key),
        scoring=json.loads(scoring),
    )
    return frame


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
default_week = int(upcoming.iloc[0]) if not upcoming.empty else int(weeks["week"].max())
week_options = [int(w) for w in weeks["week"]]

st.sidebar.title("Gridiron Lakehouse")
week = st.sidebar.selectbox(
    "Week",
    week_options,
    index=week_options.index(default_week),
    format_func=lambda w: f"Week {w}" + (" (upcoming)" if w == default_week else ""),
)
format_label = st.sidebar.radio("Scoring", list(FORMATS), key="scoring_format")
fmt = FORMATS[str(format_label)]

scoring: dict[str, Any] = {}
if fmt == "custom":
    with st.sidebar.expander("League scoring", expanded=True):
        reception = st.number_input("Points per reception", 0.0, 2.0, 1.0, 0.25, key="rec")
        pass_td = st.number_input("Passing touchdown", 2.0, 8.0, 4.0, 1.0, key="pass_td")
        pass_yards = st.number_input("Passing yards per point", 10, 50, 25, 5, key="pass_yd")
        interception = st.number_input("Interception", -6.0, 0.0, -2.0, 1.0, key="int")
        yards = st.number_input("Rushing / receiving yards per point", 5, 20, 10, 1, key="yd")
        touchdown = st.number_input("Rushing / receiving touchdown", 4.0, 8.0, 6.0, 1.0, key="td")
        fumble = st.number_input("Fumble lost", -4.0, 0.0, -2.0, 1.0, key="fumble")
        premium = st.number_input("Tight end bonus per reception", 0.0, 1.5, 0.0, 0.25, key="te")
        bonus_100 = st.number_input("100-yard rushing or receiving bonus", 0.0, 6.0, 0.0, 1.0)
        bonus_300 = st.number_input("300-yard passing bonus", 0.0, 6.0, 0.0, 1.0)
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
st.sidebar.caption(f"Data: {source().description}")
st.sidebar.markdown(
    f"Built on Databricks and Snowflake. [How it works]({REPO_URL}#readme) · "
    f"[Source code]({REPO_URL})"
)


def week_sheet(position: str) -> pd.DataFrame:
    """The selected week's sheet for a position in the selected scoring format."""
    rows = run(
        "select * from marts.mart_cheat_sheet where season = ? and week = ? and position = ?",
        (season, week, position),
    )
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
        return out.sort_values("rank").reset_index(drop=True)

    settings = json.dumps(scoring, sort_keys=True)
    projected = custom_points(
        "marts.mart_cheat_sheet",
        stats=PROJECTED_STATS,
        prefix="proj_",
        week_key=(season, week, position),
        scoring=settings,
    )
    out = rows.merge(projected, on="player_id")
    ratio = (out["custom_points"] / out["proj_ppr"].where(out["proj_ppr"] > 0)).fillna(1.0)
    out = out.assign(
        proj=out["custom_points"],
        floor=(out["floor_ppr"] * ratio).clip(upper=out["custom_points"]),
        ceiling=(out["ceiling_ppr"] * ratio).clip(lower=out["custom_points"]),
    )
    out = out.sort_values("proj", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    pool = out["rank"] <= POOL[position]
    out["tier"] = pd.Series(dtype="Int64")
    out.loc[pool, "tier"] = tiers.assign_tiers(out.loc[pool, "proj"].tolist())
    out["advice"] = [tiers.start_sit(position, int(r)) for r in out["rank"]]
    actual = custom_points(
        "marts.mart_player_game_log",
        stats=ACTUAL_STATS,
        prefix="",
        week_key=(season, week, position),
        scoring=settings,
    ).rename(columns={"custom_points": "actual_custom"})
    out = out.merge(actual, on="player_id", how="left")
    out["actual"] = out["actual_custom"].where(out["actual_ppr"].notna())
    out.loc[out["actual_ppr"].notna() & out["actual"].isna(), "actual"] = 0.0
    return out


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
st.markdown(
    f'<p style="color:{LABEL_COLOR};font-weight:600;letter-spacing:0.08em;'
    'font-size:0.8rem;margin-bottom:0">GRIDIRON LAKEHOUSE</p>',
    unsafe_allow_html=True,
)
st.title("Fantasy Football Cheat Sheet")
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

headline = headline_numbers()
projected_count = run(
    "select count(*) as n from marts.mart_cheat_sheet where season = ? and week = ?",
    (season, week),
)["n"].iloc[0]
metric_columns = st.columns(3)
metric_columns[0].metric(f"Players projected, week {week}", f"{int(projected_count):,}")
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

sheet_tab, compare_tab, risers_tab, record_tab = st.tabs(
    ["Cheat sheet", "Start / Sit", "Risers", "Track record"]
)

# Cheat sheet -------------------------------------------------------------------------------

with sheet_tab:
    position = str(st.radio("Position", POSITIONS, horizontal=True, key="position"))
    sheet = week_sheet(position)
    search = st.text_input("Find a player", key="search", placeholder="Name")
    show_all = st.toggle("Show every projected player", key="show_all")
    view = sheet if show_all else sheet[sheet["rank"] <= POOL[position]]
    if search:
        view = sheet[sheet["player_name"].str.contains(search, case=False, na=False)]
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

# Start / Sit -----------------------------------------------------------------------------

with compare_tab:
    everyone = pd.concat([week_sheet(p) for p in POSITIONS], ignore_index=True)
    everyone = everyone.sort_values("proj", ascending=False)
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
    else:
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

with risers_tab:
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
    else:
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
            st.caption(
                "Blank projection: not projected this week (bye week or not expected to play)."
            )

# Track record ----------------------------------------------------------------------------

with record_tab:
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
        "ranked in the top 24 QBs, 48 RBs, 72 WRs and 24 TEs that week. MAE is in PPR "
        "points (lower is better); rank correlation is Spearman's, within each week "
        "(higher is better). Expert rankings have no point values, so they are only "
        "compared on ranking. A calibrated 80% range should contain about 80% of outcomes."
    )

    weekly = run(
        "select season, week, position, method, kind, n, abs_error_sum, spearman "
        "from marts.mart_projection_scorecard order by season, week"
    )
    seasons = sorted(weekly["season"].unique(), reverse=True)
    if seasons:
        left, right = st.columns(2)
        record_season = left.selectbox("Season", seasons, key="record_season")
        record_position = right.radio("Position", POSITIONS, horizontal=True, key="record_position")
        detail = weekly[
            (weekly["season"] == record_season) & (weekly["position"] == record_position)
        ]
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
