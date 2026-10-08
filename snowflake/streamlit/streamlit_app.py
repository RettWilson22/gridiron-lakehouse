"""Fourth Down Explorer: NFL fourth-down decisions compared with an expected-points model.

Runs as a Streamlit in Snowflake app (reads the dbt marts in GRIDIRON) or locally against
a DuckDB build of the same marts:

    make dbt-local && streamlit run snowflake/streamlit/streamlit_app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from data_access import Source, connect

CALL_LABELS = {
    "agreed": "Agreed with model",
    "too_conservative": "Kicked, model said go",
    "too_aggressive": "Went for it, model said kick",
    "different_kick": "Different kick than model",
}
CALL_COLORS = ["#2f5d8a", "#c8553d", "#8a8a8a", "#d9a441"]
DECISION_LABELS = {"go": "Go for it", "punt": "Punt", "field_goal": "Field goal"}

st.set_page_config(page_title="Fourth Down Explorer", layout="wide")


@st.cache_resource
def source() -> Source:
    return connect()


@st.cache_data(ttl=600)
def run(sql: str, params: tuple[object, ...] = ()) -> pd.DataFrame:
    frame: pd.DataFrame = source().query(sql, list(params))
    return frame


def clock(seconds: float) -> str:
    """Game seconds remaining -> 'Q3 07:12'."""
    seconds = int(seconds)
    quarter = min(4, 4 - (seconds - 1) // 900) if seconds > 0 else 4
    in_quarter = seconds - (4 - quarter) * 900
    return f"Q{quarter} {in_quarter // 60:02d}:{in_quarter % 60:02d}"


def field_position(yardline_100: float) -> str:
    yl = int(yardline_100)
    if yl == 50:
        return "Midfield"
    return f"Opp {yl}" if yl < 50 else f"Own {100 - yl}"


# Sidebar -------------------------------------------------------------------------------
seasons = run("select distinct season from marts.mart_fourth_down_leaderboard order by season desc")
st.sidebar.header("Filters")
season = st.sidebar.selectbox("Season", seasons["season"].tolist())
teams = run(
    "select team, coach from marts.mart_fourth_down_leaderboard where season = ? order by team",
    (int(season),),
)
team = st.sidebar.selectbox(
    "Team",
    teams["team"].tolist(),
    format_func=lambda t: f"{t}  ({teams.set_index('team').at[t, 'coach']})",
)
neutral_only = st.sidebar.checkbox(
    "Neutral situations only",
    value=True,
    help="Win probability between 10% and 90% with at least five minutes left. "
    "The leaderboard always uses neutral situations.",
)
st.sidebar.caption(f"Data source: {source().name}")

st.title("Fourth Down Explorer")
st.caption(
    "Every NFL fourth down since 2015, compared with an expected-points model of going for "
    "it, punting and kicking. Built on Databricks and Snowflake."
)

# Team summary ---------------------------------------------------------------------------
summary = run(
    "select * from marts.mart_fourth_down_leaderboard where season = ? and team = ?",
    (int(season), team),
).iloc[0]
cols = st.columns(5)
cols[0].metric("Neutral fourth downs", int(summary["neutral_fourth_downs"]))
cols[1].metric("Model said go", int(summary["go_recommendations"]))
cols[2].metric(
    "Go rate when model said go",
    f"{summary['go_rate_when_recommended']:.0%}",
    f"{summary['go_rate_vs_league'] * 100:+.0f} pts vs league",
)
cols[3].metric("Aggressiveness rank", f"{int(summary['aggressiveness_rank'])} of {len(teams)}")
cols[4].metric(
    "EP left on the table",
    f"{summary['expected_points_lost']:.1f}",
    help="Sum over neutral fourth downs of (best option - chosen option), in expected points.",
)

decisions_tab, leaderboard_tab, trends_tab, calculator_tab = st.tabs(
    ["Team decisions", "Leaderboard", "Trends", "Situation calculator"]
)

# Team decisions -------------------------------------------------------------------------
with decisions_tab:
    plays = run(
        "select * from marts.mart_fourth_down_calls "
        "where season = ? and team = ? order by week, game_seconds_remaining desc",
        (int(season), team),
    )
    if neutral_only:
        plays = plays[plays["is_neutral_situation"]]
    plays = plays.assign(
        call=plays["call_quality"].map(CALL_LABELS),
        decision_label=plays["decision"].map(DECISION_LABELS),
        model_label=plays["recommendation"].map(DECISION_LABELS),
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Where the calls happened")
        chart = (
            alt.Chart(plays)
            .mark_circle(size=70, opacity=0.8)
            .encode(
                x=alt.X(
                    "yardline_100:Q",
                    title="Yards from opponent end zone",
                    scale=alt.Scale(domain=[100, 0]),
                ),
                y=alt.Y(
                    "ydstogo:Q", title="Yards to go", scale=alt.Scale(domain=[0, 20], clamp=True)
                ),
                color=alt.Color(
                    "call:N",
                    title="Call vs model",
                    scale=alt.Scale(domain=list(CALL_LABELS.values()), range=CALL_COLORS),
                ),
                tooltip=[
                    "week",
                    "opponent",
                    "ydstogo",
                    "yardline_100",
                    "decision_label",
                    "model_label",
                    alt.Tooltip("go_advantage:Q", format=".2f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.subheader("Calls by type")
        st.dataframe(
            plays.groupby("call").size().rename("plays").sort_values(ascending=False),
            use_container_width=True,
        )

    st.subheader("Every fourth down")
    table = pd.DataFrame(
        {
            "Week": plays["week"],
            "Opponent": plays["opponent"],
            "Clock": plays["game_seconds_remaining"].map(clock),
            "Situation": "4th and " + plays["ydstogo"].astype(int).astype(str),
            "Field position": plays["yardline_100"].map(field_position),
            "Score diff": plays["score_differential"].astype(int),
            "Decision": plays["decision_label"],
            "Model": plays["model_label"],
            "P(convert)": plays["p_convert"].round(2),
            "Go advantage (EP)": plays["go_advantage"].round(2),
            "Outcome": plays["outcome"],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

# Leaderboard ----------------------------------------------------------------------------
with leaderboard_tab:
    board = run(
        "select team, coach, neutral_fourth_downs, go_recommendations, "
        "go_rate_when_recommended, go_rate_vs_league, expected_points_lost, "
        "decision_quality_rank, aggressiveness_rank "
        "from marts.mart_fourth_down_leaderboard where season = ? "
        "order by aggressiveness_rank",
        (int(season),),
    )
    st.subheader(f"{season}: how often teams went for it when the model said go")
    bars = (
        alt.Chart(board)
        .mark_bar()
        .encode(
            x=alt.X(
                "go_rate_when_recommended:Q",
                title="Go rate when model said go",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("team:N", sort="-x", title=None),
            color=alt.condition(alt.datum.team == team, alt.value("#c8553d"), alt.value("#2f5d8a")),
            tooltip=[
                "team",
                "coach",
                alt.Tooltip("go_rate_when_recommended:Q", format=".0%"),
                "go_recommendations",
            ],
        )
        .properties(height=max(300, 18 * len(board)))
    )
    st.altair_chart(bars, use_container_width=True)
    most, least = st.columns(2)
    display_cols = {
        "team": "Team",
        "coach": "Coach",
        "go_rate_when_recommended": "Go rate when model said go",
        "expected_points_lost": "EP left on table",
    }
    most.markdown("**Most aggressive**")
    most.dataframe(
        board.head(5)[list(display_cols)].rename(columns=display_cols),
        hide_index=True,
        use_container_width=True,
    )
    least.markdown("**Least aggressive**")
    least.dataframe(
        board.tail(5).iloc[::-1][list(display_cols)].rename(columns=display_cols),
        hide_index=True,
        use_container_width=True,
    )

# Trends ---------------------------------------------------------------------------------
with trends_tab:
    trend = run(
        "select season, team, go_rate, go_rate_when_recommended from marts.mart_team_trends "
        "where team = ? order by season",
        (team,),
    )
    league = run(
        "select season, avg(go_rate) as go_rate, "
        "avg(go_rate_when_recommended) as go_rate_when_recommended "
        "from marts.mart_team_trends group by season order by season"
    ).assign(team="League average")
    both = pd.concat([trend, league]).melt(
        id_vars=["season", "team"], var_name="metric", value_name="rate"
    )
    both["metric"] = both["metric"].map(
        {"go_rate": "Overall go rate", "go_rate_when_recommended": "Go rate when model said go"}
    )
    st.subheader(f"{team} vs league, by season")
    lines = (
        alt.Chart(both.dropna())
        .mark_line(point=True)
        .encode(
            x=alt.X("season:O", title="Season"),
            y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("team:N", title=None, scale=alt.Scale(range=["#c8553d", "#8a8a8a"])),
            strokeDash=alt.StrokeDash("metric:N", title=None),
        )
        .properties(height=360)
    )
    st.altair_chart(lines, use_container_width=True)

# Calculator -----------------------------------------------------------------------------
with calculator_tab:
    st.subheader("What should the offense do?")
    st.caption("Evaluated by the same model, served as a Snowflake Python UDF when deployed.")
    c1, c2, c3, c4 = st.columns(4)
    ydstogo = c1.number_input("Yards to go", min_value=1, max_value=30, value=2)
    yardline = c2.slider("Yards from opponent end zone", min_value=1, max_value=99, value=38)
    score_diff = c3.number_input(
        "Score differential (offense)", min_value=-40, max_value=40, value=0
    )
    minutes = c4.slider("Minutes left in game", min_value=0, max_value=60, value=20)
    result = source().recommend(ydstogo, yardline, score_diff, minutes * 60)
    st.markdown(
        f"**Recommendation: {DECISION_LABELS[result['recommendation']]}** "
        f"(go advantage {result['go_advantage']:+.2f} expected points)"
    )
    values = pd.DataFrame(
        {
            "Option": ["Go for it", "Punt", "Field goal"],
            "Expected points": [result["ev_go"], result["ev_punt"], result["ev_field_goal"]],
        }
    )
    st.altair_chart(
        alt.Chart(values)
        .mark_bar()
        .encode(
            x=alt.X("Expected points:Q"),
            y=alt.Y("Option:N", sort=None, title=None),
            color=alt.value("#2f5d8a"),
        )
        .properties(height=160),
        use_container_width=True,
    )
    st.caption(
        f"P(convert) {result['p_convert']:.0%}, P(field goal good) {result['p_field_goal']:.0%}. "
        "Expected points ignore clock and score leverage; treat late-game output with care."
    )
