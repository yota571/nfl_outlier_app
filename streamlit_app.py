import time
from datetime import datetime

import pandas as pd
import streamlit as st

# You already use this in NBA
from DFS_Wrapper import PrizePick


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NFL Outlier-Style App (PrizePicks)",
    layout="wide",
)


# =====================================================
# BASIC HELPERS
# =====================================================
@st.cache_data(show_spinner=False)
def get_prizepicks_nfl_board() -> pd.DataFrame:
    """
    Use your existing DFS_Wrapper.PrizePick logic to pull the NFL board.
    This returns a cleaned DataFrame with at least:
    ['player', 'team', 'stat_type', 'line', 'opp', 'game_time', 'raw']
    """
    pp = PrizePick(site="PrizePicks", sport="nfl")  # adjust args to match your wrapper

    # You'll need to implement something like this in DFS_Wrapper:
    # board = pp.get_nfl_board()

    board = pp.get_board()  # <-- if your existing method auto-detects by sport
    df = pd.DataFrame(board)

    # Normalize some expected fields for the app
    # Adjust keys/columns to match your real board structure
    rename_map = {
        "name": "player",
        "team_abbr": "team",
        "projection_type": "stat_type",
        "line_score": "line",
        "opponent": "opp",
        "start_time": "game_time",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert times if available
    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def get_nfl_player_gamelogs(player_name: str, stat_type: str, games_back: int) -> pd.DataFrame:
    """
    TODO: Replace stub with real NFL data (e.g., nfl_data_py / your own DB).
    For now, this returns a fake DataFrame so the UI is wired correctly.
    """
    # Example fake data
    dates = pd.date_range(end=datetime.today(), periods=games_back, freq="7D")
    vals = pd.Series(range(10, 10 + games_back))  # dummy numbers

    df = pd.DataFrame(
        {
            "GAME_DATE": dates,
            "STAT": vals,
        }
    )
    return df


def compute_outlier_metrics(line: float, stat_series: pd.Series) -> dict:
    """
    Given a line and a series of historical stat values, compute some basic metrics.
    """
    if stat_series.empty:
        return {
            "games": 0,
            "over_hits": 0,
            "over_pct": 0.0,
            "avg": None,
            "median": None,
            "max": None,
            "min": None,
        }

    games = len(stat_series)
    over_hits = (stat_series > line).sum()
    over_pct = round(100 * over_hits / games, 1)
    return {
        "games": games,
        "over_hits": over_hits,
        "over_pct": over_pct,
        "avg": round(stat_series.mean(), 1),
        "median": round(stat_series.median(), 1),
        "max": stat_series.max(),
        "min": stat_series.min(),
    }


# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.title("NFL Outlier Controls")

sample_size = st.sidebar.slider("Games Back", min_value=3, max_value=17, value=8, step=1)

prop_family = st.sidebar.selectbox(
    "Prop Type",
    [
        "Passing Yards",
        "Rushing Yards",
        "Receiving Yards",
        "Receptions",
        "Rushing Attempts",
        "Pass TD",
        "Rush+Rec Yards",
        "Fantasy Score",
    ],
)

min_over_pct = st.sidebar.slider(
    "Minimum Over % (Filter)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
)

show_raw = st.sidebar.checkbox("Show raw PrizePicks JSON", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("NFL Outlier-style app • v1.0 (beta)")


# =====================================================
# MAIN APP LAYOUT
# =====================================================
st.title("NFL Outlier-Style App (PrizePicks)")

col_board, col_outliers = st.columns([1.1, 1.9])

with col_board:
    st.subheader("PrizePicks NFL Board")

    with st.spinner("Loading NFL board from PrizePicks..."):
        try:
            board_df = get_prizepicks_nfl_board()
        except Exception as e:
            st.error(f"Error loading NFL board: {e}")
            board_df = pd.DataFrame()

    if board_df.empty:
        st.warning("No NFL board data found. Check DFS_Wrapper / PrizePicks config.")
    else:
        # Optional: filter to just the prop family selected (depends on your stat_type naming)
        if "stat_type" in board_df.columns:
            if prop_family == "Passing Yards":
                mask = board_df["stat_type"].str.contains("Pass Yds", case=False, na=False)
            elif prop_family == "Rushing Yards":
                mask = board_df["stat_type"].str.contains("Rush Yds", case=False, na=False)
            elif prop_family == "Receiving Yards":
                mask = board_df["stat_type"].str.contains("Rec Yds", case=False, na=False)
            elif prop_family == "Receptions":
                mask = board_df["stat_type"].str.contains("Receptions", case=False, na=False)
            elif prop_family == "Rushing Attempts":
                mask = board_df["stat_type"].str.contains("Rush Att", case=False, na=False)
            elif prop_family == "Pass TD":
                mask = board_df["stat_type"].str.contains("Pass TD", case=False, na=False)
            elif prop_family == "Rush+Rec Yards":
                mask = board_df["stat_type"].str.contains("Rush.+Rec", case=False, na=False)
            elif prop_family == "Fantasy Score":
                mask = board_df["stat_type"].str.contains("Fantasy", case=False, na=False)
            else:
                mask = pd.Series([True] * len(board_df))
            board_df = board_df[mask]

        display_cols = [c for c in ["player", "team", "stat_type", "line", "opp", "game_time"] if c in board_df.columns]
        st.dataframe(board_df[display_cols].sort_values("game_time", na_position="last"))

        if show_raw:
            st.write("Raw board data:")
            st.json(board_df.to_dict(orient="records"))


with col_outliers:
    st.subheader("Outlier View (Historical Hit Rates)")

    if board_df.empty:
        st.info("Load the NFL board first to compute outliers.")
    else:
        rows = []
        with st.spinner("Computing outlier metrics (stubbed with fake NFL logs for now)..."):
            for _, row in board_df.iterrows():
                player = row.get("player")
                stat_type = row.get("stat_type")
                line = float(row.get("line", 0))

                # Get historical game logs for this player/stat
                gamelogs = get_nfl_player_gamelogs(player_name=player, stat_type=stat_type, games_back=sample_size)

                stat_series = gamelogs["STAT"] if "STAT" in gamelogs.columns else pd.Series(dtype=float)
                metrics = compute_outlier_metrics(line=line, stat_series=stat_series)

                if metrics["games"] == 0:
                    continue

                rows.append(
                    {
                        "Player": player,
                        "Team": row.get("team"),
                        "Prop": stat_type,
                        "Line": line,
                        "Games": metrics["games"],
                        "Over Hits": metrics["over_hits"],
                        "Over %": metrics["over_pct"],
                        "Avg": metrics["avg"],
                        "Median": metrics["median"],
                        "Max": metrics["max"],
                        "Min": metrics["min"],
                    }
                )

            if not rows:
                st.warning("No historical data found for any players (stub still in place).")
            else:
                out_df = pd.DataFrame(rows)
                out_df = out_df[out_df["Over %"] >= min_over_pct]
                out_df = out_df.sort_values("Over %", ascending=False)

                st.dataframe(out_df.reset_index(drop=True))
