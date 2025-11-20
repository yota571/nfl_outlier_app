import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# Try to import your existing PrizePicks wrapper
try:
    from DFS_Wrapper import PrizePick
    HAS_DFS_WRAPPER = True
except Exception:
    HAS_DFS_WRAPPER = False


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NFL Outlier-Style App (PrizePicks)",
    layout="wide",
)


# =====================================================
# FALLBACK SAMPLE DATA (so the app still runs)
# =====================================================
def _fake_nfl_board() -> pd.DataFrame:
    """Used only if DFS_Wrapper / PrizePicks is not working yet."""
    now = datetime.utcnow()
    data = [
        {
            "player": "Patrick Mahomes",
            "team": "KC",
            "stat_type": "Pass Yds",
            "line": 278.5,
            "opp": "BUF",
            "game_time": now + timedelta(hours=4),
        },
        {
            "player": "Christian McCaffrey",
            "team": "SF",
            "stat_type": "Rush Yds",
            "line": 78.5,
            "opp": "SEA",
            "game_time": now + timedelta(hours=7),
        },
        {
            "player": "CeeDee Lamb",
            "team": "DAL",
            "stat_type": "Rec Yds",
            "line": 84.5,
            "opp": "PHI",
            "game_time": now + timedelta(hours=3),
        },
        {
            "player": "Josh Allen",
            "team": "BUF",
            "stat_type": "Rush+Rec Yds",
            "line": 44.5,
            "opp": "KC",
            "game_time": now + timedelta(hours=4),
        },
    ]
    return pd.DataFrame(data)


def _fake_nfl_logs(player_name: str, stat_type: str, games_back: int) -> pd.DataFrame:
    """
    Dummy historical logs so the Outlier logic is visible even with no real NFL feed yet.
    Replace this with real NFL data when ready.
    """
    dates = pd.date_range(end=datetime.today(), periods=games_back, freq="7D")
    # Just some fake values that trend upwards, tweak as desired
    if "pass" in stat_type.lower():
        base = 230
    elif "rush" in stat_type.lower():
        base = 60
    elif "rec" in stat_type.lower():
        base = 70
    else:
        base = 50
    values = [base + i * 5 for i in range(games_back)]
    return pd.DataFrame(
        {
            "GAME_DATE": dates,
            "STAT": values,
        }
    )


# =====================================================
# REAL (OR FALLBACK) PRIZEPICKS NFL BOARD
# =====================================================
@st.cache_data(show_spinner=False)
def get_prizepicks_nfl_board() -> pd.DataFrame:
    """
    MAIN ENTRY: Loads the NFL PrizePicks board using DFS_Wrapper if available.
    Mirrors how the NBA app uses PrizePick() + get_data().
    If anything fails, falls back to a fake board so the UI still works.
    """
    # ---------- ATTEMPT REAL DFS_WRAPPER CALL ----------
    if HAS_DFS_WRAPPER:
        try:
            pp = PrizePick()
            try:
                raw = pp.get_data(organize_data=False)
            except TypeError:
                # older signature fallback, same as your NBA app
                raw = pp.get_data(False)

            if not isinstance(raw, list):
                st.warning("PrizePicks data not in expected list format. Using fake NFL board.")
                return _fake_nfl_board()

            records = []

            for item in raw:
                if not isinstance(item, dict):
                    continue

                league = item.get("league", "")
                if "NFL" not in str(league).upper():
                    continue

                status = str(item.get("status", "")).lower()
                # keep only pre-game / similar
                if status and status not in ("pre_game", "pre-game", "pregame"):
                    continue

                stat_type = item.get("stat_type")
                if not stat_type:
                    continue

                # line value
                line_val = item.get("line_score")
                if line_val is None:
                    line_val = item.get("line_value")
                if line_val is None:
                    continue
                try:
                    line_val = float(line_val)
                except Exception:
                    continue

                player_name = item.get("player_name") or item.get("name")
                if not player_name:
                    continue

                team = item.get("team")
                opponent = item.get("opponent") or ""
                start_time = item.get("start_time") or item.get("game_date_time")

                records.append(
                    {
                        "player": player_name,
                        "team": team,
                        "stat_type": stat_type,
                        "line": line_val,
                        "opp": opponent,
                        "game_time": start_time,
                    }
                )

            if not records:
                st.warning("No NFL records found in PrizePicks data. Using fake board.")
                return _fake_nfl_board()

            df = pd.DataFrame(records)
            df["line"] = pd.to_numeric(df["line"], errors="coerce")
            df = df.dropna(subset=["line"])

            df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

            # Optional: sort by time
            df = df.sort_values("game_time", na_position="last")

            return df

        except Exception as e:
            st.error(f"Error loading NFL board from DFS_Wrapper / PrizePicks: {e}")
            st.info("Using a small fake NFL board so you can at least see the UI.")
            return _fake_nfl_board()

    # ---------- NO DFS_WRAPPER: ALWAYS FAKE ----------
    st.warning("DFS_Wrapper not found. Using a small fake NFL board.")
    return _fake_nfl_board()


@st.cache_data(show_spinner=False)
def get_nfl_player_gamelogs(player_name: str, stat_type: str, games_back: int) -> pd.DataFrame:
    """
    Replace this with real NFL data when you're ready.
    For now, it's wired to a fake series just to drive the Outlier logic.
    """
    # TODO: hook up to real NFL stats
    return _fake_nfl_logs(player_name, stat_type, games_back)


def compute_outlier_metrics(line: float, stat_series: pd.Series) -> dict:
    """
    Given a PrizePicks line and a series of historical stat values,
    compute basic Outlier metrics (Over %, avg, etc.).
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
        "All",
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

show_raw = st.sidebar.checkbox("Show raw board data (debug)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("NFL Outlier-style app • v0.1 (stubbed)")


# =====================================================
# MAIN LAYOUT
# =====================================================
st.title("NFL Outlier-Style App (PrizePicks)")

col_board, col_outliers = st.columns([1.1, 1.9])

# -----------------------------------------------------
# LEFT: PRIZEPICKS BOARD
# -----------------------------------------------------
with col_board:
    st.subheader("PrizePicks NFL Board")

    with st.spinner("Loading NFL board..."):
        board_df = get_prizepicks_nfl_board()

    if board_df.empty:
        st.warning("No NFL board data available (even fake). Check the code.")
    else:
        # Filter by prop family based on stat_type text
        if "stat_type" in board_df.columns and prop_family != "All":
            stype = board_df["stat_type"].astype(str)

            if prop_family == "Passing Yards":
                mask = stype.str.contains("pass", case=False) & stype.str.contains("yd", case=False)
            elif prop_family == "Rushing Yards":
                mask = stype.str.contains("rush", case=False) & stype.str.contains("yd", case=False)
            elif prop_family == "Receiving Yards":
                mask = stype.str.contains("rec", case=False) & stype.str.contains("yd", case=False)
            elif prop_family == "Receptions":
                mask = stype.str.contains("reception", case=False)
            elif prop_family == "Rushing Attempts":
                mask = stype.str.contains("rush", case=False) & stype.str.contains("att", case=False)
            elif prop_family == "Pass TD":
                mask = stype.str.contains("pass", case=False) & stype.str.contains("td", case=False)
            elif prop_family == "Rush+Rec Yards":
                mask = stype.str.contains("rush", case=False) & stype.str.contains("rec", case=False)
            elif prop_family == "Fantasy Score":
                mask = stype.str.contains("fantasy", case=False)
            else:
                mask = pd.Series([True] * len(board_df))

            board_df = board_df[mask]

        # Display cleaned columns
        display_cols = [c for c in ["player", "team", "stat_type", "line", "opp", "game_time"] if c in board_df.columns]
        if "game_time" in display_cols:
            board_df = board_df.sort_values("game_time", na_position="last")

        st.dataframe(
            board_df[display_cols],
            use_container_width=True,
        )

        if show_raw:
            st.write("Raw board dataframe (debug):")
            st.dataframe(board_df, use_container_width=True)


# -----------------------------------------------------
# RIGHT: OUTLIER VIEW
# -----------------------------------------------------
with col_outliers:
    st.subheader("Outlier View (Historical Hit Rates)")

    if board_df.empty:
        st.info("No board data to compute outliers on.")
    else:
        rows = []
        with st.spinner("Computing outlier metrics..."):
            for _, row in board_df.iterrows():
                player = str(row.get("player", "")).strip()
                stat_type = str(row.get("stat_type", "")).strip()

                try:
                    line = float(row.get("line", 0))
                except Exception:
                    continue

                if not player or not stat_type:
                    continue

                # Pull NFL logs (stubbed or real)
                gamelogs = get_nfl_player_gamelogs(
                    player_name=player,
                    stat_type=stat_type,
                    games_back=sample_size,
                )

                if "STAT" not in gamelogs.columns:
                    continue

                metrics = compute_outlier_metrics(line=line, stat_series=gamelogs["STAT"])

                if metrics["games"] == 0:
                    continue

                rows.append(
                    {
                        "Player": player,
                        "Team": row.get("team", None),
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
            st.warning(
                "No outlier rows built. This is expected while using the stubbed logs, "
                "but check that get_nfl_player_gamelogs() returns a 'STAT' column."
            )
        else:
            out_df = pd.DataFrame(rows)
            out_df = out_df[out_df["Over %"] >= min_over_pct]
            out_df = out_df.sort_values(["Over %", "Games"], ascending=False)

            st.dataframe(out_df.reset_index(drop=True), use_container_width=True)
