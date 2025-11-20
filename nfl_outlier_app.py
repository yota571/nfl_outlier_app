import time
from datetime import datetime

import pandas as pd
import streamlit as st

from DFS_Wrapper import PrizePick

# Try to use real NFL data
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA = True
except Exception:
    HAS_NFL_DATA = False


# =====================================================
# BASIC HELPERS
# =====================================================
def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    for ch in [".", ",", "'", "`"]:
        name = name.replace(ch, "")
    name = " ".join(name.split())
    return name


# =====================================================
# NFL PRIZEPICKS LOADER (FULL-GAME MARKETS ONLY)
# =====================================================
def _map_nfl_market(stat_type_raw: str) -> str | None:
    """
    Map raw PrizePicks NFL stat_type text -> clean full-game markets,
    or None if we want to ignore the prop.
    """
    if not stat_type_raw:
        return None

    s = str(stat_type_raw).lower()

    # Filter out halves, first x attempts/receptions, combos, etc.
    bad_fragments = [
        "first 5",
        "first five",
        "first 2",
        "first two",
        "first 3",
        "first 4",
        "first reception",
        "in first",
        "combo",
        "fg made",
        "field goals made",
        "kicking",
        "extra point",
    ]
    if any(b in s for b in bad_fragments):
        return None

    # Fantasy
    if "fantasy" in s:
        return "fs"  # Fantasy Score

    # TDs
    if "pass" in s and "td" in s:
        return "pass_td"

    # Passing yards
    if "pass" in s and ("yd" in s or "yard" in s):
        return "pass_yds"

    # Rush + Rec yards
    if "rush" in s and "rec" in s and ("yd" in s or "yard" in s):
        return "rush_rec_yds"

    # Rushing attempts
    if "rush" in s and "attempt" in s:
        return "rush_att"

    # Rushing yards
    if "rush" in s and ("yd" in s or "yard" in s):
        return "rush_yds"

    # Receiving yards
    if "rec" in s and "yard" in s:
        return "rec_yds"

    # Receptions
    if "reception" in s:
        return "receptions"

    # Ignore everything else (defensive props, etc.)
    return None


@st.cache_data
def load_prizepicks_nfl_props() -> pd.DataFrame:
    """
    Pull cleaned NFL props from PrizePicks, similar to your NBA loader.

    - NFL only
    - pre-game only
    - full-game style markets only
    - one row per (player, team, opponent, market) with last line
    """
    try:
        pp = PrizePick()
        try:
            raw = pp.get_data(organize_data=False)
        except TypeError:
            raw = pp.get_data(False)
    except Exception as e:
        st.error(f"Error loading PrizePicks data: {e}")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    if not isinstance(raw, list):
        st.warning("PrizePicks data not in expected list format.")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    records = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        # league filter
        league = item.get("league", "")
        if "NFL" not in str(league).upper():
            continue

        # status filter
        status = str(item.get("status", "")).lower()
        if status and status not in ("pre_game", "pre-game", "pregame"):
            continue

        raw_stat_type = item.get("stat_type")
        market_code = _map_nfl_market(raw_stat_type)
        if market_code is None:
            continue

        # line
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

        # skip multi-player combos
        if "+" in str(player_name):
            continue

        team = item.get("team")
        opponent = item.get("opponent") or ""

        # skip split opponents
        if any(sep in opponent for sep in ("/", "|", "+")):
            continue

        start_time = item.get("start_time") or item.get("game_date_time")

        records.append(
            {
                "player_name": player_name,
                "team": team,
                "opponent": opponent,
                "market": market_code,
                "line": line_val,
                "game_time": start_time,
                "book": "PrizePicks",
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    df = pd.DataFrame(records)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    # one line per player/team/opponent/market – keep last (latest line)
    df = df.sort_values("game_time")
    df = df.groupby(
        ["player_name", "team", "opponent", "market"],
        as_index=False,
    ).last()

    return df[["player_name", "team", "opponent", "market", "line", "game_time", "book"]]


# Nice display names for NFL markets
NFL_MARKET_LABELS = {
    "pass_yds": "Passing Yards",
    "rush_yds": "Rushing Yards",
    "rec_yds": "Receiving Yards",
    "receptions": "Receptions",
    "rush_att": "Rushing Attempts",
    "pass_td": "Pass TD",
    "rush_rec_yds": "Rush+Rec Yards",
    "fs": "Fantasy Score",
}


# =====================================================
# REAL NFL STATS VIA nfl_data_py (WITH STUB FALLBACK)
# =====================================================
@st.cache_data
def get_ids_table() -> pd.DataFrame:
    """ID table for name + ESPN headshot IDs."""
    if not HAS_NFL_DATA:
        return pd.DataFrame(columns=["gsis_id", "name", "espn_id"])

    try:
        ids = nfl.import_ids()
        keep_cols = [c for c in ids.columns if c in ("gsis_id", "name", "espn_id")]
        ids = ids[keep_cols].dropna(subset=["name"])
        return ids
    except Exception:
        return pd.DataFrame(columns=["gsis_id", "name", "espn_id"])


@st.cache_data
def get_weekly_table() -> pd.DataFrame:
    if not HAS_NFL_DATA:
        return pd.DataFrame()

    try:
        years = list(range(datetime.today().year - 2, datetime.today().year + 1))
        weekly = nfl.import_weekly_data(years)
        return weekly
    except Exception:
        return pd.DataFrame()


def _match_player_record(player_name: str, ids_df: pd.DataFrame) -> dict | None:
    if ids_df.empty or not player_name:
        return None

    target = normalize_name(player_name)

    # exact match
    for _, r in ids_df.iterrows():
        if normalize_name(r["name"]) == target:
            return r.to_dict()

    # contains
    for _, r in ids_df.iterrows():
        norm = normalize_name(r["name"])
        if target and (target in norm or norm in target):
            return r.to_dict()

    # last name unique
    parts = target.split()
    if len(parts) >= 2:
        last = parts[-1]
        candidates = []
        for _, r in ids_df.iterrows():
            norm = normalize_name(r["name"])
            if last and last in norm:
                candidates.append(r.to_dict())
        if len(candidates) == 1:
            return candidates[0]

    return None


def _fake_nfl_logs(player_name: str, games_back: int) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=games_back, freq="7D")
    base_seed = (sum(ord(c) for c in player_name) % 50)

    df = pd.DataFrame({"GAME_DATE": dates})
    df["passing_yards"] = base_seed + 180 + (pd.Series(range(games_back)) * 5)
    df["rushing_yards"] = base_seed + 40 + (pd.Series(range(games_back)) * 2)
    df["receiving_yards"] = base_seed + 45 + (pd.Series(range(games_back)) * 3)
    df["receptions"] = 3 + (pd.Series(range(games_back)) % 6)
    df["carries"] = 8 + (pd.Series(range(games_back)) % 10)
    df["passing_tds"] = (pd.Series(range(games_back)) % 4) * 0.7
    df["fantasy_points_ppr"] = (
        df["passing_yards"] / 25.0
        + (df["rushing_yards"] + df["receiving_yards"]) / 10.0
        + df["passing_tds"] * 4
    )

    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    return df


@st.cache_data
def get_player_gamelog(player_name: str, gsis_id: str | None) -> pd.DataFrame:
    games = 60

    if not HAS_NFL_DATA or gsis_id is None:
        return _fake_nfl_logs(player_name, games)

    weekly = get_weekly_table()
    if weekly.empty or "player_id" not in weekly.columns:
        return _fake_nfl_logs(player_name, games)

    df = weekly[weekly["player_id"] == gsis_id].copy()
    if df.empty:
        return _fake_nfl_logs(player_name, games)

    sort_cols = [c for c in ["season", "week"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    df["GAME_DATE"] = pd.to_datetime(df.get("gameday", datetime.today()), errors="coerce")

    for col in [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "carries",
        "passing_tds",
        "fantasy_points_ppr",
    ]:
        if col not in df.columns:
            df[col] = 0

    df = df.head(games).reset_index(drop=True)
    return df


def get_market_series(gamelog_df: pd.DataFrame, market: str) -> pd.Series:
    if gamelog_df.empty:
        return pd.Series(dtype="float")

    m = (market or "").lower().strip()

    if m == "pass_yds":
        return gamelog_df["passing_yards"]
    if m == "rush_yds":
        return gamelog_df["rushing_yards"]
    if m == "rec_yds":
        return gamelog_df["receiving_yards"]
    if m == "receptions":
        return gamelog_df["receptions"]
    if m == "rush_att":
        return gamelog_df["carries"]
    if m == "pass_td":
        return gamelog_df["passing_tds"]
    if m == "rush_rec_yds":
        return gamelog_df["rushing_yards"] + gamelog_df["receiving_yards"]
    if m == "fs":
        return gamelog_df["fantasy_points_ppr"]

    return pd.Series(dtype="float")


# =====================================================
# SIDEBAR UI
# =====================================================
st.set_page_config(page_title="NFL Outlier-Style App (PrizePicks)", layout="wide")

st.sidebar.title("NFL Outlier-Style App (PrizePicks, Cloud)")

mode = st.sidebar.radio("Prop source", ["PrizePicks (live)"])

games_to_look_back = st.sidebar.slider(
    "Games to look back (N)", min_value=5, max_value=25, value=10, step=1
)
min_over_rate = st.sidebar.slider(
    "Minimum Over % (last N games)", min_value=0.0, max_value=1.0, value=0.6, step=0.05
)
min_edge = st.sidebar.number_input("Minimum Edge (Avg - Line)", value=5.0, step=0.5)
min_confidence = st.sidebar.slider(
    "Minimum Confidence %", min_value=0, max_value=100, value=60, step=5
)
only_today = st.sidebar.checkbox("Only today's games (by game_time)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Markets: passing yards, rushing yards, receiving yards, receptions, "
    "rushing attempts, pass TD, rush+rec yards, fantasy score."
)


# =====================================================
# LOAD PROPS
# =====================================================
props_df = None
if mode == "PrizePicks (live)":
    props_df = load_prizepicks_nfl_props()

if props_df is None or props_df.empty:
    st.info("No NFL props loaded yet.")
    st.stop()

if only_today and "game_time" in props_df.columns:
    today = datetime.today().date()
    props_df = props_df[props_df["game_time"].dt.date == today]

if props_df.empty:
    st.warning("No props after applying date filter.")
    st.stop()

teams = ["All"] + sorted(props_df["team"].dropna().unique().tolist())
markets = ["All"] + sorted(props_df["market"].dropna().unique().tolist())

top_col1, top_col2, top_col3 = st.columns([2, 2, 3])
with top_col1:
    team_filter = st.selectbox("Team filter", teams)
with top_col2:
    market_filter = st.selectbox(
        "Market filter (code)",
        markets,
        help="Codes: pass_yds, rush_yds, rec_yds, receptions, rush_att, pass_td, rush_rec_yds, fs",
    )
with top_col3:
    search_name = st.text_input("Search player (optional)", "")

df = props_df.copy()
if team_filter != "All":
    df = df[df["team"] == team_filter]
if market_filter != "All":
    df = df[df["market"].str.lower() == market_filter.lower()]

if df.empty:
    st.warning("No props match the selected filters.")
    st.stop()


# =====================================================
# EDGE / CONFIDENCE / PREDICTION / BET SIDE
# =====================================================
st.title("NFL Prop Edge Finder (PrizePicks, Cloud Version)")
st.caption(
    "PrizePicks full-game NFL props vs nfl_data_py stats → edges, confidence %, & prediction. "
    "(Falls back to stubbed stats if nfl_data_py is not installed.)"
)

st.write("### Calculating edges…")

rows = []
errors = []

unique_players = sorted(df["player_name"].dropna().unique().tolist())
player_logs: dict[str, pd.DataFrame] = {}
player_ids: dict[str, dict] = {}

ids_df = get_ids_table()

progress = st.progress(0.0)
status_text = st.empty()
total_players = len(unique_players) if unique_players else 1

for i, name in enumerate(unique_players):
    status_text.text(f"Building NFL stats for players… ({i+1}/{total_players})")

    rec = _match_player_record(name, ids_df) if not ids_df.empty else None
    gsis_id = rec.get("gsis_id") if rec else None
    espn_id = rec.get("espn_id") if rec else None

    player_ids[name] = {"gsis_id": gsis_id, "espn_id": espn_id}

    glog = get_player_gamelog(name, gsis_id)
    if glog.empty:
        errors.append(f"No game log for player: {name}")
        continue

    player_logs[name] = glog

    time.sleep(0.05)
    progress.progress((i + 1) / total_players)

progress.progress(1.0)
status_text.text("Computing edges, predictions, confidence, bet side…")

for _, row in df.iterrows():
    player_name = row.get("player_name")
    market = row.get("market")
    line = row.get("line")
    book = row.get("book") or "PrizePicks"

    if player_name is None or market is None or line is None:
        continue
    if player_name not in player_logs:
        continue

    try:
        line_float = float(line)
    except Exception:
        errors.append(f"Invalid line for {player_name}: {line}")
        continue

    gamelog = player_logs[player_name]
    series = get_market_series(gamelog, market).dropna()
    if series.empty:
        errors.append(f"No stats for {player_name} – market '{market}'")
        continue

    season_avg = series.mean()
    last_n = series.iloc[:games_to_look_back]
    if last_n.empty:
        errors.append(f"No recent games for {player_name}")
        continue

    avg_last_n = last_n.mean()
    over_rate_n = (last_n > line_float).mean()
    edge_n = avg_last_n - line_float

    last7 = series.iloc[:7]
    if len(last7) > 0:
        avg_last_7 = last7.mean()
        over_rate_7 = (last7 > line_float).mean()
        edge_7 = avg_last_7 - line_float
    else:
        avg_last_7 = None
        over_rate_7 = None
        edge_7 = None

    w_season = 0.4
    w_last_n = 0.4
    w_last_7 = 0.2
    avg7_for_blend = avg_last_7 if avg_last_7 is not None else avg_last_n
    predicted_score = (
        w_season * season_avg
        + w_last_n * avg_last_n
        + w_last_7 * avg7_for_blend
    )

    hit_score = over_rate_n
    if line_float != 0:
        edge_ratio = max(0.0, edge_n / max(1.0, line_float))
    else:
        edge_ratio = 0.0
    edge_score = max(0.0, min(1.0, edge_ratio * 4.0))

    confidence = 0.6 * hit_score + 0.4 * edge_score
    confidence_pct = round(confidence * 100, 1)

    delta = predicted_score - line_float
    if delta >= 0.5:
        bet_side = "Over"
    elif delta <= -0.5:
        bet_side = "Under"
    else:
        bet_side = "No clear edge"

    pid_info = player_ids.get(player_name, {})

    rows.append(
        {
            "player": player_name,
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": market,
            "line": line_float,
            "book": book,
            "bet_side": bet_side,
            "season_avg": round(season_avg, 2),
            f"avg_last_{games_to_look_back}": round(avg_last_n, 2),
            f"over_rate_last_{games_to_look_back}": round(over_rate_n, 2),
            f"edge_last_{games_to_look_back}": round(edge_n, 2),
            "avg_last_7": round(avg_last_7, 2) if avg_last_7 is not None else None,
            "over_rate_last_7": round(over_rate_7, 2) if over_rate_7 is not None else None,
            "edge_last_7": round(edge_7, 2) if edge_7 is not None else None,
            "predicted_score": round(predicted_score, 2),
            "confidence_pct": confidence_pct,
            "game_time": row.get("game_time"),
            "espn_id": pid_info.get("espn_id"),
        }
    )

if not rows:
    st.error("No edges could be calculated from the current props.")
    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            for e in errors:
                st.write("- ", e)
    st.stop()

edges_df = pd.DataFrame(rows)
edge_cols = [c for c in edges_df.columns if c.startswith("edge_last_") and "7" not in c]
rate_cols = [c for c in edges_df.columns if c.startswith("over_rate_last_") and "7" not in c]
edge_col = edge_cols[0]
rate_col = rate_cols[0]

# apply player search
if search_name.strip():
    view_df = edges_df[edges_df["player"].str.contains(search_name, case=False, na=False)]
else:
    view_df = edges_df.copy()

if view_df.empty:
    st.warning("No props after applying search filter.")
    st.stop()


# =====================================================
# TABS
# =====================================================
tab_cards, tab_table, tab_player, tab_whatif = st.tabs(
    ["Cards & Explanation", "Table", "Player Detail", "What-If Line Tester"]
)


# =====================================================
# TAB 1: CARDS + EXPLANATION
# =====================================================
with tab_cards:
    st.write("### Featured Edges (Card View)")

    filtered_edges = view_df[
        (view_df[rate_col] >= min_over_rate)
        & (view_df[edge_col] >= min_edge)
        & (view_df["confidence_pct"] >= min_confidence)
    ]

    if filtered_edges.empty:
        featured_df = view_df.copy()
        st.caption("No props match all filters yet – showing best available edges instead.")
    else:
        featured_df = filtered_edges.copy()

    featured_df = featured_df.sort_values(
        by=["confidence_pct", rate_col, edge_col], ascending=False
    ).reset_index(drop=True)

    top_n = min(12, len(featured_df))
    if top_n == 0:
        st.info("No edges available to display.")
    else:
        for idx in range(0, top_n, 2):
            cols = st.columns(2)
            for j in range(2):
                k = idx + j
                if k >= top_n:
                    break
                r = featured_df.iloc[k]
                with cols[j]:
                    card = st.container()
                    with card:
                        top_col1, top_col2 = st.columns([1, 2])

                        with top_col1:
                            espn_id = r.get("espn_id")
                            if pd.notna(espn_id) and espn_id:
                                try:
                                    img_url = f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png"
                                    st.image(img_url, use_column_width=True)
                                except Exception:
                                    st.write(" ")
                            else:
                                st.write(" ")

                        with top_col2:
                            display_market = NFL_MARKET_LABELS.get(r["market"], r["market"])
                            st.markdown(f"### {r['player']}")
                            team = r.get("team") or ""
                            opp = r.get("opponent") or ""
                            st.markdown(f"**{team} vs {opp}**")
                            st.markdown(
                                f"*{display_market}* &nbsp; | &nbsp; **Line:** `{r['line']}`"
                            )

                            side = r.get("bet_side", "No clear edge")
                            side_emoji = (
                                "⬆️" if side == "Over"
                                else "⬇️" if side == "Under"
                                else "⚖️"
                            )
                            st.markdown(f"**Recommended:** {side_emoji} **{side}**")

                            st.markdown(
                                f"Predicted: **{r['predicted_score']}**  "
                                f"(Season avg: {r['season_avg']})"
                            )

                        conf = r.get("confidence_pct", 0) or 0
                        edge_val = r.get(edge_col, 0) or 0
                        hit = r.get(rate_col, 0) or 0

                        st.markdown(
                            f"Edge: `{edge_val:.2f}` &nbsp; | &nbsp; "
                            f"Hit rate (N): `{hit:.2f}` &nbsp; | &nbsp; "
                            f"Confidence: `{conf:.1f}%`"
                        )
                        st.progress(min(max(conf / 100.0, 0.0), 1.0))

                        if st.button(
                            "Why this prop?",
                            key=f"why_{r['player']}_{r['market']}_{k}",
                        ):
                            st.session_state["explain_row_nfl"] = r.to_dict()

    st.write("### Prop Explanation (Why this prediction & confidence)")

    if "explain_row_nfl" in st.session_state:
        er = st.session_state["explain_row_nfl"]

        player_name = er.get("player")
        market = er.get("market")
        line = er.get("line")

        display_market = NFL_MARKET_LABELS.get(market, market)

        st.markdown(
            f"**Player:** {player_name}  \n"
            f"**Market:** {display_market}  \n"
            f"**Line:** `{line}`"
        )

        mask = (
            (edges_df["player"] == player_name)
            & (edges_df["market"] == market)
            & (edges_df["line"] == line)
        )
        row_matches = edges_df[mask]
        if row_matches.empty:
            st.warning("Could not find full data for this prop in the edges table.")
        else:
            r = row_matches.iloc[0]

            season_avg = r.get("season_avg")
            avg_last_n = r.get(f"avg_last_{games_to_look_back}")
            over_rate_n = r.get(f"over_rate_last_{games_to_look_back}")
            edge_n = r.get(f"edge_last_{games_to_look_back}")
            avg_last_7 = r.get("avg_last_7")
            over_rate_7 = r.get("over_rate_last_7")
            edge_7 = r.get("edge_last_7")
            predicted_score = r.get("predicted_score")
            confidence_pct = r.get("confidence_pct")
            bet_side = r.get("bet_side")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Key Numbers**")
                st.write(f"- Season average: **{season_avg}**")
                st.write(f"- Last {games_to_look_back} avg: **{avg_last_n}**")
                st.write(f"- Hit rate last {games_to_look_back}: **{over_rate_n:.2f}**")
                st.write(f"- Edge last {games_to_look_back}: **{edge_n:.2f}** vs line **{line}**")
                if avg_last_7 is not None:
                    st.write(f"- Last 7 avg: **{avg_last_7}**")
                    st.write(f"- Hit rate last 7: **{(over_rate_7 or 0):.2f}**")
                    st.write(f"- Edge last 7: **{(edge_7 or 0):.2f}**")

            with col_right:
                st.markdown("**Model Result**")
                st.write(f"- Predicted score: **{predicted_score}**")
                st.write(f"- Confidence: **{confidence_pct:.1f}%**")
                st.write(f"- Recommended side: **{bet_side}**")

            gamelog = player_logs.get(player_name)
            if gamelog is None:
                rec = player_ids.get(player_name, {})
                gamelog = get_player_gamelog(player_name, rec.get("gsis_id"))

            series = get_market_series(gamelog, market).dropna()
            last_n_vals = series.iloc[:games_to_look_back]

            if not last_n_vals.empty:
                hist_df = pd.DataFrame(
                    {
                        "GAME_DATE": gamelog["GAME_DATE"].iloc[:games_to_look_back].values,
                        display_market.upper(): last_n_vals.values,
                        "vs_line": [
                            "Over" if v > line else "Under" if v < line else "Push"
                            for v in last_n_vals.values
                        ],
                    }
                )
                st.markdown(f"**Last {games_to_look_back} games for {display_market.upper()}**")
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("Not enough recent games to show game-by-game history.")
    else:
        st.caption("Click **“Why this prop?”** on any card above to see a detailed explanation here.")


# =====================================================
# TAB 2: FULL EDGES TABLE
# =====================================================
with tab_table:
    st.write("### Full Edges Table")

    def highlight_edges(row):
        styles = [""] * len(row)
        try:
            val = float(row[edge_col])
            over = float(row[rate_col])
            conf = float(row["confidence_pct"])
            side = str(row["bet_side"])
        except Exception:
            return styles

        if val >= min_edge and over >= min_over_rate and conf >= min_confidence and side in ("Over", "Under"):
            color = "background-color: #d4f8d4"
        elif val < 0:
            color = "background-color: #f8d4d4"
        elif val > 0:
            color = "background-color: #fff4c2"
        else:
            return styles

        return [color] * len(row)

    display_cols = [
        "player",
        "team",
        "opponent",
        "market",
        "line",
        "book",
        "bet_side",
        "predicted_score",
        "season_avg",
        f"avg_last_{games_to_look_back}",
        f"over_rate_last_{games_to_look_back}",
        f"edge_last_{games_to_look_back}",
        "avg_last_7",
        "over_rate_last_7",
        "edge_last_7",
        "confidence_pct",
        "game_time",
    ]

    table_df = view_df.copy()
    for col in display_cols:
        if col not in table_df.columns:
            table_df[col] = None

    table_df["market"] = table_df["market"].apply(lambda m: NFL_MARKET_LABELS.get(m, m))

    styled_edges = (
        table_df[display_cols]
        .sort_values(by=["confidence_pct", rate_col, edge_col], ascending=False)
        .style.apply(highlight_edges, axis=1)
    )

    st.dataframe(styled_edges, use_container_width=True)

    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            for e in errors:
                st.write("- ", e)


# =====================================================
# TAB 3: PLAYER DETAIL
# =====================================================
with tab_player:
    st.write("### Player Detail (Raw Game Log)")

    player_options = [""] + sorted(edges_df["player"].unique().tolist())
    selected_player = st.selectbox("Select a player", player_options)

    if selected_player:
        rec = player_ids.get(selected_player, {})
        gamelog = player_logs.get(selected_player) or get_player_gamelog(
            selected_player, rec.get("gsis_id")
        )

        if not gamelog.empty:
            st.write(f"Recent games for **{selected_player}**")
            st.dataframe(gamelog.head(games_to_look_back), use_container_width=True)
        else:
            st.warning("No game log available for this player.")


# =====================================================
# TAB 4: WHAT-IF LINE TESTER
# =====================================================
with tab_whatif:
    st.write("### What-If Line Tester")

    if edges_df.empty:
        st.info("No edges calculated yet.")
    else:
        player_options = [""] + sorted(edges_df["player"].unique().tolist())
        sel_player = st.selectbox("Player", player_options, key="whatif_player_nfl")

        if sel_player:
            markets_for_player = sorted(
                edges_df.loc[edges_df["player"] == sel_player, "market"].dropna().unique().tolist()
            )
            sel_market = st.selectbox("Market", markets_for_player, key="whatif_market_nfl")

            if sel_market:
                row_match = edges_df[
                    (edges_df["player"] == sel_player)
                    & (edges_df["market"] == sel_market)
                ]

                if row_match.empty:
                    st.warning("No edge row found for this player/market.")
                else:
                    base_row = row_match.iloc[0]
                    current_line = float(base_row["line"])

                    st.write(f"**Current line:** `{current_line}`")
                    new_line = st.number_input(
                        "Set your own line",
                        value=current_line,
                        step=0.5,
                    )

                    rec = player_ids.get(sel_player, {})
                    gamelog = player_logs.get(sel_player) or get_player_gamelog(
                        sel_player, rec.get("gsis_id")
                    )
                    series = get_market_series(gamelog, sel_market).dropna()
                    if series.empty:
                        st.warning("No stats available for this player/market.")
                    else:
                        season_avg = series.mean()
                        last_n = series.iloc[:games_to_look_back]
                        if last_n.empty:
                            st.warning("Not enough recent games for this player.")
                        else:
                            avg_last_n = last_n.mean()
                            over_rate_n = (last_n > new_line).mean()
                            edge_n = avg_last_n - new_line

                            last7 = series.iloc[:7]
                            if len(last7) > 0:
                                avg_last_7 = last7.mean()
                                over_rate_7 = (last7 > new_line).mean()
                                edge_7 = avg_last_7 - new_line
                            else:
                                avg_last_7 = None
                                over_rate_7 = None
                                edge_7 = None

                            w_season = 0.4
                            w_last_n = 0.4
                            w_last_7 = 0.2
                            avg7_for_blend = avg_last_7 if avg_last_7 is not None else avg_last_n
                            predicted_score = (
                                w_season * season_avg
                                + w_last_n * avg_last_n
                                + w_last_7 * avg7_for_blend
                            )

                            hit_score = over_rate_n
                            if new_line != 0:
                                edge_ratio = max(0.0, edge_n / max(1.0, new_line))
                            else:
                                edge_ratio = 0.0
                            edge_score = max(0.0, min(1.0, edge_ratio * 4.0))

                            confidence = 0.6 * hit_score + 0.4 * edge_score
                            confidence_pct = round(confidence * 100, 1)

                            delta = predicted_score - new_line
                            if delta >= 0.5:
                                bet_side = "Over"
                            elif delta <= -0.5:
                                bet_side = "Under"
                            else:
                                bet_side = "No clear edge"

                            col_l, col_r = st.columns(2)
                            with col_l:
                                st.markdown("**What-if summary**")
                                st.write(f"- Season avg: **{season_avg:.2f}**")
                                st.write(f"- Last {games_to_look_back} avg: **{avg_last_n:.2f}**")
                                st.write(f"- Hit rate vs your line: **{over_rate_n:.2f}**")
                                st.write(f"- Edge vs your line: **{edge_n:.2f}**")

                                if avg_last_7 is not None:
                                    st.write(f"- Last 7 avg: **{avg_last_7:.2f}**")
                                    st.write(f"- Hit rate last 7 vs line: **{(over_rate_7 or 0):.2f}**")
                                    st.write(f"- Edge last 7 vs line: **{(edge_7 or 0):.2f}**")

                            with col_r:
                                st.markdown("**Model result at your line**")
                                st.write(f"- Predicted score: **{predicted_score:.2f}**")
                                st.write(f"- Confidence: **{confidence_pct:.1f}%**")
                                st.write(f"- Recommended side: **{bet_side}**")

                            st.progress(min(max(confidence_pct / 100.0, 0.0), 1.0))

                            last_n_vals = last_n.copy()
                            display_market = NFL_MARKET_LABELS.get(sel_market, sel_market)
                            hist_df = pd.DataFrame(
                                {
                                    "GAME_DATE": gamelog["GAME_DATE"].iloc[:games_to_look_back].values,
                                    display_market.upper(): last_n_vals.values,
                                    "vs_your_line": [
                                        "Over" if v > new_line else "Under" if v < new_line else "Push"
                                        for v in last_n_vals.values
                                    ],
                                }
                            )
                            st.markdown(f"**Last {games_to_look_back} games vs your line `{new_line}`**")
                            st.dataframe(hist_df, use_container_width=True)
