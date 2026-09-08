import json
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from core import prepare_stats, player_games, parse_board, summarize, history

st.set_page_config(page_title='NFL Prop Explorer', layout='wide')
st.markdown("""
<style>
@media (max-width: 640px) {
    .block-container { padding: 1rem 0.8rem 3rem; }
    h1 { font-size: 1.8rem !important; }
    h3 { font-size: 1.2rem !important; overflow-wrap: anywhere; }
    button { min-height: 44px; }
    [data-baseweb="tab-list"] { gap: 0.6rem; overflow-x: auto; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner='Loading PrizePicks board...')
def live_board():
    from DFS_Wrapper import PrizePick
    provider = PrizePick()
    rows = provider.get_data(organize_data=False)
    originals = {p['id']: p.get('attributes', {}) for p in provider.api_data.get('data', [])}
    for row in rows:
        attrs = originals.get(row.get('projection_id'), {})
        for key in ['event_type', 'game_id', 'description']:
            row[key] = attrs.get(key)
        row['event_metadata_checked'] = True
    return rows, pd.Timestamp.now(tz='UTC')

@st.cache_data(ttl=3600, show_spinner='Loading NFL game history...')
def load_stats(season, week=1):
    import nflreadpy as nfl
    frames, warnings = [], []
    # Load independently: an unpublished Week 1 season must not discard prior history.
    years = [season - 2, season - 1] + ([season] if week > 1 else [])
    for year in years:
        try:
            frame = pd.DataFrame(nfl.load_player_stats(seasons=year, summary_level='week').to_dicts())
            if not frame.empty:
                frames.append(frame)
            else:
                warnings.append(f'{year}: no stats published yet.')
        except Exception as exc:
            warnings.append(f'{year}: stats unavailable ({type(exc).__name__}: {exc}).')
    if not frames:
        raise RuntimeError('No season history could be loaded. Check internet access and nflreadpy.')
    return prepare_stats(pd.concat(frames, ignore_index=True)), warnings

@st.cache_data(ttl=3600, show_spinner='Loading season schedule...')
def load_schedule(season):
    import nflreadpy as nfl
    return pd.DataFrame(nfl.load_schedules(seasons=season).to_dicts())

def main():
    st.title('NFL Prop Explorer')
    st.caption('PrizePicks full-game lines compared with regular-season NFL game history.')
    st.info('General feed only: availability on your signed-in PrizePicks mobile board is unverified. A listed projection may not be selectable for you. Match the player, opponent, game date and line in mobile before using it.')
    today = datetime.now(ZoneInfo('America/Chicago'))
    season = st.sidebar.number_input('Season', min_value=2000, max_value=today.year + 1, value=today.year if today.month >= 3 else today.year - 1)
    week = st.sidebar.number_input('Week', min_value=1, max_value=18, value=1)
    n = st.sidebar.slider('Games to look back', 5, 25, 10)
    min_games = st.sidebar.slider('Minimum recorded games', 1, 25, 5)
    side_filter = st.sidebar.selectbox('Side', ['All', 'Over', 'Under'])
    hit_min = st.sidebar.slider('Minimum historical side hit rate', 0., 1., .6, .05)
    timezone = st.sidebar.selectbox('Display timezone', ['America/Chicago', 'America/New_York', 'America/Denver', 'America/Los_Angeles', 'UTC'])
    only_today = st.sidebar.checkbox("Only today's games")
    if st.sidebar.button('Refresh live data'):
        live_board.clear()
        load_stats.clear()
        load_schedule.clear()
    upload = st.sidebar.file_uploader('Optional PrizePicks board JSON', type=['json'])
    st.sidebar.caption('Upload the unorganized list returned by DFS_Wrapper if the live feed is unavailable.')
    st.info('Week 1 uses prior-season history. Rookies without NFL history remain visible without an estimate. A historical average is not a forecast or a win probability; team changes and injuries are not modeled.')
    try:
        if upload:
            raw = json.load(upload)
            fetched = None
        else:
            raw, fetched = live_board()
        board, skips = parse_board(raw)
    except Exception as exc:
        st.error(f'Board unavailable: {exc}')
        st.info('Use the JSON upload to continue with an exported board. No sample lines are substituted.')
        return
    if fetched is not None:
        st.caption(f'Board fetched: {fetched.tz_convert(timezone):%Y-%m-%d %H:%M %Z}. Cached for up to 5 minutes; started games are removed on every rerun.')
    with st.expander('Board import diagnostics'):
        st.write(skips)
        st.caption('Fantasy Score is excluded until PrizePicks scoring is explicitly implemented and verified. Alternate lines retain their odds type.')
    if board.empty:
        st.warning('No supported upcoming NFL props were returned.')
        return
    try:
        stats, warnings = load_stats(int(season), int(week))
    except Exception as exc:
        st.error(str(exc)); return
    for warning in warnings:
        st.warning('Some season history could not be loaded. Results use the available games.')
        with st.expander('Download details'):
            st.caption(warning)
    if week == 1:
        st.caption(f'Using prior-season history for {int(season)} Week 1. Current-season stats are not needed yet.')
    # No games from the target week or later may enter a historical comparison.
    stats = stats[(stats.season < season) | ((stats.season == season) & (stats.week < week))]
    try:
        schedule = load_schedule(int(season))
        selected = schedule[(schedule['game_type'] == 'REG') & (schedule['week'] == week)]
        dates = set(pd.to_datetime(selected['gameday']).dt.date)
        board = board[board.game_time.dt.tz_convert('America/New_York').dt.date.isin(dates)]
    except Exception as exc:
        st.warning(f'Week schedule unavailable ({exc}). Upcoming games are shown; verify kickoff dates manually.')
    local_times = board.game_time.dt.tz_convert(timezone)
    if only_today:
        board = board[local_times.dt.date == datetime.now(ZoneInfo(timezone)).date()]
    st.caption(f'History cutoff: before {int(season)} Week {int(week)}. Board contains upcoming games; verify kickoff for the desired week.')
    a, b, c = st.columns(3)
    team = a.selectbox('Team', ['All'] + sorted(board.team.unique()))
    market = b.selectbox('Market', ['All'] + sorted(board.market.unique()))
    search = c.text_input('Search player')
    if team != 'All': board = board[board.team.eq(team)]
    if market != 'All': board = board[board.market.eq(market)]
    if search: board = board[board.player.str.contains(search, case=False, regex=False)]
    rows = []
    for record in board.to_dict('records'):
        games = player_games(stats, record['player'])
        result = summarize(games, record['market'], record['line'], n, record['odds_type'])
        record.update(result or dict(side='No history', games=0, side_hit_rate=0.))
        rows.append(record)
    if not rows:
        st.info('No props match the selected filters.'); return
    data = pd.DataFrame(rows)
    cards, table, detail, whatif = st.tabs(['Cards & Explanation', 'Table', 'Player Detail', 'What-If Line Tester'])
    with table:
        st.dataframe(data, hide_index=True)
        st.download_button('Download results', data.to_csv(index=False), 'nfl-props.csv', 'text/csv')
    with cards:
        st.caption('Demons and Goblins are treated as More-only. If history points below the line, the app passes instead of suggesting Under.')
        featured = data[data.side.isin(['Over', 'Under']) & (data.games >= min_games) & (data.side_hit_rate >= hit_min)]
        if side_filter != 'All': featured = featured[featured.side.eq(side_filter)]
        if featured.empty:
            st.info('No props meet the filters. All imported props remain in the Table tab.')
        for idx, row in featured.sort_values('side_hit_rate', ascending=False).head(12).iterrows():
            with st.container(border=True):
                matched = player_games(stats, row['player'])
                if 'headshot_url' in matched:
                    photos = matched.headshot_url.dropna()
                    if not photos.empty and isinstance(photos.iloc[0], str) and photos.iloc[0].startswith('https://'):
                        st.image(photos.iloc[0], width=100)
                st.caption('Mobile availability: unverified')
                st.subheader(f"{row['player']} | {row['market']} | {row['line']}")
                st.write(f"{row['team']} vs {row['opponent']} | {row['odds_type']} | {row['game_time'].tz_convert(timezone):%b %d %H:%M %Z}")
                if row['side_policy'] == 'More only':
                    st.caption('More-only alternate line: Under is excluded.')
                st.write(f"Historical direction: {row['side']} | Average: {row['baseline']:.2f} | Difference: {row['edge']:+.2f}")
                st.write(f"Side hit rate: {row['side_hit_rate']:.0%} across {row['games']} games ({row['history_seasons']}); pushes: {row['push_rate']:.0%}")
                with st.expander('Why this prop?'):
                    st.caption('Direction compares the last available N-game average with the line. Hit rates include pushes in the denominator and use the displayed side. They are descriptive, not estimated probabilities.')
                    st.dataframe(history(player_games(stats, row['player']), row['market'], row['line'], n), hide_index=True)
    with detail:
        player = st.selectbox('Player', sorted(data.player.unique()), key='detail_player')
        st.dataframe(player_games(stats, player).head(n), hide_index=True)
    with whatif:
        choices = list(data.index)
        idx = st.selectbox('Choose prop', choices, format_func=lambda i: f"{data.loc[i, 'player']} / {data.loc[i, 'market']} / {data.loc[i, 'line']} / {data.loc[i, 'odds_type']} / {data.loc[i, 'game_time']}")
        row = data.loc[idx]
        line = st.number_input('Your line', min_value=0., value=float(row['line']), step=.5, key=f'line_{idx}_{row["player"]}_{row["market"]}_{row["line"]}')
        games = player_games(stats, row['player'])
        result = summarize(games, row['market'], line, n, row['odds_type'])
        if result:
            if result['side_policy'] == 'More only':
                st.info('More-only alternate line. A below-line average means Pass; changing the test line does not unlock Under.')
            st.write(result)
            st.dataframe(history(games, row['market'], line, n), hide_index=True)
        else:
            st.info('No matching NFL history for this player and market.')

if __name__ == '__main__':
    main()
