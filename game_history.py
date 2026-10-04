"""Join history by game identity and restore only snap-confirmed zero-stat games."""
import pandas as pd
from core import MARKETS, prepare_stats
from verification import team_code


def complete_history(raw, snaps, identities, schedule, season, week):
    stats = prepare_stats(raw, include_postseason=True)
    if stats.empty:
        return stats
    if schedule.empty:
        return stats.iloc[:0].copy()
    stats['team'] = stats['team'].map(team_code)
    keys = ['season', 'week', 'season_type', 'team']
    schedule = schedule[schedule.game_type.isin(['REG', 'WC', 'DIV', 'CON', 'SB', 'POST'])].copy()
    schedule['season_type'] = schedule.game_type.map(lambda x: 'REG' if x == 'REG' else 'POST')
    sides = []
    for side, other in [('home', 'away'), ('away', 'home')]:
        d = schedule[['season', 'week', 'season_type', 'game_id', 'gameday', side+'_team', other+'_team']].copy()
        d = d.rename(columns={side+'_team': 'team', other+'_team': 'opponent_team'})
        d['team'] = d.team.map(team_code)
        d['opponent_team'] = d.opponent_team.map(team_code)
        sides.append(d)
    games = pd.concat(sides, ignore_index=True)
    games = games[(games.season < season) | ((games.season == season) & games.season_type.eq('REG') & (games.week < week))]
    # A team must have exactly one game for these season/week/type keys.
    games = games.drop_duplicates()
    games = games[~games.duplicated(keys, keep=False)]
    stats = stats.drop(columns=['game_id', 'gameday', 'opponent_team'], errors='ignore').merge(games, on=keys, how='inner', validate='many_to_one')
    stats['history_source'] = 'Official weekly stats'
    stat_cols = sorted({c for cols in MARKETS.values() for c in cols})
    required_snaps = {'season', 'week', 'game_id', 'team', 'pfr_player_id', 'offense_snaps'}
    if required_snaps.issubset(snaps.columns) and {'pfr_id', 'gsis_id'}.issubset(identities.columns):
        ids = identities[['pfr_id', 'gsis_id']].dropna().drop_duplicates()
        ids = ids[~ids.pfr_id.duplicated(keep=False)].rename(columns={'gsis_id': 'player_id'})
        active = snaps[pd.to_numeric(snaps.offense_snaps, errors='coerce') > 0].copy()
        active['team'] = active.team.map(team_code)
        active = active.merge(ids, left_on='pfr_player_id', right_on='pfr_id', validate='many_to_one')
        # Use the actual game ID, never dates generated from row positions.
        active = active[['game_id', 'team', 'player_id']].drop_duplicates().merge(games, on=['game_id', 'team'], validate='many_to_one')
        player_keys = keys + ['player_id']
        active = active.merge(stats[player_keys].drop_duplicates(), on=player_keys, how='left', indicator=True)
        missing = active[active['_merge'].eq('left_only')].drop(columns='_merge').copy()
        # Missing a whole team's stats is a source outage, not a zero performance.
        available_cols = [c for c in stat_cols if c in stats]
        if available_cols:
            complete = stats.groupby(keys)[available_cols].agg(lambda s: s.notna().all()).reset_index()
            missing = missing.merge(complete, on=keys, how='inner', validate='many_to_one')
            for col in available_cols:
                missing[col] = missing[col].map(lambda complete: 0.0 if complete else float('nan'))
            missing['history_source'] = 'Offensive snaps confirmed; no official stat row'
            stats = pd.concat([stats, missing], ignore_index=True)
    stats['gameday'] = pd.to_datetime(stats.gameday, errors='coerce')
    return stats.sort_values(['gameday', 'season', 'week'], ascending=False).reset_index(drop=True)
