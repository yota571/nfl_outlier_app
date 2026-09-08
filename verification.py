"""Verified identities and event joins. All timestamps are UTC."""
import re
import pandas as pd
from core import normalize_name

TEAM_ALIASES = {'JAC':'JAX', 'LAR':'LA', 'WSH':'WAS', 'OAK':'LV', 'SD':'LAC'}
def team_code(value):
    code = str(value or '').strip().upper()
    return TEAM_ALIASES.get(code, code)

def resolve_player(row, rosters):
    if rosters.empty:
        return None, 'Current roster unavailable'
    names = rosters['_name'] if '_name' in rosters else rosters.full_name.map(normalize_name)
    candidates = rosters[names.eq(normalize_name(row['player']))].copy()
    candidates = candidates[candidates.team.map(team_code).eq(team_code(row['team']))]
    ids = candidates.gsis_id.dropna().unique()
    if len(ids) != 1:
        return None, 'Player/team identity missing or ambiguous'
    result = candidates[candidates.gsis_id.eq(ids[0])].iloc[0].to_dict()
    return result, None

def attach_games(board, schedule, season, week):
    rows, rejected = [], []
    if schedule.empty:
        return board.iloc[:0].copy(), ['Schedule unavailable; no games verified']
    eligible = schedule[(schedule.season == season) & (schedule.week == week) & schedule.game_type.eq('REG')].copy()
    for row in board.to_dict('records'):
        team, opp = team_code(row['team']), team_code(row['opponent'])
        games = eligible[((eligible.home_team.map(team_code)==team) & (eligible.away_team.map(team_code)==opp)) | ((eligible.away_team.map(team_code)==team) & (eligible.home_team.map(team_code)==opp))]
        games = games[pd.to_datetime(games.gameday).dt.date.eq(row['game_time'].tz_convert('America/New_York').date())]
        if len(games) != 1:
            rejected.append(f"{row['player']}: matchup/date not verified"); continue
        game = games.iloc[0]
        row.update(team=team, opponent=opp, game_id=game.game_id, home_away='Home' if team_code(game.home_team)==team else 'Away', stadium=game.get('stadium'), roof=game.get('roof'))
        rows.append(row)
    return pd.DataFrame(rows) if rows else board.iloc[:0].copy(), rejected

def allowed_sides(odds_type, raw=None):
    # Explicit provider restrictions always win. Missing alternate metadata is conservative.
    if raw is not None:
        tokens = raw if isinstance(raw, list) else re.split(r'[^a-z]+', str(raw).lower())
        return tuple(s for s in ('over','under') if s in tokens)
    return ('over',) if str(odds_type).lower() in ('demon','goblin') else ('over','under') if str(odds_type).lower()=='standard' else ()
