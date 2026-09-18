import math
import re
import unicodedata
import pandas as pd

MARKETS = {
    'pass_yds': ('passing_yards',), 'rush_yds': ('rushing_yards',),
    'rec_yds': ('receiving_yards',), 'receptions': ('receptions',),
    'targets': ('targets',), 'rush_att': ('carries',), 'pass_td': ('passing_tds',),
    'pass_rush_yds': ('passing_yards', 'rushing_yards'),
    'rush_rec_yds': ('rushing_yards', 'receiving_yards'),
}
ALIASES = {
    'passingyards': 'pass_yds', 'passyards': 'pass_yds', 'passyds': 'pass_yds',
    'rushingyards': 'rush_yds', 'rushyards': 'rush_yds', 'rushyds': 'rush_yds',
    'receivingyards': 'rec_yds', 'recyards': 'rec_yds', 'recyds': 'rec_yds',
    'rectargets': 'targets', 'receivingtargets': 'targets', 'targets': 'targets',
    'receptions': 'receptions', 'rushattempts': 'rush_att', 'rushingattempts': 'rush_att',
    'passtd': 'pass_td', 'passtds': 'pass_td', 'passingtds': 'pass_td',
    'passrushyds': 'pass_rush_yds', 'passrushyards': 'pass_rush_yds',
    'rushrecyds': 'rush_rec_yds', 'rushrecyards': 'rush_rec_yds',
}

def normalize_name(value):
    if not isinstance(value, str):
        return ''
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().lower()
    value = re.sub(r"[.'`’]", '', value)
    value = re.sub(r'[^a-z0-9 ]', ' ', value)
    return re.sub(r'\s+(jr|sr|ii|iii|iv)$', '', ' '.join(value.split()))

def prepare_stats(df):
    df = df.copy()
    for old, new in [('recent_team', 'team'), ('rushing_attempts', 'carries')]:
        if new not in df and old in df:
            df[new] = df[old]
    for col in ['player_id', 'player_name', 'player_display_name', 'season', 'week', 'season_type']:
        if col not in df:
            df[col] = None
    df['merge_name'] = df.player_display_name.fillna(df.player_name).map(normalize_name)
    for col in ['season', 'week']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['season', 'week'])
    df = df[df.season_type.eq('REG')]
    return df.sort_values(['season', 'week'], ascending=False).drop_duplicates(['player_id', 'merge_name', 'season', 'week']).reset_index(drop=True)

def player_games(stats, name):
    # Never substitute a more popular player with the same surname.
    matched = stats[stats.merge_name.eq(normalize_name(name))]
    if matched.player_id.dropna().nunique() > 1:
        return matched.iloc[:0]
    return matched.copy()

def market_series(games, market):
    cols = MARKETS.get(market, ())
    if not cols or any(c not in games for c in cols):
        return pd.Series(index=games.index, dtype=float)
    values = games[list(cols)].apply(pd.to_numeric, errors='coerce')
    return values.sum(axis=1, min_count=len(cols)).replace([float('inf'), -float('inf')], float('nan'))

def parse_board(raw, now=None):
    if not isinstance(raw, list):
        raise ValueError('Expected a list from PrizePicks. Upload the unorganized JSON export or check the provider.')
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    records, skipped = [], {}
    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1
    for item in raw:
        if not isinstance(item, dict):
            skip('Malformed record'); continue
        if str(item.get('league', '')).upper() != 'NFL':
            skip('Other league or partial game'); continue
        if item.get('is_live') in (True, 1, 'true', 'True') or str(item.get('status', '')).lower() not in ('pre_game', 'pre-game', 'pregame'):
            skip('Not confirmed pregame'); continue
        if item.get('event_metadata_checked') and (item.get('event_type') != 'team' or not str(item.get('game_id') or '').startswith('NFL_game_')):
            skip('Not a confirmed single NFL game'); continue
        if re.search(r'\bseason\b|season-long|futures', ' '.join(str(item.get(k) or '') for k in ['description', 'scope', 'period', 'stat_type']), re.I):
            skip('Season-long projection'); continue
        detail = ' '.join(str(item.get(k) or '') for k in ['description', 'short_description', 'stat_type', 'scope', 'period']).lower()
        if re.search(r'half|quarter|\b[12]h\b|\bq[1-4]\b|first\s+(drive|five|5)', detail):
            skip('Partial game'); continue
        market = ALIASES.get(re.sub(r'[^a-z]', '', str(item.get('stat_type') or '').lower()))
        if not market:
            skip('Unsupported market (including unverified fantasy scoring)'); continue
        name = item.get('player_name') or item.get('name')
        if not isinstance(name, str) or not name.strip() or '+' in name:
            skip('Missing name or combo'); continue
        try:
            line = float(item.get('line_score', item.get('line_value')))
            if not math.isfinite(line) or line < 0:
                raise ValueError()
        except (TypeError, ValueError):
            skip('Invalid line'); continue
        start = pd.to_datetime(item.get('start_time') or item.get('game_date_time'), utc=True, errors='coerce')
        if pd.isna(start) or start <= now:
            skip('Missing kickoff or game already started'); continue
        records.append(dict(projection_id=str(item.get('projection_id') or ''), player=name.strip(), team=str(item.get('team') or ''), opponent=str(item.get('opponent') or ''), market=market, line=line, game_time=start, odds_type=str(item.get('odds_type') or 'unknown'), book='PrizePicks'))
    cols = ['projection_id', 'player', 'team', 'opponent', 'market', 'line', 'game_time', 'odds_type', 'book']
    return pd.DataFrame(records, columns=cols).drop_duplicates().reset_index(drop=True), skipped

def summarize(games, market, line, n, odds_type="standard"):
    values = market_series(games, market).dropna()
    if values.empty:
        return None
    recent = values.head(n)
    delta = float(recent.mean() - line)
    side = 'Over' if delta > 0 else 'Under' if delta < 0 else 'No clear edge'
    more_only = str(odds_type).strip().lower() in ('demon', 'goblin')
    if more_only and side == 'Under':
        side = 'Pass (More-only)'
    over = float((recent > line).mean())
    under = float((recent < line).mean())
    return dict(side_policy='More only' if more_only else 'Standard', baseline=float(recent.mean()), edge=delta, side=side, games=len(recent), over_rate=over, under_rate=under, push_rate=float((recent == line).mean()), side_hit_rate=over if side == 'Over' else under if side == 'Under' else 0., history_seasons=', '.join(str(int(x)) for x in games.loc[recent.index, 'season'].unique()))


def predictive_summary(games, market, line, n, opponent_games=None):
    """Recency-weighted research projection with conservative probability smoothing."""
    values = market_series(games, market).dropna().head(n)
    if len(values) < 5:
        return None
    # Half-life of four games: recent usage changes matter without discarding history.
    weights = pd.Series([0.5 ** (i / 4.0) for i in range(len(values))], index=values.index)
    weights = weights / weights.sum()
    mean = float((values * weights).sum())
    variance = float((((values - mean) ** 2) * weights).sum())
    sd = max(variance ** 0.5, 0.25)
    over_weight = float(weights[values > line].sum())
    under_weight = float(weights[values < line].sum())
    push_weight = float(weights[values == line].sum())
    # Beta(2,2) prior prevents small samples and streaks from producing extreme confidence.
    effective_n = float(min(len(values), 10))
    empirical_over = (over_weight * effective_n + 2.0) / (effective_n + 4.0)
    opponent_used = 0
    if opponent_games is not None and not opponent_games.empty:
        opponent_values = market_series(opponent_games, market).dropna().head(5)
        if len(opponent_values) >= 3:
            opponent_used = len(opponent_values)
            opponent_mean = float(opponent_values.mean())
            mean = 0.80 * mean + 0.20 * opponent_mean
            opponent_over = (float((opponent_values > line).sum()) + 1.0) / (len(opponent_values) + 2.0)
            empirical_over = 0.80 * empirical_over + 0.20 * opponent_over
    # A distribution estimate adds margin information; blend it with actual hit frequency.
    normal_over = 0.5 * (1.0 + math.erf((mean - float(line)) / (sd * math.sqrt(2.0))))
    over_share = max(0.05, min(0.95, 0.65 * empirical_over + 0.35 * normal_over))
    # Preserve pushes as a real third outcome instead of redistributing them to either side.
    push = max(0.0, min(0.90, push_weight))
    decisive_mass = 1.0 - push
    over = decisive_mass * over_share
    under = decisive_mass * (1.0 - over_share)
    side = 'Over' if over > under else 'Under' if under > over else 'No clear edge'
    probability = max(over, under)
    return dict(
        side=side,
        baseline=mean,
        edge=mean - float(line),
        games=len(values),
        over_rate=over,
        under_rate=under,
        push_rate=push,
        side_hit_rate=probability,
        uncertainty=sd,
        opponent_games=opponent_used,
        method='recency-weighted + smoothed' + (' + opponent-adjusted' if opponent_used else ''),
    )


def history(games, market, line, n):
    values = market_series(games, market).dropna().head(n)
    result = games.loc[values.index, ['season', 'week']].copy()
    result['value'] = values
    result['result'] = values.map(lambda v: 'Over' if v > line else 'Under' if v < line else 'Push')
    return result


def historical_lean(games, market, line, n, sides):
    # Treat a one-sided feed as More-only before deriving the historical lean.
    odds_type = 'demon' if tuple(sides or ()) == ('over',) else 'standard'
    result = summarize(games, market, line, n, odds_type)
    if result is None:
        return 'NO HISTORY', 'No matched historical sample'
    # Derive the displayed lean from the majority of actual outcomes, not the mean.
    # This keeps the label consistent with the chart and observed side rate.
    if tuple(sides or ()) == ('over',):
        side = 'Over'
    elif result['over_rate'] > result['under_rate']:
        side = 'Over'
    elif result['under_rate'] > result['over_rate']:
        side = 'Under'
    else:
        side = None
    rate = result['over_rate'] if side == 'Over' else result['under_rate'] if side == 'Under' else max(result['over_rate'],result['under_rate'])
    games_count=result['games']
    margin=1.96 * (rate * (1-rate) / max(games_count,1)) ** 0.5
    low=max(0.0,rate-margin); high=min(1.0,rate+margin)
    detail = f"Average {result['baseline']:.1f} / {games_count} recorded games / observed side rate {rate:.0%} (approx. 95% range {low:.0%}-{high:.0%})"
    if result['games'] < 5:
        return 'INSUFFICIENT HISTORY', detail
    if result['push_rate'] >= 0.40:
        return 'PUSH-HEAVY / NO LEAN', detail
    if rate < 0.55:
        return 'WEAK HISTORY / NO LEAN', detail
    side = {'Over': 'over', 'Under': 'under'}.get(side)
    if side is None:
        return 'NO HISTORICAL LEAN', detail
    if side not in sides:
        return 'PASS / Lean side unavailable', detail
    return 'Historical lean: ' + ('MORE / OVER' if side == 'over' else 'LESS / UNDER'), detail
