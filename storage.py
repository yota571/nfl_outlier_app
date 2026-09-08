"""Optional PostgreSQL immutable research history. No local-cloud SQLite fallback."""
import hashlib
import json
from datetime import datetime, timezone

DDL='''CREATE TABLE IF NOT EXISTS nfl_research_snapshots (
 snapshot_id TEXT PRIMARY KEY,
 created_at TIMESTAMPTZ NOT NULL,
 kickoff TIMESTAMPTZ NOT NULL,
 model_version TEXT NOT NULL,
 payload JSONB NOT NULL
)'''

def snapshot_record(payload, kickoff, now=None):
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None or kickoff.tzinfo is None:
        raise ValueError('Snapshot timestamps must include a timezone')
    if kickoff<=now:
        raise ValueError('Cannot save a pregame snapshot after kickoff')
    if payload.get('hypothetical'):
        raise ValueError('Hypothetical lines cannot be saved as offered pregame predictions')
    body={**payload,'created_at':now.isoformat(),'kickoff':kickoff.isoformat()}
    encoded=json.dumps(body,sort_keys=True,allow_nan=False)
    return hashlib.sha256(json.dumps({k:v for k,v in body.items() if k != 'created_at'},sort_keys=True,allow_nan=False).encode()).hexdigest(),now,kickoff,body['model_version'],encoded

def save_snapshot(database_url,payload,kickoff):
    import psycopg
    values=snapshot_record(payload,kickoff)
    with psycopg.connect(database_url,connect_timeout=8) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute('INSERT INTO nfl_research_snapshots (snapshot_id,created_at,kickoff,model_version,payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',values)
    return values[0]

def save_board_snapshot(database_url, board, fetched_at, stats=None):
    from tracking import connect
    from core import market_series
    now=datetime.now(timezone.utc)
    observed=datetime.fromisoformat(str(fetched_at))
    if observed.tzinfo is None or not 0 <= (now-observed).total_seconds() <= 300:
        raise ValueError('Board is stale')
    values=[]
    for _,r in board.iterrows():
        kickoff=r.game_time.to_pydatetime()
        if kickoff <= now: continue
        baseline=None
        if stats is not None and not stats.empty:
            games=stats[stats.player_id.eq(r.player_id)].sort_values(['season','week'],ascending=False)
            history=market_series(games,r.market).dropna().head(10)
            if len(history)>=5:
                baseline={'mean':float(history.mean()),'more':float((history>r.line).mean()),'less':float((history<r.line).mean()),'push':float((history==r.line).mean()),'games':len(history),'version':'historical-last10-v1','status':'Uncalibrated historical baseline'}
        payload={'player':r.player,'player_id':r.player_id,'game_id':r.game_id,'market':r.market,'projection_id':str(r.projection_id),'line':float(r.line),'odds_type':r.odds_type,'sides':sorted(r.sides),'board_fetched_at':str(fetched_at),'hypothetical':False,'model_status':'Board observation; no validated recommendation','model_version':'board-v2','baseline':baseline}
        values.append(snapshot_record(payload,kickoff,now))
    inserted=0
    with connect(database_url) as conn:
        conn.execute(DDL)
        ensure_rls(conn, 'nfl_research_snapshots')
        conn.commit()
        for value in values:
            result=conn.execute('INSERT INTO nfl_research_snapshots (snapshot_id,created_at,kickoff,model_version,payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',value)
            inserted+=result.rowcount
    return inserted


def board_records(url):
    from tracking import connect
    with connect(url) as conn:
        conn.execute(DDL)
        ensure_rls(conn, 'nfl_research_snapshots')
        conn.commit()
        conn.execute('CREATE TABLE IF NOT EXISTS nfl_board_outcomes (snapshot_id TEXT PRIMARY KEY REFERENCES nfl_research_snapshots(snapshot_id), actual DOUBLE PRECISION NOT NULL, recorded_at TIMESTAMPTZ NOT NULL)')
        ensure_rls(conn, 'nfl_board_outcomes')
        conn.commit()
        rows=conn.execute("SELECT s.snapshot_id,s.payload,o.actual FROM nfl_research_snapshots s LEFT JOIN nfl_board_outcomes o USING(snapshot_id) WHERE s.model_version='board-v2' ORDER BY s.created_at DESC").fetchall()
    return [dict(**body,snapshot_id=key,actual=actual) for key,body,actual in rows]


def settle_board(url,stats,schedule,now=None):
    from tracking import connect
    from core import market_series
    now=now or datetime.now(timezone.utc)
    saved=0
    with connect(url) as conn:
        for row in board_records(url):
            if row['actual'] is not None or (now-datetime.fromisoformat(row['kickoff'])).total_seconds()<36*3600: continue
            game=schedule[schedule.game_id.eq(row['game_id'])]
            if len(game)!=1 or game[['home_score','away_score']].isna().any(axis=None): continue
            g=game.iloc[0]
            player=stats[stats.player_id.eq(row['player_id']) & stats.season.eq(g.season) & stats.week.eq(g.week) & stats.season_type.eq('REG')]
            result=market_series(player,row['market']).dropna()
            if len(result)!=1: continue
            result=conn.execute('INSERT INTO nfl_board_outcomes VALUES (%s,%s,%s) ON CONFLICT DO NOTHING',(row['snapshot_id'],float(result.iloc[0]),now))
            saved+=result.rowcount
    return saved


def ensure_rls(conn, table):
    """Avoid requesting ACCESS EXCLUSIVE on every read/write once RLS is enabled."""
    from psycopg import sql
    row=conn.execute("SELECT relrowsecurity FROM pg_class WHERE oid=to_regclass(%s)",('public.'+table,)).fetchone()
    if row is None: raise RuntimeError('Tracking table is unavailable')
    if not row[0]:
        conn.execute("SET LOCAL lock_timeout = '5s'")
        conn.execute(sql.SQL('ALTER TABLE public.{} ENABLE ROW LEVEL SECURITY').format(sql.Identifier(table)))
