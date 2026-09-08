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
    return hashlib.sha256(encoded.encode()).hexdigest(),now,kickoff,body['model_version'],encoded

def save_snapshot(database_url,payload,kickoff):
    import psycopg
    values=snapshot_record(payload,kickoff)
    with psycopg.connect(database_url,connect_timeout=8) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute('INSERT INTO nfl_research_snapshots (snapshot_id,created_at,kickoff,model_version,payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',values)
    return values[0]

def save_board_snapshot(database_url, board, fetched_at):
    """Record verified offered lines for later evaluation; idempotent per line snapshot."""
    import psycopg
    from datetime import datetime, timezone
    now=datetime.now(timezone.utc)
    with psycopg.connect(database_url,connect_timeout=8) as conn:
        conn.execute(DDL)
        for _, r in board.iterrows():
            kickoff=r.game_time.to_pydatetime() if hasattr(r.game_time,'to_pydatetime') else r.game_time
            if kickoff.tzinfo is None or kickoff <= now: continue
            payload={'player':r.player,'player_id':r.player_id,'game_id':r.game_id,'market':r.market,'line':float(r.line),'odds_type':r.odds_type,'sides':list(r.sides),'board_fetched_at':str(fetched_at),'hypothetical':False,'model_status':'Verified board observation'}
            sid,created,kick,version,encoded=snapshot_record({**payload,'model_version':'board-v1'},kickoff,now=now)
            conn.execute('INSERT INTO nfl_research_snapshots (snapshot_id,created_at,kickoff,model_version,payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',(sid,created,kick,version,encoded))
