"""Prediction ledger for PostgreSQL; all writes are parameterized and pregame-only."""
import hashlib,json,math
from datetime import datetime,timezone

SCHEMA='''
CREATE TABLE IF NOT EXISTS public.nfl_predictions (
 prediction_id text PRIMARY KEY,
 created_at timestamptz NOT NULL,
 kickoff timestamptz NOT NULL,
 player_id text NOT NULL,
 game_id text NOT NULL,
 market text NOT NULL,
 model_version text NOT NULL,
 payload jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS public.nfl_outcomes (
 prediction_id text PRIMARY KEY REFERENCES public.nfl_predictions(prediction_id),
 recorded_at timestamptz NOT NULL,
 actual double precision NOT NULL,
 source text NOT NULL
);
ALTER TABLE public.nfl_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.nfl_outcomes ENABLE ROW LEVEL SECURITY;
'''

def prepare_prediction(payload,now=None):
 now=now or datetime.now(timezone.utc)
 kickoff=datetime.fromisoformat(payload['kickoff'])
 observed=datetime.fromisoformat(payload['board_fetched_at'])
 if now.tzinfo is None or kickoff.tzinfo is None or observed.tzinfo is None: raise ValueError('Timezone required')
 if kickoff<=now: raise ValueError('Game has started')
 if not 0<=(now-observed).total_seconds()<=300: raise ValueError('Board is stale')
 if payload.get('hypothetical'): raise ValueError('Only offered lines can be tracked')
 if payload['market'] not in ('targets','rush_att'): raise ValueError('No eligible count model for this market')
 for k in ['line','mean','more','less','push']:
  if not math.isfinite(float(payload[k])): raise ValueError('Non-finite prediction')
 if any(not 0<=payload[k]<=1 for k in ['more','less','push']): raise ValueError('Invalid probability')
 if abs(payload['more']+payload['less']+payload['push']-1)>1e-8: raise ValueError('Probabilities must sum to one')
 if payload['line']<0 or payload['mean']<0: raise ValueError('Count forecast cannot be negative')
 # Idempotent across reruns at the same line and exact model inputs. Original timestamp stays immutable.
 stable={k:v for k,v in payload.items() if k not in ['created_at','board_fetched_at']}
 encoded=json.dumps(stable,sort_keys=True,allow_nan=False)
 key=hashlib.sha256(encoded.encode()).hexdigest()
 return key,now,{**payload,'created_at':now.isoformat()}

def connect(url):
 import psycopg
 from psycopg.conninfo import conninfo_to_dict
 mode=conninfo_to_dict(url).get('sslmode','require')
 if mode not in ('require','verify-ca','verify-full'): raise ValueError('Encrypted database connection required')
 return psycopg.connect(url,connect_timeout=8,prepare_threshold=None,sslmode=mode)

def initialize(url):
 with connect(url) as conn:
  conn.execute(SCHEMA)

def save(url,payload):
 key,created,body=prepare_prediction(payload)
 with connect(url) as conn:
  conn.execute('INSERT INTO public.nfl_predictions (prediction_id,created_at,kickoff,player_id,game_id,market,model_version,payload) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',(key,created,body['kickoff'],body['player_id'],body['game_id'],body['market'],body['model_version'],json.dumps(body)))
 return key

def load(url):
 with connect(url) as conn:
  rows=conn.execute('SELECT p.prediction_id,p.payload,o.actual FROM public.nfl_predictions p LEFT JOIN public.nfl_outcomes o USING(prediction_id) ORDER BY p.created_at DESC LIMIT 1000').fetchall()
 return [dict(prediction_id=r[0],**r[1],actual=r[2]) for r in rows]

def settle(url,stats,schedule,now=None):
 """Only official player rows for scored games at least 36 hours past kickoff. Missing stats remain pending."""
 now=now or datetime.now(timezone.utc)
 records=load(url); saved=0
 for row in records:
  if row['actual'] is not None: continue
  if (now-datetime.fromisoformat(row['kickoff'])).total_seconds()<36*3600: continue
  g=schedule[schedule.game_id.eq(row['game_id'])]
  if g.empty or g[['home_score','away_score']].isna().any(axis=None): continue
  s=stats[stats.game_id.eq(row['game_id']) & stats.player_id.eq(row['player_id'])]
  col='targets' if row['market']=='targets' else 'carries'
  if len(s)!=1 or col not in s:continue
  actual=float(s.iloc[0][col])
  if not math.isfinite(actual):continue
  with connect(url) as conn:
   conn.execute('INSERT INTO public.nfl_outcomes (prediction_id,recorded_at,actual,source) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',(row['prediction_id'],now,actual,'nflverse official weekly stats'))
  saved+=1
 return saved
