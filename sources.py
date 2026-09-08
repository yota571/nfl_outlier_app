"""Source adapters; failures stay visible and never become zero-valued features."""
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from core import prepare_stats, normalize_name

PBP_COLS=['game_id','season','season_type','week','posteam','defteam','play_type','play_id','pass_attempt','rush_attempt','qb_dropback','qb_scramble','qb_kneel','receiver_player_id','rusher_player_id','passer_player_id','complete_pass','passing_yards','receiving_yards','rushing_yards','air_yards','yards_after_catch','yardline_100','epa','score_differential','qtr','down','ydstogo']

def stamp(): return datetime.now(timezone.utc).isoformat()
def frame(result): return pd.DataFrame(result.to_dicts())
def result(name, fn):
    checked = stamp()
    try:
        df = fn()
        return df, dict(source=name,status='Available' if not df.empty else 'Empty',checked_at=checked,last_success_at=checked if not df.empty else None,rows=len(df),error=None)
    except Exception as exc:
        return pd.DataFrame(),dict(source=name,status='Unavailable',checked_at=checked,last_success_at=None,rows=0,error=f'{type(exc).__name__}: {exc}')

@st.cache_data(ttl=180,show_spinner=False)
def board_source():
    from DFS_Wrapper import PrizePick
    provider=PrizePick()
    rows=provider.get_data(False)
    originals={p['id']:p.get('attributes',{}) for p in provider.api_data.get('data',[])}
    for row in rows:
        attrs=originals.get(row.get('projection_id'),{})
        for key in ['event_type','game_id','description','allowed_wager_types','updated_at']:
            row[key]=attrs.get(key)
        row['event_metadata_checked']=True
    return rows,stamp()

@st.cache_data(ttl=3600,show_spinner=False)
def foundation(season,week):
    data,health={},[]
    for key,fn in [('schedule',lambda:frame(nfl.load_schedules(season))),('rosters',lambda:frame(nfl.load_rosters(season))),('depth',lambda:frame(nfl.load_depth_charts(season)))]:
        data[key],status=result('nflverse '+key,fn); health.append(status)
    if not data['rosters'].empty:
        data['rosters']['_name']=data['rosters'].full_name.map(normalize_name)
    if not data['depth'].empty:
        d=data['depth']; times=pd.to_datetime(d.dt,utc=True,errors='coerce'); d=d.assign(observed_at=times)
        d=d[d.observed_at<=pd.Timestamp.now(tz='UTC')]
        data['depth']=d[d.observed_at.eq(d.groupby('team').observed_at.transform('max'))].copy()
    frames=[]
    for year in [season-2,season-1]+([season] if week>1 else []):
        d,status=result(f'nflverse weekly stats {year}',lambda:frame(nfl.load_player_stats(year)))
        health.append(status)
        if not d.empty: frames.append(d)
    data['stats']=prepare_stats(pd.concat(frames,ignore_index=True)) if frames else pd.DataFrame()
    if not data['stats'].empty:
        d=data['stats']; data['stats']=d[(d.season<season)|((d.season==season)&(d.week<week))]
    data['snaps'],status=result(f'nflverse snaps {season-1}',lambda:frame(nfl.load_snap_counts(season-1)))
    health.append(status)
    return data,health

@st.cache_data(ttl=86400,show_spinner=False)
def play_history(season):
    def read():
        raw=nfl.load_pbp(season)
        return frame(raw.select([c for c in PBP_COLS if c in raw.columns]))
    return result(f'nflverse play-by-play {season}',read)
