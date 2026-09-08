"""Source adapters; failures stay visible and never become zero-valued features."""
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
import nflreadpy as nfl
import polars as pl
import requests
from core import prepare_stats, normalize_name

PBP_COLS=['game_id','season','season_type','week','posteam','defteam','play_type','play_id','pass_attempt','rush_attempt','qb_dropback','qb_scramble','qb_kneel','receiver_player_id','rusher_player_id','passer_player_id','complete_pass','passing_yards','receiving_yards','rushing_yards','air_yards','yards_after_catch','yardline_100','epa','score_differential','qtr','down','ydstogo']

def stamp(): return datetime.now(timezone.utc).isoformat()
def frame(result): return result.to_pandas()
def result(name, fn):
    checked = stamp()
    try:
        df = fn()
        return df, dict(source=name,status='Available' if not df.empty else 'Empty',checked_at=checked,last_success_at=checked if not df.empty else None,rows=len(df),error=None)
    except Exception as exc:
        return pd.DataFrame(),dict(source=name,status='Unavailable',checked_at=checked,last_success_at=None,rows=0,error=f'{type(exc).__name__}: {exc}')

@st.cache_data(ttl=180,max_entries=2,show_spinner=False)
def board_source():
    from DFS_Wrapper import PrizePick
    class TimedPrizePick(PrizePick):
        def _get_api_data(self, dfs_book):
            response=requests.get(self.DFS_BOOKS[dfs_book.lower()],timeout=(10,25))
            response.raise_for_status()
            return response.json()
    provider=TimedPrizePick()
    rows=[r for r in provider.get_data(False) if str(r.get("league", "")).upper()=="NFL"]
    originals={p['id']:p.get('attributes',{}) for p in provider.api_data.get('data',[])}
    for row in rows:
        attrs=originals.get(row.get('projection_id'),{})
        for key in ['event_type','game_id','description','allowed_wager_types','updated_at']:
            row[key]=attrs.get(key)
        row['event_metadata_checked']=True
    return rows,stamp()

@st.cache_data(ttl=3600,max_entries=4,show_spinner=False)
def foundation(season,week,include_history=False,include_usage=False):
    data,health={},[]
    for key,fn in [('schedule',lambda:frame(nfl.load_schedules(season))),('rosters',lambda:frame(nfl.load_rosters(season)))]:
        data[key],status=result('nflverse '+key,fn); health.append(status)
    if not data['rosters'].empty:
        data['rosters']['_name']=data['rosters'].full_name.map(normalize_name)
    data['depth'],data['snaps']=pd.DataFrame(),pd.DataFrame()
    if include_usage:
        data['depth'],status=depth_source(season); health.append(status)
        data['snaps'],status=snap_source(season-1); health.append(status)
    else:
        health.extend([dict(source='nflverse '+key,status='Not requested; opens with Player',checked_at=None) for key in ['depth','snaps']])
    if not include_history:
        data['stats']=pd.DataFrame()
        health.append(dict(source='nflverse weekly stats',status='Not requested; opens with Player or Research',checked_at=None))
        return data,health
    frames=[]
    for year in [season-2,season-1]+([season] if week>1 else []):
        d,status=result(f'nflverse weekly stats {year}',lambda:frame(nfl.load_player_stats(year)))
        health.append(status)
        if not d.empty: frames.append(d)
    data['stats']=prepare_stats(pd.concat(frames,ignore_index=True)) if frames else pd.DataFrame()
    if not data['stats'].empty:
        d=data['stats']; data['stats']=d[(d.season<season)|((d.season==season)&(d.week<week))]
    return data,health

def latest_depth(raw):
    raw=raw.with_columns(pl.col('dt').str.to_datetime(strict=False,time_zone='UTC').alias('observed_at'))
    raw=raw.filter(pl.col('observed_at')<=datetime.now(timezone.utc))
    raw=raw.filter(pl.col('observed_at')==pl.col('observed_at').max().over('team'))
    return frame(raw)

@st.cache_data(ttl=3600,max_entries=2,show_spinner=False)
def depth_source(season):
    return result('nflverse depth',lambda:latest_depth(nfl.load_depth_charts(season)))

@st.cache_data(ttl=86400,max_entries=2,show_spinner=False)
def snap_source(season):
    return result(f'nflverse snaps {season}',lambda:frame(nfl.load_snap_counts(season)))

@st.cache_data(ttl=86400,max_entries=2,show_spinner=False)
def play_history(season):
    def read():
        raw=nfl.load_pbp(season)
        return frame(raw.select([c for c in PBP_COLS if c in raw.columns]))
    return result(f'nflverse play-by-play {season}',read)
