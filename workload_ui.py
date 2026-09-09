"""Lazy UI for count-model evaluation and persistent pregame records."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from opportunity import forecast,sample_counts,VERSION
from sources import stamp

@st.cache_data(show_spinner=False)
def assets():
 root=Path(__file__).parent/'model_assets'
 return pd.read_parquet(root/'opportunity_history.parquet'),json.loads((root/'validation.json').read_text())

def validation_view(report):
 st.subheader('What has been tested?')
 st.caption('2025 expanding walk-forward test: every forecast used earlier weeks only. Counts are conditional on playing offensive snaps; this is not a PrizePicks win-rate test.')
 for kind,m in report['metrics'].items():
  st.write(f"**{kind.title()}** / {m['games']:,} player-games")
  st.write(f"Average error: model {m['mae']:.3f} / last-10 baseline {m['baseline_mae']:.3f}")
  st.caption(f"RMSE: model {m['rmse']:.3f} / baseline {m['baseline_rmse']:.3f}. Both models use the same eligible games.")
  ci=m.get('mae_gain_95pct_player_bootstrap')
  if ci:
   st.caption(f"95% player-bootstrap interval for error improvement: {ci[0]:+.3f} to {ci[1]:+.3f}."+(' This interval includes no improvement.' if ci[0]<=0 else ' This is a small historical improvement, not a guarantee.'))
 st.info('Workload counts have been tested. MORE/LESS probabilities still need calibration against recorded pregame lines. No validated betting recommendation is claimed.')
 st.caption('No current injury, route, QB-change or weather adjustments are included. The test covers same-team players with at least five prior appearances and some prior opportunities; rookies and role changes are not covered.')


def render_workload(row,season,week,board_time,url):
 table,report=assets()
 kind='carries' if row.market in ('rush_att','rush_yds') else 'targets'
 with st.expander('Targets / carries model',expanded=True):
  st.caption('Team volume x player opportunity share. Includes snap-confirmed zero-opportunity games.')
  if (season,week)>(2026,1):
   st.warning('This model asset ends at 2025 Week 18. Current-season inputs must be refreshed before producing later-week forecasts.')
   return
  pred=forecast(table,row.player_id,row.team,kind,season,week)
  if pred is None:
   st.info('No eligible workload forecast: fewer than five same-team appearances, no prior opportunities, or a team change. Historical comparisons remain available.')
  else:
   st.metric(f'Expected {kind}',f"{pred['mean']:.1f}")
   st.caption(f"Last-10 baseline {pred['baseline']:.1f} / team volume {pred['team_volume']:.1f} / {pred['history_games']} same-team games. Conditional on offensive participation and an unchanged role.")
   if row.market in ('targets','rush_att'):
    samples,_=sample_counts(pred,f'{row.player_id}|{row.game_id}|{kind}')
    more=float(np.mean(samples>row.line));less=float(np.mean(samples<row.line));push=float(np.mean(samples==row.line))
    st.write(f"Experimental outcomes at {row.line:g}: MORE {more:.1%} / LESS {less:.1%} / Push {push:.1%}")
    st.caption('These probabilities are uncalibrated. They do not override available-side restrictions or constitute a recommendation.')
    payload=dict(player_id=row.player_id,player=row.player,game_id=row.game_id,market=row.market,projection_id=str(row.get('projection_id','')),line=float(row.line),mean=pred['mean'],more=more,less=less,push=push,available_sides=list(row.sides),model_version=VERSION,model_data_version=report['created_at'],kickoff=row.game_time.isoformat(),board_fetched_at=board_time,hypothetical=False,model_status='Workload tested; probabilities experimental')
    st.download_button('Download offered-line forecast',json.dumps(payload,indent=2),'pregame_forecast.json','application/json',key='count_download')
    if url:
     if st.button('Save this pregame forecast',key='save_count'):
      try:
       from tracking import save
       save(url,payload);st.success('Forecast saved with its original line and timestamp.')
      except Exception:
       st.error('Not saved. The line may be stale or the database may not be ready. Refresh the board and check Results.')
    else: st.caption('Permanent saving requires the database setup on Results. The download is available now.')
  with st.expander('Count-model test results'):
   validation_view(report)


def render_results(url):
 _,report=assets()
 st.subheader('Prediction tracking')
 if url:
  render_board_results(url)
 if not url:
  st.info('Permanent tracking is ready to connect. No database is connected and no saved results are being claimed.')
  st.markdown('[Create your Supabase project](https://supabase.com/dashboard)')
  st.write('Create a project, run the setup SQL below, and add its session-pooler connection URI as DATABASE_URL in Streamlit Secrets. Keep the password out of chat and GitHub.')
  from tracking import SCHEMA
  st.download_button('Download database setup SQL',SCHEMA,'supabase_setup.sql','text/plain')
  st.caption('After connecting, forecasts are saved when you press Save this pregame forecast in Research. Saving is not an unattended background job.')
 else:
  from tracking import load,settle
  if st.button('Test database connection'):
   try: st.success(f'Connected. {len(load(url))} stored records found.')
   except Exception: st.error('Connection failed or tables are missing. Check DATABASE_URL and run the setup SQL.')
  try:
   records=load(url)
  except Exception:
   st.error('Could not load the ledger. The database may need its setup SQL applied.')
   from tracking import SCHEMA
   st.download_button('Download database setup SQL',SCHEMA,'supabase_setup.sql','text/plain',use_container_width=True)
   st.caption('Run this file in Supabase SQL Editor, then reboot the Streamlit app and test the connection again.')
   records=[]
  if st.button('Refresh official results'):
   try:
    import nflreadpy as nfl
    from datetime import datetime, timezone, timedelta
    cutoff=datetime.now(timezone.utc)-timedelta(hours=36)
    ready=[]
    for record in records:
     if record['actual'] is not None: continue
     try:
      kickoff=datetime.fromisoformat(str(record['kickoff'])).astimezone(timezone.utc)
     except (KeyError, TypeError, ValueError):
      continue
     if kickoff <= cutoff: ready.append(record)
    seasons=sorted({int(r['game_id'].split('_')[0]) for r in ready})
    if not seasons:
     st.info('No pending forecasts are 36 hours past kickoff yet. Saved forecasts remain listed below.')
    else:
     stats=nfl.load_player_stats(seasons).to_pandas();schedule=nfl.load_schedules(seasons).to_pandas()
     n=settle(url,stats,schedule);st.success(f'{n} results added.')
     records=load(url)
   except Exception:st.warning('Results could not be refreshed. Existing predictions are preserved.')
  st.write(f'{len(records)} saved forecasts')
  complete=[r for r in records if r['actual'] is not None]
  if complete:
   for kind in sorted({r['market'] for r in complete}):
    rows=[r for r in complete if r['market']==kind]
    # One earliest forecast per player/game/market/model avoids counting line-change snapshots as independent games.
    unique={}
    for r in sorted(rows,key=lambda r:r['created_at']):unique.setdefault((r['player_id'],r['game_id'],r['market'],r['model_version']),r)
    rows=list(unique.values());mae=float(np.mean([abs(r['mean']-r['actual']) for r in rows]))
    st.write(f'{kind}: {len(rows)} unique player-games / forecast MAE {mae:.2f}')
    brier=float(np.mean([(r['more']-float(r['actual']>r['line']))**2 for r in rows]))
    st.caption(f'MORE-event Brier score: {brier:.3f}. Lower is better; a push counts as not MORE. No hit-rate or ROI claim is made.')
    st.caption('Earliest saved forecast per player, game, market and model. Line-change snapshots are not counted as independent games.')
  else:st.caption('Insufficient verified historical data. Completed results have not been recorded yet.')
  for r in records[:25]:
   with st.expander(f"{r['player']} / {r['market']} / line {r['line']}"):
    st.write(f"Forecast {r['mean']:.1f} / actual {r['actual'] if r['actual'] is not None else 'Pending'}")
    st.caption(f"Saved {r['created_at']} / {r['model_version']}")
 with st.expander('Historical workload-model evaluation'):
  validation_view(report)


def render_board_results(url):
 with st.expander('All props / automatic board history',expanded=True):
  from storage import board_records,settle_board
  try:
   rows=board_records(url)
   st.write(f'{len(rows)} recorded line observations')
   st.caption('Collection runs when the live board loads. Repeat loads of the same fetched board are deduplicated. New source snapshots preserve line movement. Uploaded boards are excluded.')
   if st.button('Update all-prop outcomes'):
    import nflreadpy as nfl
    from core import prepare_stats
    seasons=sorted({int(r['game_id'].split('_')[0]) for r in rows if r['actual'] is None})
    if seasons:
     n=settle_board(url,prepare_stats(nfl.load_player_stats(seasons).to_pandas()),nfl.load_schedules(seasons).to_pandas())
     st.success(f'{n} outcomes added. Missing stats stay pending; games must be at least 36 hours past kickoff.')
     rows=board_records(url)
    else: st.info('No pending board observations.')
   if rows:
    view=pd.DataFrame([{'Player':r['player'],'Market':r['market'],'Line':r['line'],'Observed':r['board_fetched_at'],'Actual':r['actual'],'Status':'Pending' if r['actual'] is None else 'MORE' if r['actual']>r['line'] else 'LESS' if r['actual']<r['line'] else 'Push'} for r in rows])
    st.dataframe(view.head(200),hide_index=True,use_container_width=True)
    st.download_button('Export all-prop history',view.to_csv(index=False),'board_history.csv','text/csv')
    unique={}
    for r in sorted(rows,key=lambda x:x['board_fetched_at']):
     if r['actual'] is not None and r.get('baseline'): unique.setdefault((r['game_id'],r['player_id'],r['market']),r)
    if unique:
     comparisons=[]
     for r in unique.values():
      candidates=dict(r.get('baselines') or {})
      if r.get('baseline') and 'last10' not in candidates: candidates['last10']=r['baseline']
      if r.get('model'): candidates['workload_model']=r['model']
      for name,pred in candidates.items():
       if pred and 'mean' in pred and 'more' in pred:
        comparisons.append({'Market':r['market'],'Method':name,'Absolute error':abs(pred['mean']-r['actual']),'Brier':(pred['more']-float(r['actual']>r['line']))**2})
     if comparisons:
      metrics=pd.DataFrame(comparisons)
      leaderboard=metrics.groupby(['Market','Method']).agg(Games=('Absolute error','size'),MAE=('Absolute error','mean'),Brier=('Brier','mean')).reset_index().sort_values(['Market','MAE'])
      st.write('Model leaderboard / earliest observation per player, game and market')
      st.dataframe(leaderboard,hide_index=True,use_container_width=True)
      st.caption('MAE and Brier are computed only from completed, pregame observations. Lower is better. Methods may cover different player sets, so compare rows with similar sample sizes.')
      calibrated=[r for r in unique.values() if r.get('model') and r['actual'] is not None]
      if len(calibrated)>=10:
       cal=pd.DataFrame([{'Predicted MORE':float(r['model']['more']),'Observed MORE':float(r['actual']>r['line'])} for r in calibrated])
       cal['Band']=pd.cut(cal['Predicted MORE'],[0,.5,.6,.7,.8,.9,1.0],include_lowest=True)
       reliability=cal.groupby('Band',observed=False).agg(Games=('Observed MORE','size'),Predicted=('Predicted MORE','mean'),Observed=('Observed MORE','mean')).reset_index()
       st.write('Probability calibration / workload model')
       st.dataframe(reliability,hide_index=True,use_container_width=True)
       st.caption('Observed MORE rates should approach predicted rates as the sample grows. This is diagnostic evidence, not a betting guarantee.')
      else:
       st.caption(f'Calibration needs at least 10 completed workload predictions; {len(calibrated)} are available.')
   st.caption('Personal workload forecasts are listed below. Historical baseline probabilities are uncalibrated; collection does not automatically retrain a model.')
  except Exception:
   st.error('Automatic board history is unavailable. Check database access; no successful tracking is claimed.')
