"""Collect the currently active NFL slate without a Streamlit browser session."""
import os
import pandas as pd
from datetime import datetime,timezone
from core import parse_board
from sources import board_source,foundation
from verification import attach_games,resolve_player,allowed_sides
from storage import save_board_snapshot,settle_board,board_records
from pathlib import Path

def main():
    url=os.environ['DATABASE_URL']
    now=datetime.now(timezone.utc)
    season=now.year if now.month>=3 else now.year-1
    data,_=foundation(season,1)
    schedule=data['schedule']
    if schedule.empty: raise RuntimeError('Schedule unavailable')
    raw,fetched=board_source()
    parsed,_=parse_board(raw)
    originals={str(r.get('projection_id')):r for r in raw}
    model_table=pd.read_parquet(Path(__file__).parent/'model_assets'/'opportunity_history.parquet')
    saved=0
    for week in range(1,19):
        board,_=attach_games(parsed,schedule,season,week)
        if board.empty: continue
        history,_=foundation(season,week,include_history=True)
        records=[]
        for row in board.to_dict('records'):
            identity,reason=resolve_player(row,data['rosters'])
            if reason: continue
            row['player_id']=identity['gsis_id']
            row['sides']=allowed_sides(row['odds_type'],originals.get(str(row['projection_id']),{}).get('allowed_wager_types'))
            records.append(row)
        if records: saved+=save_board_snapshot(url,pd.DataFrame(records),fetched,history['stats'],model_table,season,week)
    print(f'Collected {saved} new verified observations')
    pending=board_records(url)
    seasons=sorted({int(r['game_id'].split('_')[0]) for r in pending if r['actual'] is None})
    settled=0
    if seasons:
        import nflreadpy as nfl
        from core import prepare_stats
        settled=settle_board(url,prepare_stats(nfl.load_player_stats(seasons).to_pandas()),nfl.load_schedules(seasons).to_pandas())
    print(f'Settled {settled} official outcomes')
if __name__=='__main__': main()
