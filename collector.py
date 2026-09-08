"""Collect the currently active NFL slate without a Streamlit browser session."""
import os
import pandas as pd
from datetime import datetime,timezone
from core import parse_board
from sources import board_source,foundation
from verification import attach_games,resolve_player,allowed_sides
from storage import save_board_snapshot,settle_board

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
        if records: saved+=save_board_snapshot(url,pd.DataFrame(records),fetched,history['stats'])
    print(f'Collected {saved} new verified observations')
    # Settlement is independently available in Results; no credentials are logged.
if __name__=='__main__': main()
