import html
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st
from core import parse_board, summarize, history, historical_lean
from verification import attach_games, resolve_player, allowed_sides
from sources import foundation, board_source, play_history, stamp, depth_source, snap_source
from research import simulate, distribution, VERSION

st.set_page_config(page_title='NFL Prop Intelligence',page_icon='🏈',layout='centered')
st.markdown('''<style>
.stApp {background:#0b1018;color:#ecf2fa}
.block-container {max-width:850px;padding-top:1.2rem;padding-bottom:5rem}
h1 {font-size:2rem!important;letter-spacing:-.06rem} h3 {font-size:1.1rem!important}
header[data-testid="stHeader"] {background:#0b1018}
.card {background:#141e2c;border:1px solid #26364b;border-radius:18px;padding:18px;margin:12px 0}
.eyebrow {font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:#77dac6;font-weight:700}
.player {font-size:22px;font-weight:750;line-height:1.25;margin:8px 0}
.muted {color:#9babc0;font-size:13px;line-height:1.6}
.line {font-size:32px;font-weight:750;color:#f5f8fc;margin-top:8px}
.badge {color:#e8c78a;font-size:12px;font-weight:650;margin-top:8px}
button {min-height:44px} [data-testid="stRadio"] {background:#101925;border-radius:12px;padding:8px}
[data-testid="stRadio"] {position:fixed;bottom:0;left:0;right:0;max-width:820px;margin:auto;z-index:999;border:1px solid #26364b;padding-bottom:max(8px,env(safe-area-inset-bottom))}
[data-testid="stRadio"] label p {font-size:13px}
@media(max-width:640px){.block-container{padding:1rem .8rem 5rem} h1{font-size:1.65rem!important}.player{font-size:20px}.card{padding:15px} [data-testid="stRadio"] div[role="radiogroup"] {gap:6px} .stApp {overflow-x:hidden}}
</style>''',unsafe_allow_html=True)
LABELS={'targets':'Receiving targets','pass_yds':'Passing yards','rush_yds':'Rushing yards','rec_yds':'Receiving yards','receptions':'Receptions','rush_att':'Rush attempts','pass_td':'Passing touchdowns','rush_rec_yds':'Rush + receiving yards','pass_rush_yds':'Pass + rushing yards'}
def database_url():
    url=os.environ.get('DATABASE_URL')
    if url: return url
    try: return st.secrets.get('DATABASE_URL')
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError): return None

def esc(x): return html.escape(str(x or ''))

@st.cache_data(ttl=86400,max_entries=24,show_spinner=False)
def cached_sim(pbp,player_id,market,season,week):
    return simulate(pbp,player_id,market,season,week)

@st.cache_data(ttl=180,max_entries=4,show_spinner=False)
def verified_players(board,rosters,raw_by_id):
    issues=[]
    verified=[]
    for row in board.to_dict('records'):
        identity,reason=resolve_player(row,rosters)
        if reason: issues.append(f"{row['player']}: {reason}"); continue
        row.update(player_id=identity['gsis_id'],position=identity['position'],headshot_url=identity.get('headshot_url'),pfr_id=identity.get('pfr_id'),roster_status=identity.get('status'),team_verified=True)
        original=raw_by_id.get(str(row.get('projection_id')), {})
        row['sides']=allowed_sides(row['odds_type'],original.get('allowed_wager_types'))
        verified.append(row)
    return pd.DataFrame(verified),issues

def main():
    st.markdown('<div class="eyebrow">NFL / WEEKLY RESEARCH</div>',unsafe_allow_html=True)
    st.title('NFL Prop Intelligence')
    st.caption('Verified matchups. Real opportunity data. Evidence before confidence.')
    now=datetime.now(ZoneInfo('America/Chicago'))
    with st.expander('Slate & settings'):
        season=st.number_input('Season',2000,now.year+1,now.year if now.month>=3 else now.year-1)
        week=st.number_input('Week',1,18,1)
        n=st.slider('Historical games',5,25,10)
        timezone=st.selectbox('Timezone',['America/Chicago','America/New_York','America/Denver','America/Los_Angeles','UTC'])
        upload=st.file_uploader('Optional board JSON',type=['json'])
        if st.button('Refresh sources',use_container_width=True):
            board_source.clear(); foundation.clear(); play_history.clear(); depth_source.clear(); snap_source.clear(); cached_sim.clear()
    nav=st.radio('Navigate',['Props','Top picks','Player','Research','Results','Health'],horizontal=True,label_visibility='collapsed')
    if nav=='Results':
        from workload_ui import render_results
        render_results(database_url())
        return
    health=[]
    with st.spinner('Checking live board and player identities...'):
        try:
            if upload: raw=json.load(upload); fetched=stamp(); origin='Uploaded board'
            else: raw,fetched=board_source(); origin='PrizePicks'
            board,skips=parse_board(raw)
            health.append(dict(source=origin,status='Available',checked_at=fetched,rows=len(board)))
        except Exception as exc:
            board=pd.DataFrame(); skips={}; fetched=stamp()
            health.append(dict(source='PrizePicks',status='Unavailable',checked_at=fetched,error=str(exc)))
        data,source_health=foundation(int(season),int(week),include_history=nav in ('Props','Top picks','Player','Research'),include_usage=nav in ('Player','Top picks')); health.extend(source_health)
    issues=[]
    if not board.empty:
        board,issues=attach_games(board,data['schedule'],season,week)
    raw_by_id={str(x.get('projection_id')):x for x in raw if isinstance(x,dict) and x.get('projection_id')} if not board.empty else {}
    board,identity_issues=verified_players(board,data['rosters'],raw_by_id)
    issues.extend(identity_issues)
    if nav=='Health':
        st.subheader('System health')
        for status in health:
            with st.expander(f"{status['source']} / {status['status']}"):
                st.write(f"Checked: {status.get('checked_at','Unknown')}")
                st.write(f"Rows: {status.get('rows',0)}")
                if status.get('error'): st.error(status['error'])
        st.warning('Current injury reports, live routes, weather forecasts and sportsbook prices are not connected. No calibrated recommendations are enabled.')
        st.caption('Roster status is not a practice report or confirmation of game-day availability. Depth charts are timestamped observations, not guaranteed starters.')
        st.write('Database: configured; connection is tested when saving.' if database_url() else 'Database: not connected. Durable prediction history is not enabled.')
        st.write(f'Model: {VERSION} / live probability calibration: unavailable / agreement: not evaluated')
        try:
            from workload_ui import assets
            _,evaluation=assets()
            for kind,metrics in evaluation.get('metrics',{}).items():
                st.caption(f'Historical {kind} workload test: MAE {metrics["mae"]:.2f} vs baseline {metrics["baseline_mae"]:.2f} across {metrics["games"]:,} player-games; this is not a betting win rate.')
        except Exception:
            st.caption('Historical workload evaluation is unavailable.')
        st.write(f'Verified props: {len(board)} / rejected mappings and games: {len(issues)}')
        with st.expander('Import details'):
            st.write(skips)
            for issue in issues[:100]: st.caption(issue)
        st.markdown('[nflverse source and availability](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)')
        return
    if board.empty:
        st.info('No verified props available for this slate. Check Health for source or mapping issues.'); return
    if database_url() and not upload:
        try:
            from storage import save_board_snapshot
            from workload_ui import assets
            model_table,_=assets()
            saved_count=save_board_snapshot(database_url(), board, fetched, data['stats'],model_table,int(season),int(week))
            st.caption(f'Board tracking connected / {saved_count} new observations saved. Collection runs when the board loads.')
        except Exception:
            st.warning('Board tracking failed. These lines were not confirmed saved. Check Results and database connectivity.')
    st.caption(f"Week {week} / {len(board)} verified props / board checked {pd.Timestamp(fetched).tz_convert(timezone):%H:%M %Z}")
    if nav=='Top picks':
        st.subheader('Top picks')
        st.caption('Ranked with tested workload forecasts when available, historical baselines otherwise. Probabilities remain uncalibrated.')
        from workload_ui import assets
        model_table,_=assets()
        ranked=[]
        for _,r in board.iterrows():
            games=data['stats'][data['stats'].player_id.eq(r.player_id)] if not data['stats'].empty else pd.DataFrame()
            result=summarize(games,r.market,r.line,n)
            side={'Over':'over','Under':'under'}.get(result['side']) if result else None
            if not result or result['games']<5 or side not in r.sides: continue
            model=None
            if r.market in ('targets','rush_att'):
                from opportunity import forecast
                kind='targets' if r.market=='targets' else 'carries'
                model=forecast(model_table,r.player_id,r.team,kind,int(season),int(week))
            reference=float(model['mean']) if model else float(result['baseline'])
            model_edge=(reference-float(r.line))/max(float(r.line),1.0)
            direction='Over' if model_edge>0 else 'Under' if model_edge<0 else result['side']
            if model and direction.lower() not in r.sides: continue
            risk=[]
            snap_share=None
            if model and direction != result['side']:
                risk.append('model/history disagreement')
            if not data['snaps'].empty and pd.notna(r.get('pfr_id')):
                recent=data['snaps'][(data['snaps'].pfr_player_id.eq(r.pfr_id)) & data['snaps'].game_type.eq('REG')].sort_values(['season','week'],ascending=False).head(n)
                if not recent.empty:
                    snap_share=float(recent.offense_pct.mean())
                    if snap_share<.55: risk.append('low snap share')
            roster_state=str(r.get('roster_status') or '').strip().upper()
            if roster_state and roster_state not in ('ACT','ACTIVE'):
                risk.append(f'roster status {roster_state}')
            if not data['depth'].empty and pd.notna(r.get('player_id')):
                depth_rows=data['depth'][data['depth'].gsis_id.eq(r.player_id)]
                if not depth_rows.empty and pd.to_numeric(depth_rows.pos_rank,errors='coerce').min()>1: risk.append('not first on depth chart')
            score=abs(model_edge) * (0.70 if 'model/history disagreement' in risk else 0.85 if risk else 1.0)
            ranked.append((score,r,result,model,risk,reference))
        for score,r,result,model,risk,reference in sorted(ranked,key=lambda x:x[0],reverse=True)[:25]:
            label='MORE / OVER' if (('Over' if (model and reference>r.line) else result['side'])=='Over') else 'LESS / UNDER'
            source='workload model' if model else 'historical baseline'
            direction_key='more' if label=='MORE / OVER' else 'less'
            estimated_prob=float(model.get(direction_key, 0.0)) if model else float(result.get('side_hit_rate', 0.0))
            probability_text=f' / estimated {estimated_prob:.0%} {label.split(" /")[0].lower()}' if estimated_prob > 0 else ''
            flags=' / risk: '+', '.join(risk) if risk else ''
            snap_text=f' / recent snaps {snap_share:.0%}' if snap_share is not None else ''
            roster=str(r.get('roster_status') or 'unknown')
            tier='STRONGER RESEARCH SUPPORT' if result['games']>=10 and not risk else 'RESEARCH WATCH'
            st.markdown(f'''<div class="card"><div class="eyebrow">{esc(r.position)} / {esc(r.odds_type)}</div><div class="player">{esc(r.player)}</div><div class="muted">{esc(r.team)} vs {esc(r.opponent)} / {r.game_time.tz_convert(timezone):%a %b %d, %I:%M %p}</div><div class="line">{r.line:g} <span style="font-size:15px;font-weight:400">{esc(LABELS.get(r.market,r.market))}</span></div><div class="badge">{label} / {tier}</div><div class="muted">Projection {reference:.1f} / {source}{probability_text} / {result['games']} history games / roster {esc(roster)}{snap_text}{flags}</div><div class="muted">Not a validated recommendation</div></div>''',unsafe_allow_html=True)
        if not ranked: st.info('No props have enough history and an available historical side.')
        return
    if nav=='Props':
        st.markdown('### NFL board')
        st.caption('Cards show a historical MORE/LESS lean, not a validated prediction. Research contains experimental simulations. Confirm the exact line in PrizePicks.')
        search=st.text_input('Find a player',placeholder='Search player name')
        with st.expander('Filter position, market & line type'):
            position=st.selectbox('Position',['All']+sorted(board.position.dropna().unique()))
            market=st.selectbox('Market',['All']+sorted(board.market.unique()),format_func=lambda m:LABELS.get(m,m))
            line_type=st.selectbox('Line type',['Standard','All','Demon','Goblin'])
        view=board.copy()
        if search: view=view[view.player.str.contains(search,case=False,regex=False)]
        if position!='All': view=view[view.position.eq(position)]
        if market!='All': view=view[view.market.eq(market)]
        if line_type!='All': view=view[view.odds_type.str.lower().eq(line_type.lower())]
        view=view.sort_values(['game_time','player','market','line'])
        pages=max(1,(len(view)+11)//12)
        page=min(st.session_state.get('board_page',1),pages)
        if view.empty: st.info('No lines match these filters.')
        for _,r in view.iloc[(page-1)*12:page*12].iterrows():
            stats=data['stats']
            games=stats[stats.player_id.eq(r.player_id)] if not stats.empty else pd.DataFrame()
            lean,lean_detail=historical_lean(games,r.market,r.line,n,r.sides)
            side_text=' / '.join('MORE' if s=='over' else 'LESS' for s in r.sides) or 'Side availability unknown'
            st.markdown(f'''<div class="card"><div class="eyebrow">{esc(r.position)} / {esc(r.odds_type)}</div><div class="player">{esc(r.player)}</div><div class="muted">{esc(r.team)} {'vs' if r.home_away=='Home' else '@'} {esc(r.opponent)} / {r.game_time.tz_convert(timezone):%a %b %d, %I:%M %p}</div><div class="line">{r.line:g} <span style="font-size:15px;font-weight:400">{esc(LABELS.get(r.market,r.market))}</span></div><div class="muted">Feed sides: {esc(side_text)}</div><div class="badge">{esc(lean)}</div><div class="muted">{esc(lean_detail)}</div><div class="muted">Historical comparison / not a model pick</div></div>''',unsafe_allow_html=True)
        st.number_input('Page',1,pages,page,key='board_page')
        st.caption(f'{len(view)} matching lines. Sorted by kickoff and player, not historical hit rate.')
        export=board.drop(columns=['sides']).copy(); export['availability']='Mobile unverified'; export['recommendation']='PASS - validation incomplete'
        st.download_button('Export verified board',export.to_csv(index=False),'verified_nfl_board.csv','text/csv',use_container_width=True)
        return
    player=st.selectbox('Player',sorted(board.player.unique()))
    identity=board[board.player.eq(player)].iloc[0]
    stats=data['stats']; games=stats[stats.player_id.eq(identity.player_id)].copy() if not stats.empty else pd.DataFrame()
    if nav=='Player':
        if isinstance(identity.headshot_url,str) and identity.headshot_url.startswith('https://'): st.image(identity.headshot_url,width=90)
        st.subheader(player)
        st.caption(f'{identity.position} / {identity.team} / roster: {identity.roster_status}')
        depth=data['depth']
        if not depth.empty:
            entries=depth[depth.gsis_id.eq(identity.player_id)]
            for _,d in entries.iterrows(): st.write(f"Depth chart: {d.pos_abb}, rank {d.pos_rank} / observed {d.observed_at:%b %d %H:%M UTC}")
        snaps=data['snaps']
        if not snaps.empty:
            recent=snaps[snaps.pfr_player_id.eq(identity.pfr_id) & snaps.game_type.eq('REG')].sort_values(['season','week'],ascending=False).head(n)
            if not recent.empty:
                st.subheader('Recorded offensive snap share')
                st.line_chart(recent.sort_values(['season','week']).set_index('week').offense_pct)
                st.caption(f'{int(recent.season.max())} season history. Not a forecast of this week\'s playing time. Routes are not inferred from snaps.')
        if games.empty: st.info('No matched NFL history. No rookie estimate is fabricated.')
        else:
            with st.expander('Game log'):
                for _,g in games.head(n).iterrows():
                    st.write(f"{int(g.season)} Week {int(g.week)} / {g.get('team','')} vs {g.get('opponent_team','')}")
                    st.caption(f"Targets: {g.get('targets','Unavailable')} / Carries: {g.get('carries','Unavailable')} / Receiving yards: {g.get('receiving_yards','Unavailable')}")
        return
    st.subheader('Opportunity lab')
    st.caption('Experimental play-level volume and efficiency simulation. This is not the full-game, calibrated model described in the product roadmap.')
    props=board[board.player.eq(player)].reset_index(drop=True)
    choice=st.selectbox('Prop',list(props.index),format_func=lambda i:f"{LABELS.get(props.loc[i,'market'],props.loc[i,'market'])} / {props.loc[i,'line']} / {props.loc[i,'odds_type']}")
    row=props.loc[choice]
    if row.market in ('targets','rush_att','rec_yds','receptions','rush_yds'):
        from workload_ui import render_workload
        render_workload(row,int(season),int(week),fetched,database_url())
    line=st.number_input('What-if line',min_value=0.,value=float(row.line),step=.5,key=f'whatif_{player}_{row.market}_{row.line}')
    st.caption('Changing this line is hypothetical. It does not create an offer in PrizePicks.')
    with st.spinner('Loading play-level history...'):
        pbp,status=play_history(int(season)-1)
        run=cached_sim(pbp,row.player_id,row.market,int(season),int(week))
    if not run:
        st.info('Insufficient play-level history or this market has no research model yet.')
    else:
        out=distribution(run,line,row.sides)
        st.warning('Research only / Uncalibrated / PASS')
        st.metric('Simulated mean',f"{out['mean']:.1f}")
        st.write(f"Median {out['median']:.1f} / 10th-90th percentile {out['p10']:.1f}-{out['p90']:.1f}")
        st.write(f"Simulated MORE {out['more']:.1%} / LESS {out['less']:.1%} / Push {out['push']:.1%}")
        if 'under' not in row.sides: st.caption('LESS is not offered by this feed for this line; it is an outcome probability only.')
        if 'over' not in row.sides: st.caption('MORE is not offered by this feed for this line; it is an outcome probability only.')
        st.write(f"{run['opportunity_label']}: {run['expected_opportunities']:.1f} expected under unchanged-role assumption")
        counts,bins=np.histogram(run['samples'],bins=25)
        chart=pd.DataFrame({'Outcome':(bins[:-1]+bins[1:])/2,'Simulations':counts}).set_index('Outcome')
        st.bar_chart(chart)
        st.caption(f"Compared with line {line:g}. 20,000 draws / {run['recorded_games']} recorded games / {run['plays']} plays.")
        with st.expander('Assumptions and risks',expanded=True):
            for risk in run['assumptions']: st.write(risk)
        payload={**out,'player':player,'player_id':row.player_id,'game_id':row.game_id,'market':row.market,'line':line,'offered_line':row.line,'hypothetical':bool(line!=row.line),'model_version':VERSION,'created_at':stamp(),'board_fetched_at':fetched,'calibrated':False,'recommendation':'PASS'}
        if database_url() and st.button('Save pregame research snapshot',disabled=line!=row.line):
            try:
                from storage import save_snapshot
                save_snapshot(database_url(),payload,row.game_time.to_pydatetime())
                st.success('Pregame research snapshot saved. Original records are never updated by this app.')
            except Exception:
                st.error('Snapshot was not saved. Check database connectivity and permissions; no result is claimed.')
        st.download_button('Download research snapshot',json.dumps(payload,indent=2),'research_snapshot.json','application/json',use_container_width=True)
    with st.expander('Historical comparison'):
        if not games.empty:
            result=summarize(games,row.market,line,n,row.odds_type)
            if result:
                st.caption(f"Historical average {result['baseline']:.1f}. These are recorded outcomes, not probabilities.")
                for _,h in history(games,row.market,line,n).iterrows(): st.write(f"{int(h.season)} W{int(h.week)}: {h.value:g} / {h.result}")
    st.caption('See Results for workload-model evaluation. Betting probability calibration and model agreement are not yet available.')

if __name__=='__main__': main()
