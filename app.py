import html
import os
import json
import time
from urllib.parse import quote_plus
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
*,*:before,*:after{box-sizing:border-box}html,body{overscroll-behavior-x:none;line-height:1.45;-webkit-text-size-adjust:100%;text-size-adjust:100%;scrollbar-gutter:stable;scroll-behavior:smooth}::selection{background:#2f6f70;color:#ffffff}input,textarea{caret-color:#77dac6}.stApp {background:#0b1018;color:#ecf2fa;color-scheme:dark;-webkit-tap-highlight-color:transparent;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased}.stSpinner>div{color:#77dac6}::-webkit-scrollbar{width:8px}::-webkit-scrollbar-track{background:#0b1018}::-webkit-scrollbar-thumb{background:#26364b;border-radius:999px}::-webkit-scrollbar-thumb:hover{background:#38506c}
.block-container {max-width:850px;padding-top:1.2rem;padding-bottom:5rem}
h1 {font-size:clamp(1.65rem,4vw,2rem)!important;letter-spacing:-.06rem} h2,h3{margin-top:1.15rem!important}h3 {font-size:1.1rem!important}hr{border-color:#26364b}[data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid #26364b;overflow-x:auto;-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain;scroll-padding-inline:12px;scrollbar-width:none}[data-baseweb="tab-list"]::-webkit-scrollbar{display:none}[data-baseweb="tab"]{color:#9babc0;padding:8px 10px;white-space:nowrap}[data-baseweb="tab"][aria-selected="true"]{color:#77dac6;border-bottom-color:#77dac6}a{color:#70b8ff;text-decoration:none}a:hover,a:focus-visible{text-decoration:underline;color:#9bd0ff}
header[data-testid="stHeader"] {background:#0b1018}
.card {background:#141e2c;border:1px solid #26364b;border-radius:18px;padding:18px;margin:12px 0;box-shadow:0 6px 18px rgba(0,0,0,.14);transition:transform .15s ease,box-shadow .15s ease}[data-testid="stDataFrame"]{border:1px solid #26364b;border-radius:12px;overflow:hidden}[data-testid="stMetric"]{background:#141e2c;border:1px solid #26364b;border-radius:14px;padding:10px 12px}[data-testid="stAlert"]{border-radius:12px}@media(hover:hover){.card:hover{transform:translateY(-1px);box-shadow:0 9px 24px rgba(0,0,0,.22)}}.card:focus-within{border-color:#38506c}[data-testid="stExpander"]{border:1px solid #26364b;border-radius:12px;overflow:hidden;background:#0f1622}[data-testid="stExpander"] summary{min-height:44px;padding:0 12px;cursor:pointer}[data-testid="stExpander"] summary:focus-visible{outline:3px solid #77dac6;outline-offset:-3px}[data-testid="stExpander"] details[open] summary{border-bottom:1px solid #26364b}
.eyebrow {font-size:11px;line-height:1.3;letter-spacing:.13em;text-transform:uppercase;color:#77dac6;font-weight:700}
.player {display:flex;align-items:center;gap:10px;font-size:22px;font-weight:750;line-height:1.25;margin:8px 0;overflow-wrap:anywhere}.player .pick-headshot{flex:0 0 56px;margin-right:0}
.muted {color:#9babc0;font-size:13px;line-height:1.6;overflow-wrap:anywhere}div[data-testid="stCaptionContainer"]{color:#9babc0;line-height:1.5}div[data-testid="stCaptionContainer"] p{margin:.25rem 0}[data-testid="stTextInput"] label p,[data-testid="stNumberInput"] label p,[data-testid="stSelectbox"] label p,[data-testid="stSlider"] label p{color:#ecf2fa;font-weight:600}
.line {display:flex;align-items:baseline;gap:6px;flex-wrap:wrap;font-size:32px;font-weight:750;color:#f5f8fc;margin-top:8px;font-variant-numeric:tabular-nums}
.badge {display:inline-flex;align-items:center;width:max-content;color:#e8c78a;background:#2b2f3d;border:1px solid #4b5363;border-radius:999px;padding:5px 9px;font-size:12px;font-weight:650;margin-top:8px}
.chips {display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.chip {display:inline-block;border-radius:999px;padding:4px 9px;font-size:11px;font-weight:700;letter-spacing:.02em}.chip-more {background:#123e39;color:#78e1cf}.chip-less {background:#26375a;color:#a9c7ff}.chip-type {background:#2b2f3d;color:#d8dce5}.chip-risk {background:#4a3020;color:#ffd18a}
.pick-headshot {display:block;width:56px;height:56px;aspect-ratio:1/1;border-radius:10px;object-fit:cover;vertical-align:middle;margin-right:9px;border:1px solid #38506c;background:#17263a}.pick-title {display:flex;align-items:center}
.prop-row {border-top:1px solid #26364b;margin-top:14px;padding-top:14px}.prop-summary {display:flex;align-items:center;justify-content:space-between;gap:12px}.prop-summary strong {font-size:26px}.prop-details summary {cursor:pointer;min-height:44px;display:flex;align-items:center;color:#9babc0;font-size:13px}.prop-details .muted {padding-bottom:8px}
button {min-height:44px;border-radius:10px!important;touch-action:manipulation}.stButton>button{background:#111b29;border:1px solid #38506c;color:#ecf2fa;font-weight:650;transition:border-color .15s ease,background .15s ease}.stButton>button:hover{background:#17263a;border-color:#77dac6;color:#ffffff}.stButton>button:active{transform:scale(.98)}[data-testid="stButton"] button[kind="primary"]{background:#159b8c;border-color:#77dac6;color:#ffffff}[data-testid="stButton"] button[kind="primary"]:hover{background:#1bb6a2}[data-testid="stButton"] button:disabled{opacity:.55;cursor:not-allowed} button:focus-visible,input:focus-visible,select:focus-visible,[role="combobox"]:focus-visible{outline:3px solid #77dac6!important;outline-offset:2px} [data-testid="stRadio"] {background:#101925;border-radius:12px;padding:8px}[data-testid="stCheckbox"] label{min-height:44px;align-items:center}[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-baseweb="select"]>div{background:#111b29;border:1px solid #26364b;color:#ecf2fa;border-radius:10px}\n[data-testid="stTextInput"]:focus-within input,[data-testid="stNumberInput"]:focus-within input,[data-baseweb="select"]:focus-within>div{border-color:#77dac6;box-shadow:0 0 0 2px rgba(119,218,198,.14)}[data-testid="stTextInput"] input::placeholder,[data-testid="stNumberInput"] input::placeholder{color:#8fa1b8;opacity:1}[data-baseweb="popover"]{background:#111b29;color:#ecf2fa}[data-baseweb="popover"] [role="listbox"]{max-height:50vh;overflow-y:auto;-webkit-overflow-scrolling:touch;overscroll-behavior:contain}[data-baseweb="popover"] [role="option"]{color:#ecf2fa;min-height:40px;display:flex;align-items:center}[data-baseweb="popover"] [role="option"]:hover{background:#17263a}
[data-testid="stRadio"] {position:fixed;bottom:0;left:0;right:0;max-width:820px;margin:auto;z-index:999;border:1px solid #26364b;box-shadow:0 -8px 24px rgba(0,0,0,.28);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);padding-bottom:max(8px,env(safe-area-inset-bottom))}
[data-testid="stRadio"] label p {font-size:13px;margin:0} [data-testid="stRadio"] label:has(input:checked) p{color:#77dac6;font-weight:700} [data-testid="stRadio"] label:has(input:checked){background:#17263a;border:1px solid #2f6f70;box-shadow:0 0 0 2px rgba(119,218,198,.12);border-radius:9px}
@media(prefers-contrast:more){.card,[data-testid="stExpander"],[data-testid="stRadio"]{border-color:#6f849d!important}.muted,div[data-testid="stCaptionContainer"]{color:#b8c7d8!important}}\n@media(prefers-reduced-motion:reduce){*,*::before,*::after{transition-duration:.01ms!important;animation-duration:.01ms!important;scroll-behavior:auto!important}}\n@media(max-width:380px){.line{font-size:28px}.player{font-size:19px}[data-testid="stRadio"] label p{font-size:11px}}\n@media(max-width:640px){[data-testid="stAlert"]{padding:.8rem 1rem}.block-container{padding:1rem .8rem calc(6.5rem + env(safe-area-inset-bottom))} [data-testid="stTextInput"] input,[data-testid="stSelectbox"]>div,[data-testid="stNumberInput"] input{min-height:46px} h1{font-size:1.65rem!important;margin-bottom:.25rem}.player{font-size:20px}.card{padding:15px;border-radius:16px}.muted{font-size:12px;line-height:1.45}.line{font-size:30px}.prop-summary strong{font-size:24px}[data-testid="stRadio"] div[role="radiogroup"]{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px}[data-testid="stRadio"] label{justify-content:center;min-height:34px;padding:3px 1px;white-space:nowrap;touch-action:manipulation;user-select:none}[data-testid="stRadio"]{left:.5rem;right:.5rem;width:auto;border-radius:14px 14px 0 0}.stApp{overflow-x:hidden}}
</style>''',unsafe_allow_html=True)
LABELS={'targets':'Receiving targets','pass_yds':'Passing yards','rush_yds':'Rushing yards','rec_yds':'Receiving yards','receptions':'Receptions','rush_att':'Rush attempts','pass_td':'Passing touchdowns','rush_rec_yds':'Rush + receiving yards','pass_rush_yds':'Pass + rushing yards'}
def database_url():
    url=os.environ.get('DATABASE_URL')
    if url: return url
    try: return st.secrets.get('DATABASE_URL')
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError): return None

def esc(x): return html.escape(str(x or ''))

def photo_markup(name, url):
    fallback=f'https://ui-avatars.com/api/?name={quote_plus(str(name or "NFL Player"))}&background=17263a&color=ffffff&bold=true&size=96'
    src=str(url or '').strip()
    if not src.startswith(('https://','http://')): src=fallback
    src=src.replace('http://','https://',1)
    return f'<img class="pick-headshot" src="{esc(src)}" loading="lazy" decoding="async" width="56" height="56" onerror="this.onerror=null;this.src=\'{esc(fallback)}\';" alt="">'

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
        # Keep both nflverse/NFL URLs and ESPN fallback photos. Some feeds return
        # http URLs or numeric ESPN IDs as floats, so normalize before rendering.
        headshot=None
        photo_source='unavailable'
        for name in ('headshot_url','headshot','headshot_url_https'):
            value=identity.get(name)
            if pd.notna(value):
                candidate=str(value).strip()
                if candidate.startswith(('https://','http://')):
                    headshot=candidate.replace('http://','https://',1)
                    photo_source='nflverse'
                    break
        if not headshot:
            espn_raw=identity.get('espn_id')
            if pd.notna(espn_raw):
                espn_text=str(espn_raw).strip()
                if espn_text.endswith('.0'):
                    espn_text=espn_text[:-2]
                if espn_text.isdigit():
                    headshot=f'https://a.espncdn.com/i/headshots/nfl/players/full/{espn_text}.png'
                    photo_source='espn'
        # Always give the card a visible image slot when a provider omits a photo.
        # The initials image is a deterministic fallback; real NFL/ESPN photos win above.
        if not headshot:
            photo_source='avatar'
            headshot=f'https://ui-avatars.com/api/?name={quote_plus(str(row.get("player","NFL Player")))}&background=17263a&color=ffffff&bold=true&size=96'
        row.update(player_id=identity['gsis_id'],position=identity['position'],headshot_url=headshot,photo_source=photo_source,pfr_id=identity.get('pfr_id'),roster_status=identity.get('status'),team_verified=True)
        original=raw_by_id.get(str(row.get('projection_id')), {})
        row['sides']=allowed_sides(row['odds_type'],original.get('allowed_wager_types'))
        verified.append(row)
    return pd.DataFrame(verified),issues

def main():
    st.markdown('<div class="eyebrow">NFL / WEEKLY RESEARCH</div>',unsafe_allow_html=True)
    st.title('NFL Prop Intelligence')
    st.caption('Verified matchups. Real opportunity data. Evidence before confidence.')
    if st.button('Refresh board',use_container_width=True,type='primary'):
        board_source.clear(); foundation.clear(); play_history.clear(); depth_source.clear(); snap_source.clear(); cached_sim.clear()
        st.rerun()
    now=datetime.now(ZoneInfo('America/Chicago'))
    with st.expander('Slate & settings'):
        season=st.number_input('Season',2000,now.year+1,now.year if now.month>=3 else now.year-1)
        week=st.number_input('Week',1,18,1)
        n=st.slider('Historical games',5,25,10)
        load_board_history=st.checkbox('Load historical context on the Props board (slower)',value=False,key='board_history')
        timezone=st.selectbox('Timezone',['America/Chicago','America/New_York','America/Denver','America/Los_Angeles','UTC'])
        upload=st.file_uploader('Optional board JSON',type=['json'])
        if st.button('Refresh sources',use_container_width=True):
            board_source.clear(); foundation.clear(); play_history.clear(); depth_source.clear(); snap_source.clear(); cached_sim.clear()
    nav=st.radio('Navigate',['Props','Top picks','Results','Player','Research','Health'],horizontal=True,label_visibility='collapsed')
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
        data,source_health=foundation(int(season),int(week),include_history=load_board_history or nav in ('Top picks','Player','Research'),include_usage=nav in ('Player','Top picks') or (nav=='Props' and load_board_history)); health.extend(source_health)
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
        # Interactive rendering stays read-only; the background collector writes
        # complete board snapshots without delaying the mobile page.
        st.caption('Background board tracking is enabled; this page stays read-only for faster loading.')
    st.caption(f"Week {week} / {len(board)} verified props / board checked {pd.Timestamp(fetched).tz_convert(timezone):%H:%M %Z}")
    try:
        board_age=(pd.Timestamp.now(tz='UTC')-pd.to_datetime(fetched,utc=True)).total_seconds()/60
        if board_age>15:
            st.warning(f'Board data is {board_age:.0f} minutes old. Refresh before using a line.')
        elif board_age>5:
            st.caption(f'Board refreshed {board_age:.0f} minutes ago; confirm the live line before using it.')
    except (TypeError,ValueError):
        st.warning('Board freshness could not be verified. Confirm every line in PrizePicks.')
    if nav=='Top picks':
        st.subheader('Top picks')
        st.caption('Ranked with tested workload forecasts when available, historical baselines otherwise. Probabilities remain uncalibrated.')
        st.caption('Context limits: injuries, weather, live routes and game-script changes are not modeled.')
        pick_mode=st.selectbox('Show', ['Qualified research candidates','Full research watchlist'], index=0)
        st.caption('Qualified view requires at least 8 history games, a 5% projection edge, and 55% side support. These remain uncalibrated research signals.')
        from workload_ui import assets
        model_table,_=assets()
        ranked=[]
        stats_by_player={pid:grp for pid,grp in data['stats'].groupby('player_id')} if not data['stats'].empty else {}
        for _,r in board.iterrows():
            games=stats_by_player.get(r.player_id,pd.DataFrame())
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
            side_prob=float(model.get('more' if direction=='Over' else 'less',0.0)) if model else float(result.get('side_hit_rate',0.0))
            if pick_mode=='Qualified research candidates' and (result['games']<8 or abs(model_edge)<0.05 or side_prob<0.55): continue
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
            side_chip='MORE / OVER' if label.startswith('MORE') else 'LESS / UNDER'
            side_class='chip-more' if side_chip.startswith('MORE') else 'chip-less'
            risk_html=''.join(f'<span class="chip chip-risk">{esc(flag)}</span>' for flag in risk[:2])
            photo_url=str(r.get('headshot_url') or '')
            st.markdown(f'''<div class="card"><div class="eyebrow">{esc(r.position)} / {esc(r.odds_type)}</div><div class="player">{photo_markup(r.player, photo_url)}{esc(r.player)}</div><div class="muted">{esc(r.team)} vs {esc(r.opponent)} / {r.game_time.tz_convert(timezone):%a %b %d, %I:%M %p}</div><div class="line">{r.line:g} <span style="font-size:15px;font-weight:400">{esc(LABELS.get(r.market,r.market))}</span></div><div class="chips"><span class="chip {side_class}">{side_chip}</span><span class="chip chip-type">{esc(r.odds_type)}</span><span class="chip chip-type">{esc(roster)}</span>{risk_html}</div><div class="badge">{tier}</div><div class="muted">Projection {reference:.1f} / {source}{probability_text} / {result['games']} history games{snap_text}</div><div class="muted">Not a validated recommendation</div></div>''',unsafe_allow_html=True)
        if not ranked: st.info('No props have enough history and an available historical side.')
        return
    if nav=='Props':
        st.markdown('### NFL board')
        st.caption('Cards show a historical MORE/LESS lean, not a validated prediction. Search or filter first, then open a card for its evidence. Confirm the exact line in PrizePicks.')
        if not load_board_history:
            st.caption('History not loaded. Browse quickly or load analysis for historical comparisons.')
            def enable_board_history():
                st.session_state['board_history']=True
            st.button('Load analysis',on_click=enable_board_history,use_container_width=True)
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
        real_photos=int(view.get('photo_source',pd.Series(index=view.index)).isin(['nflverse','espn']).sum()) if 'photo_source' in view else 0
        st.caption(f'Player photos: {real_photos}/{len(view)} provider photos available; fallback avatars fill any missing images.')
        sort_order=st.selectbox('Sort props by',['Best available evidence','Kickoff time'],index=0)
        if sort_order=='Best available evidence': st.caption('Evidence order favors verified history, model context, and lower risk flags; it is not a win probability.')
        else: st.caption('Kickoff order groups props by game start time.')
        if sort_order=='Best available evidence':
            stats_for_sort=data['stats']
            stats_by_player={pid:grp for pid,grp in stats_for_sort.groupby('player_id')} if not stats_for_sort.empty else {}
            def evidence_score(row):
                games=stats_by_player.get(row.player_id,pd.DataFrame())
                result=summarize(games,row.market,row.line,n)
                if not result or result['games']<5: return -1.0
                edge=abs(float(result['baseline'])-float(row.line))/max(float(row.line),1.0)
                return float(result['games']) + min(edge,2.0)*5.0
            view=view.copy()
            view['_evidence_score']=view.apply(evidence_score,axis=1)
            view=view.sort_values(['_evidence_score','game_time','player'],ascending=[False,True,True])
        else:
            view=view.sort_values(['game_time','player','market','line'])
        # Keep a player's complete matchup together, retaining the sorted order
        # of the first (best-ranked) prop in each group.
        groups=list(view.groupby(['player_id','game_id'],sort=False))
        pages=max(1,(len(groups)+5)//6)
        page_key='grouped_board_page'
        st.session_state[page_key]=max(1,min(st.session_state.get(page_key,1),pages))
        page=st.number_input('Player page',1,pages,key=page_key)
        if view.empty: st.info('No lines match these filters.')
        for _,player_props in groups[(page-1)*6:page*6]:
            first=player_props.iloc[0]
            photo_value=first.get('headshot_url')
            photo_url=photo_value if isinstance(photo_value,str) else ''
            stats=data['stats']
            games=stats[stats.player_id.eq(first.player_id)] if not stats.empty else pd.DataFrame()
            context_bits=[]
            if load_board_history:
                snaps=data.get('snaps',pd.DataFrame())
                if not snaps.empty and pd.notna(first.get('pfr_id')) and 'pfr_player_id' in snaps:
                    recent=snaps[(snaps.pfr_player_id.eq(first.pfr_id)) & snaps.game_type.eq('REG')].sort_values(['season','week'],ascending=False).head(n)
                    if not recent.empty:
                        context_bits.append(f"recent snap share {float(pd.to_numeric(recent.offense_pct,errors='coerce').mean()):.0%}")
                depth=data.get('depth',pd.DataFrame())
                if not depth.empty and 'gsis_id' in depth:
                    ranks=pd.to_numeric(depth[depth.gsis_id.eq(first.player_id)].pos_rank,errors='coerce').dropna()
                    if not ranks.empty: context_bits.append(f"depth rank {int(ranks.min())}")
            context_text=' / '.join(context_bits) if context_bits else 'Role context unavailable'
            rows=[]
            for _,r in player_props.iterrows():
                if load_board_history:
                    lean,lean_detail=historical_lean(games,r.market,r.line,n,r.sides)
                else:
                    lean,lean_detail='History not loaded','Use Load analysis for historical comparisons.'
                side_text=' / '.join('MORE' if s=='over' else 'LESS' for s in r.sides) or 'Availability unknown'
                rows.append(f'<div class="prop-row"><div class="prop-summary"><span>{esc(LABELS.get(r.market,r.market))}</span><strong>{r.line:g}</strong></div><div class="chips"><span class="chip chip-type">{esc(side_text)}</span><span class="chip chip-type">{esc(r.odds_type)}</span></div><div class="badge">{esc(lean)}</div><details class="prop-details"><summary>View analysis</summary><div class="muted">{esc(lean_detail)}<br>Historical comparison / not a validated prediction. Confirm availability in PrizePicks.</div></details></div>')
            st.markdown(f'''<div class="card player-group"><div class="eyebrow">{esc(first.position)} / {esc(first.team)}</div><div class="player pick-title">{photo_markup(first.player,photo_url)}{esc(first.player)}</div><div class="muted">{esc(first.team)} {'vs' if first.home_away=='Home' else '@'} {esc(first.opponent)} / {first.game_time.tz_convert(timezone):%a %b %d, %I:%M %p} / roster: {esc(first.get('roster_status') or 'unknown')}</div><div class='muted'>Role context: {esc(context_text)}</div>{''.join(rows)}</div>''',unsafe_allow_html=True)
        st.caption(f'{len(groups)} player matchups / {len(view)} matching lines. Each player stays together; evidence sorting ranks groups by their strongest historical prop.')
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
