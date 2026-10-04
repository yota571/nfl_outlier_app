"""Opportunity forecasts evaluated chronologically, conditional on offensive participation."""
from collections import defaultdict
import hashlib
import numpy as np
import pandas as pd
from verification import team_code
VERSION='team-share-v1'


def build_dataset(stats, snaps, rosters):
    stats=stats[stats.season_type.eq('REG')].copy()
    snaps=snaps[snaps.game_type.eq('REG') & (snaps.offense_snaps>0)].copy()
    for d in [stats,snaps]: d['team']=d.team.map(team_code)
    ids=rosters[['pfr_id','gsis_id']].dropna().drop_duplicates()
    counts=ids.groupby('pfr_id').gsis_id.nunique()
    ids=ids[ids.pfr_id.isin(counts[counts==1].index)]
    active=snaps.merge(ids,left_on='pfr_player_id',right_on='pfr_id',validate='many_to_one').rename(columns={'gsis_id':'player_id'})
    active=active[active.position.isin(['QB','RB','FB','WR','TE'])]
    keys=['season','week','team','player_id']
    active=active.drop_duplicates(keys)
    cols=keys+['targets','carries','receptions']
    if stats.duplicated(keys).any(): raise ValueError('Ambiguous official player-game rows')
    official=stats[cols].copy()
    for c in ['targets','carries','receptions']: official[c]=pd.to_numeric(official[c],errors='coerce')
    # Incomplete team stat columns are not converted to zero totals.
    totals=official.groupby(['season','week','team'])[['targets','carries']].agg(lambda x:x.sum() if x.notna().all() else np.nan).rename(columns={'targets':'team_targets','carries':'team_carries'}).reset_index()
    out=active.merge(official,on=keys,how='left',indicator=True,validate='one_to_one').merge(totals,on=['season','week','team'],how='inner',validate='many_to_one')
    # A snap-confirmed participant absent from a complete official game file has no recorded offensive stats.
    missing=out['_merge'].eq('left_only')
    for c in ['targets','carries','receptions']: out.loc[missing,c]=0
    out['zero_source']=np.where(missing,'Snap-confirmed; no official stat row','Official weekly counts')
    out=out.dropna(subset=['targets','carries','team_targets','team_carries'])
    out=out[(out.targets<=out.team_targets)&(out.carries<=out.team_carries)&(out.targets>=0)&(out.carries>=0)]
    return out[['game_id','season','week','team','opponent','player_id','player','position','offense_snaps','targets','carries','receptions','team_targets','team_carries','zero_source']].sort_values(['season','week','team','player_id']).reset_index(drop=True)


def forecast_records(history, team_history, kind, team):
    if kind not in ('targets','carries') or len(history)<5 or len(team_history)<5: return None
    prior=history[-10:]
    if prior[-1]['team']!=team: return None
    if sum(float(x[kind]) for x in prior)==0: return None
    current=[x for x in prior if x['team']==team]
    if len(current)<5: return None
    weights=np.power(.85,np.arange(len(current)-1,-1,-1))
    values=np.array([float(x[kind]) for x in current])
    volume=np.array([float(x['team_'+kind]) for x in current])
    if (volume<=0).any() or (values>volume).any(): return None
    # Jeffreys prior on opportunity share; fixed before evaluation, not tuned to held-out games.
    alpha=.5+float(np.dot(weights,values)); beta=.5+float(np.dot(weights,volume-values))
    team_vol=np.array(team_history[-8:],dtype=float)
    team_mean=float(team_vol.mean())
    return dict(mean=team_mean*alpha/(alpha+beta),baseline=float(np.mean([float(x[kind]) for x in prior])),team_volume=team_mean,alpha=alpha,beta=beta,history_games=len(current),kind=kind,model_version=VERSION)


def forecast(table,player_id,team,kind,season,week):
    earlier=table[(table.season<season)|((table.season==season)&(table.week<week))].sort_values(['season','week'])
    player=earlier[earlier.player_id.eq(player_id)].to_dict('records')
    team_rows=earlier[earlier.team.eq(team)].drop_duplicates(['season','week','team'])
    return forecast_records(player,team_rows['team_'+kind].tolist(),kind,team)


def sample_counts(prediction,seed_key,size=20000):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256((VERSION+seed_key).encode()).digest()[:8],'big'))
    volume=rng.poisson(prediction['team_volume'],size)
    shares=rng.beta(prediction['alpha'],prediction['beta'],size)
    counts=rng.binomial(volume,shares)
    return counts,volume


def walk_forward(table,test_season=2025):
    histories=defaultdict(list); teams=defaultdict(list); results=[]
    for (season,week),rows in table.sort_values(['season','week']).groupby(['season','week'],sort=True):
        # Predict every game in a week before adding that week's outcomes to history.
        for r in rows.to_dict('records'):
            if season==test_season:
                for kind in ['targets','carries']:
                    pred=forecast_records(histories[r['player_id']],teams[(r['team'],kind)],kind,r['team'])
                    if pred is not None:
                        results.append(dict(player_id=r['player_id'],player=r['player'],season=int(season),week=int(week),kind=kind,actual=float(r[kind]),prediction=pred['mean'],baseline=pred['baseline']))
        for r in rows.to_dict('records'): histories[r['player_id']].append(r)
        for r in rows.drop_duplicates(['season','week','team']).to_dict('records'):
            for kind in ['targets','carries']: teams[(r['team'],kind)].append(float(r['team_'+kind]))
    return pd.DataFrame(results)


def evaluate(results):
    report={}
    for kind,d in results.groupby('kind'):
        err=d.prediction-d.actual; base=d.baseline-d.actual
        report[kind]=dict(games=len(d),players=d.player_id.nunique(),mae=float(err.abs().mean()),baseline_mae=float(base.abs().mean()),rmse=float(np.sqrt((err**2).mean())),baseline_rmse=float(np.sqrt((base**2).mean())),bias=float(err.mean()),beats_baseline_mae=bool(err.abs().mean()<base.abs().mean()))
    return report
