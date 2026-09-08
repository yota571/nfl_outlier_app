"""Research-only opportunity simulation. No calibrated recommendations."""
import hashlib
import numpy as np
import pandas as pd
VERSION='opportunity-research-0.1'

def simulate(pbp, player_id, market, before_season, before_week, n_sim=20000):
    # Targets include incompletions; carries include kneels to match official outcomes.
    configs={'rec_yds':('receiver_player_id','receiving_yards','Targets'), 'receptions':('receiver_player_id',None,'Targets'), 'rush_yds':('rusher_player_id','rushing_yards','Carries'), 'rush_att':('rusher_player_id',None,'Carries'), 'pass_yds':('passer_player_id','passing_yards','Attempts')}
    if market not in configs or pbp.empty: return None
    id_col,yards_col,label=configs[market]
    d=pbp[pbp.season_type.eq('REG') & ((pbp.season<before_season)|((pbp.season==before_season)&(pbp.week<before_week)))].copy()
    d=d[d[id_col].eq(player_id)]
    if label=='Attempts': d=d[d.pass_attempt.eq(1) & d.play_type.eq('pass')]
    elif label=='Carries': d=d[d.rush_attempt.eq(1)]
    else: d=d[d.play_type.eq('pass')]
    if d.game_id.nunique()<5 or len(d)<20: return None
    d=d.sort_values(['season','week','play_id'])
    games=d.game_id.drop_duplicates().tail(16)
    d=d[d.game_id.isin(games)]
    volume=d.groupby('game_id').size().to_numpy()
    mean=float(volume.mean()); variance=float(volume.var(ddof=1))
    seed=int.from_bytes(hashlib.sha256(f'{VERSION}|{player_id}|{market}|{before_season}|{before_week}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed)
    if variance>mean:
        shape=mean**2/(variance-mean); counts=rng.negative_binomial(shape,shape/(shape+mean),n_sim)
    else: counts=rng.poisson(mean,n_sim)
    if market=='rush_att': samples=counts.astype(float)
    else:
        if market=='receptions':
            outcomes=pd.to_numeric(d.complete_pass,errors='coerce').dropna().to_numpy()
        else:
            # An incomplete pass has zero yards by definition; a missing completed-play value is unknown.
            vals=pd.to_numeric(d[yards_col],errors='coerce')
            if label in ('Targets','Attempts'): vals=vals.mask(d.complete_pass.eq(0),0)
            outcomes=vals.dropna().to_numpy()
        if len(outcomes)<20: return None
        samples=np.zeros(n_sim)
        for i in range(int(counts.max())):
            mask=counts>i
            samples[mask]+=rng.choice(outcomes,int(mask.sum()))
    return dict(samples=samples,expected_opportunities=mean,opportunity_label=label,recorded_games=len(volume),plays=len(d),model_version=VERSION,assumptions=['Previous-season role assumed unchanged.', 'Games with no recorded opportunity are not represented.', 'Injuries, routes, opponent, weather and game script are not modeled.', 'Volume and per-play efficiency are sampled independently.', 'Research simulation is uncalibrated; not eligible for recommendations.'])

def distribution(run,line,sides):
    x=run['samples']; more=float(np.mean(x>line)); less=float(np.mean(x<line)); push=float(np.mean(x==line))
    return dict(mean=float(np.mean(x)),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p25=float(np.quantile(x,.25)),p75=float(np.quantile(x,.75)),p90=float(np.quantile(x,.9)),more=more,less=less,push=push,available_sides=', '.join(sides),tier='PASS - uncalibrated')
