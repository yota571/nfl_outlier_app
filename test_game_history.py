import unittest
import pandas as pd
from core import history, historical_lean, prepare_stats
from game_history import complete_history

class GameHistoryTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame([
            dict(player_id=p, player_display_name='Adam Trautman' if p=='a' else 'Teammate', team='DEN', season=s, week=w, season_type=t, receiving_yards=y)
            for s,w,t,p,y in [(2025,16,'REG','a',14),(2025,17,'REG','a',24),(2025,20,'POST','b',30),(2026,2,'REG','a',10),(2026,3,'REG','b',40)]
        ])
        self.schedule = pd.DataFrame([
            dict(season=s,week=w,game_type=t,game_id=f'{s}_{w}',gameday=d,home_team='DEN',away_team=o)
            for s,w,t,d,o in [(2025,16,'REG','2025-12-21','JAX'),(2025,17,'REG','2025-12-25','KC'),(2025,20,'DIV','2026-01-17','BUF'),(2026,2,'REG','2026-09-14','KC'),(2026,3,'REG','2026-09-27','LA'),(2026,4,'REG','2026-10-04','SF')]
        ])
        self.ids=pd.DataFrame([dict(pfr_id='pfr-a',gsis_id='a')])
        self.snaps=pd.DataFrame([dict(season=s,week=w,game_id=f'{s}_{w}',team='DEN',pfr_player_id='pfr-a',offense_snaps=20) for s,w in [(2025,20),(2026,3),(2026,4)]])
    def build(self):
        return complete_history(self.raw,self.snaps,self.ids,self.schedule,2026,4)
    def test_last_five_include_confirmed_zeros_and_playoff(self):
        games=self.build(); games=games[games.player_id.eq('a')]
        h=history(games,'rec_yds',4.5,5)
        self.assertEqual(h.value.tolist(),[0,10,0,24,14])
        self.assertEqual(h.opponent_team.tolist(),['LA','KC','BUF','KC','JAX'])
        self.assertEqual(h.gameday.dt.strftime('%Y-%m-%d').tolist(),['2026-09-27','2026-09-14','2026-01-17','2025-12-25','2025-12-21'])
        self.assertEqual(historical_lean(games,'rec_yds',4.5,5,('over','under'))[0],'Historical lean: MORE / OVER')
        self.assertIn('60%',historical_lean(games,'rec_yds',4.5,5,('over','under'))[1])
    def test_missing_team_file_is_not_zero(self):
        self.raw=self.raw[~((self.raw.season==2026)&(self.raw.week==3))]
        self.assertFalse(((self.build().season==2026)&(self.build().week==3)).any())
    def test_no_snaps_no_zero_and_dnp_not_inserted(self):
        self.snaps.loc[:,'offense_snaps']=0
        self.assertEqual(len(self.build().query("player_id == 'a'")),3)
    def test_ambiguous_identity_not_used(self):
        self.ids=pd.concat([self.ids,pd.DataFrame([dict(pfr_id='pfr-a',gsis_id='other')])])
        self.assertEqual(len(self.build().query("player_id == 'a'")),3)
    def test_unknown_column_stays_missing(self):
        games=self.build().query("player_id == 'a'")
        self.assertTrue(history(games,'pass_td',.5,5).empty)
    def test_null_official_value_not_changed_to_zero(self):
        self.raw.loc[(self.raw.season==2026)&(self.raw.week==3),'receiving_yards']=None
        games=self.build().query("player_id == 'a'")
        self.assertTrue(pd.isna(games.iloc[0].receiving_yards))
    def test_default_prepare_stats_remains_regular_for_settlement(self):
        self.assertTrue(prepare_stats(self.raw).season_type.eq('REG').all())

if __name__=='__main__': unittest.main()
