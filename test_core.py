import unittest
import pandas as pd
from core import *

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.games = prepare_stats(pd.DataFrame([dict(player_id='a', player_display_name='Test Player', season=2025, week=w, season_type='REG', passing_yards=200+w, rushing_yards=20, receiving_yards=3, receptions=2) for w in range(1,19)]))
    def test_combined_market(self):
        self.assertEqual(market_series(self.games, 'pass_rush_yds').iloc[0], 238)
        self.assertEqual(market_series(self.games, 'rush_rec_yds').iloc[0], 23)
    def test_all_lookbacks(self):
        for n in range(5,26):
            r = summarize(self.games, 'receptions', 2.5, n)
            self.assertEqual(r['side'], 'Under')
            self.assertEqual(r['side_hit_rate'], 1)
            self.assertEqual(r['games'], min(n,18))
    def test_more_only_lines(self):
        for odds in ['demon', 'goblin', ' DEMON ']:
            result = summarize(self.games, 'receptions', 3.5, 10, odds)
            self.assertEqual(result['side'], 'Pass (More-only)')
            self.assertEqual(result['side_hit_rate'], 0)
            self.assertEqual(result['under_rate'], 1)
            self.assertEqual(summarize(self.games, 'receptions', 1.5, 10, odds)['side'], 'Over')
        self.assertEqual(summarize(self.games, 'receptions', 3.5, 10, 'standard')['side'], 'Under')

    def test_pushes(self):
        r=summarize(self.games, 'receptions', 2, 7)
        self.assertEqual(r['push_rate'],1)
        self.assertEqual(r['over_rate'],0)
        self.assertEqual(r['under_rate'],0)
    def test_missing_history_alignment(self):
        self.games.loc[0,'receiving_yards']=None
        h=history(self.games,'rec_yds',2.5,5)
        self.assertEqual(h.iloc[0]['week'],17)
        self.assertEqual(len(h),5)
    def test_missing_not_zero(self):
        self.assertTrue(market_series(self.games, 'rush_att').isna().all())
    def test_no_surname_fallback(self):
        self.assertTrue(player_games(self.games,'Other Player').empty)
        self.assertEqual(normalize_name("D'Andre Swift Jr."),'dandre swift')
    def test_ambiguous_name(self):
        other=self.games.copy(); other['player_id']='b'
        self.assertTrue(player_games(pd.concat([self.games,other]),'Test Player').empty)
    def test_board(self):
        base=dict(player_name='Test Player',league='NFL',status='pre_game',stat_type='Pass+Rush Yds',line_score=200.5,start_time='2026-09-10T20:00:00-04:00')
        raw=[base,dict(base,odds_type='demon',line_score=250.5),dict(base,is_live=True),dict(base,league='NFL1H'),dict(base,line_score='NaN'),dict(base,start_time=None),dict(base,stat_type='Pass Yards in First 5 Attempts')]
        board,skips=parse_board(raw,now='2026-09-07T00:00:00Z')
        self.assertEqual(len(board),2)
        self.assertEqual(board.iloc[0].market,'pass_rush_yds')
        self.assertEqual(sum(skips.values()),5)
    def test_event_scope(self):
        base=dict(player_name='Jakobi Meyers',league='NFL',status='pre_game',stat_type='Receptions',line_score=3.5,start_time='2026-09-13T17:00:00Z',event_metadata_checked=True,event_type='team',game_id='NFL_game_example')
        rows=[base,dict(base,game_id='NFL_season_example'),dict(base,description='Season Total'),dict(base,event_type=None)]
        board,skips=parse_board(rows,now='2026-09-07T00:00:00Z')
        self.assertEqual(len(board),1)
        self.assertEqual(sum(skips.values()),3)

    def test_started_removed(self):
        base=dict(player_name='Test Player',league='NFL',status='pre_game',stat_type='Receptions',line_score=.5,start_time='2026-09-01T00:00:00Z')
        self.assertTrue(parse_board([base],now='2026-09-07T00:00:00Z')[0].empty)

if __name__=='__main__': unittest.main()
