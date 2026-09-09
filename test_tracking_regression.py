import unittest
from datetime import datetime,timezone,timedelta
from storage import snapshot_record
class TrackingRegression(unittest.TestCase):
 def test_same_observation_is_idempotent(self):
  now=datetime(2026,9,8,tzinfo=timezone.utc)
  p={'model_version':'board-v2','line':10,'board_fetched_at':now.isoformat()}
  a=snapshot_record(p,now+timedelta(days=1),now)
  b=snapshot_record(p,now+timedelta(days=1),now+timedelta(seconds=5))
  self.assertEqual(a[0],b[0])
  self.assertNotEqual(a[0],snapshot_record({**p,'line':11},now+timedelta(days=1),now)[0])
  self.assertEqual(a[0],snapshot_record({**p,'board_fetched_at':(now+timedelta(minutes=3)).isoformat()},now+timedelta(days=1),now)[0])
  self.assertNotEqual(a[0],snapshot_record({**p,'board_fetched_at':(now+timedelta(minutes=31)).isoformat()},now+timedelta(days=1),now)[0])
if __name__=='__main__':unittest.main()
