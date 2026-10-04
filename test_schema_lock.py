import unittest
from unittest.mock import MagicMock
from storage import ensure_rls
class SchemaLockTests(unittest.TestCase):
 def test_enabled_rls_does_not_alter_table(self):
  conn=MagicMock();conn.execute.return_value.fetchone.return_value=(True,)
  ensure_rls(conn,'nfl_research_snapshots')
  self.assertEqual(conn.execute.call_count,1)
 def test_disabled_rls_is_enabled(self):
  conn=MagicMock();conn.execute.return_value.fetchone.return_value=(False,)
  ensure_rls(conn,'nfl_research_snapshots')
  self.assertEqual(conn.execute.call_count,3)
 def test_missing_table_rejected(self):
  conn=MagicMock();conn.execute.return_value.fetchone.return_value=None
  with self.assertRaises(RuntimeError):ensure_rls(conn,'nfl_research_snapshots')
if __name__=='__main__':unittest.main()
