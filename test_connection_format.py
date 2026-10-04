import unittest
from unittest.mock import patch
from tracking import normalize_database_url,connect
class ConnectionFormatTests(unittest.TestCase):
 def test_accepted_formats_preserve_password(self):
  uri='postgresql://user:p%40ss%3Dword@localhost:5432/postgres'
  for value in (uri,' '+uri+' ', '"'+uri+'"', "'"+uri+"'", 'DATABASE_URL = "'+uri+'"', "DATABASE_URL = '"+uri+"'"):
   self.assertEqual(normalize_database_url(value),uri)
 def test_invalid_formats_have_safe_errors(self):
  for value in ('',None,'https://project.supabase.co','DATABASE_URL = secret','DATABASE_URL = "secret"\nEXTRA = "secret"','postgresql://secret\nmore'):
   with self.assertRaises(ValueError) as cm: normalize_database_url(value)
   self.assertNotIn('secret',str(cm.exception))
 def test_connect_normalizes_and_keeps_ssl(self):
  uri='postgresql://u:pw@localhost/postgres'
  with patch('psycopg.connect') as call:
   connect('DATABASE_URL = "'+uri+'"')
   call.assert_called_once_with(uri,connect_timeout=8,prepare_threshold=None,sslmode='require')
 def test_cannot_disable_ssl(self):
  with self.assertRaises(ValueError):connect('postgresql://u:p@localhost/postgres?sslmode=disable')
if __name__=='__main__': unittest.main()
