"""Current core regression checks; run with deployed project dependencies."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import app
import storage

class CoreChecks(unittest.TestCase):
    def test_confirmed_category_is_preserved(self):
        self.assertEqual(app.classify('internet bill is wrong', 'Electricity')['category'], 'Electricity')
        self.assertEqual(app.classify('electricity bill', 'Telecom')['category'], 'Telecom')
    def test_readiness_is_optional_and_deterministic(self):
        self.assertIsNone(app.readiness([])['score'])
        self.assertEqual(app.readiness(app.CHECKLIST[:4])['score'], 80)
    def test_recovery_key_storage_isolation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(storage,'DB_PATH',Path(folder)/'cases.sqlite3'):
            storage.save_case('one', {'id':'PG-1','status':'Draft'})
            self.assertEqual(storage.load_cases('two'), {})
            storage.delete_case('two','PG-1')
            self.assertIn('PG-1',storage.load_cases('one'))
            storage.delete_case('one','PG-1')
            self.assertEqual(storage.load_cases('one'),{})
    def test_removed_ai_check_stays_absent(self):
        self.assertFalse(hasattr(app,'check_ai_connection'))
        self.assertEqual(len(app.AGENT_SPECS),6)

if __name__=='__main__': unittest.main()
