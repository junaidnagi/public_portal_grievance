"""Pitch-feature checks. Transport and AI calls are mocked; no complaint is sent."""
import copy
import hashlib
import importlib.util
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
import zipfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


def load_app():
    class Object:
        def __init__(self, *args, **kw): self.__dict__.update(kw)
    modules = {}
    for name, attrs in {
        'crewai': dict(Agent=Object, BaseLLM=Object, Crew=Object, Task=Object, Process=types.SimpleNamespace(sequential='sequential')),
        'groq': dict(Groq=Object, APIConnectionError=RuntimeError, APIStatusError=RuntimeError, RateLimitError=RuntimeError),
        'rag': dict(retrieve=lambda *a, **kw: [], search_index=lambda *a, **kw: [], INDEX_DIR=Path('/tmp/no-feature-test-index')),
        'storage': dict(save_case=lambda *a: None, load_cases=lambda *a: {}, delete_case=lambda *a: None)
    }.items():
        module = types.ModuleType(name); module.__dict__.update(attrs); modules[name] = module
    spec = importlib.util.spec_from_file_location('pitch_feature_app', Path(__file__).parent/'app.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules): spec.loader.exec_module(module)
    return module


class PitchFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = load_app()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.settings = {'AUTOMATION_DB_PATH': self.temp.name+'/workflow.sqlite3', 'SUBMISSION_DB_PATH': self.temp.name+'/outbox.sqlite3',
            'SMTP_HOST': 'smtp.example.com', 'SMTP_USERNAME': 'test@example.com', 'SMTP_PASSWORD': 'test-only', 'SMTP_FROM': 'test@example.com'}
        self.app.secret = lambda key, default='': self.settings.get(key, default)
        self.app.st = types.SimpleNamespace(secrets={}, session_state={'recovery_token': 'a'*64, 'demo_mode': False},
            errors=types.SimpleNamespace(StreamlitSecretNotFoundError=FileNotFoundError), caption=lambda *a, **kw: None, warning=lambda *a, **kw: None)

    def case(self, identifier='PG-FEATURE-TEST'):
        app = self.app
        case = dict(id=identifier, name='Fictional Citizen', city='Islamabad', category='Electricity', company='IESCO',
            complaint='My electricity bill is incorrect and the previous complaint remains unresolved.', subject='Billing review',
            service_number='01234567890123', incident_date='2026-09-01', previous_reference='OFFICIAL-TEST-123', requested_resolution='Review and correct the bill.',
            profile={'email': 'citizen@example.com', 'phone': '03001234567', 'address': 'Fictional address', 'cnic': '12345-1234567-1'},
            authority='IESCO', escalation_authority='NEPRA', date='2026-10-04', status='Waiting for Response', reference='OFFICIAL-TEST-123',
            follow_up='2026-10-05', analytics_consent=False, sources=[], evidence=[], submission_target='company',
            mode='Local template', demo_case=False, audit=app.readiness([]), intake=app.classify('billing', 'Electricity'),
            escalation_checks={'no_response': True, 'no_parallel': True, 'route_applicable': True})
        case['outputs'] = app.demo_outputs(case, [])
        return case

    def test_label_bound_bill_parsing(self):
        fields = self.app.recognized_bill_fields('IESCO\nReference No: ۰۱۲۳۴۵۶۷۸۹۰۱۲۳\nIssue Date: 01/09/2026\nDue Date: 20-SEP-2026\nTotal Amount: PKR 45,000\nBilling Month: SEP-2026')
        self.assertEqual(fields['reference_number'], '01234567890123')
        self.assertEqual(fields['due_date'], '2026-09-20'); self.assertEqual(fields['bill_amount'], '45000')
        self.assertEqual(fields['company'], 'IESCO')
        fields = self.app.recognized_bill_fields('Identity 12345-1234567-1\nContact 03001234567\n45000')
        self.assertFalse(fields['reference_number']); self.assertFalse(fields['bill_amount'])

    @unittest.skipUnless(shutil.which('tesseract'), 'Tesseract is optional in the test host')
    def test_actual_local_ocr(self):
        picture = Image.new('RGB', (1500, 550), 'white'); draw = ImageDraw.Draw(picture)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 42)
        draw.multiline_text((40, 40), 'IESCO\nREFERENCE NO: 01234567890123\nDUE DATE: 20-09-2026\nTOTAL AMOUNT: PKR 45000', fill='black', font=font, spacing=24)
        data = io.BytesIO(); picture.save(data, 'PNG')
        result = self.app.bill_ocr(data.getvalue())
        self.assertEqual(result['fields']['reference_number'], '01234567890123')
        self.assertEqual(result['fields']['bill_amount'], '45000')
        self.assertEqual(result['image_sha256'], hashlib.sha256(data.getvalue()).hexdigest())
        with self.assertRaises(ValueError): self.app.bill_ocr(data.getvalue(), provider='groq', api_key='fake', consent=False)
        with self.assertRaises(ValueError): self.app.bill_ocr(b'not-an-image')

    def test_unicode_pdf_and_attachment_privacy(self):
        case = self.case()
        pdf = self.app.complaint_pdf(case, [], False)
        text = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
        self.assertIn(case['service_number'], text); self.assertNotIn(case['profile']['cnic'], text)
        identified = self.app.complaint_pdf(case, [], True)
        self.assertIn(case['profile']['cnic'], '\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(identified)).pages))
        case['complaint'] = 'میرے بجلی کے بل میں غلط رقم درج ہے۔ براہ کرم بل کی جانچ کریں اور غلط رقم درست کریں۔'
        case['outputs'][3]['text'] = self.app.template_letter(case)
        self.assertTrue(self.app.complaint_pdf(case).startswith(b'%PDF-'))
        package = zipfile.ZipFile(io.BytesIO(self.app.complaint_package(case, [], False)))
        self.assertIn('complaint.pdf', package.namelist()); self.assertIn('reminders.ics', package.namelist())

    def test_legal_dates_calendar_and_changed_facts(self):
        app = self.app; case = self.case()
        self.assertEqual(app.add_rule_period(date(2026, 1, 31), 'calendar_months', 3), date(2026, 4, 30))
        self.assertEqual(app.add_rule_period(date(2026, 10, 2), 'working_days', 1, ['2026-10-05']), date(2026, 10, 6))
        with self.assertRaises(ValueError): app.add_rule_period(date(2026, 1, 1), 'working_days', -1)
        case['deadline_plan'] = dict(rule_id='nepra_prior_15', start='2026-09-01', reference='OFFICIAL-TEST-123',
            confirmed=True, receipt_confirmed=True, category=case['category'], company=case['company'], facts_fingerprint=app.deadline_facts_fingerprint(case))
        self.assertEqual(app.active_deadline(case)['due'], '2026-09-16')
        self.assertIn('can replace it', app.active_deadline(case)['scope'].lower())
        calendar = app.calendar_reminders(case).decode(); self.assertIn('CLASS:PRIVATE', calendar)
        self.assertIn('DTSTART;VALUE=DATE:20260916', calendar); self.assertIn('Personal complaint follow-up', calendar)
        self.assertNotIn(case['name'], calendar); self.assertNotIn(case['profile']['cnic'], calendar)
        self.assertTrue(all(len(line.encode()) <= 75 for line in calendar.splitlines()))
        case['complaint'] += ' Changed facts.'; self.assertIsNone(app.active_deadline(case))

    def test_escalation_trigger_once_and_exact_transport(self):
        app = self.app; case = self.case(); owner = app.analytics_owner(); sent = []
        class SMTP:
            def __init__(self, *a, **kw): pass
            def login(self, *a): pass
            def send_message(self, message, **kw): sent.append((message, kw)); return {}
            def close(self): pass
        with patch.object(app.smtplib, 'SMTP_SSL', SMTP):
            app.authorize_escalation(case, 'regulator', [], False, 'email', True)
            self.assertEqual(app.process_escalations({case['id']: case}, owner, date(2026, 10, 4)), [])
            self.assertEqual(len(sent), 0)
            app.process_escalations({case['id']: case}, owner, date(2026, 10, 5))
            app.process_escalations({case['id']: case}, owner, date(2026, 10, 6))
        self.assertEqual(len(sent), 1); self.assertEqual(sent[0][1]['to_addrs'], [app.company_routes()['NEPRA']['email']])
        self.assertIn('OFFICIAL-TEST-123', sent[0][0].get_body().get_content())
        self.assertEqual(case['status'], 'Email sent'); self.assertEqual(case['reference'], '')
        self.assertEqual(case['submission_target'], 'regulator')

    def test_changes_closure_tampering_and_demo_block(self):
        app = self.app; owner = app.analytics_owner()
        for variant in ('edit', 'closed', 'tampered'):
            case = self.case('PG-' + variant); app.authorize_escalation(case, 'regulator', [], False, 'prepare', True)
            if variant == 'edit': case['profile']['email'] = 'changed@example.com'
            if variant == 'closed': case['status'] = 'Resolved'
            if variant == 'tampered':
                with app.feature_database() as db: db.execute('UPDATE escalation_jobs SET signature=? WHERE case_id=?', ('bad', case['id']))
            app.process_escalations({case['id']: case}, owner, date(2026, 10, 5))
            self.assertEqual(case['automatic_escalation']['state'], 'Needs review'); self.assertNotIn('escalation_packet', case)
        app.st.session_state['demo_mode'] = True
        with self.assertRaises(ValueError): app.authorize_escalation(self.case(), 'regulator', [], False, 'email', True)
        with self.assertRaises(ValueError): app.authorize_escalation(self.case(), 'regulator', [], False, 'prepare', False)

    def test_portal_preparation_and_ombudsman_eligibility(self):
        app = self.app; case = self.case(); case.update(category='Telecom', company='Ufone', service_number='03001234567')
        case['outputs'] = app.demo_outputs(case, [])
        app.authorize_escalation(case, 'regulator', [], False, 'prepare', True)
        app.process_escalations({case['id']: case}, app.analytics_owner(), date(2026, 10, 5))
        self.assertEqual(case['status'], 'Escalation Required'); self.assertNotIn('submission', case)
        self.assertEqual(app.receiving_organization(case['escalation_packet']['snapshot']), 'PTA')
        self.assertTrue(app.escalation_eligibility(case, 'mohtasib'))
        case['escalation_checks'].update(federal_maladministration=True, first_notice_date='2026-09-01',
            federal_respondent='Fictional federal agency', maladministration='The agency failed to examine the recorded complaint.')
        self.assertEqual(app.escalation_eligibility(case, 'mohtasib'), [])
        snapshot = app.escalation_snapshot(case, 'mohtasib')
        self.assertEqual(app.receiving_organization(snapshot), 'Wafaqi Mohtasib')
        self.assertEqual(app.dispatch_destination(snapshot), 'escalation:MOHTASIB')
        self.assertTrue(all('mohtasib.gov.pk' in p['url'] for p in app.portal_choices(snapshot)))
        self.assertIn('Fictional federal agency', app.complaint_body(snapshot))
        case['escalation_checks']['first_notice_date'] = '2026-01-01'
        self.assertTrue(any('period has passed' in p for p in app.escalation_eligibility(case, 'mohtasib')))

    def test_public_heatmap_consent_threshold_and_withdrawal(self):
        app = self.app; case = self.case(); case.update(public_city='Islamabad', public_analytics_consent=True)
        for i in range(3):
            case['id'] = 'PG-' + str(i); app.sync_public_analytics(case, 'owner-one')
        self.assertEqual(app.heatmap_cells(), [])
        for i in range(2):
            case['id'] = 'PG-owner-' + str(i); app.sync_public_analytics(case, 'owner-' + str(i))
        self.assertEqual(app.heatmap_cells()[0]['count'], 5)
        self.assertEqual(set(app.heatmap_cells()[0]), {'city', 'category', 'count', 'lat', 'lon'})
        case['public_analytics_consent'] = False; app.sync_public_analytics(case, 'owner-1')
        self.assertEqual(app.heatmap_cells(), [])
        case.update(public_analytics_consent=True, demo_case=True); app.sync_public_analytics(case, 'owner-1')
        self.assertEqual(app.heatmap_cells(), [])

    def test_worker_persists_private_case_without_browser(self):
        spec = importlib.util.spec_from_file_location('pitch_scheduler', Path(__file__).parent/'scheduler.py')
        scheduler = importlib.util.module_from_spec(spec); spec.loader.exec_module(scheduler)
        app = self.app; case = self.case(); case['follow_up'] = '2026-10-04'; owner = app.analytics_owner()
        app.authorize_escalation(case, 'regulator', [], False, 'prepare', True)
        database = Path(self.temp.name)/'cases.sqlite3'
        with sqlite3.connect(database) as db:
            db.execute('CREATE TABLE cases (owner TEXT,id TEXT,body TEXT,PRIMARY KEY(owner,id))')
            db.execute('INSERT INTO cases VALUES (?,?,?)', (owner, case['id'], json.dumps(case)))
        self.assertEqual(scheduler.run_once(app, database)['cases_updated'], 1)
        with sqlite3.connect(database) as db: saved = json.loads(db.execute('SELECT body FROM cases').fetchone()[0])
        self.assertEqual(saved['automatic_escalation']['state'], 'Prepared')
        self.assertEqual(scheduler.run_once(app, database)['cases_updated'], 0)


if __name__ == '__main__': unittest.main()
