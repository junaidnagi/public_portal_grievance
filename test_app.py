import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import app

class Checks(unittest.TestCase):
    def test_pdf_error(self):
        self.assertTrue(app.extract_pdf(b'invalid')[1])
    def test_readiness_retrieval(self):
        self.assertEqual(app.readiness(app.CHECKLIST[:4])['score'], 80)
        self.assertEqual(app.retrieve('electricity', [{'text':'electricity bill','page':2}])[0]['page'],2)
        self.assertEqual(app.retrieve('telecom',[{'text':'electricity'}]),[])
    def test_actual_crew_with_mock_provider(self):
        with patch('app.Groq'), patch.object(app.GroqLLM,'call',return_value='Final Answer: A cautious demonstration result.'):
            outputs=app.run_agents({'complaint':'electricity bill','audit':app.readiness([])},[],'test',app.DEFAULT_MODEL)
        self.assertEqual(len(outputs),6)
        self.assertEqual(outputs[3]['agent'],'Petition')
    def test_groq_errors(self):
        import httpx
        from groq import RateLimitError, AuthenticationError
        from types import SimpleNamespace
        with patch('app.Groq'):
            llm = app.GroqLLM('test', app.DEFAULT_MODEL)
        request = httpx.Request('POST', 'https://api.groq.com/')
        rate = RateLimitError('sensitive', response=httpx.Response(429, request=request, headers={'retry-after':'1'}), body={})
        good = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='done'))])
        llm.client.chat.completions.create.side_effect = [rate, good]
        with patch('app.time.sleep') as sleep:
            self.assertEqual(llm.call('hello'), 'done')
            sleep.assert_called_once_with(1)
        auth = AuthenticationError('sensitive', response=httpx.Response(401, request=request), body={})
        llm.client.chat.completions.create.side_effect = auth
        with self.assertRaisesRegex(RuntimeError, 'invalid') as error:
            llm.call('hello')
        self.assertNotIn('sensitive', str(error.exception))
    def test_analysis_results(self):
        from types import SimpleNamespace
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Final Answer: Demonstration output.'))])
        with patch('groq.Groq') as client:
            client.return_value.chat.completions.create.return_value = response
            at = AppTest.from_file('app.py')
            at.secrets['GROQ_API_KEY'] = 'test'
            at.run(timeout=40)
            at.sidebar.radio[0].set_value('New Complaint').run()
            at.text_input[0].set_value('Demo Citizen')
            at.text_input[1].set_value('Islamabad')
            at.text_area[0].set_value('My electricity bill is unusually high.')
            next(b for b in at.button if b.label=='Analyze Complaint').click().run(timeout=40)
            self.assertFalse(at.exception)
            self.assertEqual(len(at.tabs), 6)
            self.assertEqual(at.tabs[3].label, 'Complaint letter')
            self.assertTrue(any('Not assessed' in x.value for x in at.info))
    def test_classification(self):
        self.assertEqual(app.classify('internet bill is wrong', 'Electricity')['category'], 'Telecom')
        self.assertEqual(app.classify('My electricity bill is too high', 'Telecom')['category'], 'Electricity')
    def test_sqlite_isolation(self):
        import tempfile
        import storage
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory, patch.object(storage, 'DB_PATH', Path(directory)/'cases.sqlite3'):
            storage.save_case('owner-a', {'id':'PG-1', 'status':'Draft'})
            self.assertEqual(storage.load_cases('owner-b'), {})
            self.assertEqual(storage.load_cases('owner-a')['PG-1']['status'], 'Draft')
            storage.delete_case('owner-b', 'PG-1')
            self.assertIn('PG-1', storage.load_cases('owner-a'))
            storage.delete_case('owner-a', 'PG-1')
            self.assertEqual(storage.load_cases('owner-a'), {})
    def test_no_key_demo_and_readiness(self):
        at = AppTest.from_file('app.py').run(timeout=40)
        at.sidebar.radio[0].set_value('New Complaint').run()
        at.text_input[0].set_value('Demo Citizen')
        at.text_input[1].set_value('Islamabad')
        at.text_area[0].set_value('My electricity bill is Rs 45000.')
        next(b for b in at.button if b.label=='Analyze Complaint').click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 6)
        case_id = at.session_state['current_case']
        self.assertEqual(at.session_state['cases'][case_id]['mode'], 'Demo / template')
        at.sidebar.radio[0].set_value('Document preparation').run()
        for box in at.checkbox[:4]:
            box.set_value(True)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value, '80%')
        at.sidebar.radio[0].set_value('My Cases').run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.tabs), 6)
    def test_voice_validation_and_mock_transcription(self):
        from types import SimpleNamespace
        with self.assertRaises(ValueError):
            app.transcribe_audio(b'', 'test')
        with patch('app.Groq') as client:
            client.return_value.audio.transcriptions.create.return_value = SimpleNamespace(text='My electricity bill is wrong.')
            self.assertEqual(app.transcribe_audio(b'a'*200, 'test'), 'My electricity bill is wrong.')
    def test_media_classification(self):
        self.assertEqual(app.classify('A television channel broadcast objectionable content', 'Other / Unsure')['category'], 'Media / Broadcasting')
    def test_ui(self):
        at=AppTest.from_file('app.py').run(timeout=40)
        self.assertFalse(at.exception)
        at.sidebar.radio[0].set_value('New Complaint').run()
        next(b for b in at.button if b.label=='Analyze Complaint').click().run()
        self.assertIn('Enter a complaint',at.warning[0].value)
        self.assertEqual([x.label for x in at.text_input], ['Name', 'City'])
        self.assertEqual(at.text_area[0].label, 'Complaint description')
        self.assertEqual(at.selectbox[0].label, 'Complaint category')
        self.assertTrue(any(b.label=='Analyze Complaint' for b in at.button))
        for page in ['Home','Document preparation','My Cases','Regulations','Analytics','About']:
            at.sidebar.radio[0].set_value(page).run()
            self.assertFalse(at.exception,page)

if __name__=='__main__': unittest.main()
