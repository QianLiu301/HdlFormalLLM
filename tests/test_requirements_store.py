"""Step 0: reading the model's JSON reply (no database, no API key needed)."""
import contextlib
import io
import json

from src import requirements_store as R

GOOD = {'requirements': [{'id': 'REQ-1', 'text': 'ADD gives a + b'}],
        'questions': [{'id': 'Q1', 'question': 'Reset synchronous?', 'options': ['yes', 'no'],
                       'default': 0, 'why': 'timing of reset'}]}
# the kind of reply that failed in practice: a missing comma between objects,
# an unescaped quote inside a string, and a trailing comma
BROKEN = ('{"requirements": [{"id": "REQ-1", "text": "ADD gives a + b"} '
          '{"id": "REQ-2", "text": "the "zero" output is 1 when result is 0"}], '
          '"questions": [{"id": "Q1", "question": "Reset synchronous?", "options": ["yes", "no"], '
          '"default": 0, "why": "timing of reset"},]}')


def test_valid_reply_is_not_marked_repaired():
    a, err = R.parse_analysis(json.dumps(GOOD))
    assert err is None and a['json_repaired'] is False and len(a['requirements']) == 1


def test_broken_reply_is_repaired_and_marked():
    a, err = R.parse_analysis(BROKEN)
    assert err is None and a['json_repaired'] is True
    assert [r['text'] for r in a['requirements']] == ['ADD gives a + b',
                                                      'the "zero" output is 1 when result is 0']
    assert a['questions'][0]['options'] == ['yes', 'no']


def test_reply_without_json_is_an_error():
    a, err = R.parse_analysis('Sorry, I cannot help with that.')
    assert a is None and 'JSON' in err


def test_analyze_endpoint_retries_once(monkeypatch):
    with contextlib.redirect_stdout(io.StringIO()):
        import main

    class FakeLLM:
        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def _call_api(self, prompt, max_tokens=None, system_prompt=None):
            self.prompts.append(prompt)
            return self.replies.pop(0)

    def run(replies):
        fake = FakeLLM(replies)
        monkeypatch.setattr(main.LLMFactory, 'create_provider', staticmethod(lambda *a, **k: fake))
        r = main.app.test_client().post('/api/requirements/analyze', json={
            'llm': 'deepseek', 'module_type': 'alu', 'bitwidth': 32,
            'description': '32-bit ALU with ADD and SUB'})
        return r.status_code, r.get_json(), fake

    # first reply unreadable, second fine -> success after one retry
    code, body, fake = run(['no json at all', json.dumps(GOOD)])
    assert code == 200 and body['success'] and body['retried'] is True and len(fake.prompts) == 2
    assert 'could not be parsed' in fake.prompts[1]
    # the fixed overflow question is still added after a retry
    assert body['questions'][-1]['question'] == R.OVERFLOW_QUESTION
    # repairable reply -> no retry
    code, body, fake = run([BROKEN])
    assert code == 200 and body['json_repaired'] is True and body['retried'] is False and len(fake.prompts) == 1
    # unreadable twice -> clear error with advice
    code, body, fake = run(['nope', 'still nope'])
    assert code == 502 and 'choose another LLM' in body['error'] and len(fake.prompts) == 2
