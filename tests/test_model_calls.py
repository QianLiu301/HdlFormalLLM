"""
Regression: an empty answer is a failed call, and one caused by the output
budget gets one more try.

All 7 empty answers of the trials of 2026-11-03/04 (6 DeepSeek, 1 gpt-5-mini) had
finish_reason 'length': a reasoning model used its whole budget before writing
anything. They were logged as successful calls, and the empty Step 0 of trial
cpu_v2 did not show in the summary although that run had no requirements.

Run from the repository root:  python -m pytest tests/test_model_calls.py
"""
import json
import sqlite3
from pathlib import Path

import pytest

from src import experiment_logger as X


class FakeModel:
    """Answers only with a budget of at least `needs` tokens."""
    model = 'fake'

    def __init__(self, needs, finish='length'):
        self.needs, self.finish, self.budgets = needs, finish, []

    def _answers(self, max_tokens):
        self.budgets.append(max_tokens)
        ok = max_tokens >= self.needs
        X.record_call_meta(finish_reason='stop' if ok else self.finish)
        return ok

    def _call_api(self, prompt, max_tokens=None, system_prompt=None):
        return 'answer' if self._answers(max_tokens) else ''

    def _call_api_stream(self, prompt, max_tokens=None, system_prompt=None):
        if self._answers(max_tokens):
            yield 'ans'
            yield 'wer'


X.instrument_class(FakeModel)


@pytest.fixture
def calls(tmp_path, monkeypatch):
    db = tmp_path / 'calls.db'
    conn = sqlite3.connect(db)
    conn.executescript(X._SCHEMA)
    conn.close()
    monkeypatch.setattr(X, 'DB_PATH', str(db))

    def rows():
        conn = sqlite3.connect(db)
        try:
            return [(ok, err, budget, json.loads(extra or '{}')) for ok, err, budget, extra in conn.execute(
                'SELECT success, error, max_tokens, extra FROM llm_calls ORDER BY id')]
        finally:
            conn.close()
    return rows


def test_an_answer_is_logged_as_before(calls):
    assert FakeModel(needs=10)._call_api('p', max_tokens=100) == 'answer'
    assert [r[:3] for r in calls()] == [(1, None, 100)]


def test_budget_exhausted_is_retried_with_twice_the_budget(calls):
    m = FakeModel(needs=8000)
    assert m._call_api('p', max_tokens=6000, system_prompt='s') == 'answer'
    assert m.budgets == [6000, 12000]
    first, second = calls()
    assert first[:3] == (0, 'empty answer (finish_reason=length)', 6000)
    assert second[:3] == (1, None, 12000) and second[3]['budget_retry'] is True
    assert second[3]['system_prompt_sha256'] == first[3]['system_prompt_sha256']


def test_the_prompt_details_stay_on_the_retry(calls):
    X.record_call_meta(prompt_stage='req_analysis')
    FakeModel(needs=8000)._call_api('p', max_tokens=6000)
    assert [r[3].get('prompt_stage') for r in calls()] == ['req_analysis', 'req_analysis']


def test_there_is_only_one_retry(calls):
    m = FakeModel(needs=10 ** 9)
    assert m._call_api('p', max_tokens=6000) == ''
    assert m.budgets == [6000, 12000]
    assert [r[0] for r in calls()] == [0, 0]


def test_the_budget_stops_at_the_limit(calls):
    m = FakeModel(needs=10 ** 9)
    m._call_api('p', max_tokens=20000)
    assert m.budgets == [20000, X.BUDGET_RETRY_CAP]
    m = FakeModel(needs=10 ** 9)
    m._call_api('p', max_tokens=X.BUDGET_RETRY_CAP)
    assert m.budgets == [X.BUDGET_RETRY_CAP]


def test_an_empty_answer_for_another_reason_is_failed_but_not_retried(calls):
    m = FakeModel(needs=10 ** 9, finish='stop')
    assert m._call_api('p', max_tokens=6000) == ''
    assert m.budgets == [6000]
    assert [r[:2] for r in calls()] == [(0, 'empty answer (finish_reason=stop)')]


def test_a_stream_retries_before_anything_was_sent(calls):
    m = FakeModel(needs=8000)
    assert list(m._call_api_stream('p', max_tokens=6000)) == ['ans', 'wer']
    assert m.budgets == [6000, 12000]
    assert [r[:3] for r in calls()] == [(0, 'empty answer (finish_reason=length)', 6000), (1, None, 12000)]


def test_step0_says_that_the_budget_ran_out(calls, monkeypatch):
    import main
    monkeypatch.setattr(main.LLMFactory, 'create_provider', staticmethod(lambda *a, **k: FakeModel(needs=10 ** 9)))
    monkeypatch.setattr(main, 'SITE_PASSWORD', '', raising=False)
    c = main.app.test_client()
    r = c.post('/api/requirements/analyze', json={'module_type': 'alu', 'llm': 'openai', 'bitwidth': 32,
                                                  'description': 'A 32-bit ALU'})
    body = r.get_json()
    assert r.status_code == 502 and not body['success']
    assert body['error'].startswith('The model used its whole output budget')


def test_the_trial_summary_shows_runs_without_requirements():
    from src.xdb.trial import summarize
    run = {'llm': 'openai', 'design': 'cpu', 'rep': 1, 'stage': 'done', 'error': None,
           'initial': {'verdicts': {'pass': 1}}, 'final': {}, 'rounds': [],
           'step0': {'ok': True}, 'llm_calls': 4, 'llm_calls_ok': 4}
    bad = dict(run, rep=2, step0={'ok': False, 'error': 'The model used its whole output budget'}, llm_calls_ok=3)
    text = summarize([run, bad])
    assert '| requirements |' in text and '| 1/2 |' in text
    assert '1 run(s) have no requirements because Step 0 failed' in text
    assert '- openai cpu #2: The model used its whole output budget' in text
    assert '1 model call(s) failed or came back empty' in text
