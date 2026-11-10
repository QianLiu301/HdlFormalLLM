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
import re
import sqlite3
from pathlib import Path

import pytest
import requests

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

    def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
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


# ---- providers swallow their errors: a failed call does not always look empty ----
FALLBACK = 'Test ALU operation with various input values and verify correct output'


class SwallowingModel(FakeModel):
    """Like the GWDG providers: an error becomes a note and a fallback text."""

    def _call_api(self, prompt, max_tokens=None, system_prompt=None):
        if self._answers(max_tokens):
            return 'answer'
        X.record_call_meta(api_error='ValueError: empty content (finish_reason=length)')
        return FALLBACK

    def _call_api_stream(self, prompt, max_tokens=None, system_prompt=None):
        self._answers(max_tokens)
        X.record_call_meta(api_error='ValueError: empty content (finish_reason=length)')
        yield FALLBACK


X.instrument_class(SwallowingModel)


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} Client Error', response=self)

    def json(self):
        return self.body


def _answer(content, finish='stop'):
    return FakeResponse(200, {'model': 'm', 'choices': [{'finish_reason': finish, 'message': {'content': content}}]})


def _gwdg(monkeypatch, *responses):
    """Serve these responses to a GWDG provider; returns the request bodies."""
    from src import llm_providers as L
    seen, queue = [], list(responses)

    def post(url, headers=None, json=None, proxies=None, timeout=None):
        seen.append(json)
        return queue.pop(0)
    monkeypatch.setattr(L.requests, 'post', post)
    return seen


def test_a_fallback_text_after_an_api_error_is_a_failed_call(calls):
    import src.llm_providers  # noqa: F401  registers its fallback check
    m = SwallowingModel(needs=10 ** 9, finish='stop')
    assert m._call_api('p', max_tokens=6000) == FALLBACK
    assert m.budgets == [6000]
    assert [r[:2] for r in calls()] == [(0, 'ValueError: empty content (finish_reason=length)')]


def test_a_stale_api_error_does_not_fail_a_real_answer(calls):
    import src.llm_providers  # noqa: F401

    class Recovered(FakeModel):
        def _call_api(self, prompt, max_tokens=None, system_prompt=None):
            X.record_call_meta(api_error='Timeout on the first attempt')
            return 'module m; endmodule'
    X.instrument_class(Recovered)
    Recovered(needs=1)._call_api('p', max_tokens=100)
    assert [r[0] for r in calls()] == [1]


def test_budget_exhausted_behind_a_fallback_text_is_retried(calls):
    import src.llm_providers  # noqa: F401
    m = SwallowingModel(needs=8000)
    assert m._call_api('p', max_tokens=6000) == 'answer'
    assert m.budgets == [6000, 12000]
    assert [r[0] for r in calls()] == [0, 1]


def test_a_stream_that_already_sent_text_is_not_retried(calls):
    import src.llm_providers  # noqa: F401
    m = SwallowingModel(needs=10 ** 9)
    assert list(m._call_api_stream('p', max_tokens=6000)) == [FALLBACK]
    assert m.budgets == [6000]
    assert [r[0] for r in calls()] == [0]


def test_gwdg_empty_content_is_retried_with_a_larger_budget(calls, monkeypatch):
    from src import llm_providers as L
    seen = _gwdg(monkeypatch, _answer(None, 'length'), _answer('answer'))
    assert L.GlmProvider(api_key='k')._call_api('p', max_tokens=1000) == 'answer'
    assert [b['max_tokens'] for b in seen] == [1000, 2000]
    first, second = calls()
    assert first[0] == 0 and 'empty content' in first[1] and second[:2] == (1, None)


def test_only_glm_limits_its_reasoning(calls, monkeypatch):
    from src import llm_providers as L
    seen = _gwdg(monkeypatch, _answer('a'), _answer('b'), _answer('c'))
    L.GlmProvider(api_key='k')._call_api('p', max_tokens=100)
    L.QwenProvider3(api_key='k')._call_api('p', max_tokens=100)
    L.GemmaProvider(api_key='k')._call_api('p', max_tokens=100)
    assert [b.get('reasoning_effort') for b in seen] == ['low', None, None]
    assert [b['model'] for b in seen] == ['glm-5.3-flash', 'qwen3-coder-next', 'gemma-4-31b-it']


def test_the_providers_own_words_are_kept(calls, monkeypatch):
    from src import llm_providers as L
    _gwdg(monkeypatch, FakeResponse(403, {'error': {'message': 'Not authorized to access this Model: glm-4.7'}}))
    assert L.GlmProvider(api_key='k')._call_api('p', max_tokens=100) == FALLBACK
    assert calls()[0][:2] == (0, 'HTTPError 403: Not authorized to access this Model: glm-4.7')


def test_the_page_shows_why_a_model_could_not_answer(calls, monkeypatch):
    import main
    from src.alu_generator import ALUGenerator

    class Refused:
        model = 'fake'

        def _call_api(self, prompt, max_tokens=None, system_prompt=None):
            return FALLBACK

        def _call_api_stream(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
            X.record_call_meta(api_error='HTTPError 403: Not authorized to access this Model: glm-4.7')
            yield FALLBACK
    X.instrument_class(Refused)
    for cls in {ALUGenerator, main.ALUGenerator}:
        monkeypatch.setattr(cls, '_setup_llm', lambda self: Refused())
    r = main.app.test_client().post('/api/generate-hardware-stream', json={
        'module_type': 'alu', 'llm': 'qwen', 'bitwidth': 8, 'workflow_mode': 'implementation', 'run_id': 'refused'})
    assert 'QWEN could not answer: HTTPError 403: Not authorized to access this Model: glm-4.7' in r.get_data(as_text=True)


def _menu(select_id):
    html = (Path(__file__).resolve().parent.parent / 'static/bdd_generator.html').read_text(encoding='utf-8')
    block = re.search(rf'<select id="{select_id}".*?</select>', html, re.S).group(0)
    return re.findall(r'^\s*<option value="([^"]+)"', block, re.M)


@pytest.mark.parametrize('select_id', ['req-llm', 'hw-llm', 'bdd-llm'])
def test_the_menus_offer_no_retired_model(select_id):
    values = _menu(select_id)
    assert {'qwen', 'gemma', 'glm'} <= set(values)
    assert not {'gptoss', 'together'} & set(values)


def test_gemma_is_registered_and_gpt_oss_is_not(monkeypatch):
    from src import llm_providers as L
    monkeypatch.setenv('GEMMA_API_KEY', 'k')
    monkeypatch.delenv('GEMMA_MODEL', raising=False)
    p = L.LLMFactory.create_provider('gemma')
    assert type(p).__name__ == 'GemmaProvider' and p.model == 'gemma-4-31b-it'
    assert type(L.LLMFactory.create_provider('gptoss')).__name__ == 'LocalLLMProvider'


def test_gemma_can_use_the_qwen_key(monkeypatch):
    from src import llm_providers as L
    for name in ('GEMMA_API_KEY', 'ACADEMIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('QWEN_API_KEY', 'school')
    assert L.GemmaProvider().api_key == 'school'
