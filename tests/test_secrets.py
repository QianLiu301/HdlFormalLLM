"""No credential may leave the machine (2026-11-04: a Gemini key reached the
public repository inside a 429 message quoted by a trial result).

Covers src/redact.py, the Gemini key in a header instead of the URL, the
error messages of the providers and of the server, the trial outputs and the
pre-commit check tools/check_secrets.py. No network, no database."""
import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from src.redact import known_secrets, redact_obj, redact_secrets  # noqa: E402

# shapes of real credentials, assembled at run time so that no line of this
# file looks like a key to a scanner (GitHub secret scanning, check_secrets)
FAKE_GOOGLE = 'AI' + 'za' + 'SyDq3k9TzW1mN8vB2' + 'xR7cL5pH4jF6gK0sYt'
FAKE_PW = 'npg_' + 's3cretValue'
NEON_URL = 'postgresql://owner:' + FAKE_PW + '@ep-x.neon.tech/db'
FAKE_ANT = 'sk-' + 'ant-api03-abcdefghijklmnopqrst'
FAKE_GROQ = 'gs' + 'k_abcdefghijklmnopqrstuv'
URL_429 = ('HTTPError: 429 Client Error: Too Many Requests for url: https://generativelanguage.googleapis.com'
           f'/v1beta/models/gemini-2.5-flash:generateContent?key={FAKE_GOOGLE}')


def test_patterns_mask_keys_and_passwords():
    assert FAKE_GOOGLE not in redact_secrets(URL_429, [])
    assert redact_secrets(URL_429, []).endswith(':generateContent?key=***')
    assert redact_secrets(NEON_URL, []) == \
        'postgresql://owner:***@ep-x.neon.tech/db'
    assert redact_secrets('Authorization: Bearer abcdefghijklmnopqrstuvwxyz', []) == 'Authorization: Bearer ***'
    assert redact_secrets(f'{FAKE_ANT} {FAKE_GROQ}', []) == 'sk-ant-*** gsk_***'


def test_ordinary_text_is_left_alone():
    for text in ('design line 12: error: key is not declared', 'assign y = a & b; // sk-learn',
                 'The model call failed: Timeout: no answer from DeepSeek within 300 s (2 attempts)'):
        assert redact_secrets(text, []) == text


def test_configured_keys_are_masked_whatever_their_format(tmp_path, monkeypatch):
    """Keys without a known prefix (Mistral, Together) are caught by value."""
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/llm_config.json').write_text(json.dumps(
        {'providers': {'mistral': {'api_key': 'Qm7xZt2LpV9wRk4N'}, 'groq': {'api_key': 'gsk_xxxx'}}}),
        encoding='utf-8')
    (tmp_path / '.env.save').write_text('DATABASE_URL=postgresql://o:Pw9secretPw9@h/db\n', encoding='utf-8')
    monkeypatch.setenv('TOGETHER_API_KEY', 'f00dfeedcafebabe1234')
    s = known_secrets(tmp_path)
    assert {'Qm7xZt2LpV9wRk4N', 'Pw9secretPw9', 'f00dfeedcafebabe1234'} <= set(s)
    assert 'gsk_xxxx' not in s                                       # the example placeholder
    text = 'mistral said Qm7xZt2LpV9wRk4N; together f00dfeedcafebabe1234'
    assert redact_secrets(text, s) == 'mistral said ***; together ***'
    assert redact_obj({'a': [text, 3, None]}, s) == {'a': ['mistral said ***; together ***', 3, None]}


def test_gemini_sends_its_key_in_a_header(monkeypatch):
    from src import llm_providers as LP
    sent = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'candidates': [{'content': {'parts': [{'text': 'module m; endmodule'}]}}]}

    def fake_post(url, **kw):
        sent.update(url=url, headers=kw.get('headers') or {})
        return _Resp()
    monkeypatch.setattr(LP.requests, 'post', fake_post)
    g = LP.GeminiProvider(api_key=FAKE_GOOGLE, model='gemini-2.5-flash')
    assert g._call_api_rest('hello', max_tokens=100).startswith('module')
    assert FAKE_GOOGLE not in sent['url'] and 'key=' not in sent['url']
    assert sent['headers'].get('x-goog-api-key') == FAKE_GOOGLE


def test_recorded_api_errors_hold_no_key(monkeypatch):
    from src import llm_providers as LP
    meta = {}
    monkeypatch.setattr(LP, '_record_meta', lambda **kw: meta.update(kw))
    g = LP.GeminiProvider(api_key=FAKE_GOOGLE, model='gemini-2.5-flash')
    g._note_api_error(RuntimeError(URL_429))
    assert FAKE_GOOGLE not in meta['api_error'] and '429' in meta['api_error']


def test_deepseek_timeouts_are_recorded_not_hidden(monkeypatch):
    """Before: three 60-s timeouts, then the fallback text and no reason at all."""
    from src import llm_providers as LP
    meta, calls = {}, []
    monkeypatch.setattr(LP, '_record_meta', lambda **kw: meta.update(kw))
    monkeypatch.setattr(LP.time, 'sleep', lambda s: None)

    def timeout(*a, **kw):
        calls.append(kw.get('timeout'))
        raise LP.requests.exceptions.Timeout()
    monkeypatch.setattr(LP.requests, 'post', timeout)
    d = LP.DeepSeekProvider(api_key='sk-' + 'x' * 30)
    with contextlib.redirect_stdout(io.StringIO()):
        out = d._call_api('write a module', max_tokens=100)
    assert LP.is_fallback_text(out) or 'module' not in out
    assert calls == [LP.DeepSeekProvider.TIMEOUT] * 2                 # one retry, not three 60-s waits
    assert meta['api_error'].startswith('Timeout: no answer from DeepSeek within 300 s')


def test_deepseek_does_not_retry_a_bad_key(monkeypatch):
    from src import llm_providers as LP
    meta, calls = {}, []
    monkeypatch.setattr(LP, '_record_meta', lambda **kw: meta.update(kw))
    monkeypatch.setattr(LP.time, 'sleep', lambda s: None)

    class _R:
        status_code = 402
        text = '{"error":{"message":"Insufficient Balance"}}'

    monkeypatch.setattr(LP.requests, 'post', lambda *a, **kw: calls.append(1) or _R())
    d = LP.DeepSeekProvider(api_key='sk-' + 'x' * 30)
    with contextlib.redirect_stdout(io.StringIO()):
        d._call_api('write a module', max_tokens=100)
    assert len(calls) == 1
    assert meta['api_error_status'] == 402 and 'Insufficient Balance' in meta['api_error']


def _main():
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    return main


def test_server_names_the_failed_call_not_the_empty_answer(monkeypatch):
    main = _main()

    class _Gen:
        last_error = "The model's answer (70 characters) contained no complete Verilog module."
    monkeypatch.setattr(main, '_last_call_meta', lambda run_id: {'api_error': URL_429})
    reason = main._model_failure_reason(_Gen(), 'r1')
    assert reason.startswith('The model call failed: HTTPError: 429') and FAKE_GOOGLE not in reason
    monkeypatch.setattr(main, '_last_call_meta', lambda run_id: {})
    assert main._model_failure_reason(_Gen(), 'r1') == _Gen.last_error


def test_server_error_responses_are_redacted():
    main = _main()
    resp = main.app.response_class(json.dumps({'success': False, 'error': URL_429}), status=500,
                                   mimetype='application/json')
    out = main._redact_error_responses(resp).get_data(as_text=True)
    assert FAKE_GOOGLE not in out and '429' in out


def test_trial_outputs_are_redacted_and_api_failures_counted(tmp_path):
    from src.xdb import trial as TR
    lost = {'tag': 't', 'llm': 'gemini', 'design': 'cpu', 'rep': 1, 'stage': 'duv', 'rounds': [],
            'error': f'design: The model call failed: {URL_429}'}
    bad = {'tag': 't', 'llm': 'gemini', 'design': 'cpu', 'rep': 2, 'stage': 'duv', 'rounds': [],
           'error': "design: The model's answer (900 characters) contained no complete Verilog module."}
    assert (TR.failure_kind(lost), TR.failure_kind(bad), TR.failure_kind({'error': None})) == ('api', 'model', None)
    assert TR.failure_kind({'error': "KeyError: 'initial'"}) == 'tool'
    summary = TR.write_outputs(tmp_path, [lost, bad])
    for f in ('runs.jsonl', 'runs.csv', 'summary.md'):
        assert FAKE_GOOGLE not in (tmp_path / f).read_text(encoding='utf-8'), f
    assert '| design ×1, design (API) ×1 |' in summary
    assert '1 run(s) stopped because the provider did not answer' in summary


# ------------------------------------------------------------- pre-commit check

def test_check_secrets_flags_keys_files_and_known_values():
    import check_secrets as C
    names = ['.env.save', 'config/llm_config.json', '.env.example', 'data/x.jsonl', 'tests/t.py', 'docs/d.md']
    lines = {
        'data/x.jsonl': [(4, json.dumps({'error': URL_429}))],
        'tests/t.py': [(1, "URL = 'postgresql://bdx:bdx@localhost/bdx_test'"),
                       (2, "m = 'Qm7xZt2LpV9wRk4N'")],
        'docs/d.md': [(1, 'DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx.neon.tech/db'),
                      (2, 'DATABASE_URL=postgresql://user:pass@host/db'),
                      (3, 'DATABASE_URL=' + NEON_URL)],
    }
    found = C.findings(names, lines, ['Qm7xZt2LpV9wRk4N'])
    where = sorted(f.split(': ')[0] for f in found)
    assert where == ['.env.save', 'config/llm_config.json', 'data/x.jsonl:4', 'docs/d.md:3', 'tests/t.py:2']
    assert not any(FAKE_GOOGLE in f or FAKE_PW in f or 'Qm7xZt2LpV9wRk4N' in f for f in found)
    # the files that define the patterns are exempt from them, never from the configured keys
    assert C.findings(['tests/test_secrets.py'], {'tests/test_secrets.py': [(1, 'Bearer ' + 'a' * 30)]}, []) == []
    assert C.findings(['src/redact.py'], {'src/redact.py': [(9, "k = 'Qm7xZt2LpV9wRk4N'")]},
                      ['Qm7xZt2LpV9wRk4N']) != []


@pytest.mark.parametrize('path', ['data/trials', 'src', 'tests', 'docs'])
def test_repository_holds_no_credentials(path):
    import check_secrets as C
    names, lines = C.tracked()
    names = [n for n in names if n.startswith(path)]
    assert C.findings(names, {n: lines[n] for n in names if n in lines}, []) == []
