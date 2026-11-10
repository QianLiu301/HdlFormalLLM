"""The words in the Step 1 and Step 2 boxes reach the model (D41).

No database and no API key: the mock model answers and records each prompt.
"""
import contextlib
import io
from pathlib import Path

import pytest

from src import design_templates as T
from src import prompt_store as P
from src import requirements_store as R
from src.simulation_runner import WebSimulationRunner
from src.xdb import trial as TR

WORDS = 'Keep the carry chain short, and report any metastable path together.'
LEAD = "The user's request for this step"
STEP0 = 'IMPORTANT - Confirmed Requirements and Decisions:'


@pytest.fixture
def web(monkeypatch, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    proj = tmp_path / 'proj'
    (proj / 'output').mkdir(parents=True)
    monkeypatch.setattr(main, 'PROJECT_ROOT', proj)
    monkeypatch.setattr(main, 'simulation_runner', WebSimulationRunner(project_root=str(proj)))
    monkeypatch.setattr('src.experiment_logger.DB_PATH', str(tmp_path / 'calls.db'))   # not the real log
    TR.install_mock()
    prompts = []
    answer = TR.MockLLM._call_api

    def recording(self, prompt, *a, **k):
        prompts.append(prompt)
        return answer(self, prompt, *a, **k)
    monkeypatch.setattr(TR.MockLLM, '_call_api', recording)

    def post(path, **body):
        prompts.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            r = main.app.test_client().post(path, json={'llm': 'mock', **body})
            r.get_data()                      # a stream runs while it is read
        assert r.status_code == 200, r.get_data(as_text=True)
        assert prompts, 'the model was not called'
        return prompts[-1]
    post.proj = proj
    return post


def saved_step0(proj, module_type='alu'):
    rec, err = R.build_record({'module_type': module_type, 'config': {'bitwidth': 32},
                               'requirements': [{'text': 'ADD adds', 'source': 'llm'}], 'questions': []})
    assert err is None
    return R.save(rec, proj).relative_to(proj).as_posix()


def test_no_words_no_block():
    assert P.user_request_block(None) == '' and P.user_request_block('  \n ') == ''


def test_the_block_keeps_the_words_as_typed():
    block = P.user_request_block(f'  {WORDS}\n- second line\n')
    assert LEAD in block and f'{WORDS}\n- second line\n\n' in block


@pytest.mark.parametrize('path', ['/api/generate', '/api/generate-stream'])
def test_step2_sends_the_box(web, path):
    desc = T.description('alu', {'bitwidth': 32})
    prompt = web(path, module_type='alu', bitwidth=32, input=desc)
    assert 'Contrast SLT against SLTU on the same operand pair' in prompt
    assert P.user_request_block(desc, for_scenarios=True) in prompt
    assert 'Do not invent new steps' in prompt and 'in the same clock cycle' in prompt


@pytest.mark.parametrize('path', ['/api/generate', '/api/generate-stream'])
def test_step2_step0_comes_after_the_box_and_wins(web, path):
    prompt = web(path, module_type='alu', bitwidth=32, input=WORDS,
                 requirements_filepath=saved_step0(web.proj))
    assert prompt.index(LEAD) < prompt.index(STEP0)


@pytest.mark.parametrize('design', ['alu', 'counter', 'regfile', 'cpu'])
def test_step1_sends_the_box_on_both_paths(web, design):
    body = dict(module_type=design, bitwidth=32, depth=32, pipeline_stages=5, input=WORDS)
    plain = web('/api/generate-hardware', **body)
    stream = web('/api/generate-hardware-stream', **body)
    block = P.user_request_block(WORDS)
    assert plain.endswith(block) and stream.endswith(block)
    assert 'Do not invent new steps' not in block


@pytest.mark.parametrize('path', ['/api/generate-hardware', '/api/generate-hardware-stream'])
def test_step1_step0_comes_after_the_box_and_wins(web, path):
    prompt = web(path, module_type='alu', bitwidth=32, input=WORDS,
                 requirements_filepath=saved_step0(web.proj))
    assert prompt.index(LEAD) < prompt.index(STEP0)


def test_a_repair_keeps_the_words_without_parsing_them_again(web):
    """A repair sends input ''. If its words were parsed, '8-bit' would change the width."""
    prompt = web('/api/generate-hardware', module_type='alu', bitwidth=32, input='',
                 user_request='8-bit style comments, please')
    assert P.user_request_block('8-bit style comments, please') in prompt
    assert '32-bit' in prompt and '8-bit ALU' not in prompt


def test_an_empty_box_leaves_the_prompts_as_they_were(web):
    assert LEAD not in web('/api/generate-hardware', module_type='alu', bitwidth=32, input='')


def test_a_generator_used_outside_the_web_flow_adds_nothing(tmp_path):
    from src.feature_generator_llm import FeatureGeneratorLLM
    TR.install_mock()
    with contextlib.redirect_stdout(io.StringIO()):
        g = FeatureGeneratorLLM(llm_provider='mock', project_root=str(tmp_path))
        prompt = g._create_prompt(g.parse_user_input(WORDS, module_type='alu', bitwidth=32))
    assert WORDS not in prompt and LEAD not in prompt


@pytest.mark.parametrize('text, expected', [
    # a box without a width leaves the selected one alone
    ('ALU with ADD only', {'bitwidth': None, 'module_type': 'alu', 'llm': None}),
    ('16-bit ALU', {'bitwidth': 16, 'module_type': 'alu', 'llm': None}),
    # sentences: words that merely contain a keyword change nothing
    ('Keep the reset value at zero', {'module_type': None, 'llm': None}),
    ('Report any metastable path', {'module_type': None, 'llm': None}),
    ('add the two operands together', {'module_type': None, 'llm': None}),
    # what the box was made for still works
    ('32x32 register file', {'module_type': 'regfile'}),
    ('5-stage RISC-V CPU', {'module_type': 'cpu'}),
    ('32-bit counter with deepseek', {'bitwidth': 32, 'module_type': 'counter', 'llm': 'deepseek'}),
    ('16位计数器', {'bitwidth': 16, 'module_type': 'counter'}),
    ('using Together AI', {'llm': 'together'}),
])
def test_what_the_box_changes_in_the_configuration(text, expected):
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    parsed = main.parse_hardware_natural_language(text)
    assert {k: parsed[k] for k in expected} == expected


def test_a_box_without_a_width_keeps_the_selected_width(web):
    prompt = web('/api/generate-hardware', module_type='alu', bitwidth=32, input='ALU with ADD only')
    assert '32-bit' in prompt and '16-bit' not in prompt
