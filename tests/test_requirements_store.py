"""Step 0: reading the model's JSON reply (no database, no API key needed)."""
import contextlib
import io
import json
import re
from pathlib import Path

from src import design_templates as T
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

        def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
            self.prompts.append(prompt)
            self.sampling = sampling
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
    assert fake.sampling == main.ANALYSIS_SAMPLING
    assert 'could not be parsed' in fake.prompts[1]
    # the fixed overflow question is still added after a retry
    assert body['questions'][-1]['question'] == R.OVERFLOW_QUESTION
    # repairable reply -> no retry
    code, body, fake = run([BROKEN])
    assert code == 200 and body['json_repaired'] is True and body['retried'] is False and len(fake.prompts) == 1
    # unreadable twice -> clear error with advice
    code, body, fake = run(['nope', 'still nope'])
    assert code == 502 and 'choose another LLM' in body['error'] and len(fake.prompts) == 2


def _question(text):
    return {'id': 'Q9', 'question': text, 'options': ['one', 'two'], 'default': 0, 'why': ''}


# Questions the model really asked (output/requirements, 2026-11): the first group is
# settled by the reference model, the second is not.
SETTLED_QUESTIONS = ['For SLL/SRL/SRA, should the shift amount be taken as B[4:0] (5 bits) or B[31:27] (high bits)?',
                     'For SRA on a negative operand, does sign-extension mean replicating the sign bit?',
                     'For SLT/SLTU, what should the result be when A == B?',
                     'How should the 32-bit result register be filled for comparison opcodes (SLT, SLTU)?',
                     'What shift amount should SRL and SRA use?']
OPEN_QUESTIONS = ['What should the ALU do for unused opcode values 1010 through 1111?',
                  'On reset (rst=1), should overflow and negative flags be 0?']


def test_points_the_reference_fixes_are_not_questions():
    a = {'requirements': [], 'questions': [_question(q) for q in SETTLED_QUESTIONS + OPEN_QUESTIONS]}
    out = R.add_settled(a, 'alu', 8)
    assert [q['question'] for q in out['questions']] == OPEN_QUESTIONS
    assert [q['id'] for q in out['questions']] == ['Q1', 'Q2']
    assert [f['id'] for f in out['settled']] == ['F1', 'F2', 'F3', 'F4', 'F5']
    assert out['settled'][0]['fact'] == 'A shift moves by the low 3 bits of B; the other bits of B are ignored.'
    assert 'wrap to 8 bits' in out['settled'][3]['fact']


def test_other_design_types_keep_their_questions():
    a = {'requirements': [], 'questions': [_question('What does SRA mean here?')]}
    assert R.add_settled(a, 'counter', 8) == a
    assert R.settled_facts('cpu', 32) == [] and 'nothing is fixed' in R.settled_lines('counter', 8)


def test_the_fixed_rules_stay_in_the_prompt_when_the_page_hides_them():
    """The page shows them only for an edited description; both generators always get them."""
    from src import prompt_store
    rec, err = R.build_record({'module_type': 'alu', 'config': {'bitwidth': 8}, 'source': 'template',
                               'requirements': [{'text': 'ADD adds', 'source': 'template'}], 'questions': []})
    assert err is None and len(rec['settled']) == 5
    assert 'Fixed by the reference model the tests are judged against:' in prompt_store.decisions_block(rec)


def test_the_settled_facts_are_what_the_reference_does():
    from src.testbench_generator import TestbenchGenerator as T
    ref = T._reference_result
    assert ref('SLL', 1, 33, 32) == 2                        # low 5 bits of 33 are 1
    assert ref('SRL', 0x80000000, 33, 32) == 0x40000000
    assert ref('SRA', 0x80000000, 1, 32) == 0xC0000000       # copies the sign bit
    assert ref('SLT', 5, 5, 32) == 0 and ref('SLTU', 5, 5, 32) == 0
    assert ref('SLT', 0xFFFFFFFF, 0, 32) == 1 and ref('SLTU', 0xFFFFFFFF, 0, 32) == 0
    assert ref('ADD', 0xFFFFFFFF, 1, 32) == 0 and ref('SUB', 0, 1, 32) == 0xFFFFFFFF


def test_both_generators_receive_the_settled_facts():
    from src import prompt_store
    rec, err = R.build_record({'module_type': 'alu', 'config': {'bitwidth': 16},
                               'requirements': [{'text': 'ADD adds', 'source': 'llm'}], 'questions': [],
                               'settled': [{'id': 'F1', 'fact': 'whatever the page sent'}]})
    assert err is None and rec['settled'] == R.settled_facts('alu', 16)
    block = prompt_store.decisions_block(rec)
    assert 'Fixed by the reference model the tests are judged against:' in block
    assert '- F1: A shift moves by the low 4 bits of B; the other bits of B are ignored.' in block
    old = {k: v for k, v in rec.items() if k != 'settled'}       # records saved before this change
    assert 'reference model' not in prompt_store.decisions_block(old)


# ---------------------------------------------------------------- the standard descriptions

CONFIGS = [('alu', {'bitwidth': 16}), ('counter', {'bitwidth': 8}), ('regfile', {'bitwidth': 32, 'depth': 16}),
           ('cpu', {'pipeline_stages': 3})]


def test_the_page_fills_in_the_same_descriptions():
    html = (Path(__file__).parents[1] / 'static/bdd_generator.html').read_text(encoding='utf-8')
    block = html[html.index('const BDD_TEMPLATES = {'):html.index('const BDD_EXAMPLES')]

    def page(name, **values):
        text = re.search(rf'\b{name}: \([^)]*\) => `(.*?)`', block, re.S).group(1)
        for k, v in values.items():
            text = text.replace('${' + k.replace('_log', 'Math.log2(bw)') + '}', str(v))
        return text.replace('${depth || 32}', str(values.get('depth'))).replace('${stages || 5}', str(values.get('stages')))

    assert page('alu', bw=16, _log=4) == T.description('alu', {'bitwidth': 16})
    assert page('counter', bw=8) == T.description('counter', {'bitwidth': 8})
    assert page('regfile', bw=32, depth=16) == T.description('regfile', {'bitwidth': 32, 'depth': 16})
    assert page('cpu', stages=3) == T.description('cpu', {'pipeline_stages': 3})


def test_an_unchanged_description_needs_no_model(monkeypatch):
    with contextlib.redirect_stdout(io.StringIO()):
        import main

    def no_model(*a, **k):
        raise AssertionError('a standard description must not call a model')
    monkeypatch.setattr(main.LLMFactory, 'create_provider', staticmethod(no_model))
    client = main.app.test_client()
    for module, config in CONFIGS:
        body = client.post('/api/requirements/analyze', json={
            'module_type': module, **config, 'description': T.description(module, config)}).get_json()
        assert body['success'] and body['source'] == 'template' and body['run_id'] is None, body
        assert body['requirements'] and all(r['source'] == 'template' for r in body['requirements'])
        again = client.post('/api/requirements/analyze', json={
            'module_type': module, **config, 'description': '  ' + T.description(module, config) + '\n'}).get_json()
        assert again == body                                # the same every time


def test_each_design_gets_its_own_decisions():
    def asked(module, config):
        a = T.standard_analysis(module, config, T.description(module, config))
        return R.add_fixed_questions(R.add_settled(a, module, config.get('bitwidth')), module)

    alu = asked('alu', {'bitwidth': 16})
    assert [q['question'] for q in alu['questions']] == [R.OVERFLOW_QUESTION]
    assert len(alu['settled']) == 5
    assert 'B[3:0]' in alu['requirements'][5]['text'] and 'result[15]' in alu['requirements'][10]['text']
    assert [q['id'] for q in asked('counter', {'bitwidth': 8})['questions']] == ['Q1', 'Q2', 'Q3']
    assert asked('regfile', {'bitwidth': 32})['questions'] == asked('cpu', {})['questions'] == []


def test_an_edited_description_goes_to_a_model():
    text = T.description('alu', {'bitwidth': 32}) + '\n- Also a rotate operation'
    assert T.standard_analysis('alu', {'bitwidth': 32}, text) is None
    assert T.standard_analysis('alu', {'bitwidth': 16}, T.description('alu', {'bitwidth': 32})) is None   # another width
    assert T.standard_analysis('other', {}, 'anything') is None


def test_the_alu_description_leaves_the_overflow_meaning_open():
    text = T.description('alu', {'bitwidth': 32}).lower()
    assert 'signed overflow' not in text and 'carry' not in text


def test_the_saved_record_says_no_model_was_involved():
    a = T.standard_analysis('regfile', {'bitwidth': 32}, T.description('regfile', {'bitwidth': 32}))
    rec, err = R.build_record({'module_type': 'regfile', 'config': {'bitwidth': 32}, 'source': 'template',
                               'requirements': a['requirements'], 'questions': []})
    assert err is None and rec['analysis_source'] == 'template'
    assert {r['source'] for r in rec['requirements']} == {'template'}

