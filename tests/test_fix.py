"""Local fixes (D46): only the located lines change, and a change is kept only if it
compiles, breaks no example that passed, and is right on random inputs too.

The design is a real one (session 109, 2026-11-10, qwen3-coder-next, 32-bit ALU,
tests/data/qwen_alu_32): SLT compares the wrong way round (line 92) and SRA
shifts an unsigned operand, so it fills with zeros (lines 62-67).
"""
import json
import shutil
from pathlib import Path

import pytest

from src.xdb import fix as F

pytestmark = pytest.mark.skipif(not shutil.which('iverilog'), reason='needs iverilog')
DATA = Path(__file__).parent / 'data' / 'qwen_alu_32'
DESIGN = (DATA / 'alu.v').read_text(encoding='utf-8')
TB = (DATA / 'alu_tb.v').read_text(encoding='utf-8')
GOOD_SLT = [{'line': 92, 'new': "            4'b1000: temp_result = {31'b0, ($signed(a) < $signed(b))};  // SLT"}]


def _checks():
    idx = json.loads((DATA / 'alu_tb.scenarios.json').read_text(encoding='utf-8'))
    return [{'test_no': r['test'], 'tag': 'BDD:' + tag, 'op': r['operation'], 'width': r['width'],
             'ref': r['expected_ref'], 'used': r['expected_used']}
            for tag, rows in idx['trace']['detail'].items() for r in rows if r.get('port') == 'result']


@pytest.fixture(scope='module')
def base():
    return F.baseline(DESIGN, TB, _checks(), 32)


def test_the_lines_of_each_operation_are_found():
    assert [ln['no'] for ln in F.locate(DESIGN, 'SLT')] == [92]
    # SRA: the case item reads sra_stage0: the shifter stages, the shift amount, and their declarations
    assert [ln['no'] for ln in F.locate(DESIGN, 'SRA')] == [32, 33, 38, 62, 63, 64, 65, 66, 67, 91]
    # ADD: its own case item only, not every line that writes temp_result
    assert [ln['no'] for ln in F.locate(DESIGN, 'ADD')] == [76, 77, 78, 79, 80]


def test_the_baseline_matches_what_the_page_showed(base):
    checks = _checks()
    assert sum(base['scenario'].get(c['test_no']) == c['ref'] for c in checks) == 42 and len(checks) == 50
    assert base['random']['SLT'] > 0 and base['random']['SRA'] > 0 and base['random']['ADD'] == 0


def test_a_real_fix_is_accepted(base):
    new, err = F.apply_changes(DESIGN, GOOD_SLT, [92])
    ev = F.evaluate(DESIGN, new, TB, _checks(), 32, ['SLT'], base)
    assert ev['accepted'], ev['reason']
    assert (ev['scenario_right_before'], ev['scenario_right_after'], ev['scenario_newly_wrong']) == (42, 47, 0)
    assert ev['random_wrong_after'] == base['random']['SRA']           # only SRA is left
    assert new.splitlines()[:91] == DESIGN.splitlines()[:91]            # nothing else moved
    assert new.splitlines()[92:] == DESIGN.splitlines()[92:]


def test_a_fix_for_the_shown_examples_only_is_rejected(base):
    cheat = [{'line': 92, 'new': "            4'b1000: temp_result = (a == 0 && b == 0) ? 32'd0 : "
                                 "{31'b0, (a[31] ^ b[31]) ? (a[31] ? 1'b0 : 1'b1) : (a[31] ? (a < b) : (a >= b))};"}]
    new, _ = F.apply_changes(DESIGN, cheat, [92])
    ev = F.evaluate(DESIGN, new, TB, _checks(), 32, ['SLT'], base)
    assert not ev['accepted'] and 'still wrong' in ev['reason']


def test_a_change_that_breaks_another_operation_is_rejected(base):
    # line 33 is the shift amount, shared by SLL and SRL: halving it breaks them
    new, _ = F.apply_changes(DESIGN, [{'line': 33, 'new': '    assign shift_amt = {1\'b0, b[3:0]};'}],
                             [33, 62, 63, 64, 65, 66, 67, 91])
    ev = F.evaluate(DESIGN, new, TB, _checks(), 32, ['SRA'], base)
    assert not ev['accepted']
    assert 'were right are now wrong' in ev['reason'] or 'worse' in ev['reason']


def test_lines_outside_the_spot_are_refused():
    new, err = F.apply_changes(DESIGN, [{'line': 10, 'new': ''}], [92])
    assert new is None and 'not given: 10' in err


def test_a_change_that_does_not_compile_is_rejected(base):
    new, _ = F.apply_changes(DESIGN, [{'line': 92, 'new': "            4'b1000: temp_result = (;"}], [92])
    ev = F.evaluate(DESIGN, new, TB, _checks(), 32, ['SLT'], base)
    assert not ev['accepted'] and 'does not compile' in ev['reason']


def test_the_models_answer_is_read():
    p, err = F.parse_changes('Sure.\n{"changes": [{"line": 92, "new": "x"}], "why": "signed", "needs_other_lines": [33]}')
    assert err == '' and p['changes'] == [{'line': 92, 'new': 'x'}] and p['needs_other_lines'] == [33]
    assert F.parse_changes('no json here')[0] is None
