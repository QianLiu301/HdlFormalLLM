"""
Regression: which ALU operation a Scenario Outline is mapped to.

When no tag names an operation, FeatureParser._get_alu_operation falls back to
the scenario's title and steps. That text is English, and six of the sixteen
mnemonics are English words (add, sub, and, or, not, nor). The fallback used to
match them as plain words, so

    @compare
    Scenario Outline: Compare signed and unsigned less-than on the same operands

was mapped to AND, and its Examples rows became AND tests that usually pass on
the golden design (a missing expected column defaults to 0, and x & 0 = 0).
The Gherkin step keyword `And` matched too: a scenario reduced to its Examples
table (ablation variant E) had every row turned into an AND test.

These tests pin the stricter rule (operation-like context only; ambiguous ->
unmapped), the unchanged tag path, and byte-identical testbenches for features
that were already mapped correctly.

Run from the repository root:  python -m pytest tests/test_alu_operation_mapping.py
No API key, database or simulator needed.
"""
import contextlib
import datetime as _dt
import io
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import testbench_generator as T  # noqa: E402

ABLATION = ROOT / 'artifact' / 'ablation'
DUT = {'module_type': 'alu', 'module_name': 'alu_32bit', 'bitwidth': 32}

COMPARE_TITLE = 'Compare signed and unsigned less-than on the same operands'
COMPARE_OUTLINE = f"""
  @compare
  Scenario Outline: {COMPARE_TITLE}
    Given I have operand A = <A>
    And I have operand B = <B>
    When I compare A and B both ways
    Then the signed result should be <Signed> and the unsigned result <Unsigned>

    Examples:
      | A          | B | Signed | Unsigned |
      | 4294967295 | 0 | 1      | 0        |
"""
ADD_OUTLINE = """
  @add @arithmetic
  Scenario Outline: Verify ADD operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the ADD operation with opcode 0000
    Then the result should be <Expected_Result>

    Examples:
      | A | B | Expected_Result |
      | 1 | 2 | 3               |
"""


def _feature(tmp_path, *outlines, name='alu.feature'):
    path = tmp_path / name
    path.write_text('Feature: 32-bit ALU Verification\n' + ''.join(outlines),
                    encoding='utf-8')
    return path


def _parse(path):
    return T.FeatureParser(str(path), module_type_override='alu').parse()


def _generate(path, out_dir):
    gen = T.TestbenchGenerator(output_dir=str(out_dir))
    with contextlib.redirect_stdout(io.StringIO()):
        return gen.generate_single(str(path), dict(DUT))


@pytest.fixture
def frozen_clock(monkeypatch):
    """The testbench header carries a timestamp; freeze it to compare bytes."""
    class Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1, 0, 0, 0)
    monkeypatch.setattr(T, 'datetime', Frozen)


# ---------------------------------------------------------------------------
# The reported case
# ---------------------------------------------------------------------------
def test_compare_scenario_is_unmapped_not_and(tmp_path):
    path = _feature(tmp_path, ADD_OUTLINE, COMPARE_OUTLINE)
    spec = _parse(path)

    assert [s['operation'] for s in spec['scenarios']] == ['ADD']
    assert spec['parse_stats']['scenarios_unmapped'] == 1

    res = _generate(path, tmp_path / 'tb')
    assert res['success']
    tb = Path(res['filepath']).read_text(encoding='utf-8')
    assert "opcode = 4'b0010" not in tb             # no AND stimulus at all
    assert tb.count("opcode = 4'b0000") == 1        # only the ADD row
    # Where the generator also records a per-scenario catalog
    # (*_tb.scenarios.json on the explainability branch), the scenario must be
    # listed as unmapped. Without a catalog this loop has nothing to check.
    for entry in res.get('scenario_catalog') or []:
        if entry.get('title') == COMPARE_TITLE:
            assert entry['status'] == 'unmapped'
            assert entry['operation'] is None


def test_feature_with_only_the_compare_scenario_yields_no_testbench(tmp_path):
    path = _feature(tmp_path, COMPARE_OUTLINE)
    res = _generate(path, tmp_path / 'tb')
    assert not res['success']
    assert res['parse_stats']['scenarios_unmapped'] == 1
    assert 'skipped' in res['error']


def test_gherkin_and_keyword_is_not_an_operation():
    """Ablation variant E keeps only the Examples tables and step keywords."""
    spec = _parse(ABLATION / 'variant_E_tags_steps_titles_blanked.feature')
    assert spec['scenarios'] == []
    assert spec['parse_stats']['scenarios_unmapped'] == spec['parse_stats']['outlines_total'] == 10


# ---------------------------------------------------------------------------
# Already-correct features: unchanged, byte for byte
# ---------------------------------------------------------------------------
# SLTU has 7 rows: until 2026-11-04 the parser dropped the last Examples row of
# a file without a final newline (all ablation variants), so 6 were counted.
BASELINE_OPS = {'ADD': 7, 'SUB': 7, 'AND': 6, 'OR': 6, 'XOR': 6,
                'SLL': 7, 'SRL': 7, 'SRA': 8, 'SLT': 7, 'SLTU': 7}


@pytest.mark.parametrize('variant', [
    'variant_B_steps_blanked',               # tags + titles left
    'variant_C_tags_removed',                # titles + When steps left
    'variant_D_tags_removed_steps_blanked',  # titles only
    'variant_F_titles_blanked',              # tags + When steps left
])
def test_ablation_variants_reproduce_the_baseline_testbench(tmp_path, frozen_clock, variant):
    base = _generate(ABLATION / 'variant_A_original.feature', tmp_path / 'a')
    other = _generate(ABLATION / f'{variant}.feature', tmp_path / 'v')
    assert base['success'] and other['success']
    assert Path(other['filepath']).read_bytes() == Path(base['filepath']).read_bytes()


def test_baseline_operation_counts():
    spec = _parse(ABLATION / 'variant_A_original.feature')
    assert Counter(s['operation'] for s in spec['scenarios']) == BASELINE_OPS
    assert spec['parse_stats']['scenarios_unmapped'] == 0


# ---------------------------------------------------------------------------
# The fallback rule, case by case
# ---------------------------------------------------------------------------
def _block(title, when=(), other=()):
    steps = ['Given I have operand A = <A>', *when, *other]
    return (f' {title}\n' + ''.join(f'    {s}\n' for s in steps)
            + '\n    Examples:\n      | A | B | Expected_Result |\n      | 1 | 2 | 3 |\n')


def _op(tag, block):
    return T.FeatureParser('unused.feature')._get_alu_operation(tag, block)[0]


@pytest.mark.parametrize('title, when, expected', [
    # upper-case mnemonic
    ('Verify ADD operation', [], 'ADD'),
    ('Verify SLTU operation (Unsigned Less Than)', [], 'SLTU'),
    ('Bitwise AND of all-ones with zero', [], 'AND'),
    ('Logical shift left (SLL) by 31', [], 'SLL'),
    # "<name> operation", any case
    ('Check the add operation near overflow', [], 'ADD'),
    ('Check the and operation on masks', [], 'AND'),
    ('Verify Xor Operation', [], 'XOR'),
    # When step, including an And that continues it, and opcode literals
    ('Edge cases', ['When I perform the SRA operation with opcode 0111'], 'SRA'),
    ('Edge cases', ['When I set the inputs', 'And I perform the SUB operation'], 'SUB'),
    ('Edge cases', ['When I apply opcode 1001'], 'SLTU'),
    ('Edge cases', ["When I apply opcode 4'b1000"], 'SLT'),
    ('Edge cases', ['When I apply opcode 0x4'], 'XOR'),
    # "<name> operation" outranks a bare mnemonic
    ('Verify SLTU operation (unsigned version of SLT)', [], 'SLTU'),
])
def test_operation_like_context_maps(title, when, expected):
    assert _op('', _block(title, when)) == expected


@pytest.mark.parametrize('title, when, other', [
    # English words and the step keyword
    (COMPARE_TITLE, ['When I compare A and B both ways'],
     ['Then the signed result should be <Signed> and the unsigned result <Unsigned>']),
    ('vvvvv', ['When yyyyy'], ['And wwwww', 'And wwwww']),
    ('Signed or unsigned comparison', [], []),
    ('Zero flag and negative flag behaviour', [], []),
    ('Add two numbers and check the flags', ['When I add A and B'], []),
    ('Sub-word boundary values', [], []),
    ('Verify reset and operation mode', [], []),
    ('Edge cases', ['When I apply the inputs', 'And operation completes in one cycle'], []),
    # capitalised for emphasis
    ('Result must NOT overflow for small operands', [], []),
    # Then steps describe outputs, not the action
    ('Edge cases', ['When I apply the inputs'],
     ['Then the result should be <R> AND the zero flag <Z>']),
    # ambiguous: more than one operation named
    ('Verify SLT against SLTU on the same operands', [], []),
    ('SRA vs SRL on negative operands', [], []),
    ('Edge cases', ['When I perform the ADD operation with opcode 0001'], []),
])
def test_english_or_ambiguous_text_stays_unmapped(title, when, other):
    assert _op('@compare', _block(title, when, other)) is None


def test_steps_of_a_following_plain_scenario_are_not_read(tmp_path):
    """block_content runs up to the next Scenario Outline; a plain Scenario in
    between must not lend its title to the outline before it."""
    plain = """
  Scenario: Verify ADD operation resets cleanly
    When I perform the ADD operation with opcode 0000
    Then nothing happens
"""
    spec = _parse(_feature(tmp_path, COMPARE_OUTLINE, plain))
    assert spec['scenarios'] == []
    assert spec['parse_stats']['scenarios_unmapped'] == 1


@pytest.mark.parametrize('tag, expected', [
    ('@and @logic', 'AND'), ('@logic @and', 'AND'), ('@add_arithmetic', 'ADD'),
    ('@xor', 'XOR'), ('@sltu @compare', 'SLTU'), ('@arithmetic @sub', 'SUB'),
])
def test_tag_path_is_unchanged(tag, expected):
    # The title would map elsewhere (or nowhere); the tag decides.
    assert _op(tag, _block('Compare signed and unsigned values')) == expected


# ---------------------------------------------------------------------------
# Rows that carry their own Operation column
# ---------------------------------------------------------------------------
ROWS_OUTLINE = f"""
  @compare
  Scenario Outline: {COMPARE_TITLE}
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the <Operation> operation with opcode <Opcode>
    Then the result should be <Expected_Result>

    Examples:
      | Operation | A          | B | Opcode | Expected_Result |
      | SLT       | 4294967295 | 1 | 1000   | 1               |
      | SLTU      | 4294967295 | 1 | 1001   | 0               |
      | LESS      | 1          | 2 | 1000   | 1               |
"""


def test_rows_with_their_own_operation_survive_an_unmappable_title(tmp_path):
    """The prompt template's SLT-vs-SLTU outline names the operation per row.
    The title naming none must not discard those rows; a row whose Operation
    cell is unrecognised is dropped rather than given an operation."""
    spec = _parse(_feature(tmp_path, ROWS_OUTLINE))
    assert [(s['operation'], s['opcode']) for s in spec['scenarios']] == [
        ('SLT', '1000'), ('SLTU', '1001')]
    assert spec['parse_stats']['scenarios_unmapped'] == 0
    assert spec['parse_stats']['rows_unmapped'] == 1


def test_operation_column_with_no_recognisable_row_is_unmapped(tmp_path):
    outline = ROWS_OUTLINE.replace('| SLT       |', '| LT        |').replace(
        '| SLTU      |', '| LTU       |')
    spec = _parse(_feature(tmp_path, outline))
    assert spec['scenarios'] == []
    assert spec['parse_stats']['scenarios_unmapped'] == 1


def test_dropped_rows_are_reported_not_hidden(tmp_path):
    """A row dropped because its Operation cell is unrecognised must appear in
    the scenario results as skipped (with its reason), so the scenario reads
    'partly tested' rather than fully tested; a scenario all of whose rows are
    dropped says why, instead of 'no Examples rows'."""
    from src import scenario_results as SR
    out = _generate(_feature(tmp_path, ROWS_OUTLINE), tmp_path / 'tb')
    index = SR.load_index(out['filepath'])
    sc = SR.summarize(index, '')['scenarios'][0]
    rows = {r['id'].rsplit('/r', 1)[1]: r for r in sc['rows']}
    assert rows['3'] == {'id': rows['3']['id'], 'status': 'skipped',
                         'reason': 'row names no known ALU operation'}
    assert sc['rows_total'] == 3

    outline = ROWS_OUTLINE.replace('| SLT       |', '| LT        |').replace(
        '| SLTU      |', '| LTU       |')
    path = _feature(tmp_path, ADD_OUTLINE, outline, name='none.feature')
    out = _generate(path, tmp_path / 'tb2')
    entry = SR.load_index(out['filepath'])['catalog'][1]
    assert (entry['status'], entry['reason']) == ('unmapped', 'tag maps to no known ALU operation')
