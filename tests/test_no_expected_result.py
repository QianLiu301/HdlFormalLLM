"""Regression tests: BDD rows that state no expected result produce no assertion.

Before this change the ALU and counter templates defaulted a missing expected
value to 0, so a row such as `@sltu | A | B | Signed | Unsigned |` became
`if (result == 32'd0)`: it passes on any design whose output is 0, inflates the
test count and dilutes the mutation score.

Run: python -m pytest tests/
"""
import datetime as _dt
import hashlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import testbench_generator as T  # noqa: E402

ALU = {'module_type': 'alu', 'module_name': 'alu_32bit', 'bitwidth': 32}
COUNTER = {'module_type': 'counter', 'module_name': 'counter_8bit', 'bitwidth': 8}
REASON = 'scenario states no expected result'


class _FrozenDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0)


def _generate(tmp_path, text, dut, name='t.feature', **kw):
    feature = tmp_path / name
    feature.write_text(text, encoding='utf-8')
    gen = T.TestbenchGenerator(output_dir=str(tmp_path / 'out'))
    res = gen.generate_single(str(feature), dut, **kw)
    tb = Path(res['filepath']).read_text(encoding='utf-8') if res['success'] else None
    return res, tb


def _asserts(tb):
    """Every `result == <const>` / `count == <const>` comparison in the TB."""
    return re.findall(r'if \((?:result|count) == ([^)]+)\)', tb)


ALU_HEAD = """Feature: 32-bit ALU

"""

SLTU_NO_EXPECTED = """  @sltu
  Scenario Outline: unsigned compare
    Given A is <A> and B is <B>
    Then the comparison is evaluated

    Examples:
      | A | B | Signed | Unsigned |
      | 1 | 2 | 1      | 1        |
      | 5 | 3 | 0      | 0        |
"""

ADD_WITH_EXPECTED = """  @add
  Scenario Outline: addition
    Given A is <A> and B is <B>
    Then the result is <Expected_Result>

    Examples:
      | A | B | Expected_Result |
      | 2 | 3 | 5               |
      | 7 | 1 | 8               |
"""


# ---------------------------------------------------------------- ALU, bdd oracle

def test_alu_row_without_expected_column_is_not_asserted(tmp_path):
    res, tb = _generate(tmp_path, ALU_HEAD + SLTU_NO_EXPECTED + ADD_WITH_EXPECTED, ALU)
    assert res['success']
    # only the two ADD rows are checked; no check against an invented 0
    assert _asserts(tb) == ["32'd5", "32'd8"]
    assert tb.count('total = total + 1;') == 2
    assert res['test_count'] == 2
    # numbering is not consumed by skipped rows
    assert '// Test 1' in tb and '// Test 2' in tb and '// Test 3' not in tb
    assert '// Test Cases: 2' in tb

    ev = res['quality']['expected_values']
    assert res['expected_values'] == ev
    assert ev['rows'] == 4 and ev['asserted'] == 2 and ev['skipped_no_expected'] == 2
    assert [s['row'] for s in ev['skipped']] == [1, 2]
    assert all(s['reason'] == REASON for s in ev['skipped'])
    assert all(s['operation'] == 'SLTU' for s in ev['skipped'])


def test_reason_string_defined_once_with_exact_text():
    assert T.TestbenchGenerator.NO_EXPECTED_RESULT == REASON
    src = (ROOT / 'src' / 'testbench_generator.py').read_text(encoding='utf-8')
    assert src.count(f"'{REASON}'") == 1


def test_alu_mixed_rows_only_rows_with_expected_are_asserted(tmp_path):
    # same table, some rows with a value in the expected column, one table without it
    text = ALU_HEAD + """  @and
  Scenario Outline: and with expected
    Examples:
      | A  | B | Result |
      | 12 | 10 | 8     |

  @and
  Scenario Outline: and without expected
    Examples:
      | A  | B |
      | 12 | 0 |
      | 7  | 0 |
"""
    res, tb = _generate(tmp_path, text, ALU)
    assert res['success']
    assert _asserts(tb) == ["32'd8"]
    ev = res['expected_values']
    assert (ev['asserted'], ev['skipped_no_expected']) == (1, 2)
    assert [s['row'] for s in ev['skipped']] == [2, 3]


def test_alu_all_rows_without_expected_fails_with_explanation(tmp_path):
    res, tb = _generate(tmp_path, ALU_HEAD + SLTU_NO_EXPECTED, ALU)
    assert not res['success']
    assert tb is None
    assert not list((tmp_path / 'out').rglob('*_tb.v'))
    assert 'state no expected result' in res['error']
    assert '2 Examples rows' in res['error']
    assert res['parse_stats']['rows_without_expected'] == 2
    assert res['expected_values']['asserted'] == 0


# --------------------------------------------------------------- ALU, spec oracle

def test_alu_spec_oracle_still_asserts_reference_value(tmp_path):
    res, tb = _generate(tmp_path, ALU_HEAD + SLTU_NO_EXPECTED, ALU, oracle_source='spec')
    assert res['success']
    # SLTU(1,2)=1, SLTU(5,3)=0 from the reference model
    assert _asserts(tb) == ["32'd1", "32'd0"]
    assert res['expected_values']['skipped_no_expected'] == 0


def test_alu_spec_oracle_skips_when_no_reference_and_no_expected(tmp_path):
    # MUL is outside the reference model's 10 operations
    text = ALU_HEAD + ADD_WITH_EXPECTED + """  @mul
  Scenario Outline: multiply
    Examples:
      | A | B |
      | 3 | 4 |

  @mul
  Scenario Outline: multiply with expected
    Examples:
      | A | B | Expected_Result |
      | 3 | 4 | 12              |
"""
    res, tb = _generate(tmp_path, text, ALU, oracle_source='spec')
    assert res['success']
    # ADD rows from the reference model, MUL row with expected from the BDD
    assert _asserts(tb) == ["32'd5", "32'd8", "32'd12"]
    ev = res['expected_values']
    assert ev['skipped_no_expected'] == 1
    assert ev['skipped'][0]['operation'] == 'MUL' and ev['skipped'][0]['row'] == 3


# ------------------------------------------------------------------------ counter

COUNTER_HEAD = """Feature: 8-bit counter

"""


def test_counter_row_without_expected_value_is_skipped(tmp_path):
    text = COUNTER_HEAD + """  @count_up
  Scenario Outline: up without expected
    Examples:
      | Initial_Value | Cycles |
      | 3             | 2      |

  @count_up
  Scenario Outline: up with expected
    Examples:
      | Initial_Value | Cycles | Expected_Value |
      | 10            | 4      | 14             |
"""
    res, tb = _generate(tmp_path, text, COUNTER)
    assert res['success']
    # the UP row without expected is gone; DOWN/UPDOWN defaults unchanged
    assert 'UP mode, initial=3' not in tb
    assert 'UP mode, initial=10, cycles=4' in tb
    assert _asserts(tb) == ["8'd14", "8'd95", "8'd0", "8'd0", "8'd255"]
    assert 'UPDOWN mode, initial=0, cycles=5' in tb
    assert 'UPDOWN mode, initial=100, cycles=10' in tb
    ev = res['expected_values']
    assert (ev['rows'], ev['asserted'], ev['skipped_no_expected']) == (2, 1, 1)
    assert ev['skipped'] == [{'row': 1, 'reason': REASON, 'mode': '00'}]


def test_counter_mode_whose_rows_are_all_skipped_gets_no_default_cases(tmp_path):
    text = COUNTER_HEAD + """  @count_down
  Scenario Outline: down without expected
    Examples:
      | Initial_Value | Cycles |
      | 9             | 2      |

  @count_up
  Scenario Outline: up with expected
    Examples:
      | Initial_Value | Cycles | Expected |
      | 1             | 1      | 2        |
"""
    res, tb = _generate(tmp_path, text, COUNTER)
    assert res['success']
    # neither the BDD DOWN row nor the generator's DOWN defaults are asserted:
    # defaults would credit the BDD with tests the generator wrote
    assert 'DOWN mode, initial=' not in tb.replace('UPDOWN mode', '')
    assert _asserts(tb) == ["8'd2"]
    assert res['expected_values']['skipped_no_expected'] == 1


def test_counter_updown_rows_are_kept(tmp_path):
    # the UPDOWN check is a range check that never reads the expected value
    text = COUNTER_HEAD + """  @count_updown
  Scenario Outline: updown
    Examples:
      | Initial_Value | Cycles |
      | 4             | 3      |
"""
    res, tb = _generate(tmp_path, text, COUNTER)
    assert res['success']
    assert 'UPDOWN mode, initial=4, cycles=3' in tb
    assert res['expected_values']['skipped_no_expected'] == 0


def test_counter_all_rows_without_expected_fails(tmp_path):
    text = COUNTER_HEAD + """  @count_up
  Scenario Outline: up without expected
    Examples:
      | Initial_Value | Cycles |
      | 3             | 2      |
"""
    res, _ = _generate(tmp_path, text, COUNTER)
    assert not res['success']
    assert 'state no expected result' in res['error']


def test_counter_default_cases_unchanged_without_bdd_rows():
    gen = T.TestbenchGenerator.__new__(T.TestbenchGenerator)
    spec = {'bitwidth': 8, 'scenarios': []}
    tb, qa = gen._generate_counter_testbench(spec, 'c', 'x')
    assert _asserts(tb) == ["8'd5", "8'd110", "8'd255", "8'd0",
                            "8'd95", "8'd0", "8'd0", "8'd255"]
    assert qa['expected_values']['skipped_no_expected'] == 0


# -------------------------------------------------------------- byte identity

# sha256 of the output for variant A with the timestamp frozen at 2026-01-01
# 00:00:00. Every row there has an Expected_Result column, so this change must
# not alter a single byte. On master (85a55ef) the value was db809ca2...b047;
# on bdx-explainability the testbench also carries the scenario tags
# ([BDD:...]), so the value is that branch's output before this change was
# merged (8cddb13), checked identical for all ablation variants.
VARIANT_A_SHA256 = 'aa829d26d92b8b28b46b6aef7350d6d9f2a6c3b47b62a776ce3ac84535cf334b'


def test_ablation_variant_a_byte_identical_to_master(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'datetime', _FrozenDatetime)
    feature = ROOT / 'artifact' / 'ablation' / 'variant_A_original.feature'
    res = T.TestbenchGenerator(output_dir=str(tmp_path)).generate_single(str(feature), ALU)
    assert res['success']
    data = Path(res['filepath']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == VARIANT_A_SHA256
    assert res['test_count'] == 67
    ev = res['expected_values']
    assert (ev['rows'], ev['asserted'], ev['skipped_no_expected']) == (67, 67, 0)


# ------------------------------------------- register file and CPU (same rule)
# Their templates used the reference value for a row without an expected value.
# A check the scenario never asked for would be counted as the scenario's, so
# they now skip the row as well, with the same reason.

RF = {'module_type': 'regfile', 'module_name': 'regfile', 'bitwidth': 32, 'depth': 32}
RF_FEATURE = """Feature: 32x32-bit Register File

  @write_read
  Scenario Outline: Write and read back
    When I write <data> to register <addr>
    And I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | data | expected |
      | 7    | 42   | 42       |

  @write_read
  Scenario Outline: Write and read back, no value stated
    When I write <data> to register <addr>
    And I read register <addr>

    Examples:
      | addr | data |
      | 9    | 5    |

  @read
  Scenario Outline: Read a register
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected |
      | 9    | 5        |
"""


def _index(res):
    from src import scenario_results as SR
    return SR.load_index(res['filepath'])


def test_register_file_read_without_expected_value_is_not_checked(tmp_path):
    res, tb = _generate(tmp_path, RF_FEATURE, RF)
    ev = res['expected_values']
    assert (ev['asserted'], ev['skipped_no_expected']) == (2, 1)
    # the write of the skipped row still happens: the later read of R9 sees 5
    assert "waddr = 5'd9; wdata = 32'd5;" in tb and "if (rdata1 == 32'd5)" in tb
    assert tb.count('total = total + 1;') == 3            # 2 reads from the scenarios + the R0 check
    trace = _index(res)['trace']
    skipped = [k for k, v in trace['skipped'].items() if v == REASON]
    assert len(skipped) == 1
    assert trace['detail'][skipped[0]][0]['operation'] == 'WRITE'


def test_register_file_spec_oracle_still_checks_the_row(tmp_path):
    res, tb = _generate(tmp_path, RF_FEATURE, RF, oracle_source='spec')
    assert res['expected_values']['skipped_no_expected'] == 0
    assert tb.count('total = total + 1;') == 4


CPU = {'module_type': 'cpu', 'module_name': 'riscv_cpu', 'bitwidth': 32, 'pipeline_stages': 5}
CPU_FEATURE = """Feature: RISC-V CPU

  @arithmetic
  Scenario Outline: Register results
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction    | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | add x3, x1, x2 | 1   | 10        | 2   | 20        | 3  | 30       |

  @arithmetic
  Scenario Outline: Register results without a value
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>

    Examples:
      | instruction    | rs1 | rs1_value | rs2 | rs2_value | rd |
      | sub x3, x1, x2 | 1   | 10        | 2   | 20        | 3  |

  @branch
  Scenario Outline: Branch without a stated decision
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>

    Examples:
      | instruction   | rs1 | rs1_value | rs2 | rs2_value |
      | beq x1, x2, 8 | 1   | 10        | 2   | 10        |
"""


def test_cpu_row_without_expected_value_is_not_checked(tmp_path):
    res, tb = _generate(tmp_path, CPU_FEATURE, CPU)
    ev = res['expected_values']
    assert (ev['asserted'], ev['skipped_no_expected']) == (1, 2)
    assert tb.count('total = total + 1;') == 1
    assert sorted(_index(res)['trace']['skipped'].values()) == [REASON, REASON]


def test_cpu_spec_oracle_still_checks_the_rows(tmp_path):
    res, tb = _generate(tmp_path, CPU_FEATURE, CPU, oracle_source='spec')
    assert res['expected_values']['skipped_no_expected'] == 0
    assert tb.count('total = total + 1;') == 3


def test_feature_whose_only_cpu_rows_state_nothing_fails_with_the_reason(tmp_path):
    only = CPU_FEATURE.split('  @arithmetic\n  Scenario Outline: Register results without')[0].split(
        '  @arithmetic')[0] + '  @arithmetic\n  Scenario Outline: Register results without' + \
        CPU_FEATURE.split('  @arithmetic\n  Scenario Outline: Register results without')[1]
    f = tmp_path / 'only.feature'
    f.write_text(only, encoding='utf-8')
    res = T.TestbenchGenerator(output_dir=str(tmp_path / 'tb')).generate_single(str(f), dict(CPU))
    assert not res['success'] and 'state no expected result' in res['error']
