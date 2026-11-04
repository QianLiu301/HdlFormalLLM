"""
Regression: the template testbenches judge what the scenario says.

CPU (TestbenchGenerator._generate_cpu_testbench)
  * a store row in the prompt's own format (rd column = 0) used to be checked
    against x0 instead of the memory the store writes -- a correct CPU failed it;
  * load/store offsets were ignored (lw x4, 4(x1) placed the value at x1);
  * rows whose instruction names other registers than the rs1/rs2/rd columns,
    rows using x29/x30 (the testbench's own registers), unaligned addresses,
    addresses in the result area, and memory words two rows fight over tested
    something else; they are now skipped and counted.
(ALU / counter rows without an expected value: tests/test_no_expected_result.py.)

Every CPU check below runs on a small correct RV32I CPU, so any failure is a
testbench error. Needs iverilog.

Run from the repository root:  python -m pytest tests/test_template_testbenches.py
"""
import contextlib
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import testbench_generator as T  # noqa: E402

needs_iverilog = pytest.mark.skipif(shutil.which('iverilog') is None, reason='iverilog not installed')

# One instruction per cycle; a load takes two (the testbench's data memory reads
# synchronously). Implements the RV32I subset the CPU template uses.
CPU = """// Minimal RV32I subset CPU for the explainability tests: one instruction per
// cycle, a load takes two (the testbench's data memory reads synchronously).
module riscv_cpu (
    input  wire        clk,
    input  wire        rst_n,
    output wire [31:0] imem_addr,
    input  wire [31:0] imem_data,
    output wire [31:0] dmem_addr,
    output wire [31:0] dmem_wdata,
    output wire        dmem_wen,
    output wire [3:0]  dmem_be,
    input  wire [31:0] dmem_rdata,
    output wire [31:0] debug_pc,
    output wire [31:0] debug_inst
);
    reg [31:0] pc;
    reg [31:0] x [0:31];
    reg        load_wait;
    integer k;

    wire [31:0] inst = imem_data;
    wire [6:0]  opc  = inst[6:0];
    wire [4:0]  rd   = inst[11:7];
    wire [2:0]  f3   = inst[14:12];
    wire [4:0]  rs1  = inst[19:15];
    wire [4:0]  rs2  = inst[24:20];
    wire [6:0]  f7   = inst[31:25];
    wire [31:0] a    = (rs1 == 0) ? 32'd0 : x[rs1];
    wire [31:0] b    = (rs2 == 0) ? 32'd0 : x[rs2];
    wire [31:0] imm_i = {{20{inst[31]}}, inst[31:20]};
    wire [31:0] imm_s = {{20{inst[31]}}, inst[31:25], inst[11:7]};
    wire [31:0] imm_b = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0};

    wire is_r = opc == 7'b0110011, is_i = opc == 7'b0010011, is_lw = opc == 7'b0000011,
         is_sw = opc == 7'b0100011, is_br = opc == 7'b1100011;
    wire [31:0] op2 = is_r ? b : imm_i;

    reg [31:0] alu;
    always @(*) begin
        case (f3)
            3'b000: alu = (is_r && f7[5]) ? a - op2 : a + op2;
            3'b010: alu = ($signed(a) < $signed(op2)) ? 32'd1 : 32'd0;     // SLT / SLTI
            3'b100: alu = a ^ op2;
            3'b110: alu = a | op2;
            3'b111: alu = a & op2;
            default: alu = 32'd0;
        endcase
    end

    reg taken;
    always @(*) begin
        case (f3)
            3'b000: taken = a == b;
            3'b001: taken = a != b;
            3'b100: taken = $signed(a) < $signed(b);
            3'b101: taken = $signed(a) >= $signed(b);
            default: taken = 1'b0;
        endcase
    end

    assign imem_addr  = pc;
    assign dmem_addr  = is_sw ? a + imm_s : a + imm_i;
    assign dmem_wdata = b;
    assign dmem_wen   = is_sw & rst_n;
    assign dmem_be    = 4'b1111;
    assign debug_pc   = pc;
    assign debug_inst = inst;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; load_wait <= 0;
            for (k = 0; k < 32; k = k + 1) x[k] <= 0;
        end else if (is_lw && !load_wait) begin
            load_wait <= 1;                                   // address out, data next cycle
        end else begin
            load_wait <= 0;
            if ((is_r || is_i) && rd != 0) x[rd] <= alu;
            if (is_lw && rd != 0) x[rd] <= dmem_rdata;
            pc <= (is_br && taken) ? pc + imm_b : pc + 4;
        end
    end
endmodule
"""

CPU_DUT = {'module_type': 'cpu', 'module_name': 'riscv_cpu', 'bitwidth': 32, 'pipeline_stages': 5}

HEADER = """Feature: RISC-V CPU Verification (5-stage pipeline)

  Background:
    Given the CPU is a 5-stage pipeline RV32I processor
"""


def _outline(tag, rows, then='Then register x<rd> should contain <expected>'):
    head = '| instruction | rs1 | rs1_value | rs2 | rs2_value | rd | mem_value | expected |'
    body = '\n'.join(f'      | {r} |' for r in rows)
    return f"""
  @{tag}
  Scenario Outline: {tag} rows
    Given register x<rs1> contains <rs1_value>
    When I execute <instruction>
    {then}

    Examples:
      {head}
{body}
"""


def _generate(tmp_path, text, dut, name='t.feature', oracle='bdd'):
    f = tmp_path / name
    f.write_text(text, encoding='utf-8')
    with contextlib.redirect_stdout(io.StringIO()):
        r = T.TestbenchGenerator(output_dir=str(tmp_path / 'tb')).generate_single(str(f), dict(dut),
                                                                                  oracle_source=oracle)
    assert r['success'], r
    return r


def _run(tmp_path, tb_path, design_text, name='dut.v'):
    d = tmp_path / name
    d.write_text(design_text, encoding='utf-8')
    vvp = tmp_path / 'sim.vvp'
    subprocess.run(['iverilog', '-g2012', '-o', str(vvp), str(tb_path), str(d)], check=True)
    log = subprocess.run(['vvp', str(vvp)], capture_output=True, text=True,
                         encoding='utf-8', errors='replace').stdout
    return [l for l in log.splitlines() if re.search(r'\b(?:PASS|FAIL)\s', l)]   # not the summary line


@needs_iverilog
def test_store_rows_and_offsets_pass_on_a_correct_cpu(tmp_path):
    rows = ['sw x2, 0(x1)  | 1 | 0x140 | 2 | 0x7BC | 0 | 0     | 0x7BC',     # prompt format: rd = 0
            'sw x2, 8(x1)  | 1 | 0x200 | 2 | 77    | 0 | 0     | 77',        # store with an offset
            'lw x3, 4(x1)  | 1 | 0x100 | 0 | 0     | 3 | 0x55  | 0x55',      # load with an offset
            'lw x4, -4(x1) | 1 | 0x180 | 0 | 0     | 4 | 0x66  | 0x66',
            'add x3, x1, x2 | 1 | 10   | 2 | 20    | 3 | 0     | 30']
    r = _generate(tmp_path, HEADER + _outline('memory', rows), CPU_DUT)
    lines = _run(tmp_path, r['filepath'], CPU)
    assert len(lines) == 5 and not any('FAIL' in l for l in lines), lines


def test_rows_that_test_something_else_are_skipped(tmp_path):
    rows = ['add x3, x4, x2 | 1 | 10 | 2 | 20 | 3 | 0 | 30',      # instruction reads x4, row sets x1
            'add x5, x1, x2 | 1 | 10 | 2 | 20 | 3 | 0 | 30',      # writes x5, row reads the result from x3
            'add x3, x1, x29 | 1 | 10 | 29 | 20 | 3 | 0 | 30',    # x29 is the testbench's marker register
            'lw x3, 2(x1)  | 1 | 0x100 | 0 | 0 | 3 | 5 | 5',      # unaligned address
            'lw x3, 0(x1)  | 1 | 0x400 | 0 | 0 | 3 | 5 | 5',      # inside the testbench's result area
            'sw x2, 0(x1)  | 1 | 0x140 | 2 | 1 | 0 | 0 | 1',
            'sw x2, 0(x1)  | 1 | 0x140 | 2 | 2 | 0 | 0 | 2',      # second store to the same word
            'lw x3, 0(x1)  | 1 | 0x140 | 0 | 0 | 3 | 9 | 9',      # reads a word a store changed
            'lw x3, 0(x1)  | 1 | 0x180 | 0 | 0 | 3 | 7 | 7',
            'lw x4, 0(x1)  | 1 | 0x180 | 0 | 0 | 4 | 7 | 7']      # same word, same value: fine
    f = tmp_path / 'skip.feature'
    f.write_text(HEADER + _outline('memory', rows), encoding='utf-8')
    with contextlib.redirect_stdout(io.StringIO()):
        spec = T.FeatureParser(str(f), module_type_override='cpu').parse()
        text, qa = T.TestbenchGenerator(output_dir=str(tmp_path / 'tb'))._generate_cpu_testbench(
            spec, 'riscv_cpu', 'test')
    st = qa['cpu_program']
    assert st['checks'] == 3                     # the first store to 0x140, both loads of 0x180
    assert (st['skipped_register_mismatch'], st['skipped_reserved_register'],
            st['skipped_bad_address'], st['skipped_memory_conflict']) == (2, 1, 2, 2)
    assert text.count('total = total + 1;') == 3


@needs_iverilog
def test_prompt_example_passes_on_a_correct_cpu(tmp_path):
    """The example of prompts/bdd_cpu/v1.yaml (store value reduced to fit one ADDI)."""
    rows = ['add x3, x1, x2   | 1 | 10    | 2 | 20     | 3 | 0      | 30',
            'sub x3, x1, x2   | 1 | 30    | 2 | 10     | 3 | 0      | 20',
            'addi x3, x1, 100 | 1 | 50    | 0 | 0      | 3 | 0      | 150',
            'slt x3, x1, x2   | 1 | -1    | 2 | 1      | 3 | 0      | 1',
            'slti x3, x1, 0   | 1 | -1    | 0 | 0      | 3 | 0      | 1',
            'lw x3, 0(x1)     | 1 | 0x100 | 0 | 0      | 3 | 0x1234 | 0x1234',
            'sw x2, 0(x1)     | 1 | 0x140 | 2 | 0x7CD  | 0 | 0      | 0x7CD']
    r = _generate(tmp_path, HEADER + _outline('mixed', rows), CPU_DUT)
    lines = _run(tmp_path, r['filepath'], CPU)
    assert len(lines) == 7 and not any('FAIL' in l for l in lines), lines


def test_last_examples_row_is_kept_without_a_final_newline(tmp_path):
    """The web's BDD generator saves features without a final newline; the
    Examples regex needed one, so the last row of every such file was lost."""
    feature = HEADER + _outline('arith', ['add x3, x1, x2 | 1 | 10 | 2 | 20 | 3 | 0 | 30',
                                          'sub x3, x1, x2 | 1 | 30 | 2 | 10 | 3 | 0 | 20'])
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    with_nl = _generate(tmp_path / 'a', feature, CPU_DUT)
    without = _generate(tmp_path / 'b', feature.rstrip('\n'), CPU_DUT)
    count = lambda r: Path(r['filepath']).read_text(encoding='utf-8').count('total = total + 1;')
    assert count(with_nl) == count(without) == 2
