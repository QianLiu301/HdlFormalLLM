"""
Stage one ALU run for the browser checks in tests/ui/ (no LLM involved).

Writes, where the page itself would put them (provider folder "gemini"):
    output/bdd/gemini/xdb_ui_alu.feature       scenarios with two wrong expected
                                               results (SUB r7, SRA r5) and one
                                               wrong zero flag (ADD r7)
    output/dut/gemini/xdb_ui_alu.v             design whose SLTU compares signed
    output/dut/gemini/xdb_ui_alu_fixed.v       the same design with SLTU fixed
                                               (what a repaired design looks like)
    output/requirements/xdb_ui_req.json        Step-0 record
and prints the page state as JSON. The testbench and the simulation are left
to the page (Step 3 / Step 4), so the browser check runs the real flow.

    python tests/ui/stage_run.py            # ALU
    python tests/ui/stage_run.py regfile    # register file (32 x 32 bit)
    python tests/ui/stage_run.py cpu        # RV32I CPU (SLT compares unsigned)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.xdb.samples import (ALU, CPU_BUGGY, CPU_FEATURE, CPU_OK, CPU_RECORD, RF_BUGGY,  # noqa: E402
                             RF_FEATURE, RF_FIXED, RF_RECORD)
# ALU: SLTU compares signed. Register file: register 0 is not protected; the
# feature reads R1 expecting 100 after a later row wrote 300.


def regfile():
    out = ROOT / 'output'
    for d in ('bdd/gemini', 'dut/gemini', 'requirements'):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / 'bdd/gemini/xdb_ui_rf.feature').write_text(RF_FEATURE, encoding='utf-8')
    (out / 'dut/gemini/xdb_ui_rf.v').write_text(RF_BUGGY, encoding='utf-8')
    (out / 'dut/gemini/xdb_ui_rf_fixed.v').write_text(RF_FIXED, encoding='utf-8')
    (out / 'requirements/xdb_ui_rf_req.json').write_text(json.dumps(RF_RECORD), encoding='utf-8')
    print(json.dumps({
        'bdd': {'filename': 'xdb_ui_rf.feature', 'filepath': 'output/bdd/gemini/xdb_ui_rf.feature',
                'llm': 'gemini'},
        'dut': {'moduleType': 'regfile', 'moduleName': 'Register File', 'bitwidth': 32, 'depth': 32,
                'pipelineStages': 5, 'filename': 'xdb_ui_rf.v', 'source': 'llm', 'llm': 'gemini'},
        'requirements': {'filepath': 'output/requirements/xdb_ui_rf_req.json', 'module_type': 'regfile'},
        'fixed_design': {'filename': 'xdb_ui_rf_fixed.v', 'content': RF_FIXED},
        'fake_repair': {'failures': 2, 'operations': ['READ']},
    }))


def cpu():
    out = ROOT / 'output'
    for d in ('bdd/gemini', 'dut/gemini', 'requirements'):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / 'bdd/gemini/xdb_ui_cpu.feature').write_text(CPU_FEATURE, encoding='utf-8')
    (out / 'dut/gemini/xdb_ui_cpu.v').write_text(CPU_BUGGY, encoding='utf-8')
    (out / 'dut/gemini/xdb_ui_cpu_fixed.v').write_text(CPU_OK, encoding='utf-8')
    (out / 'requirements/xdb_ui_cpu_req.json').write_text(json.dumps(CPU_RECORD), encoding='utf-8')
    print(json.dumps({
        'bdd': {'filename': 'xdb_ui_cpu.feature', 'filepath': 'output/bdd/gemini/xdb_ui_cpu.feature',
                'llm': 'gemini'},
        'dut': {'moduleType': 'cpu', 'moduleName': 'RISC-V CPU', 'bitwidth': 32, 'depth': 32,
                'pipelineStages': 5, 'filename': 'xdb_ui_cpu.v', 'source': 'llm', 'llm': 'gemini'},
        'requirements': {'filepath': 'output/requirements/xdb_ui_cpu_req.json', 'module_type': 'cpu'},
        'fixed_design': {'filename': 'xdb_ui_cpu_fixed.v', 'content': CPU_OK},
        'fake_repair': {'failures': 2, 'operations': ['SLT', 'SLTI']},
    }))


def main():
    out = ROOT / 'output'
    for d in ('bdd/gemini', 'dut/gemini', 'requirements'):
        (out / d).mkdir(parents=True, exist_ok=True)
    feat = (ROOT / 'artifact/ablation/variant_A_original.feature').read_text(encoding='utf-8')
    feat = (feat.replace('@add @arithmetic', '@add @arithmetic @req_1')
                .replace('@sltu', '@sltu @req_2', 1)
                .replace('| 300             | False     |', '| 300             | True      |', 1))
    (out / 'bdd/gemini/xdb_ui_alu.feature').write_text(feat, encoding='utf-8')
    alu = ALU.replace('module alu_32bit', 'module alu')
    (out / 'dut/gemini/xdb_ui_alu.v').write_text(alu, encoding='utf-8')
    fixed = alu.replace("4'b1001: result = ($signed(a) < $signed(b)) ? 1 : 0;   // BUG: SLTU compares signed",
                        "4'b1001: result = (a < b) ? 1 : 0;")
    assert fixed != alu
    (out / 'dut/gemini/xdb_ui_alu_fixed.v').write_text(fixed, encoding='utf-8')
    record = {'module_type': 'alu', 'created_at': '2026-11-03T10:00:00',
              'requirements': [
                  {'id': 'REQ-1', 'text': 'ADD returns a + b modulo 2^32', 'source': 'llm'},
                  {'id': 'REQ-2', 'text': 'SLTU compares unsigned', 'source': 'llm'}],
              'questions': [], 'decisions': []}
    (out / 'requirements/xdb_ui_req.json').write_text(json.dumps(record), encoding='utf-8')
    print(json.dumps({
        'bdd': {'filename': 'xdb_ui_alu.feature', 'filepath': 'output/bdd/gemini/xdb_ui_alu.feature',
                'llm': 'gemini'},
        'dut': {'moduleType': 'alu', 'moduleName': 'ALU', 'bitwidth': 32, 'depth': 32,
                'pipelineStages': 5, 'filename': 'xdb_ui_alu.v', 'source': 'llm', 'llm': 'gemini'},
        'requirements': {'filepath': 'output/requirements/xdb_ui_req.json', 'module_type': 'alu'},
        'fixed_design': {'filename': 'xdb_ui_alu_fixed.v', 'content': fixed},
        'fake_repair': {'failures': 3, 'operations': ['SLTU']},
    }))


if __name__ == '__main__':
    {'regfile': regfile, 'cpu': cpu}.get((sys.argv[1:] or [''])[0], main)()
