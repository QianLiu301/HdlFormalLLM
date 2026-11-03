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

    python tests/ui/stage_run.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from test_xdb import ALU  # noqa: E402  (the test design: SLTU compares signed)


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
    }))


if __name__ == '__main__':
    main()
