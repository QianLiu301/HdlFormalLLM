"""The ALU reference model R agrees with Verilog itself.

R is written in Python (TestbenchGenerator._reference_result for the result,
src/xdb/library.py for the rules and the flags). A design is judged against it,
so it must compute exactly what the Verilog operators compute. Here an ALU
written only with those operators (tests/data/reference_alu/ref_alu.v) is
simulated by iverilog and every output is compared with R:
8 bits exhaustively (10 operations x 256 x 256), 16, 32 and 64 bits on boundary
values, every shift amount and random pairs.
"""
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from src.testbench_generator import TestbenchGenerator
from src.xdb import library as L

pytestmark = pytest.mark.skipif(not shutil.which('iverilog'), reason='needs iverilog')

DATA = Path(__file__).parent / 'data' / 'reference_alu'
OPS = ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']


def _vectors(w):
    if w == 8:
        return [(k, a, b) for k in range(10) for a in range(256) for b in range(256)]
    m = (1 << w) - 1
    top = 1 << (w - 1)
    edge = {0, 1, 2, m, m - 1, top, top - 1, top + 1, m // 3, (m // 3) * 2}
    shifts = set(range(0, 2 * w + 2)) | {m, top}
    rng = random.Random(w)
    out = [(k, a, b) for k in range(10) for a in edge for b in edge]
    out += [(k, a, s) for k in (5, 6, 7) for a in edge for s in shifts]
    out += [(k, rng.getrandbits(w), rng.getrandbits(w)) for k in range(10) for _ in range(1500)]
    return out


def _simulate(tmp_path, w, vectors):
    (tmp_path / 'vectors.txt').write_text(''.join(f'{k:x} {a:x} {b:x}\n' for k, a, b in vectors),
                                          encoding='utf-8')
    subprocess.run(['iverilog', '-g2012', '-P', f'tb_ref_alu.W={w}', '-o', 'sim.vvp',
                    str(DATA / 'tb_ref_alu.v'), str(DATA / 'ref_alu.v')],
                   cwd=tmp_path, check=True)
    out = subprocess.run(['vvp', '-n', 'sim.vvp'], cwd=tmp_path, check=True, capture_output=True,
                         text=True, encoding='utf-8', errors='replace').stdout
    rows = [ln.split() for ln in out.splitlines() if ln and ln[0] in '0123456789abcdef' and len(ln.split()) == 8]
    assert len(rows) == len(vectors), f'{len(rows)} of {len(vectors)} rows simulated'
    return rows


@pytest.mark.parametrize('w', [8, 16, 32, 64])
def test_reference_matches_the_verilog_operators(tmp_path, w):
    ref = TestbenchGenerator._reference_result
    signed_ovf = L.OVERFLOW_DEFINITIONS['signed']['fn']
    carry = L.OVERFLOW_DEFINITIONS['carry']['fn']
    wrong = []
    for (k, a, b), row in zip(_vectors(w), _simulate(tmp_path, w, _vectors(w))):
        op = OPS[k]
        y, z, n, os_, oc = int(row[3], 16), int(row[4]), int(row[5]), int(row[6]), int(row[7])
        want = {
            'result': ref(op, a, b, w),
            'rule': L.apply_rule('alu', op, a, b, w),          # the second, independent Python copy
            'zero': L.flag_value('alu', 'zero', op, a, b, w),
            'negative': L.flag_value('alu', 'negative', op, a, b, w),
        }
        got = {'result': y, 'rule': y, 'zero': z, 'negative': n}
        if op in ('ADD', 'SUB'):
            want.update(signed=signed_ovf(op, a, b, w), carry=carry(op, a, b, w))
            got.update(signed=os_, carry=oc)
        for key, v in want.items():
            if v != got[key]:
                wrong.append(f'{op} a={a:#x} b={b:#x} {key}: R={v} Verilog={got[key]}')
    assert not wrong, f'{len(wrong)} differences, e.g. ' + '; '.join(wrong[:5])
