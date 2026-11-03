"""
Flag probe (design doc D23): observe outputs the testbench does not compare.

The ALU testbench compares only `result`; the flags zero / negative / overflow
are stated in the scenarios but never checked. The probe is a copy of the
generated testbench with one extra $display per test, placed right after the
test's settling delay (`#10;`), i.e. at the moment the testbench compares
`result`. Same DUT instantiation, same clock, same stimulus. The original
testbench file is not touched, and the probe never contributes to the pass
rate: its observations only feed the explanations.

    [PROBE] <test> <zero> <negative> <overflow> <result>
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

FLAGS = ('zero', 'negative', 'overflow')
_TEST_BLOCK = re.compile(r'(?P<head>[ \t]*// Test (?P<n>\d+)\b[^\n]*\n'
                         r'(?:[ \t]*(?!// Test )[^\n]*\n){0,8}?'
                         r'(?P<ind>[ \t]*)#\d+;\n)')
_PROBE_LINE = re.compile(r'\[PROBE\]\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)')


def build_probe(tb_text: str) -> Optional[str]:
    """Probe testbench text, or None if the testbench has no flag outputs."""
    declared = {f for f in FLAGS if re.search(rf'\bwire\b[^;]*\b{f}\b', tb_text)}
    if declared != set(FLAGS):
        return None
    n = 0

    def _add(m):
        nonlocal n
        n += 1
        ind = m.group('ind')
        return (m.group(0) + f'{ind}$display("[PROBE] {m.group("n")} %b %b %b %0d", '
                f'zero, negative, overflow, result);\n')

    out = _TEST_BLOCK.sub(_add, tb_text)
    if n == 0:
        return None
    # no waveform for the probe
    out = re.sub(r'^[ \t]*\$dump(?:file|vars)\b[^\n]*\n', '', out, flags=re.M)
    return out


def parse_probe_log(log: str) -> Dict[int, Dict[str, str]]:
    vals = {}
    for m in _PROBE_LINE.finditer(log or ''):
        vals[int(m.group(1))] = dict(zip(FLAGS + ('result',), m.groups()[1:]))
    return vals


def run_probe(tb_text: str, duv_path: Path, timeout: int = 60) -> Dict:
    """Compile and run the probe against the design.
    Returns {status, probe_text, log, values{test: {zero, negative, overflow, result}}}."""
    probe = build_probe(tb_text)
    if probe is None:
        return {'status': 'not_applicable', 'probe_text': None, 'log': '', 'values': {}}
    with tempfile.TemporaryDirectory() as tmp:
        tb = Path(tmp) / 'probe_tb.v'
        tb.write_text(probe, encoding='utf-8')
        try:
            c = subprocess.run(['iverilog', '-g2012', '-o', str(Path(tmp) / 'probe.vvp'), str(tb),
                                str(duv_path)], capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=timeout, cwd=tmp)
            if c.returncode != 0:
                return {'status': 'compile_error', 'probe_text': probe, 'log': c.stderr, 'values': {}}
            r = subprocess.run(['vvp', str(Path(tmp) / 'probe.vvp')], capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=timeout, cwd=tmp)
        except FileNotFoundError:
            return {'status': 'no_simulator', 'probe_text': probe, 'log': '', 'values': {}}
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'probe_text': probe, 'log': '', 'values': {}}
    values = parse_probe_log(r.stdout)
    return {'status': 'ok' if values else 'no_results', 'probe_text': probe,
            'log': r.stdout, 'values': values}
