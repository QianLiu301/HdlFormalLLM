"""
Per-scenario simulation results.

The testbench generator tags every assertion with the BDD row it came from
(``[BDD:ALU-ADD-001/r3]``) and every check it added on its own
(``[GEN:RF-R0]``), and writes a scenario index next to the testbench
(``*_tb.scenarios.json``) listing every scenario in the feature file,
including the ones that produced no assertion. This module joins the two:
the simulation log says what passed and failed, the index says what was
never tested and why.

A scenario that produced no assertion must not disappear from the report.
Before this, such scenarios were simply absent from the log, so the overall
pass rate was computed over the scenarios that happened to compile into a
check and looked better than the BDD deserved.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

TAG_RE = re.compile(r'\[(BDD|GEN):([^\]\s]+)\]')
FAIL_RE = re.compile(r'✗|✗|\bFAIL\b')
PASS_RE = re.compile(r'✓|✓|\bPASS\b')
# Verilog 字符串没有 \u 转义，CPU testbench 里的 "✓" 在日志里印成 "u2713"
MARK_RE = re.compile(r'^(?:[✓✗✓✗]|u27(?:13|17))\s*')


def index_path(tb_path) -> Path:
    return Path(tb_path).with_suffix('.scenarios.json')


def load_index(tb_path) -> Optional[Dict]:
    p = index_path(tb_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def parse_log(log: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Collect tagged PASS/FAIL lines: {'BDD': {row_id: [..]}, 'GEN': {id: [..]}}."""
    out: Dict[str, Dict[str, List[Dict]]] = {'BDD': {}, 'GEN': {}}
    for line in (log or '').splitlines():
        m = TAG_RE.search(line)
        if not m:
            continue
        if FAIL_RE.search(line):
            status = 'fail'
        elif PASS_RE.search(line):
            status = 'pass'
        else:
            continue
        message = MARK_RE.sub('', TAG_RE.sub('', line).strip()).strip()
        out[m.group(1)].setdefault(m.group(2), []).append(
            {'status': status, 'message': message})
    return out


def summarize(index: Dict, log: str) -> Dict:
    """Join the scenario index with a simulation log."""
    hits = parse_log(log)
    trace = index.get('trace') or {}
    checks = trace.get('checks', {})       # row_id -> number of assertions emitted
    skipped = trace.get('skipped', {})     # row_id -> why no assertion
    setup = trace.get('setup', {})         # row_id -> setup step, no check of its own
    weak = trace.get('weak', {})           # row_id -> check that cannot fail

    scenarios = []
    for entry in index.get('catalog', []):
        sc = {'id': entry['id'], 'title': entry.get('title', ''),
              'tags': entry.get('tags', []), 'reason': entry.get('reason'),
              'rows': [], 'checks': 0, 'passed': 0, 'failed': 0}

        if entry.get('status') != 'mapped':
            sc['status'] = 'not_run'
            sc['rows_total'] = 0
            scenarios.append(sc)
            continue

        for rid in entry.get('rows', []):
            row = {'id': rid}
            results = hits['BDD'].get(rid, [])
            if rid in skipped:
                row.update(status='skipped', reason=skipped[rid])
            elif rid in setup:
                row.update(status='setup', reason=setup[rid])
            elif rid in checks and not results:
                # 登记过断言，日志里却没有：仿真中途停了
                row.update(status='missing',
                           reason='assertion was generated but never reported; '
                                  'simulation may have stopped early')
            elif results:
                failed = sum(r['status'] == 'fail' for r in results)
                row['status'] = 'fail' if failed else 'pass'
                row['message'] = next((r['message'] for r in results
                                       if r['status'] == row['status']),
                                      results[0]['message'])
                sc['checks'] += len(results)
                sc['failed'] += failed
                sc['passed'] += len(results) - failed
                if rid in weak:
                    row['weak'] = weak[rid]
            else:
                row.update(status='skipped', reason='no assertion was generated')
            sc['rows'].append(row)

        for k in entry.get('rows_malformed', []):
            sc['rows'].append({'id': f"{entry['id']}/r{k}", 'status': 'skipped',
                               'reason': 'row malformed (column count differs '
                                         'from header)'})
        sc['rows'].sort(key=lambda r: int(r['id'].rsplit('/r', 1)[-1]))
        sc['rows_total'] = len(sc['rows'])

        st = [r['status'] for r in sc['rows']]
        if 'fail' in st:
            sc['status'] = 'fail'
        elif 'missing' in st:
            sc['status'] = 'incomplete'
        elif sc['checks'] == 0:
            sc['status'] = 'setup' if st and all(s == 'setup' for s in st) else 'no_check'
            if sc['status'] == 'no_check' and not sc['reason']:
                reasons = sorted({r.get('reason') for r in sc['rows'] if r.get('reason')})
                sc['reason'] = '; '.join(reasons) or 'no assertion was generated'
        elif any(r.get('weak') for r in sc['rows']):
            sc['status'] = 'weak'
            sc['reason'] = sc['reason'] or next(r['weak'] for r in sc['rows'] if r.get('weak'))
        elif 'skipped' in st:
            sc['status'] = 'partial'
        else:
            sc['status'] = 'pass'
        scenarios.append(sc)

    generated = []
    for g in trace.get('generated', []):
        results = hits['GEN'].get(g['id'], [])
        if results:
            status = 'fail' if any(r['status'] == 'fail' for r in results) else 'pass'
            message = results[0]['message']
        else:
            status, message = 'missing', 'never reported'
        generated.append({'id': g['id'], 'label': g.get('label', ''),
                          'status': status, 'message': message})

    counts: Dict[str, int] = {}
    for sc in scenarios:
        counts[sc['status']] = counts.get(sc['status'], 0) + 1
    bdd_checks = sum(sc['checks'] for sc in scenarios)
    bdd_passed = sum(sc['passed'] for sc in scenarios)
    return {
        'feature': index.get('feature'),
        'oracle_source': index.get('oracle_source'),
        'scenarios': scenarios,
        'generated': generated,
        'totals': {
            'scenarios': len(scenarios),
            'by_status': counts,
            'bdd_checks': bdd_checks,
            'bdd_passed': bdd_passed,
            'generated_checks': len(generated),
            'generated_passed': sum(g['status'] == 'pass' for g in generated),
        },
    }


def summarize_for_testbench(tb_path, log: str) -> Optional[Dict]:
    """None when the testbench predates scenario tagging (no index file)."""
    index = load_index(tb_path)
    return summarize(index, log) if index else None
