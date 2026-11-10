"""
Run report: what was asked, what was decided, what was generated, what was
tested, and how it came out — for one pass through the four steps.

Inputs are files the steps already wrote: the confirmed requirements
(Step 0, optional), the feature file (Step 2), the design (Step 1), the
testbench and its scenario index (Step 3), and the simulation log (Step 4).
Nothing here calls a model.

Requirement links come from @req_N tags the BDD generator was asked to put
on each scenario. They are the model's claim, not a reviewed mapping, and
the report says so wherever it shows them.
"""

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from src import scenario_results as _sr
except ImportError:
    import scenario_results as _sr

LINK_SOURCE = 'proposed by the model that wrote the scenarios; not reviewed'


def safe_path(path, project_root) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a path the browser sent; only files under output/ may be read."""
    if not path:
        return None, None
    root = (Path(project_root) / 'output').resolve()
    p = Path(path)
    if not p.is_absolute():
        p = Path(project_root) / p
    try:
        p = p.resolve()
    except OSError as e:
        return None, f'bad path {path}: {e}'
    if root not in p.parents:
        return None, f'only files under output/ can be exported: {path}'
    if not p.is_file():
        return None, f'file not found: {path}'
    return p, None


def _req_status(statuses: List[str]) -> str:
    s = [x for x in statuses if x != 'setup']   # 准备步骤本身不验证任何东西
    if not statuses:
        return 'not_covered'
    if any(x in ('fail', 'incomplete') for x in s):
        return 'fail'
    if s and all(x == 'pass' for x in s):
        return 'pass'
    return 'unverified'


def traceability(record: Optional[Dict], results: Optional[Dict]) -> Dict:
    reqs = [r for r in ((record or {}).get('requirements') or [])]
    scenarios = (results or {}).get('scenarios') or []
    known = {r['id'] for r in reqs}

    by_req: Dict[str, List[Dict]] = {r['id']: [] for r in reqs}
    unknown_refs = []
    for sc in scenarios:
        for rid in sc.get('requirements') or []:
            if rid in by_req:
                by_req[rid].append(sc)
            else:
                unknown_refs.append({'scenario': sc['id'], 'requirement': rid})

    req_rows = []
    for r in reqs:
        linked = by_req[r['id']]
        unchecked = sorted({c for s in linked for c in (s.get('unchecked_columns') or [])})
        req_rows.append({
            'id': r['id'], 'text': r.get('text', ''), 'source': r.get('source'),
            'scenarios': [{'id': s['id'], 'title': s.get('title', ''),
                           'status': s['status']} for s in linked],
            'status': _req_status([s['status'] for s in linked]),
            # 链接场景里 testbench 不比较的列。需求若恰好说的是这些列（例如溢出
            # 标志），"pass" 并不代表它被验证过——工具无法自动判断，只能指出来。
            'unchecked_columns': unchecked,
        })
    counts: Dict[str, int] = {}
    for r in req_rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1

    return {
        'link_source': LINK_SOURCE,
        'requirements': req_rows,
        'by_status': counts,
        # 有需求时才谈得上"孤儿"：没确认过需求，所有场景都没有链接是正常的
        'orphan_scenarios': ([{'id': s['id'], 'title': s.get('title', ''),
                               'status': s['status']}
                              for s in scenarios if not s.get('requirements')]
                             if reqs else []),
        'unknown_requirement_refs': unknown_refs,
    }


def build(project_root, paths: Dict[str, Optional[str]]) -> Tuple[Optional[Dict], Optional[str]]:
    """paths: requirements / bdd / duv / testbench / log (any may be missing)."""
    resolved = {}
    for key, value in paths.items():
        p, err = safe_path(value, project_root)
        if err:
            return None, err
        resolved[key] = p

    record = None
    if resolved.get('requirements'):
        record = json.loads(resolved['requirements'].read_text(encoding='utf-8'))

    results = None
    if resolved.get('testbench') and resolved.get('log'):
        log = resolved['log'].read_text(encoding='utf-8', errors='replace')
        results = _sr.summarize_for_testbench(resolved['testbench'], log)

    rel = lambda p: p.relative_to(Path(project_root)).as_posix() if p else None
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'files': {k: rel(v) for k, v in resolved.items()},
        'requirements_record': record,
        'scenario_results': results,
        'traceability': traceability(record, results) if record else None,
    }, None


# ---------------------------------------------------------------- markdown

_MARK = {'pass': '✓ pass', 'fail': '✗ FAIL', 'incomplete': '… incomplete',
         'weak': '⚠ weak check', 'partial': '◐ partial', 'no_check': '⚠ not tested',
         'not_run': '⚠ not tested', 'setup': '· setup', 'unverified': '⚠ unverified',
         'not_covered': '⚠ NOT COVERED'}


def _md_requirements(record: Dict, trace: Optional[Dict]) -> str:
    out = ['# Requirements', '',
           f"Confirmed in Step 0 for **{record.get('module_type')}**"
           f" on {record.get('created_at', '')}.", '']
    rows = {r['id']: r for r in (trace or {}).get('requirements', [])}
    out += ['| ID | Requirement | Source | Covered by | Status | Columns not checked |',
            '|---|---|---|---|---|---|']
    for r in record.get('requirements') or []:
        t = rows.get(r['id'], {})
        cov = ', '.join(s['id'] for s in t.get('scenarios', [])) or '—'
        unc = ', '.join(t.get('unchecked_columns') or []) or '—'
        out.append(f"| {r['id']} | {r['text']} | {r.get('source', '')} | {cov} | "
                   f"{_MARK.get(t.get('status'), t.get('status', ''))} | {unc} |")
    out += ['', f'Links between requirements and scenarios are {LINK_SOURCE}.',
            '"Pass" means every assertion generated from the linked scenarios passed. '
            'The testbench compares only the main output; columns listed under '
            '"Columns not checked" appear in the scenarios\' Examples tables but are '
            'never compared, so a requirement about them is not verified by a pass.', '']
    return '\n'.join(out)


def _md_decisions(record: Dict) -> str:
    out = ['# Decisions', '',
           'Points the description left open, and the answer that was given. The same '
           'text was added to the design prompt (Step 1) and the BDD prompt (Step 2).', '']
    if record.get('accepted_all_defaults'):
        out += ['Every preselected answer was kept.', '']
    qs = {q['id']: q for q in record.get('questions') or []}
    for d in record.get('decisions') or []:
        q = qs.get(d.get('question_id'), {})
        out.append(f"## {d['id']}. {d.get('question', '')}")
        if q.get('why'):
            out += ['', f"*Why it matters:* {q['why']}"]
        out.append('')
        for i, o in enumerate(q.get('options') or []):
            chosen = '**→**' if o == d['answer'] else '  '
            rec = ' *(preselected)*' if i == q.get('default') else ''
            out.append(f"- {chosen} {o}{rec}")
        if d.get('source') == 'custom':
            out.append(f"- **→** {d['answer']} *(custom answer)*")
        out += ['', f"Answer source: {d.get('source')}", '']
    if not record.get('decisions'):
        out.append('No open points were raised.')
    return '\n'.join(out)


def _md_scenarios(results: Dict, feature_text: Optional[str]) -> str:
    out = ['# BDD scenarios and results', '',
           'Each assertion in the testbench carries the ID of the BDD row it came from '
           '(`ALU-ADD-001/r3` = scenario ALU-ADD-001, Examples row 3).', '']
    for sc in results.get('scenarios') or []:
        req = ', '.join(sc.get('requirements') or []) or '—'
        count = f"{sc['passed']}/{sc['checks']}" if sc.get('checks') else '—'
        out += [f"## {sc['id']} — {sc.get('title', '')}", '',
                f"Status: **{_MARK.get(sc['status'], sc['status'])}** · checks passed: {count}"
                f" · requirements: {req}"]
        if sc.get('reason'):
            out.append(f"Note: {sc['reason']}")
        if sc.get('unchecked_columns'):
            out.append(f"Columns in the Examples table the testbench does not compare: "
                       f"{', '.join(sc['unchecked_columns'])}")
        span = sc.get('span')
        if feature_text and span:
            out += ['', '```gherkin', feature_text[span[0]:span[1]].strip('\n'), '```']
        if sc.get('rows'):
            out += ['', '| Row | Result | Detail |', '|---|---|---|']
            for r in sc['rows']:
                detail = (r.get('message') or r.get('reason') or '').replace('|', '\\|')
                if r.get('weak'):
                    detail += f" — ⚠ {r['weak']}"
                out.append(f"| {r['id'].rsplit('/', 1)[-1]} | {_MARK.get(r['status'], r['status'])} | {detail} |")
        out.append('')
    gen = results.get('generated') or []
    if gen:
        out += ['## Checks added by the generator (not from the BDD)', '',
                '| ID | Check | Result |', '|---|---|---|']
        out += [f"| {g['id']} | {g.get('label', '')} | {_MARK.get(g['status'], g['status'])} |"
                for g in gen]
    return '\n'.join(out)


def _md_readme(report: Dict) -> str:
    sr = report.get('scenario_results') or {}
    tot = sr.get('totals') or {}
    tr = report.get('traceability')
    lines = ['# Run report', '', f"Generated {report['generated_at']}.", '',
             '| File | Contents |', '|---|---|']
    have = report['files']
    if have.get('requirements'):
        lines += ['| requirements.md | Requirements confirmed in Step 0, with the scenarios that cover each |',
                  '| decisions.md | Open points, the options offered, and the answer given |']
    lines += ['| bdd-scenarios.md | Every scenario, its Gherkin text, and its result row by row |',
              '| traceability.json | The same, machine-readable |']
    for key, name in (('duv', 'design.v'), ('bdd', 'scenarios.feature'),
                      ('testbench', 'testbench.v'), ('log', 'simulation.log')):
        if have.get(key):
            lines.append(f'| {name} | {key} as generated |')
    lines += ['', '## Summary', '']
    if tot:
        lines.append(f"- {tot.get('scenarios', 0)} scenarios; "
                     + ', '.join(f"{n} {_MARK.get(k, k)}" for k, n in sorted((tot.get('by_status') or {}).items())))
        lines.append(f"- BDD assertions passed: {tot.get('bdd_passed', 0)}/{tot.get('bdd_checks', 0)}")
    else:
        lines.append('- No simulation results with scenario IDs were available.')
    if tr:
        lines.append(f"- {len(tr['requirements'])} requirements; "
                     + ', '.join(f"{n} {_MARK.get(k, k)}" for k, n in sorted(tr['by_status'].items())))
        if tr['orphan_scenarios']:
            lines.append(f"- {len(tr['orphan_scenarios'])} scenarios are not linked to any requirement")
        lines += ['', f"Requirement links are {LINK_SOURCE}."]
    return '\n'.join(lines) + '\n'


def to_zip(report: Dict, project_root) -> bytes:
    files = report['files']
    root = Path(project_root)
    feature_text = ((root / files['bdd']).read_text(encoding='utf-8', errors='replace')
                    if files.get('bdd') else None)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.md', _md_readme(report))
        rec = report.get('requirements_record')
        if rec:
            z.writestr('requirements.md', _md_requirements(rec, report.get('traceability')))
            z.writestr('decisions.md', _md_decisions(rec))
        if report.get('scenario_results'):
            z.writestr('bdd-scenarios.md', _md_scenarios(report['scenario_results'], feature_text))
        z.writestr('traceability.json', json.dumps(report, ensure_ascii=False, indent=1))
        for key, name in (('duv', 'design.v'), ('bdd', 'scenarios.feature'),
                          ('testbench', 'testbench.v'), ('log', 'simulation.log')):
            if files.get(key):
                z.write(root / files[key], name)
    return buf.getvalue()
