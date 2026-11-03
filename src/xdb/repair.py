"""
Repair (milestone M4): act on the verdicts.

    scenario_corrections(s, run)   every example whose scenario states a value
                                   that disagrees with the reference model
                                   (result) or the flag rule (flags)
    apply_corrections(text, ...)   the feature file with those Examples cells
                                   set to the reference value
    repair_scenarios(...)          write that as a NEW feature file (the old one
                                   is kept), record it as feedback, store it as a
                                   new version of the feature (parent_id, reason
                                   "feedback:<id>")
    design_feedback(...)           the failing examples of the design, with the
                                   correct values and the rules, as text for the
                                   design model's prompt

Nothing here calls a model; regenerating the design is done by the Step-1
endpoint with this feedback attached (main.py).
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from src import xdb
from src.xdb import library as L
from src.xdb import models as M

_FLAG_KEYS = {'zero': 'ALU.Q.ZERO', 'negative': 'ALU.Q.NEGATIVE', 'overflow': 'ALU.Q.OVERFLOW'}
_FIXABLE = ('scenario_wrong', 'both_wrong', 'consistent_wrong')


def find_run(s, project_root, paths: Dict) -> Optional[Tuple[M.SimRun, M.Testbench]]:
    """The simulation whose testbench and log are these files."""
    from src import run_report
    tb, e1 = run_report.safe_path(paths.get('testbench_path') or paths.get('testbench'), project_root)
    log, e2 = run_report.safe_path(paths.get('log_file') or paths.get('log'), project_root)
    if e1 or e2 or not tb or not log:
        return None
    tb_sha = hashlib.sha256(tb.read_bytes()).hexdigest()
    log_sha = hashlib.sha256(log.read_bytes()).hexdigest()
    return s.execute(select(M.SimRun, M.Testbench).join(M.Testbench, M.Testbench.id == M.SimRun.testbench_id)
                     .where(M.Testbench.artifact == tb_sha, M.SimRun.log_artifact == log_sha,
                            M.SimRun.probe_of.is_(None))
                     .order_by(M.SimRun.id.desc()).limit(1)).first()


def _column_of(s, scenario_id: int, key: str) -> Optional[str]:
    """Examples column that holds the value of a canonical outcome step."""
    row = s.execute(select(M.StepMapping.params)
                    .join(M.CanonicalStep, M.CanonicalStep.id == M.StepMapping.canonical_step_id)
                    .where(M.StepMapping.scenario_id == scenario_id, M.StepMapping.method == 'rule',
                           M.CanonicalStep.params['key'].astext == key).limit(1)).first()
    if not row or not row[0]:
        return None
    vp = L.value_param(key, row[0].keys())
    return (row[0].get(vp) or {}).get('col')


def _fmt_result(R: int, old: str, w: int) -> str:
    """The reference value in the style of the cell it replaces."""
    old = (old or '').strip()
    if old.lower().startswith('0x'):
        return f'0x{R:X}'
    if old.startswith('-') and R >= 1 << (w - 1):
        return str(R - (1 << w))
    return str(R)


def _fmt_flag(v: int, old: str) -> str:
    old = (old or '').strip()
    if old.lower() in ('true', 'false'):
        word = 'True' if v else 'False'
        return word.lower() if old.islower() else (word.upper() if old.isupper() else word)
    return str(v)


def scenario_corrections(s, run_id: int) -> List[Dict]:
    from src.xdb.explain import _probe_results, _probe_run_id
    out: List[Dict] = []
    run = s.get(M.SimRun, run_id)
    sess = s.get(M.Session, s.get(M.Testbench, run.testbench_id).session_id)
    design = s.get(M.Design, sess.design_id) if sess.design_id else None
    outcome_key = (L.library_for(design.module_type if design else None) or {}).get('outcome_key',
                                                                                    'ALU.Q.RESULT')
    for ci, cr in s.execute(select(M.CheckItem, M.CheckResult)
                            .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                            .where(M.CheckResult.sim_run_id == run_id, M.CheckItem.seq == 1)):
        if cr.verdict not in _FIXABLE or ci.expected_ref is None or not ci.scenario_id:
            continue
        er = s.get(M.ExampleRow, ci.example_row_id) if ci.example_row_id else None
        col = _column_of(s, ci.scenario_id, outcome_key)
        if er is None or not col or col not in (er.vals or {}):
            continue
        old = er.vals[col]
        out.append({'tag': ci.tag, 'scenario_key': ci.tag[4:].rsplit('/r', 1)[0], 'row_no': er.row_no,
                    'port': ci.port or 'result', 'column': col, 'old': old,
                    'new': _fmt_result(int(ci.expected_ref), old, ci.width or 32), 'width': ci.width,
                    'operation': ci.operation, 'stimulus': ci.stimulus})
    for (tag, port), o in _probe_results(s, _probe_run_id(s, run_id)).items():
        if o['verdict'] not in _FIXABLE or o['stated'] is None or o['rule_value'] is None:
            continue
        er = s.get(M.ExampleRow, o['example_row_id']) if o['example_row_id'] else None
        col = _column_of(s, o['scenario_id'], _FLAG_KEYS[port]) if o['scenario_id'] else None
        if er is None or not col or col not in (er.vals or {}):
            continue
        old = er.vals[col]
        out.append({'tag': tag, 'scenario_key': tag[4:].rsplit('/r', 1)[0], 'row_no': er.row_no,
                    'port': port, 'column': col, 'old': old, 'new': _fmt_flag(o['rule_value'], old),
                    'width': design.bitwidth if design else None,
                    'operation': o.get('operation'), 'stimulus': o.get('stimulus')})
    for c in out:
        c['text'] = _describe(c)
    return sorted(out, key=lambda c: (c['scenario_key'], c['row_no'], c['column']))


def _describe(c: Dict) -> str:
    """One plain line for the page: inputs, then old -> new cell value."""
    from src.xdb.explain import _pv
    st, w = c.get('stimulus') or {}, c.get('width')
    if 'addr' in st:                       # register file: the read the value belongs to
        from src.xdb.explain import _example
        ex = _example('regfile', c.get('operation'), st, w)
        return f"{ex[0].upper() + ex[1:]}: expected value {c['old']} → {c['new']}"
    inputs = ', '.join(f"{k.upper()} = {_pv(st[k], w)}" for k in ('a', 'b') if st.get(k) is not None)
    what = 'result' if c['port'] == 'result' else f"{c['port']} flag"
    return f"{c.get('operation') or 'Row ' + str(c['row_no'])}{' with ' + inputs if inputs else ''}: " \
           f"{what} {c['old']} → {c['new']}"


def _cells(line: str) -> List[str]:
    return line.rstrip('\n').split('|')


def apply_corrections(feature_text: str, catalog: List[Dict], corrections: List[Dict]) -> Tuple[str, List, List]:
    """Set Examples cells to their corrected values. Rows are counted like the
    testbench generator counts them (all data rows of the scenario's Examples
    tables in order), so row_no addresses the same row as the check tag."""
    spans = {e['id']: e.get('span') for e in catalog}
    by_scn: Dict[str, List[Dict]] = {}
    for c in corrections:
        by_scn.setdefault(c['scenario_key'], []).append(c)
    applied, skipped = [], []
    # last span first, so earlier offsets stay valid
    for key in sorted(by_scn, key=lambda k: -(spans.get(k) or [0])[0]):
        span = spans.get(key)
        if not span:
            skipped += by_scn[key]
            continue
        block = feature_text[span[0]:span[1]]
        lines = block.splitlines(keepends=True)
        want = {(c['row_no'], c['column'].strip().lower()): c for c in by_scn[key]}
        in_ex, header, row_no = False, None, 0
        for i, line in enumerate(lines):
            st = line.strip()
            if re.match(r'^(Examples|Scenarios)\s*:', st):
                in_ex, header = True, None
                continue
            if not in_ex or not st.startswith('|'):
                if in_ex and st and not st.startswith('#'):
                    in_ex = False
                continue
            parts = _cells(line)
            names = [p.strip().lower() for p in parts]
            if header is None:
                header = names
                continue
            row_no += 1
            for idx, name in enumerate(header):
                c = want.get((row_no, name))
                if c is None or idx >= len(parts):
                    continue
                if parts[idx].strip() != str(c['old']).strip():
                    skipped.append(c)                  # the file changed since it was stored
                    continue
                cell = parts[idx]
                width = len(cell)
                parts[idx] = (' ' + c['new']).ljust(max(width, len(c['new']) + 2))
                applied.append(c)
            nl = '\n' if line.endswith('\n') else ''
            lines[i] = '|'.join(parts) + nl
        feature_text = feature_text[:span[0]] + ''.join(lines) + feature_text[span[1]:]
    done = {(c['tag'], c['column']) for c in applied}
    skipped += [c for c in corrections if (c['tag'], c['column']) not in done and c not in skipped]
    in_file_order = lambda c: ((spans.get(c['scenario_key']) or [0])[0], c['row_no'], c['column'])
    return feature_text, sorted(applied, key=in_file_order), skipped


def _next_fix_path(path: Path) -> Path:
    stem = re.sub(r'_fix\d+$', '', path.stem)
    n = 1
    while True:
        cand = path.with_name(f'{stem}_fix{n}{path.suffix}')
        if not cand.exists():
            return cand
        n += 1


def repair_scenarios(project_root, paths: Dict, run_id: Optional[str], data: Dict,
                     url=None) -> Tuple[Optional[Dict], Optional[str]]:
    from src import run_report, scenario_results
    from src.xdb.ingest import ingest_feature
    feat_path, err = run_report.safe_path(paths.get('bdd'), project_root)
    if err or not feat_path:
        return None, err or 'no feature file in the request'
    tb_path, _ = run_report.safe_path(paths.get('testbench'), project_root)
    index = scenario_results.load_index(tb_path) if tb_path else None
    if not index:
        return None, 'the testbench has no scenario index'
    raw = feat_path.read_bytes()
    with xdb.session_scope(url) as s:
        hit = find_run(s, project_root, {'testbench': paths.get('testbench'), 'log': paths.get('log')})
        if hit is None:
            return None, 'this simulation is not in the explainability database yet; try again in a moment'
        run, tb = hit
        feat = s.get(M.Feature, tb.feature_id) if tb.feature_id else None
        if feat is None or feat.artifact != hashlib.sha256(raw).hexdigest():
            # the catalog spans and the stored rows describe the file that was
            # simulated; a different file would be edited at the wrong places
            return None, 'the feature file is not the one this simulation was generated from'
        corrections = scenario_corrections(s, run.id)
        if not corrections:
            return None, 'nothing to correct: no scenario value disagrees with the reference model'
        new_text, applied, skipped = apply_corrections(raw.decode('utf-8', errors='replace'),
                                                       index.get('catalog') or [], corrections)
        if not applied:
            return None, 'the corrections could not be applied to the feature file'
        sess = s.get(M.Session, tb.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        # the new version belongs to the session of the simulation it repairs
        run_id = sess.run_id or run_id
        design_data = {'module_type': design.module_type, 'bitwidth': design.bitwidth} if design else {}
        fb = M.Feedback(subject_type='sim_run', subject_id=run.id, kind='correction',
                        body=f'Set {len(applied)} scenario values to the reference model',
                        proposed={'action': 'apply_reference_values', 'feature_id': feat.id,
                                  'changes': applied},
                        status='applied')
        s.add(fb)
        s.flush()
        fb_id, run_pk = fb.id, run.id
    new_path = _next_fix_path(feat_path)
    new_path.write_text(new_text, encoding='utf-8')
    rel = new_path.relative_to(Path(project_root).resolve()).as_posix() \
        if new_path.is_absolute() else new_path.as_posix()
    res = ingest_feature(project_root, rel, run_id,
                         {**data, **design_data, '_version_reason': f'feedback:{fb_id}'}, url=url)
    with xdb.session_scope(url) as s:
        s.get(M.Feedback, fb_id).result_ref = f"feature:{res.get('feature_id')}"
    return {'feature_path': rel, 'feature_filename': new_path.name, 'feedback_id': fb_id,
            'feature_id': res.get('feature_id'), 'run_id': run_id,
            'corrections': applied, 'skipped': skipped, 'repaired_sim_run': run_pk}, None


# ----------------------------------------------------------------- design

_DESIGN_FAULT = ('design_wrong', 'both_wrong', 'consistent_wrong')


def _spread(items: List[Tuple[str, str]], limit: int) -> List[str]:
    """At most `limit` lines, taking one per operation in turn, so a design with
    many SLTU failures still shows its single ADD failure."""
    by_op: Dict[str, List[str]] = {}
    for op, line in items:
        by_op.setdefault(op or '', []).append(line)
    out: List[str] = []
    while len(out) < limit and any(by_op.values()):
        for op in list(by_op):
            if by_op[op] and len(out) < limit:
                out.append(by_op[op].pop(0))
    return out


def _rf_item(ci, cr, h: Optional[Dict], w: int) -> str:
    """One failing read of a register file, with the writes and resets it depends on."""
    st = ci.stimulus or {}
    a = st.get('addr')
    got = str(int(cr.observed_value)) if cr.observed_value is not None else 'x (unknown)'
    line = (f"- read of R{a} at step {ci.test_no} of the sequence: rdata1 must be "
            f"{int(ci.expected_ref)}, the design gave {got}")
    if not h:
        return line
    c = h['cause']
    before = [w_ for w_ in h.get('earlier') or []]
    if c['kind'] == 'write':
        before.append(c)
    ev = [f"step {x['test']} wrote {x['value']} to R{a}" + (' (must be ignored)' if x.get('ignored') else '')
          for x in sorted(before, key=lambda x: x['test'])]
    if c['kind'] == 'reset':
        ev.append(f"step {c['test']} reset the register file")
    if c['kind'] == 'x0':
        ev.append('register 0 must always read 0')
    return line + ('. Before it: ' + '; '.join(ev) if ev else '')


def design_feedback(project_root, paths: Dict, url=None, limit: int = 12) -> Tuple[Optional[Dict], Optional[str]]:
    """The design's failing checks as prompt text: inputs, correct value, the
    design's value, and the rules those checks follow; plus the Verilog that
    was simulated."""
    from src.xdb.explain import _probe_results, _probe_run_id, _overflow_mode, _pv, _rf_history
    from src.xdb.ingest import artifact_text
    with xdb.session_scope(url) as s:
        hit = find_run(s, project_root, paths)
        if hit is None:
            return None, 'this simulation is not in the explainability database yet; try again in a moment'
        run, tb = hit
        sess = s.get(M.Session, tb.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        mt = design.module_type if design else None
        W = (design.bitwidth if design else None) or 32
        items: List[Tuple[str, str]] = []
        ops, main_bad, ports = set(), set(), set()
        stateful = bool((L.library_for(mt) or {}).get('stateful'))
        hist = _rf_history(s, tb.id, W) if stateful else {}
        for ci, cr in s.execute(select(M.CheckItem, M.CheckResult)
                                .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                                .where(M.CheckResult.sim_run_id == run.id, M.CheckItem.seq == 1)
                                .order_by(M.CheckItem.test_no)):
            if ci.tag.startswith('GEN:') and cr.status == 'fail':
                items.append(('', f"- the testbench's own check “{ci.label}” failed: {cr.message or ''}".rstrip()))
                continue
            if stateful and cr.verdict in _DESIGN_FAULT and ci.expected_ref is not None:
                main_bad.add(ci.tag)
                ops.add(ci.operation)
                ports.add(ci.port)
                items.append((ci.operation, _rf_item(ci, cr, hist.get(ci.tag), W)))
                continue
            if cr.verdict in _DESIGN_FAULT and ci.expected_ref is not None:
                st, w = ci.stimulus or {}, ci.width or W
                main_bad.add(ci.tag)
                ops.add(ci.operation)
                ports.add(ci.port or 'result')
                got = _pv(int(cr.observed_value), w) if cr.observed_value is not None else 'x (unknown)'
                items.append((ci.operation, f"- {ci.operation} with a = {_pv(st.get('a'), w)}, "
                                            f"b = {_pv(st.get('b'), w)} (opcode {st.get('opcode')}): "
                                            f"{ci.port or 'result'} must be {_pv(int(ci.expected_ref), w)}, "
                                            f"the design gave {got}"))
        ov_mode, _ = _overflow_mode(s, sess.id)
        for (tag, port), o in sorted(_probe_results(s, _probe_run_id(s, run.id)).items()):
            if o['verdict'] in _DESIGN_FAULT and tag not in main_bad:
                st = o.get('stimulus') or {}
                ops.add(o.get('operation'))
                ports.add(port)
                got = o['observed'] if o['observed'] is not None else f"{o['raw']} (unknown)"
                items.append((o.get('operation'), f"- {o.get('operation')} with a = {_pv(st.get('a'), W)}, "
                                                  f"b = {_pv(st.get('b'), W)} (opcode {st.get('opcode')}): "
                                                  f"{port} flag must be {o['rule_value']}, the design gave {got}"))
        if not items:
            return None, 'nothing to repair: no check shows a design error'
        lib = L.library_for(mt) or {}
        rules = [f"- {r['op']}: {r['text']}" for r in (lib.get('rules') or {}).values()
                 if r.get('op') in ops or (stateful and r.get('op') in ('WRITE', 'RESET'))]
        for port in ('zero', 'negative', 'overflow'):
            if port in ports:
                fr = L.flag_rule(mt, port, ov_mode)
                if fr:
                    rules.append(f"- {port} flag: {fr['text']}")
        shown = _spread(items, limit)
        text = ('The previous version of this design failed these checks '
                f'({len(items)} in total'
                + (f'; {len(shown)} of them below'
                   + (', covering every failing operation' if len({o for o, _ in items}) <= limit else '')
                   if len(items) > len(shown) else '')
                + '):\n' + '\n'.join(shown)
                + ('\n\nThe rules these checks follow:\n' + '\n'.join(rules) if rules else ''))
        duv = s.get(M.Duv, run.duv_id) if run.duv_id else None
        art = s.get(M.Artifact, duv.artifact) if duv else None
        previous = artifact_text(art) if art else None
        return {'text': text, 'count': len(items), 'operations': sorted(o for o in ops if o),
                'ports': sorted(ports), 'sim_run_id': run.id, 'duv_id': duv.id if duv else None,
                'session_id': sess.id, 'run_id': sess.run_id, 'module_type': mt,
                'previous_verilog': previous}, None


def open_design_feedback(fb: Dict, llm: Optional[str], url=None) -> int:
    """Record that the design is being regenerated from these failures."""
    with xdb.session_scope(url) as s:
        row = M.Feedback(subject_type='sim_run', subject_id=fb['sim_run_id'], kind='correction',
                         body=fb['text'],
                         proposed={'action': 'regenerate_design', 'llm': llm, 'duv_id': fb.get('duv_id'),
                                   'failures': fb['count'], 'operations': fb['operations']},
                         status='open')
        s.add(row)
        s.flush()
        return row.id


def close_feedback(fb_id: int, status: str, result_ref: Optional[str] = None, url=None) -> None:
    with xdb.session_scope(url) as s:
        row = s.get(M.Feedback, fb_id)
        if row is not None:
            row.status, row.result_ref = status, result_ref


# Module types whose design Step 1 can regenerate with feedback (main.py).
DESIGN_REPAIR_TYPES = ('alu', 'regfile')


def available(s, run_id: int, module_type: Optional[str]) -> Dict:
    """What the Step-4 page can offer for this simulation, and whether its
    feature / design are themselves repaired versions."""
    from src.xdb.explain import _probe_results, _probe_run_id
    main_bad = {tag for tag, verdict in s.execute(
        select(M.CheckItem.tag, M.CheckResult.verdict)
        .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
        .where(M.CheckResult.sim_run_id == run_id, M.CheckItem.seq == 1,
               M.CheckItem.expected_ref.is_not(None))) if verdict in _DESIGN_FAULT}
    probe_bad = {tag for (tag, _), o in _probe_results(s, _probe_run_id(s, run_id)).items()
                 if o['verdict'] in _DESIGN_FAULT}
    # checks the testbench adds by itself (not from a scenario) can only fail on the design
    gen_bad = set(s.scalars(select(M.CheckItem.tag)
                            .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                            .where(M.CheckResult.sim_run_id == run_id, M.CheckItem.tag.like('GEN:%'),
                                   M.CheckResult.status == 'fail')))
    run = s.get(M.SimRun, run_id)
    tb = s.get(M.Testbench, run.testbench_id)
    feat = s.get(M.Feature, tb.feature_id) if tb.feature_id else None
    duv = s.get(M.Duv, run.duv_id) if run.duv_id else None
    return {'scenario_values': len(scenario_corrections(s, run_id)),
            'design_failures': len(main_bad | probe_bad | gen_bad),
            'design_regeneration': (module_type or '') in DESIGN_REPAIR_TYPES,
            'feature_reason': feat.reason if feat else None,
            'duv_reason': duv.reason if duv else None}
