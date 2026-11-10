"""
Explanation engine v1 (design doc §6; milestone M2). Deterministic: no model
is called. Every explanation is computed from stored data, stored as a
structure with its evidence (D19), and rendered to English text on demand.

    explain_check(tag, ...)        local: why did this assertion pass / fail,
                                   and who is wrong (verdict, D14), with a
                                   contrasting row of the same scenario (D20)
    explain_global(feature, port, value)
                                   global (BDX, P_T = ∅): in which situations
                                   does the specification state port = value
    explain_requirement(req, ...)  coverage of one requirement and the verdicts
                                   of the checks of its scenarios

BDX mapping (arXiv 2607.24881 §IV-C): steps are canonical steps (Σ, via
step_mapping); a target is T = (P_T, Q_T); the selected scenarios S_T are
those with Q_T ⊆ Q_s and the largest overlap with P_T; the explanation lists
their Given/When steps. Here a "scenario" is one Examples row of an outline,
i.e. the outline instantiated with that row's values.
"""

import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from src import xdb
from src.xdb import library as L
from src.xdb import models as M

# Bump whenever the structure changes, so stored explanations are recomputed
# instead of reused: v1.1 removed Chinese text, v1.2 added the rule check of
# global explanations and alternative meanings of contradicting flags.
ALGORITHM = 'bdx_v1.16'  # v1.16: what the testbench cannot show is never called verified

_OP_SYMBOL = {'ADD': '+', 'SUB': '−', 'AND': '&', 'OR': '|', 'XOR': '^', 'SLL': '<<',
              'SRL': '>>', 'SRA': '>>>', 'SLT': '<ₛ', 'SLTU': '<ᵤ'}


# ----------------------------------------------------------------- loading helpers

def _step_rows(s, scenario_id: int, canon: Optional[Dict] = None) -> Tuple[Dict, Dict, List]:
    """What _steps reads from the database. It is the same for every row of a
    scenario, so a loop over rows loads it once: at 75 ms per query (a laptop to
    Neon) three queries per row made the Step-4 summary take over 20 seconds."""
    maps = {m.position: m for m in s.scalars(select(M.StepMapping).where(
        M.StepMapping.scenario_id == scenario_id, M.StepMapping.method == 'rule'))}
    if canon is None:
        canon = {c.id: c for c in s.scalars(select(M.CanonicalStep))} if maps else {}
    return maps, canon, list(s.scalars(select(M.ScenarioStep).where(M.ScenarioStep.scenario_id == scenario_id)
                                       .order_by(M.ScenarioStep.position)))


def _steps(s, scenario_id: int, row_vals: Optional[Dict], module_type: str, width: Optional[int],
           loaded: Optional[Tuple] = None):
    """Scenario steps instantiated for one row, with their canonical mapping."""
    maps, canon, rows = loaded or _step_rows(s, scenario_id)
    out = []
    for st in rows:
        m = maps.get(st.position)
        c = canon.get(m.canonical_step_id) if m and m.canonical_step_id else None
        vals = L.resolve(m.params, row_vals) if m and m.params else {}
        port = (c.predicate or {}).get('port') if c else None
        vp = L.value_param((c.params or {}).get('key') if c else None, vals.keys())
        value = vals.get(vp) if port and vp else None
        out.append({'position': st.position, 'keyword': st.keyword, 'role': st.role,
                    'background': st.from_background,
                    'text': L.instantiate(st.surface_text, row_vals),
                    'canonical': (c.params or {}).get('key') if c else None,
                    'template': c.template if c else None,
                    'port': port, 'values': vals,
                    'value': value, 'value_int': L.to_int(value, width) if port else None})
    return out


def _overflow_mode(s, session_id: Optional[int]) -> Tuple[Optional[str], str]:
    """(definition, basis) of the overflow rule for a session: the Step-0 answer
    when the fixed question was answered, else the design template's."""
    from src import requirements_store as R
    d = s.scalar(select(M.Decision).where(M.Decision.session_id == session_id,
                                          M.Decision.question == R.OVERFLOW_QUESTION)) \
        if session_id else None
    if d is None:
        return L.DEFAULT_OVERFLOW_MODE, "the tool's design template, prompts/duv_alu"
    mode = L.overflow_mode_from_answer(d.answer)
    return mode, (f'decided in Step 0 ({d.question_key})' if mode
                  else f'custom answer in Step 0 ({d.question_key}), not checked')


def _width_from_steps(s, feature_id: int, module_type: str) -> Optional[int]:
    row = s.execute(select(M.StepMapping.params).join(M.Scenario, M.Scenario.id == M.StepMapping.scenario_id)
                    .join(M.CanonicalStep, M.CanonicalStep.id == M.StepMapping.canonical_step_id)
                    .where(M.Scenario.feature_id == feature_id,
                           M.CanonicalStep.params['key'].astext == 'ALU.P.WIDTH').limit(1)).first()
    return L.to_int((row[0] or {}).get('W', {}).get('lit')) if row else None


def _probe_run_id(s, main_run_id: Optional[int]) -> Optional[int]:
    if not main_run_id:
        return None
    return s.scalar(select(M.SimRun.id).where(M.SimRun.probe_of == main_run_id)
                    .order_by(M.SimRun.id.desc()).limit(1))


def _probe_results(s, probe_run_id: Optional[int]) -> Dict[Tuple[str, str], Dict]:
    """(row tag, flag port) -> observation of the flag probe (src/xdb/probe.py).

    The observation is a fact of the simulation; the rule value and the verdict
    depend on which overflow definition applies, so they are recomputed with the
    session's current one (a Step-0 answer may be recorded after the probe ran)."""
    from src.xdb.verdict import classify_flag
    if not probe_run_id:
        return {}
    pr = s.get(M.SimRun, probe_run_id)
    tb = s.get(M.Testbench, pr.testbench_id)
    sess = s.get(M.Session, tb.session_id)
    design = s.get(M.Design, sess.design_id) if sess.design_id else None
    mt = design.module_type if design else None
    ov_mode, _ = _overflow_mode(s, sess.id)
    out = {}
    for ci, cr in s.execute(select(M.CheckItem, M.CheckResult)
                            .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                            .where(M.CheckResult.sim_run_id == probe_run_id)):
        stim = ci.stimulus or {}
        w = (design.bitwidth if design else None) or 32
        R = L.flag_value(mt, ci.port, ci.operation, stim.get('a'), stim.get('b'), w, ov_mode)
        if R is None:
            continue
        O, B = _int(cr.observed_value), _int(ci.expected_bdd)
        raw = (cr.observed or {}).get(ci.port)
        out[(ci.tag, ci.port)] = {'observed': O, 'stated': B, 'rule_value': R,
                                  'verdict': classify_flag(O if O is not None else raw, B, R),
                                  'raw': raw, 'scenario_id': ci.scenario_id,
                                  'example_row_id': ci.example_row_id,
                                  'operation': ci.operation, 'stimulus': stim}
    return out


_OUTPUT_WORDS = ('result', 'zero', 'negative', 'overflow', 'carry')


def requirement_subject(text: str, outputs: Optional[List[str]] = None) -> Optional[str]:
    """The output a requirement is about: the first output port its text names
    (rule, reproducible). "The zero output is 1 when the result is 0" is about
    zero; "When opcode is ADD, result = a + b" is about result."""
    import re
    names = [o for o in (outputs or _OUTPUT_WORDS)]
    best = None
    for n in names:
        m = re.search(rf'\b{re.escape(n)}\b', text or '', flags=re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), n)
    return best[1] if best else None


def _subject(module_type: Optional[str], text: str, outputs: List[str]) -> Optional[str]:
    """The output a requirement is judged on. A register file is observed only
    through its read ports: everything it stores shows up on rdata1, unless the
    requirement names the second read port."""
    import re
    lib = L.library_for(module_type) or {}
    if lib.get('reference'):              # CPU: a branch decision, a stored word, or a register
        t = text or ''
        if re.search(r'\b(?:branch\w*|beq|bne|blt|bge|jump\w*|pc)\b', t, flags=re.I):
            return 'taken'
        return None                       # results reach the check through rd or memory alike
    if lib.get('stateful'):
        return 'rdata2' if re.search(r'\b(?:rdata2|raddr2|read port 2|second read port|port 2|'
                                     r'both read ports|two read ports)\b',
                                     text or '', flags=re.I) else 'rdata1'
    return requirement_subject(text, [o for o in outputs if o in _OUTPUT_WORDS] or None)


# A requirement that depends on these is not shown by the probe, which only applies operands.
_NEEDS_STATE = re.compile(r'\b(reset|rst|clock|edge|enable|load|initial)', re.I)


def _requirement_status(s, q: M.Requirement, sim_run_id: Optional[int]) -> Dict:
    """Status of one requirement from the evidence that exists for its subject port.

    verified           every check of the subject port in the linked scenarios passed (testbench)
    verified_by_probe  the testbench does not compare the port; the flag probe observed it
                       in every linked row and the design agrees with the rule
    fail               some check / probe observation of the subject port is not a pass
    stated_only        the scenarios state the port, nothing simulated it
    not_tested         linked scenarios, but no check ran
    checked_unlinked   no scenario claims it, but it is a flag's definition and the probe
                       observed that flag in every example
    not_covered        no scenario claims the requirement

    cannot_test (library) overrules every status but not_covered: passing rows say
    nothing about a requirement the testbench cannot show, and a linked scenario is
    only the model's claim (D43).
    """
    outputs = [p.name for p in s.scalars(select(M.Port).where(M.Port.design_id == q.design_id,
                                                              M.Port.direction == 'output'))]
    design = s.get(M.Design, q.design_id) if q.design_id else None
    subject = _subject(design.module_type if design else None, q.text, outputs)
    links = s.execute(select(M.ScenarioRequirement, M.Scenario)
                      .join(M.Scenario, M.Scenario.id == M.ScenarioRequirement.scenario_id)
                      .where(M.ScenarioRequirement.requirement_id == q.id, M.Scenario.is_current)).all()
    probe = _probe_results(s, _probe_run_id(s, sim_run_id))
    run = s.get(M.SimRun, sim_run_id) if sim_run_id else None
    # Which outputs the testbench compares is a property of the whole testbench: a
    # scenario that produced no checks does not make "result" uncompared.
    tb_ports = ({c for c in s.scalars(select(M.CheckItem.port).where(M.CheckItem.testbench_id == run.testbench_id))
                 if c} if run else set())
    scen, failing, untested = [], [], []
    main_ports_checked, main_n, probe_n = set(tb_ports), 0, 0
    lib_present = bool(L.library_for(design.module_type if design else None))
    unchecked = 0
    for link, sc in links:
        counts: Dict[str, int] = {}
        res = s.execute(select(M.CheckItem, M.CheckResult)
                        .join(M.CheckResult, (M.CheckResult.check_id == M.CheckItem.id)
                              & (M.CheckResult.sim_run_id == sim_run_id))
                        .where(M.CheckItem.scenario_id == sc.id)).all() if sim_run_id else []
        for ci, cr in res:
            if cr.status == 'setup':
                continue                 # a write or reset row checks nothing by itself
            if cr.verdict == 'undecided' and lib_present:
                unchecked += 1           # the example cannot be checked as written (D28, D29)
                continue
            main_ports_checked.add(ci.port)
            if subject and ci.port and ci.port != subject:
                continue
            main_n += 1
            k = cr.verdict or cr.status
            counts[k] = counts.get(k, 0) + 1
            if cr.status == 'fail':
                failing.append({'tag': ci.tag, 'port': ci.port, 'verdict': cr.verdict, 'via': 'testbench'})
        if subject and subject not in main_ports_checked:
            for (tag, port), o in probe.items():
                if port == subject and o['scenario_id'] == sc.id:
                    probe_n += 1
                    counts[o['verdict']] = counts.get(o['verdict'], 0) + 1
                    if o['verdict'] != 'pass':
                        failing.append({'tag': tag, 'port': port, 'verdict': o['verdict'], 'via': 'flag probe'})
        scen.append({'scenario_key': sc.scenario_key, 'title': sc.title, 'method': link.method,
                     'counts': counts})
        if not counts:
            untested.append({'scenario_key': sc.scenario_key, 'title': sc.title})
    compared = bool(subject is None or subject in main_ports_checked)
    seen = ([o for (_, port), o in probe.items() if port == subject]
            if not scen and subject in FLAG_PORTS and not _NEEDS_STATE.search(q.text or '') else [])
    if seen:
        status, evidence = 'checked_unlinked', 'flag probe'
    elif not scen:
        status, evidence = 'not_covered', None
    elif failing:
        status, evidence = 'fail', ('testbench' if compared else 'flag probe')
    elif untested and len(untested) == len(scen):
        status, evidence = 'not_tested', None
    elif compared and main_n:
        status, evidence = 'verified', 'testbench'
    elif not compared and probe_n:
        status, evidence = 'verified_by_probe', 'flag probe'
    elif not compared:
        status, evidence = 'stated_only', None
    else:
        status, evidence = 'not_tested', None
    limit = L.cannot_test(design.module_type if design else None, q.text)
    if limit and status != 'not_covered':
        status, evidence = 'not_tested', None
    return {'subject': subject, 'compared_by_testbench': compared, 'status': status,
            'cannot_test': limit, 'untested_scenarios': untested, 'unchecked': unchecked,
            'evidence': evidence, 'scenarios': scen, 'failing': failing,
            'probe_rows': len(seen), 'probe_right': sum(o['verdict'] == 'pass' for o in seen)}


def _call_summary(s, prov_ref: Optional[str]) -> Optional[Dict]:
    if not prov_ref or not prov_ref.startswith('llm_call:'):
        return None
    c = s.get(M.LlmCall, int(prov_ref.split(':', 1)[1]))
    if c is None:
        return None
    provider = (c.provider or '').removesuffix('Provider') or c.provider
    return {'llm_call_id': c.id, 'provider': provider, 'model': c.model_served or c.model_requested,
            'prompt_version': c.prompt_version}


def _int(v) -> Optional[int]:
    return int(v) if v is not None else None


def _store(s, target: Dict, structure: Dict, selected: List, evidence: Dict,
           auto_checks: Optional[Dict] = None) -> M.Explanation:
    """Reuse the explanation of an identical target; otherwise store a new one."""
    q = select(M.ExplanationTarget).where(
        M.ExplanationTarget.subject_type == target['subject_type'],
        M.ExplanationTarget.subject_id == target['subject_id'],
        M.ExplanationTarget.scope == target['scope'])
    for col in ('feature_id', 'sim_run_id'):
        v = target.get(col)
        q = q.where(getattr(M.ExplanationTarget, col).is_(None) if v is None
                    else getattr(M.ExplanationTarget, col) == v)
    for t in s.scalars(q):
        if t.q_t == target['q_t'] and t.p_t == target['p_t']:
            ex = s.scalar(select(M.Explanation).where(M.Explanation.target_id == t.id,
                                                      M.Explanation.algorithm == ALGORITHM,
                                                      M.Explanation.is_current))
            if ex is not None:
                return ex
    t = M.ExplanationTarget(entry=target.get('entry', 'click'), question=target.get('question'),
                            subject_type=target['subject_type'], subject_id=target['subject_id'],
                            feature_id=target.get('feature_id'), sim_run_id=target.get('sim_run_id'),
                            p_t=target['p_t'], q_t=target['q_t'], scope=target['scope'],
                            mode=target.get('mode', 'complete'))
    s.add(t)
    s.flush()
    ex = M.Explanation(target_id=t.id, algorithm=ALGORITHM, selected=selected, structure=structure,
                       text_rule=render(structure), evidence=evidence, auto_checks=auto_checks)
    s.add(ex)
    s.flush()
    return ex


def _out(ex: M.Explanation) -> Dict:
    return {'explanation_id': ex.id, 'structure': ex.structure, 'text': render(ex.structure),
            'plain': render_plain(ex.structure)}


# ----------------------------------------------------------------- local

def _latest_run_with(s, tag: str, session_id: Optional[int]) -> Optional[Tuple[M.CheckItem, M.SimRun]]:
    q = (select(M.CheckItem, M.SimRun)
         .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
         .join(M.SimRun, M.SimRun.id == M.CheckResult.sim_run_id)
         .join(M.Testbench, M.Testbench.id == M.CheckItem.testbench_id)
         .where(M.CheckItem.tag == tag, M.Testbench.generator != 'probe'))
    if session_id:
        q = q.where(M.Testbench.session_id == session_id)
    r = s.execute(q.order_by(M.SimRun.id.desc(), M.CheckItem.seq).limit(1)).first()
    return (r[0], r[1]) if r else None


# ----------------------------------------------------------------- stateful designs

def _rf_history(s, testbench_id: int, width: Optional[int]) -> Dict[str, Dict]:
    """Replay of the register file over the rows this testbench executed, by
    the library's own model (L.regfile_replay): value and cause of every read."""
    events = []
    for ci in s.scalars(select(M.CheckItem).where(M.CheckItem.testbench_id == testbench_id,
                                                  M.CheckItem.test_no.is_not(None),
                                                  M.CheckItem.seq == 1)):
        st = ci.stimulus or {}
        events.append({'tag': ci.tag, 'test': ci.test_no, 'op': ci.operation,
                       'addr': st.get('addr'), 'data': st.get('data')})
    return L.regfile_replay(events, width or 32)


def _row_ref(s, testbench_id: int, tag: Optional[str]) -> Dict:
    """Scenario title and example number of the row a tag names (what a user can find)."""
    if not tag:
        return {}
    row = s.execute(select(M.Scenario.title, M.ExampleRow.row_no)
                    .select_from(M.CheckItem)
                    .join(M.Scenario, M.Scenario.id == M.CheckItem.scenario_id)
                    .outerjoin(M.ExampleRow, M.ExampleRow.id == M.CheckItem.example_row_id)
                    .where(M.CheckItem.testbench_id == testbench_id, M.CheckItem.tag == tag)
                    .limit(1)).first()
    return {'title': row[0], 'row_no': row[1]} if row else {}


def _history_for(s, testbench_id: int, tag: str, hist: Optional[Dict], B: Optional[int]) -> Optional[Dict]:
    """Where the value of one read comes from, and which earlier write the
    scenario's value (if different) belongs to."""
    if not hist:
        return None
    cause = {**hist['cause'], **_row_ref(s, testbench_id, hist['cause'].get('tag'))}
    cause['same_row'] = hist['cause'].get('tag') == tag
    earlier = [{**w, **_row_ref(s, testbench_id, w['tag'])} for w in hist['earlier']]
    match = (next((w for w in earlier if w['value'] == B), None)
             if B is not None and B != hist['value'] else None)
    return {'value': hist['value'], 'cause': cause, 'earlier': earlier, 'scenario_value_from': match}


def explain_check(tag: str, sim_run_id: Optional[int] = None, session_id: Optional[int] = None,
                  seq: int = 1, url=None) -> Optional[Dict]:
    """Local explanation of one assertion, e.g. tag='BDD:ALU-SUB-001/r7'."""
    tag = tag if ':' in tag else f'BDD:{tag}'
    with xdb.session_scope(url) as s:
        if sim_run_id is None:
            hit = _latest_run_with(s, tag, session_id)
            if hit is None:
                return None
            ci, run = hit
        else:
            run = s.get(M.SimRun, sim_run_id)
            ci = s.scalar(select(M.CheckItem).where(M.CheckItem.testbench_id == run.testbench_id,
                                                    M.CheckItem.tag == tag, M.CheckItem.seq == seq))
        if ci is None or run is None:
            return None
        cr = s.get(M.CheckResult, (run.id, ci.id))
        tb = s.get(M.Testbench, ci.testbench_id)
        sess = s.get(M.Session, tb.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        mt = design.module_type if design else None
        sc = s.get(M.Scenario, ci.scenario_id) if ci.scenario_id else None
        er = s.get(M.ExampleRow, ci.example_row_id) if ci.example_row_id else None
        width = ci.width or (design.bitwidth if design else None)
        steps = _steps(s, sc.id, er.vals if er else None, mt, width) if sc else []

        B, R, O = _int(ci.expected_bdd), _int(ci.expected_ref), _int(cr.observed_value) if cr else None
        stim = ci.stimulus or {}
        rule = (L.library_for(mt) or {}).get('rules', {}).get((ci.operation or '').upper())
        rule_val = (L.apply_rule(mt, ci.operation, stim.get('a'), stim.get('b'), width)
                    if rule and width else None)
        from src.xdb import mapping as _mapping
        blocked = _mapping.blocked_steps(s, sc.id) if sc is not None and L.library_for(mt) else []
        history, cpu = None, None
        if (L.library_for(mt) or {}).get('reference') and stim:
            cpu = L.cpu_reference(stim)
            rule_val = cpu['value']
        if (L.library_for(mt) or {}).get('stateful'):
            history = _history_for(s, ci.testbench_id, ci.tag,
                                   _rf_history(s, ci.testbench_id, width).get(ci.tag), B)
            rule_val = history['value'] if history else None

        # scenario value vs. what was applied (e.g. -1 written, 2^W-1 applied: same bits)
        differs = []
        for st in steps:
            if st['role'] == 'P' and st['port'] in stim and st['port'] != 'opcode':
                if st['value_int'] is not None and stim[st['port']] is not None \
                        and st['value_int'] != stim[st['port']]:
                    differs.append({'port': st['port'], 'scenario': st['value'],
                                    'applied': stim[st['port']]})
        # Then steps no check of this row compares (e.g. ALU flags)
        compared = {c.port for c in s.scalars(select(M.CheckItem).where(
            M.CheckItem.testbench_id == ci.testbench_id, M.CheckItem.tag == ci.tag))}
        for port_, also in ((L.library_for(mt) or {}).get('port_aliases') or {}).items():
            if port_ in compared:
                compared |= set(also)
        unverified, probed = [], []
        ov_mode, _ = _overflow_mode(s, sess.id)
        probe = _probe_results(s, _probe_run_id(s, run.id))
        for st in steps:
            if st['role'] == 'Q' and st['port'] and st['port'] not in compared \
                    and (ci.tag, st['port']) in probe:
                o = probe[(ci.tag, st['port'])]
                probed.append({'port': st['port'], 'stated': st['value'], 'observed': o['observed'],
                               'raw': o['raw'], 'rule_value': o['rule_value'], 'verdict': o['verdict']})
                continue
            if st['role'] == 'Q' and st['port'] and st['port'] not in compared:
                # not simulated, but the scenario can still contradict the rule library
                expect = (L.flag_value(mt, st['port'], ci.operation, stim.get('a'), stim.get('b'), width,
                                       ov_mode) if width and stim.get('a') is not None else None)
                stated = L.to_int(st['value'], 1)
                consistent = None if expect is None or stated is None else expect == stated
                alts = ([a['name'] for a in L.flag_alternatives(mt, st['port'], ci.operation,
                                                                 stim.get('a'), stim.get('b'), width, ov_mode)
                         if a['value'] == stated] if consistent is False else [])
                unverified.append({'port': st['port'], 'value': st['value'], 'text': st['text'],
                                   'rule_value': expect, 'consistent': consistent,
                                   'matches_alternative': alts})
        for p in FLAG_PORTS:       # the flags the scenario leaves out: the probe saw them too
            o = probe.get((ci.tag, p))
            if o and p not in compared and p not in {x['port'] for x in probed}:
                probed.append({'port': p, 'stated': o['stated'], 'observed': o['observed'], 'raw': o['raw'],
                               'rule_value': o['rule_value'], 'verdict': o['verdict']})
        probed.sort(key=lambda x: FLAG_PORTS.index(x['port']))

        # the cause of a wrong result, when a known mix-up reproduces every example
        _alu_op = L.alu_has_cause(ci.operation) and not (L.library_for(mt) or {}).get('reference') \
            and not (L.library_for(mt) or {}).get('stateful')
        cause = None
        if _alu_op and cr and cr.verdict in ('design_wrong', 'both_wrong', 'consistent_wrong'):
            same_op = s.execute(select(M.CheckItem, M.CheckResult)
                                .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                                .where(M.CheckResult.sim_run_id == run.id, M.CheckItem.operation == ci.operation,
                                       M.CheckItem.seq == 1)).all()
            cause = L.alu_mixup(ci.operation, _alu_rows(same_op, ci.operation), width)

        contrast = None
        if sc is not None:
            # An ALU example has two operands, so the nearest example of the same
            # operation is the one with the fewest differing bits, in any scenario
            # (BDX: a minimal change). Elsewhere: the nearest row of this scenario.
            bits = _alu_op and stim.get('a') is not None and stim.get('b') is not None
            others = s.execute(
                select(M.CheckItem, M.CheckResult, M.ExampleRow)
                .join(M.CheckResult, (M.CheckResult.check_id == M.CheckItem.id)
                      & (M.CheckResult.sim_run_id == run.id))
                .join(M.ExampleRow, M.ExampleRow.id == M.CheckItem.example_row_id)
                .where((M.CheckItem.operation == ci.operation) if bits else (M.CheckItem.scenario_id == sc.id),
                       M.CheckItem.testbench_id == ci.testbench_id, M.CheckItem.seq == 1, M.CheckItem.id != ci.id,
                       M.CheckResult.status.in_(('pass', 'fail')))).all()
            mine_row = er.row_no if er else 0
            cands = [o for o in others if (o[1].status != (cr.status if cr else None))
                     or (o[1].verdict != (cr.verdict if cr else None))]

            def distance(o):
                if not bits or (o[0].stimulus or {}).get('a') is None:
                    return abs(o[2].row_no - mine_row)
                return (bin(int(o[0].stimulus['a']) ^ int(stim['a'])).count('1')
                        + bin(int(o[0].stimulus['b']) ^ int(stim['b'])).count('1'))
            # never compare with an example that only passes because the scenario is wrong too
            cands.sort(key=lambda o: (o[1].status == (cr.status if cr else None),
                                      o[1].verdict != 'pass', distance(o)))
            if cands:
                oc, ocr, oer = cands[0]
                contrast = {'row_no': oer.row_no, 'tag': oc.tag, 'status': ocr.status,
                            'same_with_fault': (cause is not None and L.alu_mixup_value(
                                ci.operation, int(oc.stimulus['a']), int(oc.stimulus['b']), width, cause)
                                == _int(oc.expected_ref)),
                            'verdict': ocr.verdict, 'stimulus': oc.stimulus,
                            'expected_bdd': _int(oc.expected_bdd), 'observed': _int(ocr.observed_value),
                            'expected_ref': _int(oc.expected_ref)}

        duv = s.get(M.Duv, run.duv_id) if run.duv_id else None
        structure = {
            'kind': 'local', 'version': 1,
            'target': {'tag': ci.tag, 'scenario_key': sc.scenario_key if sc else None,
                       'title': sc.title if sc else None, 'row_no': er.row_no if er else None,
                       'port': ci.port, 'status': cr.status if cr else None},
            'verdict': cr.verdict if cr else None,
            'values': {'observed': O, 'expected_bdd': B, 'expected_ref': R, 'width': width},
            'operation': ci.operation, 'module_type': mt, 'history': history, 'cpu': cpu, 'cause': cause,
            'blocked': blocked,
            'scenario': {'P': [x for x in steps if x['role'] == 'P'],
                         'Q': [x for x in steps if x['role'] == 'Q']},
            'stimulus': stim, 'stimulus_differs': differs,
            'rule': ({'op': rule['op'], 'expr': rule['expr'], 'text': rule['text'],
                      'value': rule_val, 'source': f'rule library {L.LIBRARY_VERSION}, hand-written'}
                     if rule else None),
            'unverified': unverified, 'probed': probed, 'contrast': contrast,
            'provenance': {'scenario': _call_summary(s, sc.prov_ref) if sc else None,
                           'design': _call_summary(s, duv.prov_ref) if duv else None,
                           'design_origin': duv.origin if duv else None,
                           'oracle': ci.oracle},
            'evidence': {'check_item_id': ci.id, 'sim_run_id': run.id, 'testbench_id': tb.id,
                         'feature_id': tb.feature_id, 'duv_id': run.duv_id,
                         'example_row_id': er.id if er else None, 'log_line': cr.message if cr else None},
        }
        p_t = [{'canonical': x['canonical'], 'values': x['values']} for x in steps
               if x['role'] == 'P' and x['canonical']]
        # the overflow definition in force is part of the target: a different
        # Step-0 answer must not reuse an explanation computed under the other one
        q_t = [{'port': ci.port, 'value': O, 'overflow_mode': ov_mode}]
        ex = _store(s, {'subject_type': 'check_item', 'subject_id': ci.id, 'scope': 'local',
                        'feature_id': tb.feature_id, 'sim_run_id': run.id, 'p_t': p_t, 'q_t': q_t,
                        'mode': 'complete'},
                    structure,
                    selected=[{'scenario_id': sc.id if sc else None,
                               'example_row_id': er.id if er else None, 'overlap': len(p_t)}],
                    evidence=structure['evidence'],
                    auto_checks={'rule_agrees_with_reference': (rule_val == R) if rule_val is not None
                                 and R is not None else None})
        return _out(ex)


# ----------------------------------------------------------------- global

def explain_global(feature_id: int, port: str, value, url=None) -> Dict:
    """In which situations does the specification (one feature version) state port = value?"""
    with xdb.session_scope(url) as s:
        feat = s.get(M.Feature, feature_id)
        sess = s.get(M.Session, feat.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        mt, width = (design.module_type, design.bitwidth) if design else (None, None)
        ov_mode, ov_basis = _overflow_mode(s, sess.id)
        if width is None:     # fall back to the Background step ("initialized with 16-bit operands")
            width = _width_from_steps(s, feature_id, mt)
        target_w = width if port == 'result' else 1
        want = L.to_int(value, target_w)
        matches, missed, total_rows = [], [], 0
        tally: Dict[str, Dict] = {}
        for sc in s.scalars(select(M.Scenario).where(M.Scenario.feature_id == feature_id,
                                                     M.Scenario.is_current)
                            .order_by(M.Scenario.scenario_key)):
            for er in s.scalars(select(M.ExampleRow).where(M.ExampleRow.scenario_id == sc.id,
                                                           M.ExampleRow.is_current)
                                .order_by(M.ExampleRow.row_no)):
                total_rows += 1
                steps = _steps(s, sc.id, er.vals, mt, width)
                q = [x for x in steps if x['role'] == 'Q' and x['port'] == port]
                op = next((x['values'].get('OP') for x in steps if x['canonical'] == 'ALU.A.OP'), None)
                a = next((x['value'] for x in steps if x['canonical'] == 'ALU.P.A'), None)
                b = next((x['value'] for x in steps if x['canonical'] == 'ALU.P.B'), None)
                rule_v = (L.flag_value(mt, port, op, a, b, width, ov_mode)
                          if port != 'result' and width else None)
                stated = L.to_int(q[0]['value'], target_w) if q and q[0]['value_int'] is not None else None
                if stated != want:
                    _tally_meanings(tally, mt, port, op, a, b, width, ov_mode, stated, rule_v)
                    if q and stated is not None and rule_v == want:
                        alts_m = [x['name'] for x in L.flag_alternatives(mt, port, op, a, b, width, ov_mode)
                                  if x['value'] == stated]
                        missed.append({'scenario_key': sc.scenario_key, 'row_no': er.row_no,
                                       'operation': op, 'stated': q[0]['value'], 'a': a, 'b': b,
                                       'alternatives': alts_m})
                    continue
                alts = [x for x in L.flag_alternatives(mt, port, op, a, b, width, ov_mode)
                        if x['value'] == want]
                _tally_meanings(tally, mt, port, op, a, b, width, ov_mode, stated, rule_v)
                matches.append({'scenario_key': sc.scenario_key, 'row_no': er.row_no, 'operation': op,
                                'P': [x['text'] for x in steps if x['role'] == 'P' and not x['background']],
                                'example_row_id': er.id, 'rule_value': rule_v,
                                'consistent': None if rule_v is None else rule_v == want,
                                'alternatives': [x['name'] for x in alts]})
        by_op: Dict[str, int] = {}
        for m in matches:
            by_op[m['operation'] or '?'] = by_op.get(m['operation'] or '?', 0) + 1
        verified = s.scalar(select(M.CheckItem.id).join(M.Testbench)
                            .where(M.Testbench.feature_id == feature_id, M.CheckItem.port == port,
                                   M.Testbench.generator != 'probe')
                            .limit(1)) is not None
        flag_rule = L.flag_rule(mt, port, ov_mode, ov_basis if port == 'overflow' else None)
        main_run = s.scalar(select(M.SimRun.id).join(M.Testbench)
                            .where(M.Testbench.feature_id == feature_id, M.SimRun.probe_of.is_(None))
                            .order_by(M.SimRun.id.desc()).limit(1))
        probe_stats: Optional[Dict] = None
        pr = [o for (tag, p_), o in _probe_results(s, _probe_run_id(s, main_run)).items() if p_ == port]
        if pr:
            probe_stats = {'rows': len(pr), 'design_agrees': sum(o['observed'] == o['rule_value'] for o in pr),
                           'verdicts': {}}
            for o in pr:
                probe_stats['verdicts'][o['verdict']] = probe_stats['verdicts'].get(o['verdict'], 0) + 1
        judged = [m for m in matches if m['consistent'] is not None]
        contra = [m for m in judged if not m['consistent']]
        alt_names = {n for m in contra for n in m['alternatives']}
        alternative = None
        for name in sorted(alt_names):
            if contra and all(name in m['alternatives'] for m in contra):
                alt = L.alternative_by_name(name)
                alternative = {'name': name, 'text': alt['text'], 'explains': len(contra),
                               'follows_rule': tally.get(name, {}).get('rule', 0),
                               'follows_other': tally.get(name, {}).get('other', 0),
                               'explains_missed': sum(name in m['alternatives'] for m in missed),
                               'missed_total': len(missed)}
        rule_check = ({'judged': len(judged), 'consistent': len(judged) - len(contra),
                       'contradicting': len(contra), 'missed': missed[:50], 'missed_count': len(missed),
                       'alternative': alternative} if flag_rule else None)
        structure = {'kind': 'global', 'version': 1,
                     'target': {'port': port, 'value': value, 'feature_id': feature_id,
                                'feature_title': feat.title},
                     'matches': matches, 'match_count': len(matches), 'rows_total': total_rows,
                     'by_operation': by_op, 'verified_by_testbench': verified, 'rule_check': rule_check,
                     'probe': probe_stats,
                     'rule': ({**flag_rule, 'source': f'rule library {L.LIBRARY_VERSION}, hand-written'}
                              if flag_rule else None),
                     'provenance': {'scenario': _call_summary(s, feat.prov_ref)}}
        ex = _store(s, {'subject_type': 'feature', 'subject_id': feature_id, 'scope': 'global',
                        'feature_id': feature_id, 'p_t': [],
                        'q_t': [{'port': port, 'value': want, 'overflow_mode': ov_mode}]},
                    structure, selected=[{'example_row_id': m['example_row_id']} for m in matches],
                    evidence={'feature_id': feature_id})
        return _out(ex)


# ----------------------------------------------------------------- requirement

def explain_requirement(requirement_id: int, sim_run_id: Optional[int] = None,
                        url=None) -> Optional[Dict]:
    with xdb.session_scope(url) as s:
        q = s.get(M.Requirement, requirement_id)
        if q is None:
            return None
        if sim_run_id is None:
            sim_run_id = s.scalar(select(M.SimRun.id).join(M.Testbench)
                                  .where(M.Testbench.session_id == q.session_id,
                                         M.SimRun.probe_of.is_(None))
                                  .order_by(M.SimRun.id.desc()).limit(1))
        st = _requirement_status(s, q, sim_run_id)
        structure = {'kind': 'requirement', 'version': 2,
                     'target': {'req_key': q.req_key, 'text': q.text},
                     **st,
                     'link_source': 'tag written by the model that generated the scenarios (not reviewed)'}
        ex = _store(s, {'subject_type': 'requirement', 'subject_id': q.id, 'scope': 'global',
                        'sim_run_id': sim_run_id, 'p_t': [], 'q_t': [{'requirement': q.req_key}]},
                    structure, selected=[{'scenario_key': x['scenario_key']} for x in st['scenarios']],
                    evidence={'sim_run_id': sim_run_id, 'requirement_id': q.id})
        return _out(ex)


# ----------------------------------------------------------------- rendering

_VERDICT = {'pass': 'Design and scenario both agree with the reference model.',
            'scenario_wrong': 'The scenario is wrong, not the design.',
            'design_wrong': 'The design is wrong; the scenario is right.',
            'both_wrong': 'Both the scenario and the design are wrong.',
            'consistent_wrong': 'The test passes, but scenario and design share the same mistake.',
            'undecided': 'There is no reference model for this check, so it cannot be decided who is wrong.',
            None: 'This check was not executed.'}

_SHORT = {'pass': 'pass', 'scenario_wrong': 'scenario wrong', 'design_wrong': 'design wrong',
          'both_wrong': 'both wrong', 'consistent_wrong': 'same mistake in both',
          'undecided': 'undecided', 'not_tested': 'not tested', 'setup': 'setup',
          'fail': 'fail', 'not_covered': 'not covered'}

_R_LINK = 'tag written by the model that generated the scenarios (not reviewed)'

_REQ_STATUS = {'verified': 'verified by simulation',
               'verified_by_probe': 'verified by the flag probe (the testbench does not compare it)',
               'fail': 'failing', 'stated_only': 'stated in the scenarios, not simulated',
               'not_tested': 'not tested', 'not_covered': 'not covered by any scenario',
               'checked_unlinked': 'no scenario linked; the flag probe checked it', 'pass': 'pass'}


def _num(v, width=None) -> str:
    if v is None:
        return '—'
    if width and width > 8 and v >= 1 << 8:
        return f'{v} (0x{v:0{(width + 3) // 4}X})'
    return str(v)


def _hex(v, width) -> str:
    return f'0x{v:0{(width + 3) // 4}X}'


def render(st: Dict) -> str:
    kind = st.get('kind')
    if kind == 'local':
        tg, vals, w = st['target'], st['values'], st['values'].get('width')
        status = 'pass' if tg.get('status') == 'pass' else 'fail'
        lines = [f"❓ Why did {tg.get('scenario_key')} row {tg.get('row_no')} {status}?   [local]",
                 '', f"Conclusion: {_VERDICT.get(st.get('verdict'), _VERDICT[None])}", '']
        P = '; '.join(f"{x['keyword']} {x['text']}" for x in st['scenario']['P'] if not x['background'])
        Q = [x for x in st['scenario']['Q'] if x['port'] == tg.get('port')]
        q_txt = f"Then {Q[0]['text']}" if Q else f"Then {tg.get('port')} = {_num(vals.get('expected_bdd'), w)}"
        lines.append(f"① Scenario: {P}; {q_txt}")
        stim = st.get('stimulus') or {}
        if stim and _is_cpu(st.get('module_type')):
            lines.append("② Applied: " + _cpu_example(stim)
                         + (f" (testbench reference {stim['tb_reference']})"
                            if stim.get('tb_reference') is not None else ''))
            for x in (st.get('cpu') or {}).get('issues') or []:
                lines.append(f"   ✗ {x['fault']} issue: {x['text']}")
            for x in (st.get('cpu') or {}).get('notes') or []:
                lines.append(f"   note: {x['text']}")
        elif stim:
            lines.append("② Applied: " + ', '.join(
                f"{k} = {_value(st.get('operation'), k, v, w) if k != 'opcode' else v}" for k, v in stim.items()))
        r = st.get('rule')
        h = st.get('history')
        if r:
            sym = _OP_SYMBOL.get(r['op'], r['op'])
            sa, sb = (L.signed_value(r['op'], 'a', stim.get('a'), w),
                      L.signed_value(r['op'], 'b', stim.get('b'), w))
            calc = (f"{_num(stim.get('a'), w) if sa is None else sa} {sym} "
                    f"{_num(stim.get('b'), w) if sb is None else sb} = {_value(r['op'], 'result', r.get('value'), w)}"
                    if stim.get('a') is not None else _num(r.get('value'), w))
            lines.append(f"③ Rule: {r['text']} → {calc}   ({r['source']})")
        if h:
            c = h['cause']
            src = {'write': lambda: f"last write: step {c['test']} ({_where(c)}), value {_num(c['value'], w)}",
                   'reset': lambda: f"cleared by the reset in step {c['test']} ({_where(c)})",
                   'initial': lambda: 'never written since the start (initial reset)',
                   'x0': lambda: 'register 0, hardwired to 0'}[c['kind']]()
            prev = '; '.join(f"step {x['test']} wrote {_num(x['value'], w)}" + (' (ignored, register 0)'
                                                                                 if x.get('ignored') else '')
                             for x in sorted(h.get('earlier') or [], key=lambda x: x['test']))
            lines.append(f"   Replay of the rows before this read: {src}" + (f"; earlier: {prev}" if prev else ''))
        o = vals.get('observed')
        agree = (('agrees with the rule' if r.get('value') == o else 'does not agree with the rule')
                 if r and o is not None else '')
        lines.append(f"④ Design output: {tg.get('port')} = {_value(st.get('operation'), 'result', o, w)}  "
                     f"{('— ' + agree) if agree else ''}")
        c = st.get('contrast')
        if c:
            cs = c.get('stimulus') or {}
            c_status = 'passed' if c['status'] == 'pass' else 'failed'
            what = (_example(st.get('module_type'), st.get('operation'), cs, w) if _stateful(st.get('module_type'))
                    else f"a = {_operand(st.get('operation'), 'a', cs, w)}, "
                         f"b = {_operand(st.get('operation'), 'b', cs, w)}")
            where = c['tag'][4:] if (c.get('tag') or '').startswith('BDD:') else f"row {c['row_no']}"
            lines.append(f"⑤ Contrast: {where} ({what}) {c_status} — {_VERDICT.get(c.get('verdict'), '')}")
        for d in st.get('stimulus_differs') or []:
            lines.append(f"Note: the scenario writes {d['scenario']} for {d['port']}; the testbench applied "
                         f"{d['applied']} (same bits as a {w}-bit value)")
        lines.append('')
        pv = (st.get('provenance') or {}).get('scenario')
        ev = st.get('evidence') or {}
        src = (f"; scenario written by {pv['provider']} {pv['model']} (prompt {pv.get('prompt_version') or '?'})"
               if pv else '')
        lines.append(f"🔎 Evidence: check [{tg.get('tag')}], simulation #{ev.get('sim_run_id')}{src}")
        if st.get('probed'):
            parts = []
            for x in st['probed']:
                ok = x['verdict'] == 'pass'
                obs = x['observed'] if x['observed'] is not None else x['raw']
                parts.append(f"{x['port']} = {obs} {'✓' if ok else '✗ ' + _SHORT.get(x['verdict'], x['verdict'])}"
                             f" (scenario {x['stated'] if x['stated'] is not None else 'not stated'}, rule {x['rule_value']})")
            lines.append("🔬 Flags, observed by a separate probe simulation (not part of the pass rate): "
                         + '; '.join(parts))
        if st.get('unverified'):
            lines.append("⚠ Not verified: " + ', '.join(x['text'] for x in st['unverified'])
                         + " — stated in the scenario, but the testbench does not compare these")
            bad = [x for x in st['unverified'] if x.get('consistent') is False]
            if bad:
                lines.append("   ✗ and the scenario contradicts the rule library here: " + ', '.join(
                    f"{x['port']} = {x['value']} (rule: {x['rule_value']}"
                    + (f"; matches {', '.join(x['matches_alternative'])}" if x.get('matches_alternative') else '')
                    + ')' for x in bad))
        return '\n'.join(lines)

    if kind == 'global':
        tg = st['target']
        lines = [f"❓ When does the specification say {tg['port']} = {tg['value']}?   [global]", '']
        if not st['matches']:
            lines.append('No row of this specification states it.')
        else:
            ops = ', '.join(f'{k} ×{n}' for k, n in st['by_operation'].items())
            lines.append(f"{st['match_count']} of {st['rows_total']} rows state it ({ops}):")
            # rows that contradict the rule first: they are the point of the explanation
            ordered = sorted(st['matches'], key=lambda m: m.get('consistent') is not False)
            for m in ordered[:20]:
                mark = '' if m.get('consistent') is None else (
                    '   [✓ rule]' if m['consistent'] else f"   [✗ rule gives {m['rule_value']}]")
                lines.append(f"  • {m['scenario_key']} r{m['row_no']}: " + '; '.join(m['P']) + mark)
            if len(st['matches']) > 20:
                lines.append(f"  … (+{len(st['matches']) - 20})")
        if st.get('rule'):
            basis = st['rule'].get('basis')
            lines += ['', f"General rule: {st['rule']['text']}"
                          + (f" ({basis})" if basis else '') + f"   ({st['rule']['source']})"]
        rc = st.get('rule_check')
        if rc and (rc['judged'] or rc['missed_count']):
            n = rc['contradicting']
            lines.append(f"Check against the rule: {rc['consistent']} of {rc['judged']} rows agree"
                         + (f", {n} contradict{'s' if n == 1 else ''} it." if n else '.'))
            if rc['missed_count']:
                lines.append(f"Rows where the rule gives {tg['port']} = {tg['value']} but the specification "
                             f"states otherwise ({rc['missed_count']}): " + ', '.join(
                                 f"{x['scenario_key']} r{x['row_no']} (A = {x['a']}, B = {x['b']}, stated {x['stated']})"
                                 for x in rc['missed'][:10]))
            alt = rc.get('alternative')
            if alt:
                also = ''
                if alt.get('missed_total'):
                    n, k = alt['missed_total'], alt['explains_missed']
                    if k == n:
                        also = (', and so does the one row it omits' if n == 1
                                else f', and so do all {n} rows it omits')
                    elif k:
                        also = f', and so do {k} of the {n} rows it omits'
                lines.append(f"The {alt['explains']} contradicting rows all match a different definition: "
                             f"{alt['name']} ({alt['text']}){also}. "
                             + (f"Where the two differ, {_examples(alt['follows_other'])} it and "
                                f"{_examples(alt['follows_rule'])} the rule, so the specification mixes both."
                                if alt.get('follows_rule') else
                                f"The specification seems to use \"{tg['port']}\" in that sense, unlike the rule"
                                f"{' (' + st['rule']['basis'] + ')' if (st.get('rule') or {}).get('basis') else ''}."))
        pb = st.get('probe')
        if pb:
            lines += ['', f"🔬 Design output, observed by the flag probe: {tg['port']} agrees with the rule in "
                          f"{pb['design_agrees']} of {pb['rows']} rows"
                          + (f"; {pb['rows'] - pb['design_agrees']} differ." if pb['rows'] != pb['design_agrees'] else '.')]
        if not st.get('verified_by_testbench'):
            lines += ['', f"⚠ The testbench never compares {tg['port']}"
                          + (", so the rows above are stated, checked by the probe only." if pb
                             else ", so these rows are stated, not verified.")]
        return '\n'.join(lines)

    if kind == 'requirement':
        tg = st['target']
        lines = [f"Requirement {tg['req_key']}: {tg['text']}", '',
                 f"Status: {_REQ_STATUS.get(st['status'], st['status'])}"]
        if st.get('subject'):
            how = ('compared by the testbench' if st.get('compared_by_testbench')
                   else 'not compared by the testbench' + ('; observed by the flag probe'
                                                           if st.get('evidence') == 'flag probe' else ''))
            lines.append(f"About: {st['subject']} (the first output the requirement names) — {how}.")
        if not st['scenarios']:
            lines.append('No scenario claims to cover this requirement.')
        for x in st['scenarios']:
            cnt = ', '.join(f"{_SHORT.get(k, k)} {n}" for k, n in sorted(x['counts'].items())) or '—'
            lines.append(f"  • {x['scenario_key']} {x['title'] or ''}: {cnt}")
        if st['failing']:
            lines.append('Failing checks: ' + ', '.join(
                f"{f['tag']}{' (' + f['port'] + ', ' + f['via'] + ')' if f.get('via') == 'flag probe' else ''}"
                f" → {_SHORT.get(f['verdict'], f['verdict'])}" for f in st['failing'][:20])
                + (f" … (+{len(st['failing']) - 20})" if len(st['failing']) > 20 else ''))
        lines.append(f'Links: {_R_LINK}')
        return '\n'.join(lines)
    return ''


# ----------------------------------------------------------------- whole-specification check

FLAG_PORTS = ('zero', 'negative', 'overflow')


def _examples(n: int, where: str = '') -> str:
    """'1 example follows', '3 examples, in ADD and SUB, follow'."""
    return f"{n} example{where} follows" if n == 1 else f"{n} examples{where} follow"


def _tally_meanings(tally: Dict, mt, port, op, a, b, width, mode, stated, rule_v):
    """Where the rule and another meaning of a flag give different values, which
    one did the example follow? Only such examples can tell the two apart."""
    if rule_v is None or stated is None:
        return
    for x in L.flag_alternatives(mt, port, op, a, b, width, mode):
        if x['value'] != rule_v and stated in (rule_v, x['value']):
            t = tally.setdefault(x['name'], {'rule': 0, 'other': 0})
            t['rule' if stated == rule_v else 'other'] += 1


def check_specification(feature_id: int, sim_run_id: Optional[int] = None, url=None) -> List[Dict]:
    """For every flag the scenarios state, compare each row with the rule.

    This is a check of the specification itself: flags are usually not
    compared by the testbench, so a simulation can pass while the scenarios
    state wrong flags (design doc §6a, first field observation)."""
    with xdb.session_scope(url) as s:
        feat = s.get(M.Feature, feature_id)
        sess = s.get(M.Session, feat.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        mt, width = (design.module_type, design.bitwidth) if design else (None, None)
        if not (L.library_for(mt) or {}).get('flags'):
            return []                      # nothing but ALU flags is checked this way
        width = width or _width_from_steps(s, feature_id, mt)
        ov_mode, ov_basis = _overflow_mode(s, sess.id)
        per: Dict[str, Dict] = {}
        canon = {c.id: c for c in s.scalars(select(M.CanonicalStep))}
        for sc in s.scalars(select(M.Scenario).where(M.Scenario.feature_id == feature_id,
                                                     M.Scenario.is_current)
                            .order_by(M.Scenario.scenario_key)):
            loaded = _step_rows(s, sc.id, canon)
            for er in s.scalars(select(M.ExampleRow).where(M.ExampleRow.scenario_id == sc.id,
                                                           M.ExampleRow.is_current)
                                .order_by(M.ExampleRow.row_no)):
                steps = _steps(s, sc.id, er.vals, mt, width, loaded)
                op = next((x['values'].get('OP') for x in steps if x['canonical'] == 'ALU.A.OP'), None)
                a = next((x['value'] for x in steps if x['canonical'] == 'ALU.P.A'), None)
                b = next((x['value'] for x in steps if x['canonical'] == 'ALU.P.B'), None)
                for x in steps:
                    if x['role'] != 'Q' or x['port'] not in FLAG_PORTS:
                        continue
                    p = per.setdefault(x['port'], {'port': x['port'], 'rows_stated': 0, 'judged': 0,
                                                   'contradicting': 0, 'examples': [], '_alts': [],
                                                   '_tally': {}})
                    p['rows_stated'] += 1
                    stated = L.to_int(x['value'], 1)
                    rule_v = L.flag_value(mt, x['port'], op, a, b, width, ov_mode) if width else None
                    if rule_v is None or stated is None:
                        continue
                    p['judged'] += 1
                    _tally_meanings(p['_tally'], mt, x['port'], op, a, b, width, ov_mode, stated, rule_v)
                    if rule_v != stated:
                        p['contradicting'] += 1
                        alts = [al['name'] for al in L.flag_alternatives(mt, x['port'], op, a, b, width,
                                                                          ov_mode) if al['value'] == stated]
                        p['_alts'].append(alts)
                        if len(p['examples']) < 10:
                            p['examples'].append({'scenario_key': sc.scenario_key, 'row_no': er.row_no,
                                                  'operation': op, 'a': a, 'b': b, 'stated': x['value'],
                                                  'rule_value': rule_v})
        if sim_run_id is None:
            sim_run_id = s.scalar(select(M.SimRun.id).join(M.Testbench)
                                  .where(M.Testbench.feature_id == feature_id, M.SimRun.probe_of.is_(None))
                                  .order_by(M.SimRun.id.desc()).limit(1))
        probe = _probe_results(s, _probe_run_id(s, sim_run_id))
        out = []
        for port, p in per.items():
            alts, meanings = p.pop('_alts'), p.pop('_tally')
            pr = [o for (tag, p_), o in probe.items() if p_ == port]
            p['probe'] = ({'rows': len(pr), 'design_agrees': sum(o['observed'] == o['rule_value'] for o in pr)}
                          if pr else None)
            alternative = None
            for name in sorted({n for x in alts for n in x}):
                if alts and all(name in x for x in alts):
                    alternative = {**L.alternative_by_name(name), 'explains': len(alts),
                                   'follows_rule': meanings.get(name, {}).get('rule', 0)}
            verified = s.scalar(select(M.CheckItem.id).join(M.Testbench)
                                .where(M.Testbench.feature_id == feature_id, M.CheckItem.port == port,
                                       M.Testbench.generator != 'probe')
                                .limit(1)) is not None
            out.append({**p, 'alternative': alternative, 'verified_by_testbench': verified,
                        'rule': L.flag_rule(mt, port, ov_mode, ov_basis if port == 'overflow' else None)})
        return sorted(out, key=lambda x: FLAG_PORTS.index(x['port']))


def render_spec_check(items: List[Dict]) -> List[str]:
    lines = []
    for it in items:
        head = f"{it['port']} flag: stated in {it['rows_stated']} rows"
        if not it['rule']:
            lines.append(f"{head}; no rule to check it against.")
            continue
        if not it['judged']:
            lines.append(f"{head}; none could be checked against the rule.")
        elif not it['contradicting']:
            lines.append(f"{head}; all {it['judged']} agree with the rule.")
        else:
            alt = it.get('alternative')
            n = it['contradicting']
            lines.append(f"{head}; {n} of {it['judged']} contradict{'s' if n == 1 else ''} the rule"
                         + (f" — all of them match {alt['name']} ({alt['text']})" if alt else '') + '.')
        pb = it.get('probe')
        if not it['verified_by_testbench'] and pb:
            lines[-1] += (f" Not compared by the testbench; the flag probe saw the design agree with the rule "
                          f"in {pb['design_agrees']} of {pb['rows']} rows.")
        elif not it['verified_by_testbench']:
            lines[-1] += ' Not compared by the testbench, and not simulated.'
    return lines


# ----------------------------------------------------------------- website summary

def summarize_run(project_root, paths: Dict[str, Optional[str]], url=None) -> Optional[Dict]:
    """What the Step-4 page needs, for the simulation whose testbench and log
    are these files. None until the run has been written to the database."""
    import hashlib
    from src import run_report
    tb, err1 = run_report.safe_path(paths.get('testbench_path') or paths.get('testbench'), project_root)
    log, err2 = run_report.safe_path(paths.get('log_file') or paths.get('log'), project_root)
    if err1 or err2 or not tb or not log:
        return None
    tb_sha = hashlib.sha256(tb.read_bytes()).hexdigest()
    log_sha = hashlib.sha256(log.read_bytes()).hexdigest()
    with xdb.session_scope(url) as s:
        hit = s.execute(select(M.SimRun, M.Testbench).join(M.Testbench, M.Testbench.id == M.SimRun.testbench_id)
                        .where(M.Testbench.artifact == tb_sha, M.SimRun.log_artifact == log_sha)
                        # the same files stored again without their scenarios and design
                        # (a call with only these two paths) must not replace the full run
                        .order_by(M.Testbench.feature_id.is_(None), M.SimRun.id.desc()).limit(1)).first()
        if hit is None:
            return None
        run, tbrow = hit
        sess = s.get(M.Session, tbrow.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        checks, verdicts = {}, {}
        for ci, cr in s.execute(select(M.CheckItem, M.CheckResult)
                                .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                                .where(M.CheckResult.sim_run_id == run.id)):
            key = cr.verdict or cr.status
            verdicts[key] = verdicts.get(key, 0) + 1
            if ci.seq == 1:
                checks[ci.tag] = {'verdict': cr.verdict, 'status': cr.status}
        reqs = []
        for q in s.scalars(select(M.Requirement).where(M.Requirement.session_id == sess.id,
                                                       M.Requirement.is_current).order_by(M.Requirement.id)):
            rs = _requirement_status(s, q, run.id)
            reqs.append({'req_key': q.req_key, 'requirement_id': q.id, 'status': rs['status'],
                         'cannot_test': rs.get('cannot_test'),
                         'text': q.text, 'untested_scenarios': rs.get('untested_scenarios') or [],
                         'subject': rs['subject'], 'compared_by_testbench': rs['compared_by_testbench'],
                         'unchecked': rs.get('unchecked', 0), 'probe_rows': rs['probe_rows'],
                         'probe_right': rs['probe_right'], 'evidence': rs['evidence'],
                         'failing_verdicts': sorted({f['verdict'] for f in rs['failing'] if f['verdict']})})
        prid = _probe_run_id(s, run.id)
        probe_info = None
        if prid:
            pr = s.get(M.SimRun, prid)
            pv: Dict[str, int] = {}
            issues: Dict[str, List[Dict]] = {}
            per_port: Dict[str, Dict[str, int]] = {}
            for (tag, port), o in _probe_results(s, prid).items():
                pv[o['verdict']] = pv.get(o['verdict'], 0) + 1
                pp = per_port.setdefault(port, {'rows': 0, 'design_agrees': 0})
                pp['rows'] += 1
                pp['design_agrees'] += int(o['observed'] == o['rule_value'])
                if o['verdict'] != 'pass':
                    issues.setdefault(tag, []).append({'port': port, 'verdict': o['verdict']})
            probe_info = {'sim_run_id': prid, 'status': pr.status, 'verdicts': pv,
                          'per_port': per_port, 'issues': issues}
        compared_ports = sorted({c for c in s.scalars(select(M.CheckItem.port)
                                                      .where(M.CheckItem.testbench_id == tbrow.id)) if c})
        feature_id, session_id, sim_run_id = tbrow.feature_id, sess.id, run.id
        mt = design.module_type if design else None
        design_name = design.name if design else None
    spec = check_specification(feature_id, sim_run_id=sim_run_id, url=url) if feature_id else []
    from src import scenario_results as _sr
    try:
        sr = _sr.summarize_for_testbench(tb, log.read_text(encoding='utf-8', errors='replace'))
    except Exception:
        sr = None
    from src.xdb import repair as _repair
    with xdb.session_scope(url) as s:
        problems, overview = build_problems(s, sim_run_id, sr, reqs, spec)
        repairs = _repair.available(s, sim_run_id, mt)
    return {'sim_run_id': sim_run_id, 'session_id': session_id, 'feature_id': feature_id,
            'repairs': repairs,
            'design': design_name, 'module_type': mt,
            'has_rule_library': L.library_for(mt) is not None,
            'verdicts': verdicts, 'checks': checks, 'requirements': reqs,
            'compared_ports': compared_ports, 'probe': probe_info,
            'spec_check': spec, 'spec_check_text': render_spec_check(spec),
            'problems': problems, 'overview': overview,
            'algorithm': ALGORITHM}



# ----------------------------------------------------------------- plain language
# What a user sees first: no internal ids (scenario keys, check tags, run
# numbers), one short sentence per point. The technical rendering above stays
# available behind "Technical details".

_PLAIN_REASON = {
    'tag maps to no known ALU operation':
        'the tool could not tell which single operation it tests, so it made no tests from it',
    'tag maps to no known register-file operation':
        'the tool could not tell which register-file operation it tests, so it made no tests from it',
    'no Examples rows': 'it has no Examples table, so there was nothing to test',
    'scenario states no expected result': 'it states no expected result, so there was nothing to compare',
    'rows name no known ALU operation':
        'the operation its examples name is not one the tool knows, so it made no tests from them',
    'row names no known ALU operation':
        'the operation its examples name is not one the tool knows, so it made no tests from them',
    'rows malformed (column count differs from header)':
        'its Examples rows do not match the table header, so they could not be read',
    'plain Scenario without an Examples table':
        'it is a plain scenario without an Examples table; only Scenario Outlines are turned into tests',
    'no assertion was generated': 'no test could be generated from its rows',
}


def plain_reason(reason: Optional[str]) -> str:
    if not reason:
        return 'no test could be generated from it'
    for k, v in _PLAIN_REASON.items():
        if reason.startswith(k):
            return v
    return reason


def _pv(v, w=None) -> str:
    """A value as a user reads it: decimal, plus hex when it is wide."""
    if v is None:
        return '—'
    if isinstance(v, str):
        return v
    if w and w > 8 and v >= 1 << 8:
        return f'{v} (0x{v:0{(w + 3) // 4}X})'
    return str(v)


def _with_width(text: str, w: Optional[int]) -> str:
    if not w:
        return text
    return text.replace('2^W', f'2^{w}').replace('log2(W)', str((w - 1).bit_length()))


def _value(op, name, v, w) -> str:
    """A value as the operation reads it: 0xFFFF is -1 for SLT, 65535 for SLTU."""
    signed = L.signed_value(op, name, v, w)
    return _pv(v, w) if signed is None else f'{_hex(v, w)} ({signed} as signed)'


def _operand(op, name, stim, w) -> str:
    return _value(op, name, (stim or {}).get(name), w)


def _overflow_basis(s, session_id) -> str:
    """Which meaning of overflow a card relies on, and who decided it."""
    mode, basis = _overflow_mode(s, session_id)
    name = L.OVERFLOW_DEFINITIONS.get(mode, {}).get('name')
    if not name:
        return ''
    if basis.startswith('decided'):
        return f'Overflow is judged as {name}, as decided in Step 0.'
    return f"Overflow is judged as {name}, the design template's definition, because Step 0 did not decide it."


def _inputs(op, stim, w) -> str:
    return f"{op or 'operation'} with A = {_operand(op, 'a', stim, w)}, B = {_operand(op, 'b', stim, w)}"


def _alu_rows(main, op) -> List[Tuple[int, int, int]]:
    """(a, b, design output) of every simulated example of an ALU operation."""
    rows = []
    for ci, cr in main:
        st = ci.stimulus or {}
        if (ci.tag.startswith('BDD:') and ci.operation == op and cr.status in ('pass', 'fail')
                and cr.observed_value is not None and st.get('a') is not None and st.get('b') is not None):
            rows.append((int(st['a']), int(st['b']), int(cr.observed_value)))
    return rows


def _lost_bit_card(main, by_op, same_mistake, total_op, ops: List[str], cause: Dict, width: int) -> Dict:
    """One card for every shift that loses the same bit of the shift amount, and a
    word on the shifts whose examples pass only because they cannot show it."""
    names = ' and '.join(ops)
    ci, cr = by_op[('design_wrong', ops[0])][0]
    also = sum(len(same_mistake.get(o) or []) for o in ops)
    n = sum(len(by_op[('design_wrong', o)]) for o in ops) + also
    fit = sum(len(_alu_rows(main, o)) for o in ops)
    res = lambda x, op_=ops[0]: _value(op_, 'result', x, width)
    detail = (f"{n} of {sum(total_op.get(o, 0) for o in ops)} {names} examples give a wrong result."
              + (f' {also} of them pass the test, because the scenario makes the same mistake.' if also else '')
              + f" E.g. {_inputs(ops[0], ci.stimulus, width)}: the design gives {res(_int(cr.observed_value))}, "
                f"correct is {res(_int(ci.expected_ref))}. All {fit} {names} examples fit this, so {cause['fix']}.")
    for other in [o for o in L.SHIFTS if o not in ops]:
        rows = _alu_rows(main, other)
        if L.alu_shift_hidden(other, rows, cause['bit'], width):
            a, b, right, bad = L.alu_shift_witness(other, cause['bit'], width)
            detail += (f" The {len(rows)} {other} examples pass, but none of them can show this fault, so {other} "
                       f"may have it too: {_inputs(other, {'a': a, 'b': b}, width)} would show it, correct is "
                       f"{res(right, other)}, with this fault {res(bad, other)}.")
    return {'severity': 'error', 'who': 'design', 'title': f"The design {cause['says']}", 'detail': detail,
            'explain': {'kind': 'check', 'tag': ci.tag}, 'ops': ops}


def _with_repeats(detail: str, shown_alone: int, repeats: int) -> str:
    """A scenario card, when in some of its examples the design makes the same
    mistake: the test passes there, so the card must not say the design is right."""
    if not repeats:
        return detail
    if not shown_alone:
        return detail.replace(' The design is right.', ' The design gives the same wrong value, so the test passes.')
    return detail + f' In {repeats} of them the design gives the same wrong value, so the test passes.'


def _swapped_scenarios(s, main, width) -> Tuple[List[Dict], set]:
    """Scenarios whose two operations are exchanged throughout: every example
    of both gives the other operation's result. Returns the cards and the
    examples they cover."""
    by_scenario: Dict[int, List] = {}
    for ci, cr in main:
        st = ci.stimulus or {}
        if (ci.tag.startswith('BDD:') and ci.scenario_id and L.alu_swap_partner(ci.operation)
                and ci.expected_bdd is not None and ci.expected_ref is not None
                and st.get('a') is not None and st.get('b') is not None):
            by_scenario.setdefault(ci.scenario_id, []).append((ci, cr))
    cards, covered = [], set()
    for sid, rows in by_scenario.items():
        ops = {ci.operation for ci, _ in rows}
        if len(ops) != 2 or L.alu_swap_partner(min(ops)) != max(ops):
            continue
        wrong = [(ci, cr) for ci, cr in rows if _int(ci.expected_bdd) != _int(ci.expected_ref)]
        fits = all(L.alu_mixup_value(ci.operation, int(ci.stimulus['a']), int(ci.stimulus['b']), width)
                   == _int(ci.expected_bdd) for ci, _ in rows)
        if not fits or {ci.operation for ci, _ in wrong} != ops:
            continue
        first = next((ci for ci, cr in wrong if cr.verdict == 'scenario_wrong'), wrong[0][0])
        sc = s.get(M.Scenario, sid)
        a, b = sorted(ops)
        cards.append({'severity': 'warning', 'who': 'scenario', 'title': f'A scenario swaps {a} and {b}',
                      'detail': f"“{sc.title}”: all {len(rows)} examples give the other operation's result. "
                                f"E.g. {_inputs(first.operation, first.stimulus, width)}: the scenario expects "
                                f"{_value(first.operation, 'result', _int(first.expected_bdd), width)}, correct is "
                                f"{_value(first.operation, 'result', _int(first.expected_ref), width)}.",
                      'explain': {'kind': 'check', 'tag': first.tag}})
        covered |= {ci.tag for ci, _ in rows}
    return cards, covered


_FLAG_WORD = {'zero': 'zero flag', 'negative': 'negative flag', 'overflow': 'overflow flag'}


def _flag_list(ports: List[str]) -> str:
    """'zero flag', 'zero and negative flags', 'zero, negative and overflow flags'."""
    if len(ports) == 1:
        return f'{ports[0]} flag'
    return f"{', '.join(ports[:-1])} and {ports[-1]} flags"


def _stateful(module_type: Optional[str]) -> bool:
    return bool((L.library_for(module_type) or {}).get('stateful'))


def _example(module_type: Optional[str], op, stim, w) -> str:
    """One example in the words of its design type."""
    if not _stateful(module_type):
        return _inputs(op, stim, w)
    stim = stim or {}
    a, d = stim.get('addr'), stim.get('data')
    op = (op or '').upper()
    if op == 'WRITE_READ':
        return f'writing {_pv(d, w)} to R{a} and reading it back'
    if op == 'WRITE':
        return f'writing {_pv(d, w)} to R{a}'
    if op == 'RESET':
        return 'reset'
    return f'reading R{a}'


def _rv(v, w=None) -> str:
    """A register value: decimal, hex only when it is large (data, not bit patterns)."""
    return _pv(v, w if isinstance(v, int) and v >= 1 << 16 else None)


def _where(ref: Dict, other: Optional[Dict] = None) -> str:
    """'example 4 of “Write values into registers”' -- what a user can look up;
    'example 4 of the same scenario' when `other` names the same scenario."""
    if not ref or not ref.get('title'):
        return 'an earlier row'
    if other and other.get('title') == ref['title'] and ref.get('row_no'):
        return f"example {ref['row_no']} of the same scenario"
    return (f"example {ref['row_no']} of “{ref['title']}”" if ref.get('row_no')
            else f"“{ref['title']}”")


def _cause_text(h: Dict, a, w) -> str:
    """Why a register holds the value it holds at a read."""
    c = h['cause']
    if c['kind'] == 'x0':
        return 'register 0 always reads 0; writes to it are ignored'
    if c['kind'] == 'initial':
        return f'nothing wrote R{a} since the simulation started, and every register starts at 0'
    if c['kind'] == 'reset':
        return f'the reset in {_where(c)} cleared it, and nothing wrote R{a} after that'
    if c.get('same_row'):
        return f'this row wrote {_rv(c["value"], w)} to R{a} just before reading it'
    return f'{_where(c)} wrote {_rv(c["value"], w)} to R{a}, and nothing wrote R{a} after that'


def _scenario_value_text(h: Dict, B, a, w) -> Optional[str]:
    """Where the scenario's (wrong) value comes from: an earlier write that a
    later write, a reset, or the register-0 rule made obsolete."""
    m = h.get('scenario_value_from')
    if not m:
        return None
    c = h['cause']
    if m.get('ignored'):
        return f'{_rv(B, w)} is what {_where(m)} tried to write to register 0, and writes to register 0 are ignored'
    if c['kind'] == 'reset' and m['test'] < c.get('test', 0):
        return (f'{_rv(B, w)} was written by {_where(m)}, but the reset in {_where(c, m)} '
                f'cleared it before this read')
    return (f'{_rv(B, w)} was written by {_where(m)}, but {_where(c, m)} overwrote it with '
            f'{_rv(c.get("value"), w)} before this read')


def _design_value_text(h: Dict, O, a, w) -> Optional[str]:
    """A hint at what the design did, when its value is recognisable."""
    if O is None:
        return None
    c = h['cause']
    old = next((x for x in h.get('earlier') or [] if x['value'] == O and not x.get('ignored')), None)
    if old is not None:
        return (f'{_rv(O, w)} is the value written by {_where(old)}: the design behaves as if '
                f'{"the reset" if c["kind"] == "reset" else "the later write"} had not taken effect.')
    if O == 0 and c['kind'] == 'write':
        return 'The design returns 0, as if the write had not taken effect.'
    ign = next((x for x in h.get('earlier') or [] if x['value'] == O and x.get('ignored')), None)
    if ign is not None:
        return (f'{_rv(O, w)} is what {_where(ign)} tried to write to register 0; '
                f'the design stored it, but that write must be ignored.')
    return None


def _plain_local_stateful(st: Dict) -> Dict:
    v, vals, w = st.get('verdict'), st['values'], st['values'].get('width')
    O, B, R = vals.get('observed'), vals.get('expected_bdd'), vals.get('expected_ref')
    stim = st.get('stimulus') or {}
    a = stim.get('addr')
    h = st.get('history')
    title = _example(st.get('module_type'), st.get('operation'), stim, w)
    title = title[0].upper() + title[1:]
    lines = []
    if v == 'pass':
        lines.append(f'Correct: R{a} reads {_rv(O, w)}, as the scenario expects.')
    elif v == 'scenario_wrong':
        lines.append(f'The scenario is wrong, not the design. It expects {_rv(B, w)}, but R{a} holds '
                     f'{_rv(R, w)} at this point — and that is what the design returns.')
    elif v == 'design_wrong':
        lines.append(f'The design is wrong. R{a} should read {_rv(R, w)}, as the scenario expects, '
                     f'but the design returns {_rv(O, w)}.')
    elif v == 'both_wrong':
        lines.append(f'Both are wrong: the scenario expects {_rv(B, w)}, the design returns {_rv(O, w)}, '
                     f'R{a} should read {_rv(R, w)}.')
    elif v == 'consistent_wrong':
        lines.append(f'The test passed, but scenario and design are wrong in the same way: both say '
                     f'{_rv(O, w)}, R{a} should read {_rv(R, w)}.')
    else:
        lines.append('This check was not executed.')
    if h and v is not None:
        x = _scenario_value_text(h, B, a, w) if v in ('scenario_wrong', 'both_wrong') else None
        lines.append(x[0].upper() + x[1:] + '.' if x else f'Why {_rv(R, w)}: {_cause_text(h, a, w)}.')
        if v in ('design_wrong', 'both_wrong', 'consistent_wrong'):
            y = _design_value_text(h, O, a, w)
            if y:
                lines.append(y)
    c = st.get('contrast')
    if c and v not in ('pass', None):
        lines.append(f"For comparison, {_example(st.get('module_type'), st.get('operation'), c.get('stimulus'), w)} "
                     f"in the same scenario {'passed' if c['status'] == 'pass' else 'failed'}.")
    unv = st.get('unverified') or []
    if unv:
        lines.append('Not checked: ' + ', '.join(x['text'] for x in unv) + '.')
    return {'title': title, 'verdict': v, 'lines': lines}


def _is_cpu(module_type: Optional[str]) -> bool:
    return bool((L.library_for(module_type) or {}).get('reference'))


def _cv(v) -> str:
    """A CPU value: signed decimal, hex when it is large."""
    if v is None:
        return '—'
    v &= 0xFFFFFFFF
    if v >= 1 << 31 and v >= 0xFFFF0000:
        return str(v - (1 << 32))
    return f'0x{v:X}' if v >= 1 << 16 else str(v)


def _cpu_example(stim: Dict) -> str:
    """'sub x3, x1, x2 with x1 = 10, x2 = 30' -- the instruction and what it reads."""
    stim = stim or {}
    ins = (stim.get('instruction') or '?').strip()
    ops = L.cpu_operands(ins)
    if not ops:
        return ins
    pre = {}
    for reg, val in ((stim.get('rs1'), stim.get('rs1_value')), (stim.get('rs2'), stim.get('rs2_value'))):
        if reg not in (None, 0) and val is not None:
            pre[reg] = val
    reads = [ops[k] for k in ('rs1', 'rs2') if ops.get(k) is not None]
    addr = ops['rs1'] if ops['kind'] in ('LW', 'SW') else None
    parts = [f'x{r} = ' + (f'0x{pre[r] & 0xFFFFFFFF:X}' if r == addr else _cv(pre[r]))
             for r in dict.fromkeys(reads) if r in pre]
    if ops['kind'] == 'LW' and stim.get('mem_value') is not None:
        parts.append(f"{_cv(stim['mem_value'])} in memory")
    return ins + (' with ' + ', '.join(parts) if parts else '')


def _cpu_target(stim: Dict, port: str) -> str:
    ops = L.cpu_operands((stim or {}).get('instruction') or '') or {}
    if port == 'taken':
        return 'the branch'
    if port == 'mem':
        return 'the stored word'
    return f"x{ops.get('rd') if ops.get('rd') is not None else (stim or {}).get('rd')}"


def _cpu_val(v, port) -> str:
    if port == 'taken':
        return '—' if v is None else ('taken' if v else 'not taken')
    return _cv(v)


def _cpu_why(st: Dict, R) -> Optional[str]:
    """'10 − 30 = −20' style working for one row (operands as the CPU sees them)."""
    stim, cpu = st.get('stimulus') or {}, st.get('cpu') or {}
    ops = L.cpu_operands(stim.get('instruction') or '')
    if not ops or R is None:
        return None
    vals = cpu.get('operands') or {}
    a = 0 if ops.get('rs1') == 0 else vals.get(f"x{ops.get('rs1')}")
    b = (ops['imm'] if ops['kind'] == 'I' else
         (0 if ops.get('rs2') == 0 else vals.get(f"x{ops.get('rs2')}")))
    if a is None or b is None:
        return None
    sa, sb = L._s32(a & 0xFFFFFFFF), L._s32(b & 0xFFFFFFFF)
    op = ops['op']
    wraps = lambda x: (f', which is {_cv(R)} as a 32-bit value'
                       if ops.get('rd') != 0 and x != L._s32(R & 0xFFFFFFFF) else '')
    if op in ('ADD', 'ADDI'):
        return f'{sa} + {sb} = {sa + sb}' + wraps(sa + sb)
    if op == 'SUB':
        return f'{sa} − {sb} = {sa - sb}' + wraps(sa - sb)
    if op in ('SLT', 'SLTI', 'BLT'):
        return f"{sa} < {sb} is {'true' if sa < sb else 'false'}"
    if op == 'BGE':
        return f"{sa} ≥ {sb} is {'true' if sa >= sb else 'false'}"
    if op in ('BEQ', 'BNE'):
        return f"{sa} and {sb} are {'equal' if sa == sb else 'different'}"
    return None


def _cpu_design_hint(st: Dict, O) -> Optional[str]:
    """What a wrong value of the design looks like, when it is recognisable."""
    stim, cpu = st.get('stimulus') or {}, st.get('cpu') or {}
    ops = L.cpu_operands(stim.get('instruction') or '')
    if not ops or O is None:
        return None
    vals = cpu.get('operands') or {}
    a = 0 if ops.get('rs1') == 0 else vals.get(f"x{ops.get('rs1')}")
    b = ops['imm'] & 0xFFFFFFFF if ops['kind'] == 'I' and ops.get('imm') is not None else (
        0 if ops.get('rs2') == 0 else vals.get(f"x{ops.get('rs2')}"))
    op = ops['op']
    if a is not None and b is not None:
        unsigned = {'SLT': int(a < b), 'SLTI': int(a < b), 'BLT': int(a < b), 'BGE': int(a >= b)}.get(op)
        signed = (L.CPU_RULES.get(op) or {}).get('cfn')
        if unsigned is not None and signed and int(signed(a, b)) != unsigned and O == unsigned:
            return 'The design seems to compare the operands as unsigned numbers.'
        if op == 'SUB' and O == (a + b) & 0xFFFFFFFF:
            return 'The design seems to add instead of subtract.'
        if op in ('ADD', 'ADDI') and O == (a - b) & 0xFFFFFFFF:
            return 'The design seems to subtract instead of add.'
    if st['target'].get('port') == 'rd' and O == 0:
        return f"The design stores 0, as if the instruction had not written x{ops.get('rd')}."
    return None


def _cpu_issue_text(cpu: Dict) -> Optional[str]:
    if not cpu or not cpu.get('issues'):
        return None
    i = cpu['issues'][0]
    tail = (' This is a limitation of the testbench, not a fault of the design or the scenario.'
            if i['fault'] == 'test' else ' The example has to be fixed before it can test anything.')
    return f"This example cannot be checked as written: {i['text']}.{tail}"


def _plain_local_cpu(st: Dict) -> Dict:
    v, vals = st.get('verdict'), st['values']
    O, B, R = vals.get('observed'), vals.get('expected_bdd'), vals.get('expected_ref')
    stim, port = st.get('stimulus') or {}, st['target'].get('port') or 'rd'
    cpu = st.get('cpu') or {}
    who, f = _cpu_target(stim, port), (lambda x: _cpu_val(x, port))
    lines = []
    if v == 'undecided' and cpu.get('issues'):
        lines.append(_cpu_issue_text(cpu))
    elif v == 'pass':
        lines.append(f'Correct: {who} is {f(O)}, as the scenario expects.' if port != 'taken'
                     else f'Correct: the branch is {f(O)}, as the scenario expects.')
    elif v == 'scenario_wrong':
        lines.append(f'The scenario is wrong, not the design. It expects {who} to be {f(B)}, but the correct '
                     f'value is {f(R)} — and that is what the design gives.' if port != 'taken' else
                     f'The scenario is wrong, not the design. It says the branch is {f(B)}, but it is {f(R)} '
                     f'— and the design does that.')
    elif v == 'design_wrong' and port == 'taken' and L.cpu_astray((st.get('evidence') or {}).get('log_line')):
        lines.append(f'The design is wrong. The branch should be {f(R)}, but after it the program did not arrive '
                     f'where it should, so the branch sent it somewhere else.')
    elif v == 'design_wrong':
        got = f(O) if O is not None else 'neither (the result store went wrong)'
        lines.append(f'The design is wrong: {who} should be {f(R)}, as the scenario expects, '
                     f'but the design gives {got}.' if port != 'taken' else
                     f'The design is wrong. The branch should be {f(R)}, as the scenario expects, but the design '
                     f'result is {got}.')
    elif v == 'both_wrong':
        lines.append(f'Both are wrong: the scenario expects {f(B)}, the design gives {f(O)}, correct is {f(R)}.')
    elif v == 'consistent_wrong':
        lines.append(f'The test passed, but scenario and design are wrong in the same way: both say {f(O)}, '
                     f'correct is {f(R)}.')
    elif v == 'undecided':
        lines.append(f'There is no reference value for this example. Expected {f(B)}, the design gave {f(O)}.')
    elif L.cpu_not_reached((st.get('evidence') or {}).get('log_line')):
        lines.append('The program never reached this example, so it says nothing about this instruction. '
                     'An instruction before it sent the program somewhere else.')
    else:
        lines.append('This check was not executed.')
    r = st.get('rule')
    if r and v not in ('pass', None) and not cpu.get('issues'):
        work = _cpu_why(st, R)
        lines.append(f"Why {f(R)}: {r['text']}" + (f'; here {work}' if work else '') + '.')
    for n in cpu.get('notes') or []:
        lines.append(n['text'][0].upper() + n['text'][1:] + '.')
    if v in ('design_wrong', 'both_wrong', 'consistent_wrong'):
        h = _cpu_design_hint(st, O)
        if h:
            lines.append(h)
    c = st.get('contrast')
    if c and v not in ('pass', None) and not cpu.get('issues'):
        lines.append(f"For comparison, {_cpu_example(c.get('stimulus'))} "
                     f"{'passed' if c['status'] == 'pass' else 'failed'}.")
    return {'title': _cpu_example(stim), 'verdict': v, 'lines': lines}


def _blocked_line(b: Dict) -> str:
    return f"the scenario says “{b['text']}”, but {b['why']}"


def plain_local(st: Dict) -> Dict:
    if st.get('blocked'):
        mt = st.get('module_type')
        base = (_plain_local_cpu if _is_cpu(mt) else _plain_local_stateful if _stateful(mt)
                else _plain_local_combinational)(dict(st, blocked=None))
        lines = [f"This example cannot be checked as written: {_blocked_line(st['blocked'][0])}.",
                 'So this simulation cannot show whether the design or the scenario is right; '
                 'it is not used to judge either.']
        lines += [f"Also: {_blocked_line(b)}." for b in st['blocked'][1:3]]
        return {'title': base['title'], 'verdict': st.get('verdict'), 'lines': lines}
    if _is_cpu(st.get('module_type')):
        return _plain_local_cpu(st)
    if _stateful(st.get('module_type')):
        return _plain_local_stateful(st)
    return _plain_local_combinational(st)


def _plain_local_combinational(st: Dict) -> Dict:
    v, vals, w = st.get('verdict'), st['values'], st['values'].get('width')
    O, B, R = vals.get('observed'), vals.get('expected_bdd'), vals.get('expected_ref')
    title = _inputs(st.get('operation'), st.get('stimulus'), w)
    port = st['target'].get('port') or 'result'
    val = lambda x: _value(st.get('operation'), 'result', x, w)
    lines = []
    if v == 'pass':
        lines.append(f'Correct: the design gives {port} = {val(O)}, as the scenario expects.')
    elif v == 'scenario_wrong':
        lines.append(f'The scenario is wrong, not the design. The scenario expects {val(B)}, '
                     f'but the correct {port} is {val(R)} — and that is what the design gives.')
    elif v == 'design_wrong':
        lines.append(f'The design is wrong. It gives {port} = {val(O)}; the correct value is '
                     f'{val(R)}, as the scenario expects.')
    elif v == 'both_wrong':
        lines.append(f'Both are wrong: the scenario expects {val(B)}, the design gives {val(O)}, '
                     f'the correct value is {val(R)}.')
    elif v == 'consistent_wrong':
        lines.append(f'The test passed, but scenario and design are wrong in the same way: both say '
                     f'{val(O)}, the correct value is {val(R)}.')
    elif v == 'undecided':
        lines.append(f'There is no reference rule for this design type. Expected {val(B)}, '
                     f'the design gave {val(O)}.')
    else:
        lines.append('This check was not executed.')
    r = st.get('rule')
    stim = st.get('stimulus') or {}
    if r and v not in ('pass', None):
        work = L.alu_working(st.get('operation'), stim.get('a'), stim.get('b'), w)
        lines.append(f"Why {val(R)}: {_with_width(r['text'], w)}" + (f"; here {work}." if work else '.'))
    cause = st.get('cause')
    if cause and v not in ('pass', None):
        lines.append(f"Cause: the design {cause['says']}. That reproduces all {cause['rows']} "
                     f"{st.get('operation')} examples, so {cause['fix']}.")
    c = st.get('contrast')
    if c and v not in ('pass', None):
        lines.append(f"For comparison, {_inputs(st.get('operation'), c.get('stimulus'), w)} "
                     f"{'passed' if c['status'] == 'pass' else 'failed'}"
                     + (': the fault does not show on these values.' if c.get('same_with_fault') else '.'))
    flags = st.get('probed') or []
    bad = [x for x in flags if x['verdict'] != 'pass']
    for x in bad:
        word = _FLAG_WORD.get(x['port'], x['port'])
        obs = x['observed'] if x['observed'] is not None else x['raw']
        if x['verdict'] == 'scenario_wrong':
            lines.append(f"The scenario states the {word} as {x['stated']}, but it should be "
                         f"{x['rule_value']} — the design gives {obs}, which is correct.")
        elif x['verdict'] == 'design_wrong':
            lines.append(f"The design's {word} is {obs}, but it should be {x['rule_value']}"
                         + (' — a consequence of the wrong result.' if v in ('design_wrong', 'both_wrong') else '.'))
        else:
            lines.append(f"The {word}: scenario {x['stated']}, design {obs}, correct {x['rule_value']}.")
    good = [x['port'] for x in flags if x['verdict'] == 'pass']
    if good:
        lines.append(f"The {_flag_list(good)} {'is' if len(good) == 1 else 'are'} correct.")
    unstated = [x['port'] for x in flags if x['stated'] is None]
    if unstated:
        lines.append(f'The scenario does not state the {_flag_list(unstated)}.')
    unv = st.get('unverified') or []
    if unv:
        lines.append('Not checked: ' + ', '.join(x['text'] for x in unv) + '.')
    return {'title': title, 'verdict': v, 'lines': lines}


def plain_global(st: Dict) -> Dict:
    tg = st['target']
    word = _FLAG_WORD.get(tg['port'], tg['port'])
    title = f"When is the {word} {1 if str(tg['value']).lower() in ('1', 'true') else 0}?"
    lines = []
    if st.get('rule'):
        lines.append(f"Rule: {st['rule']['text']}.")
    rc = st.get('rule_check') or {}
    n = st.get('match_count', 0)
    if not n:
        lines.append('No example in the scenarios states this.')
    elif rc.get('judged'):
        bad = rc.get('contradicting', 0)
        lines.append(f'The scenarios state it in {n} examples; '
                     + ('all of them agree with the rule.' if not bad
                        else f'{bad} of them contradict the rule.'))
    for m in [m for m in st.get('matches', []) if m.get('consistent') is False][:3]:
        lines.append(f"Wrong: {m['operation']} — {'; '.join(m['P'])}.")
    if rc.get('missed_count'):
        lines.append(f"{rc['missed_count']} more examples should state it but do not.")
    alt = rc.get('alternative')
    if alt and alt.get('follows_rule'):
        lines.append(f"The scenarios mix two meanings of “{tg['port']}”: where they differ, "
                     f"{_examples(alt['follows_other'])} {alt['name']} ({alt['text']}) and "
                     f"{_examples(alt['follows_rule'])} the rule.")
    elif alt:
        lines.append(f"The scenarios seem to use “{tg['port']}” to mean {alt['name']} ({alt['text']}).")
    pb = st.get('probe')
    if pb:
        lines.append(f"In simulation, the design's {word} was right in {pb['design_agrees']} of {pb['rows']} examples.")
    elif not st.get('verified_by_testbench'):
        lines.append(f'The {word} was not simulated.')
    return {'title': title, 'verdict': None, 'lines': lines}


def plain_requirement(st: Dict) -> Dict:
    title = st['target']['text']
    s_ = st.get('status')
    subj = st.get('subject') or 'its output'
    lines = []
    if s_ == 'verified':
        lines.append(f'Verified: every simulated example of {subj} passed.'
                     + (f" {st['unchecked']} more could not be checked as written." if st.get('unchecked') else ''))
    elif s_ == 'verified_by_probe':
        lines.append(f"Verified: the {_FLAG_WORD.get(subj, subj)} was right in every example "
                     f"(checked by an extra simulation).")
    elif s_ == 'fail':
        kinds = sorted({f['verdict'] for f in st.get('failing', []) if f.get('verdict')})
        who = {'design_wrong': 'the design', 'scenario_wrong': 'the scenario', 'both_wrong': 'both',
               'consistent_wrong': 'both (in the same way)'}
        lines.append(f"Failing in {len(st.get('failing', []))} examples; wrong: "
                     + ', '.join(who.get(k, k) for k in kinds) + '.')
    elif st.get('cannot_test'):
        lines.append(f"Not tested: {st['cannot_test']}, so no scenario can show this.")
    elif s_ == 'not_tested':
        sc = st.get('untested_scenarios') or []
        lines.append('Not tested: ' + (f"the scenario “{sc[0]['title']}” could not be turned into tests."
                                      if sc else 'no test ran for it.'))
    elif s_ == 'stated_only':
        lines.append(f'Stated in the scenarios, but {subj} was not simulated.')
    elif s_ == 'checked_unlinked':
        n, ok = st.get('probe_rows', 0), st.get('probe_right', 0)
        lines.append(f"No scenario is linked to it, but the extra simulation checked the {_FLAG_WORD.get(subj, subj)} "
                     f"in all {n} examples: {ok} right, {n - ok} wrong.")
    elif s_ == 'not_covered':
        lines.append('Not covered: no scenario tests this requirement.')
    lines.append('Which scenario covers which requirement was suggested by the model that wrote the scenarios.')
    return {'title': title, 'verdict': s_, 'lines': lines}


def render_plain(st: Dict) -> Dict:
    return {'local': plain_local, 'global': plain_global,
            'requirement': plain_requirement}.get(st.get('kind'), lambda _: {'title': '', 'lines': []})(st)


# ----------------------------------------------------------------- problems list

_WHO = {'design_wrong': 'design', 'both_wrong': 'design', 'scenario_wrong': 'scenario',
        'consistent_wrong': 'design'}


_PORT_WORD = {'rdata1': 'read port 1', 'rdata2': 'read port 2 (the testbench never reads through it)'}
_RF_OP_WORD = {'READ': 'register reads', 'WRITE_READ': 'reads right after a write'}


def _stateful_problem(s, testbench_id, v, op, n, m, ci, B, R, O, w, h) -> Tuple[str, str]:
    """Title and detail of one problem group of a register file."""
    st = ci.stimulus or {}
    a = st.get('addr')
    ex = _example('regfile', op, st, w)
    hh = _history_for(s, testbench_id, ci.tag, h, B) if h else None
    why = f' ({_cause_text(hh, a, w)})' if hh else ''
    what = _RF_OP_WORD.get(op, 'register reads')
    if v == 'design_wrong':
        hint = _design_value_text(hh, O, a, w) if hh else None
        return (f'The design returns wrong values on {what}',
                f'{n} of {m} {what} give a wrong value. E.g. {ex}: the design returns {_rv(O, w)}, '
                f'correct is {_rv(R, w)}{why}.' + (f' {hint}' if hint else ''))
    if v == 'both_wrong':
        return (f'{what.capitalize()}: both the design and the scenario are wrong',
                f'{n} of {m}. E.g. {ex}: scenario {_rv(B, w)}, design {_rv(O, w)}, correct {_rv(R, w)}{why}.')
    if v == 'consistent_wrong':
        return (f'{what.capitalize()}: the design repeats a mistake of the scenario',
                f'{n} of {m} pass the test, but both give the wrong value. E.g. {ex}: both say '
                f'{_rv(O, w)}, correct is {_rv(R, w)}{why}.')
    x = _scenario_value_text(hh, B, a, w) if hh else None
    title = ('A scenario expects a wrong register value' if n == 1
             else f'{n} scenario examples expect a wrong register value')
    if x:
        return title, (f'E.g. {ex}: the scenario expects {_rv(B, w)}, but R{a} holds {_rv(R, w)}. '
                       f'{x[0].upper() + x[1:]}. The design is right.')
    return title, (f'E.g. {ex}: the scenario expects {_rv(B, w)}, but R{a} holds {_rv(R, w)} at that '
                   f'point{why}. The design is right.')


def _cpu_problem(v, op, n, m, ci, B, R, O, astray=False) -> Tuple[str, str]:
    stim, port = ci.stimulus or {}, ci.port or 'rd'
    ex = _cpu_example(stim)
    f = lambda x: _cpu_val(x, port)
    branch = port == 'taken'
    st = {'stimulus': stim, 'cpu': L.cpu_reference(stim), 'target': {'port': port}}
    if v == 'design_wrong' and astray:
        return (f'A {op} branch sends the program to the wrong place',
                f'{n} of {m} {op} examples. E.g. {ex}: the branch should be {f(R)}, but after it the program '
                f'did not arrive where it should.')
    if v == 'design_wrong':
        hint = _cpu_design_hint(st, O)
        got = f(O) if O is not None else 'neither (the result store went wrong)'
        return ((f'The design decides {op} wrong' if branch else f'The design executes {op} wrong'),
                f'{n} of {m} {op} examples. E.g. {ex}: '
                + (f'the design result is {got}, but the branch is {f(R)}.' if branch
                   else f'the design gives {got}, correct is {f(R)}.') + (f' {hint}' if hint else ''))
    if v == 'both_wrong':
        return (f'{op}: both the design and the scenario are wrong',
                f'{n} of {m} {op} examples. E.g. {ex}: scenario {f(B)}, design {f(O)}, correct {f(R)}.')
    if v == 'consistent_wrong':
        return (f'{op}: the design repeats a mistake of the scenario',
                f'{n} of {m} {op} examples pass the test, but both give the wrong value. E.g. {ex}: '
                f'both say {f(O)}, correct is {f(R)}.')
    notes = ' '.join(x['text'][0].upper() + x['text'][1:] + '.' for x in st['cpu']['notes'])
    title = ((f'A scenario expects the wrong {op} decision' if branch else f'A scenario expects a wrong {op} result')
             if n == 1 else (f'{n} scenario examples expect the wrong {op} decision' if branch
                             else f'{n} scenario examples expect a wrong {op} result'))
    return title, (f'E.g. {ex}: the scenario says {f(B)}, but it is {f(R)}.' if branch else
                   f'E.g. {ex}: the scenario expects {f(B)}, correct is {f(R)}.') + \
        (f' {notes}' if notes else '') + ' The design is right.'


def _blocked_problems(s, main) -> List[Dict]:
    """Examples whose scenario asks for something the testbench does not do (D29)."""
    from src.xdb import mapping as _mapping
    cache: Dict[int, List[Dict]] = {}
    hits = []
    for ci, cr in main:
        if not ci.tag.startswith('BDD:') or cr.verdict != 'undecided' or not ci.scenario_id:
            continue
        if ci.scenario_id not in cache:
            cache[ci.scenario_id] = _mapping.blocked_steps(s, ci.scenario_id)
        if cache[ci.scenario_id]:
            hits.append(ci)
    if not hits:
        return []
    first = hits[0]
    sc = s.get(M.Scenario, first.scenario_id)
    n_sc = len({ci.scenario_id for ci in hits})
    b = cache[first.scenario_id][0]
    return [{'severity': 'warning', 'who': 'test',
             'title': ('The testbench cannot run one example as the scenario describes it' if len(hits) == 1
                       else f'The testbench cannot run {len(hits)} examples as the scenarios describe them'),
             'detail': f"E.g. “{sc.title}”: {_blocked_line(b)}."
                       + (f' ({n_sc} scenarios.)' if n_sc > 1 else '')
                       + ' These results are not used to judge the design or the scenarios.',
             'explain': {'kind': 'check', 'tag': first.tag}}]


def _cpu_not_reached(main) -> List[Dict]:
    """Examples the program never got to. They prove nothing about their own
    instruction; the fault is in what ran before them."""
    order = lambda ci: (ci.stimulus or {}).get('order', 0)
    missed = sorted((ci for ci, cr in main if ci.tag.startswith('BDD:') and cr.status == 'not_tested'
                     and L.cpu_not_reached(cr.message)), key=order)
    if not missed:
        return []
    ran = [ci for ci, cr in main if ci.tag.startswith('BDD:') and cr.status in ('pass', 'fail')
           and order(ci) < order(missed[0])]
    n = len(missed)
    if ran:
        detail = (f'The last example the program reached is {_cpu_example(max(ran, key=order).stimulus)}. '
                  'After it the program went somewhere else, so the examples behind it never ran and say '
                  'nothing about their instructions.')
    else:
        detail = 'Not even the first example ran: the CPU does not get through its first instructions.'
    return [{'severity': 'error', 'who': 'design',
             'title': ('The program never reached one example' if n == 1
                       else f'The program never reached {n} examples'),
             'detail': detail, 'explain': {'kind': 'check', 'tag': missed[0].tag}}]


def _cpu_unchecked(s, main) -> List[Dict]:
    """Examples that ran but cannot mean what they say (library: cpu_reference issues)."""
    from src.xdb import mapping as _mapping
    groups: Dict[str, List] = {}
    for ci, cr in main:
        if not ci.tag.startswith('BDD:') or cr.verdict != 'undecided' or not ci.stimulus:
            continue
        if ci.scenario_id and _mapping.blocked_steps(s, ci.scenario_id):
            continue                                   # reported with the blocked examples
        ref = L.cpu_reference(ci.stimulus)
        if ref['issues']:
            groups.setdefault(ref['issues'][0]['fault'], []).append((ci, ref['issues'][0]))
    out = []
    for fault in ('scenario', 'test'):
        items = groups.get(fault) or []
        if not items:
            continue
        ci, iss = items[0]
        n = len(items)
        if fault == 'scenario':
            out.append({'severity': 'warning', 'who': 'scenario',
                        'title': ('An example cannot be checked as written' if n == 1
                                  else f'{n} examples cannot be checked as written'),
                        'detail': f"E.g. {_cpu_example(ci.stimulus)}: {iss['text']}. The example has to be "
                                  f"fixed before it can test anything.",
                        'explain': {'kind': 'check', 'tag': ci.tag}})
        else:
            out.append({'severity': 'warning', 'who': 'test',
                        'title': ('The testbench cannot check one example' if n == 1
                                  else f'The testbench cannot check {n} examples'),
                        'detail': f"E.g. {_cpu_example(ci.stimulus)}: {iss['text']}. This is a limitation of "
                                  f"the testbench, not a fault of the design or the scenario.",
                        'explain': {'kind': 'check', 'tag': ci.tag}})
    return out


def build_problems(s, run_id: int, scenario_results: Optional[Dict], reqs: List[Dict],
                   spec: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Everything that is wrong or untested in one simulation, in plain words,
    most severe first. Each item: severity (error | warning), who (design |
    scenario | test | coverage), title, detail, and what Details opens."""
    run = s.get(M.SimRun, run_id)
    main = s.execute(select(M.CheckItem, M.CheckResult)
                     .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                     .where(M.CheckResult.sim_run_id == run_id)).all()
    width = next((ci.width for ci, _ in main if ci.width), None)
    by_op: Dict[Tuple[str, str], List] = {}
    total_op: Dict[str, int] = {}
    for ci, cr in main:
        if not ci.tag.startswith('BDD:') or cr.status not in ('pass', 'fail'):
            continue
        total_op[ci.operation or '?'] = total_op.get(ci.operation or '?', 0) + 1
        if cr.verdict in _WHO:
            by_op.setdefault((cr.verdict, ci.operation or '?'), []).append((ci, cr))
    # One card per thing to fix. Examples where the design repeats the scenario's
    # mistake are the same fault as those where it fails the test: they only pass
    # because the scenario is wrong too (D37).
    same_mistake: Dict[str, List] = {}
    for key in [k for k in by_op if k[0] == 'consistent_wrong' and ('design_wrong', k[1]) in by_op]:
        same_mistake[key[1]] = by_op.pop(key)
    probs: List[Dict] = []
    tbrow = s.get(M.Testbench, run.testbench_id)
    sess = s.get(M.Session, tbrow.session_id)
    design = s.get(M.Design, sess.design_id) if sess.design_id else None
    mt = design.module_type if design else None
    hist = _rf_history(s, run.testbench_id, width) if _stateful(mt) else {}

    swap_cards: List[Dict] = []
    swapped: set = set()
    if not _is_cpu(mt) and not _stateful(mt) and width:
        swap_cards, swapped = _swapped_scenarios(s, main, width)
        for key in [k for k in by_op if k[0] == 'scenario_wrong']:
            by_op[key] = [x for x in by_op[key] if x[0].tag not in swapped]
            if not by_op[key]:
                del by_op[key]
    # In those same examples the scenario is wrong too: fixing the design alone
    # would turn them into failures, so the scenario gets its card now.
    repeated: Dict[str, List] = {}
    for op, rows in same_mistake.items():
        repeated[op] = [x for x in rows if x[0].tag not in swapped]
        if repeated[op]:
            by_op.setdefault(('scenario_wrong', op), [])

    lost_bit: Dict[int, List[str]] = {}
    if not _is_cpu(mt) and not _stateful(mt) and width:
        for op in sorted(o for v_, o in by_op if v_ == 'design_wrong'):
            c_ = L.alu_mixup(op, _alu_rows(main, op), width)
            if c_ and 'bit' in c_:
                lost_bit.setdefault(c_['bit'], []).append(op)

    order = ['design_wrong', 'both_wrong', 'consistent_wrong', 'scenario_wrong']
    result_card: Dict[str, Dict] = {}       # operation -> the card of its wrong scenario results
    for (v, op), items in sorted(by_op.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        repeats = repeated.get(op, []) if v == 'scenario_wrong' else []
        rows = items + repeats
        ci, cr = rows[0]
        also = len(same_mistake.get(op) or []) if v == 'design_wrong' else 0
        n, m = len(rows) + also, total_op.get(op, len(rows))
        passing = (f' {also} of them pass the test, because the scenario makes the same mistake.' if also else '')
        B, R, O = _int(ci.expected_bdd), _int(ci.expected_ref), _int(cr.observed_value)
        if _is_cpu(mt):
            lost = next(((c_, r_) for c_, r_ in items if L.cpu_astray(r_.message)), None)
            if lost:
                ci, cr = lost
                B, R, O = _int(ci.expected_bdd), _int(ci.expected_ref), _int(cr.observed_value)
            title, detail = _cpu_problem(v, op, n, m, ci, B, R, O, astray=bool(lost))
            detail = _with_repeats(detail + passing, len(items), len(repeats))
            probs.append({'severity': 'error' if _WHO[v] == 'design' else 'warning', 'who': _WHO[v],
                          'title': title, 'detail': detail, 'explain': {'kind': 'check', 'tag': ci.tag}})
            continue
        if _stateful(mt):
            title, detail = _stateful_problem(s, run.testbench_id, v, op, n, m, ci, B, R, O, width,
                                              hist.get(ci.tag))
            detail = _with_repeats(detail + passing, len(items), len(repeats))
            probs.append({'severity': 'error' if _WHO[v] == 'design' else 'warning', 'who': _WHO[v],
                          'title': title, 'detail': detail, 'explain': {'kind': 'check', 'tag': ci.tag}})
            continue
        ex = _inputs(op, ci.stimulus, width)
        res = lambda x, op_=op: _value(op_, 'result', x, width)
        cause = L.alu_mixup(op, _alu_rows(main, op), width) if v != 'scenario_wrong' else None
        if v == 'design_wrong' and cause and 'bit' in cause:
            if op == min(lost_bit[cause['bit']]):      # one card for all the shifts that lose this bit
                probs.append(_lost_bit_card(main, by_op, same_mistake, total_op, lost_bit[cause['bit']],
                                            cause, width))
            continue
        if v == 'design_wrong':
            title = f"The design {cause['says']}" if cause else f'The design computes {op} wrong'
            detail = (f'{n} of {m} {op} examples give a wrong result.{passing} E.g. {ex}: the design gives '
                      f'{res(O)}, correct is {res(R)}.')
        elif v == 'both_wrong':
            title = f'{op}: both the design and the scenario are wrong'
            detail = f'{n} of {m} {op} examples. E.g. {ex}: scenario {res(B)}, design {res(O)}, correct {res(R)}.'
        elif v == 'consistent_wrong':
            title = f'{op}: the design repeats a mistake of the scenario'
            detail = (f'{n} of {m} {op} examples pass the test, but both give the wrong value. '
                      f'E.g. {ex}: both say {res(O)}, correct is {res(R)}.')
        else:
            title = f'A scenario expects a wrong {op} result' if n == 1 else f'{n} scenario examples expect a wrong {op} result'
            detail = f'E.g. {ex}: the scenario expects {res(B)}, correct is {res(R)}. The design is right.'
        if cause:
            detail += f" All {cause['rows']} {op} examples fit this, so {cause['fix']}."
        detail = _with_repeats(detail, len(items), len(repeats))
        probs.append({'severity': 'error' if _WHO[v] == 'design' else 'warning', 'who': _WHO[v],
                      'title': title, 'detail': detail, 'explain': {'kind': 'check', 'tag': ci.tag}})
        if v == 'scenario_wrong':
            result_card[op] = probs[-1]

    probs += swap_cards
    probs += _blocked_problems(s, main)
    if _is_cpu(mt):
        probs += _cpu_not_reached(main)
        probs += _cpu_unchecked(s, main)
        n_bad = sum(len(x) for (v_, _), x in by_op.items() if v_ in ('design_wrong', 'both_wrong'))
        bad_ops = {op_ for (v_, op_) in by_op if v_ in ('design_wrong', 'both_wrong')}
        n_all = sum(total_op.values())
        if len(bad_ops) >= 4 and n_all and n_bad >= 0.8 * n_all:
            probs.insert(0, {'severity': 'error', 'who': 'design',
                             'title': 'Almost every instruction gives a wrong result',
                             'detail': f'{n_bad} of {n_all} checks fail, across {len(bad_ops)} instructions. Every '
                                       'result is observed the same way (operands loaded with ADDI, the result '
                                       'stored with SW), so the fault is more likely in that path or in the '
                                       'pipeline than in each instruction.', 'explain': None})

    # Flags seen by the probe. A wrong flag goes to whoever must fix it: the design's card
    # when its result is wrong too, the scenario's card when that example is on it anyway,
    # and otherwise a card of its own, so that none is lost.
    probe = _probe_results(s, _probe_run_id(s, run_id))
    fl: Dict[Tuple[str, str, str], List] = {}
    consequences: Dict[str, set] = {}       # operation -> flags wrong only because the result is
    also_stated: Dict[str, Dict[str, int]] = {}    # operation -> flags its result card must mention
    row_verdict = {ci.tag: cr.verdict for ci, cr in main}
    on_card = {ci.tag for op_ in result_card for ci, _ in by_op[('scenario_wrong', op_)]}
    for (tag, port), o in probe.items():
        op_ = o.get('operation') or '?'
        if o['verdict'] == 'pass':
            continue
        if o['verdict'] == 'scenario_wrong' and tag in on_card:
            flags_ = also_stated.setdefault(op_, {})
            flags_[port] = flags_.get(port, 0) + 1
        elif o['verdict'] != 'scenario_wrong' and row_verdict.get(tag) in ('design_wrong', 'both_wrong',
                                                                           'consistent_wrong'):
            consequences.setdefault(op_, set()).add(port)
        else:
            fl.setdefault((o['verdict'], port, op_), []).append((tag, o))
    for op_, flags_ in also_stated.items():
        one = len(by_op[('scenario_wrong', op_)]) == 1
        for port, n in sorted(flags_.items()):
            word = _FLAG_WORD.get(port, port)
            result_card[op_]['detail'] += (f' The {word} of this example is stated wrongly too.' if one
                                           else f' In {n} of them the {word} is stated wrongly too.')
    # Every wrong overflow statement fits one other meaning: that is one thing to fix.
    alt = next((it['alternative'] for it in spec if it['port'] == 'overflow' and it.get('alternative')), None)
    wrong_over = [k for k in fl if k[0] == 'scenario_wrong' and k[1] == 'overflow']
    over_rows = sorted((x for k in wrong_over for x in fl[k]), key=lambda x: x[0])
    if alt and over_rows and alt.get('explains', 0) >= len(over_rows):
        tag, o = over_rows[0]
        ops = ' and '.join(sorted({k[2] for k in wrong_over}))
        rule_name = L.OVERFLOW_DEFINITIONS.get(_overflow_mode(s, sess.id)[0], {}).get('name')
        kept = alt.get('follows_rule', 0)     # examples that contradict the other meaning
        probs.append({'severity': 'warning', 'who': 'scenario',
                      'title': ('The scenarios mix two meanings of overflow' if kept
                                else f"The scenarios treat overflow as {alt['name']}"),
                      'detail': (f"Where the two meanings differ, {_examples(len(over_rows), f', in {ops},')} "
                                 f"{alt['name']} and {_examples(kept)} {rule_name}. E.g. " if kept else
                                 f"All {len(over_rows)} wrong overflow flags, in {ops}, fit that meaning. E.g. ")
                                + f"{_inputs(o.get('operation'), o.get('stimulus'), width)}: the scenario says the "
                                f"overflow flag is {o['stated']}, but it should be {o['rule_value']}. "
                                f"{_overflow_basis(s, sess.id)} The design is right.",
                      'explain': {'kind': 'global', 'port': 'overflow'}})
        for k in wrong_over:
            del fl[k]
    for (v, port, op), items in sorted(fl.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 9, kv[0][1], kv[0][2])):
        tag, o = items[0]
        word = _FLAG_WORD.get(port, port)
        ex = _inputs(op, o.get('stimulus'), width)
        obs = o['observed'] if o['observed'] is not None else o['raw']
        if v == 'scenario_wrong':
            title = f'A scenario states a wrong {word} for {op}' if len(items) == 1 else \
                f'{len(items)} scenario examples state a wrong {word} for {op}'
            right = ('The design is right.' if row_verdict.get(tag) == 'pass'
                     else f"The design's {word} is right, though its result is wrong.")
            detail = (f"E.g. {ex}: the scenario says the {word} is {o['stated']}, but it should be "
                      f"{o['rule_value']}. {right}")
        else:
            title = f"The design's {word} is wrong for {op}"
            detail = (f"{len(items)} examples. E.g. {ex}: the design's {word} is {obs}, but it should be "
                      f"{o['rule_value']}.")
        if port == 'overflow':
            detail = f'{detail} {_overflow_basis(s, sess.id)}'.rstrip()
        probs.append({'severity': 'error' if _WHO.get(v) == 'design' else 'warning', 'who': _WHO.get(v, 'design'),
                      'title': title, 'detail': detail, 'explain': {'kind': 'check', 'tag': tag}})
    for it in spec:
        alt = it.get('alternative')
        if alt and it['port'] == 'overflow':
            for pr in probs:
                if pr['who'] == 'scenario' and 'overflow flag' in pr['title']:
                    pr['detail'] += f" The scenarios seem to use “overflow” to mean {alt['name']}."
                    pr['explain'] = {'kind': 'global', 'port': 'overflow'}
                    break
        # no probe: contradictions in what the scenarios state, unsimulated
        if not probe and it.get('contradicting'):
            ex = (it.get('examples') or [{}])[0]
            probs.append({'severity': 'warning', 'who': 'scenario',
                          'title': f"Scenarios state the {_FLAG_WORD.get(it['port'], it['port'])} wrongly",
                          'detail': f"{it['contradicting']} of {it['judged']} examples contradict the rule"
                                    + (f"; they seem to mean {alt['name']}" if alt else '') + '.',
                          'explain': {'kind': 'global', 'port': it['port']}})

    # scenarios that produced no tests, and the requirements left untested by them
    req_by_scn: Dict[str, List[str]] = {}
    for r in reqs:
        for u in r.get('untested_scenarios') or []:
            req_by_scn.setdefault(u['scenario_key'], []).append(r['text'])
    for sc in (scenario_results or {}).get('scenarios') or []:
        if sc['status'] in ('not_run', 'no_check', 'no_examples'):
            affected = req_by_scn.get(sc['id'], [])
            probs.append({'severity': 'warning', 'who': 'test',
                          'title': f"Not tested: “{sc.get('title') or sc['id']}”",
                          'detail': f"This scenario was skipped: {plain_reason(sc.get('reason'))}."
                                    + (f" Requirement left untested: “{affected[0].rstrip('.')}”." if affected else ''),
                          'explain': None})
        elif any(r['status'] == 'skipped' for r in sc.get('rows') or []):
            # also when other rows of the scenario failed: a skipped row is untested either way
            skipped = [r for r in sc.get('rows') or [] if r['status'] == 'skipped']
            why = plain_reason(skipped[0].get('reason')) if skipped and skipped[0].get('reason') else ''
            probs.append({'severity': 'warning', 'who': 'test',
                          'title': f"Partly tested: “{sc.get('title') or sc['id']}”",
                          'detail': f'{len(skipped)} of its examples could not be turned into tests'
                                    + (f' (e.g. {why})' if why else '') + '.', 'explain': None})
    for r in reqs:
        if r.get('cannot_test'):
            probs.append({'severity': 'warning', 'who': 'test', 'title': 'The testbench cannot test a requirement',
                          'detail': f"“{r['text'].rstrip('.')}”: {r['cannot_test']}.",
                          'explain': {'kind': 'requirement', 'id': r['requirement_id']}})
        elif r['status'] == 'not_covered':
            probs.append({'severity': 'warning', 'who': 'coverage', 'title': 'A requirement is not tested by any scenario',
                          'detail': f"“{r['text'].rstrip('.')}”", 'explain': {'kind': 'requirement', 'id': r['requirement_id']}})
        elif r['status'] == 'stated_only':
            probs.append({'severity': 'warning', 'who': 'coverage', 'title': 'A requirement was not simulated',
                          'detail': f"“{r['text']}” — the scenarios state it, but nothing simulated "
                                    f"{r.get('subject') or 'it'}.",
                          'explain': {'kind': 'requirement', 'id': r['requirement_id']}})
    # checks the testbench adds by itself (not from a scenario), e.g. "register 0
    # reads as 0 after a write": a failure is the design's
    gen_fail = [ci for ci, cr in main if ci.tag.startswith('GEN:') and cr.status == 'fail']
    if gen_fail:
        labels = [(ci.label or ci.tag[4:]) for ci in gen_fail]
        first_design = next((pr for pr in probs if pr['who'] == 'design'), None)
        note = ('The check the testbench adds by itself, “' + '”, “'.join(labels) + '”, fails as well.')
        if first_design is not None:
            first_design['detail'] += ' ' + note
        else:
            probs.insert(0, {'severity': 'error', 'who': 'design',
                             'title': 'The design fails a check the testbench adds by itself',
                             'detail': '“' + '”, “'.join(labels) + '”. This check does not come from '
                                       'the scenarios; the testbench adds it for this design type.',
                             'explain': None})
    op_of = {ci.tag: ci.operation for ci, _ in main}
    for pr in probs:
        ops_ = pr.pop('ops', None) or [op_of.get((pr.get('explain') or {}).get('tag'))]
        ports = set().union(*(consequences.get(o_, set()) for o_ in ops_))
        if pr['who'] == 'design' and ports:
            words = ' and '.join(_FLAG_WORD.get(p_, p_) for p_ in sorted(ports))
            pr['detail'] += f" The {words} of these examples {'are' if len(ports) > 1 else 'is'} wrong too, as a consequence."

    counts = {'design': 0, 'scenario': 0, 'test': 0, 'coverage': 0}
    for pr in probs:
        counts[pr['who']] = counts.get(pr['who'], 0) + 1
    n_checks = sum(total_op.values())
    if _is_cpu(mt):
        checked = [f'{n_checks} instructions were simulated, each in its own short program (the operands '
                   f'loaded with ADDI, the result stored with SW), and compared with a reference model of RV32I.',
                   'Not checked: the pipeline timing. Each instruction runs on its own, with NOPs around it. '
                   'A branch is judged by whether it skips the instruction after it and the program then '
                   'arrives at the instruction behind that one.']
    elif _stateful(mt):
        checked = [f'{n_checks} register reads were simulated, in the order of the scenarios, and each was '
                   f'compared with a reference model that replays every write and reset before it.']
        compared = {ci.port for ci, _ in main if ci.port}
        unread = [p_ for p_ in (L.library_for(mt) or {}).get('outputs', ()) if p_ not in compared]
        if unread:
            checked.append('Not checked: ' + ', '.join(_PORT_WORD.get(p_, p_) for p_ in unread) + '.')
    else:
        checked = [f'{n_checks} examples were simulated and compared with a reference model.']
    if probe:
        flags = sorted({port for (_, port) in probe}, key=lambda x: FLAG_PORTS.index(x) if x in FLAG_PORTS else 9)
        checked.append('The flags (' + ', '.join(flags) + ') were checked in an extra simulation.')
    elif spec:
        checked.append('The flags were not simulated; only what the scenarios state about them was checked.')
    if reqs:
        ok = sum(1 for r in reqs if r['status'] in ('verified', 'verified_by_probe'))
        checked.append(f'{ok} of {len(reqs)} requirements are verified.')
    overview = {'ok': not probs, 'counts': counts, 'checked': checked}
    return probs, overview
