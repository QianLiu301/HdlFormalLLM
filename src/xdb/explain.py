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

from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from src import xdb
from src.xdb import library as L
from src.xdb import models as M

# Bump whenever the structure changes, so stored explanations are recomputed
# instead of reused: v1.1 removed Chinese text, v1.2 added the rule check of
# global explanations and alternative meanings of contradicting flags.
ALGORITHM = 'bdx_v1.6'   # v1.6: requirement 'compared' decided by the whole testbench

_OP_SYMBOL = {'ADD': '+', 'SUB': '−', 'AND': '&', 'OR': '|', 'XOR': '^', 'SLL': '<<',
              'SRL': '>>', 'SRA': '>>>', 'SLT': '<ₛ', 'SLTU': '<ᵤ'}


# ----------------------------------------------------------------- loading helpers

def _steps(s, scenario_id: int, row_vals: Optional[Dict], module_type: str, width: Optional[int]):
    """Scenario steps instantiated for one row, with their canonical mapping."""
    maps = {m.position: m for m in s.scalars(select(M.StepMapping).where(
        M.StepMapping.scenario_id == scenario_id, M.StepMapping.method == 'rule'))}
    canon = {c.id: c for c in s.scalars(select(M.CanonicalStep))} if maps else {}
    out = []
    for st in s.scalars(select(M.ScenarioStep).where(M.ScenarioStep.scenario_id == scenario_id)
                        .order_by(M.ScenarioStep.position)):
        m = maps.get(st.position)
        c = canon.get(m.canonical_step_id) if m and m.canonical_step_id else None
        vals = L.resolve(m.params, row_vals) if m and m.params else {}
        port = (c.predicate or {}).get('port') if c else None
        value = next(iter(vals.values()), None) if port else None
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


def _requirement_status(s, q: M.Requirement, sim_run_id: Optional[int]) -> Dict:
    """Status of one requirement from the evidence that exists for its subject port.

    verified           every check of the subject port in the linked scenarios passed (testbench)
    verified_by_probe  the testbench does not compare the port; the flag probe observed it
                       in every linked row and the design agrees with the rule
    fail               some check / probe observation of the subject port is not a pass
    stated_only        the scenarios state the port, nothing simulated it
    not_tested         linked scenarios, but no check ran
    not_covered        no scenario claims the requirement
    """
    outputs = [p.name for p in s.scalars(select(M.Port).where(M.Port.design_id == q.design_id,
                                                              M.Port.direction == 'output'))]
    subject = requirement_subject(q.text, [o for o in outputs if o in _OUTPUT_WORDS] or None)
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
    for link, sc in links:
        counts: Dict[str, int] = {}
        res = s.execute(select(M.CheckItem, M.CheckResult)
                        .join(M.CheckResult, (M.CheckResult.check_id == M.CheckItem.id)
                              & (M.CheckResult.sim_run_id == sim_run_id))
                        .where(M.CheckItem.scenario_id == sc.id)).all() if sim_run_id else []
        for ci, cr in res:
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
    if not scen:
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
    return {'subject': subject, 'compared_by_testbench': compared, 'status': status,
            'untested_scenarios': untested,
            'evidence': evidence, 'scenarios': scen, 'failing': failing}


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

        contrast = None
        if sc is not None:
            others = s.execute(
                select(M.CheckItem, M.CheckResult, M.ExampleRow)
                .join(M.CheckResult, (M.CheckResult.check_id == M.CheckItem.id)
                      & (M.CheckResult.sim_run_id == run.id))
                .join(M.ExampleRow, M.ExampleRow.id == M.CheckItem.example_row_id)
                .where(M.CheckItem.scenario_id == sc.id, M.CheckItem.id != ci.id,
                       M.CheckResult.status.in_(('pass', 'fail')))).all()
            mine_row = er.row_no if er else 0
            cands = [o for o in others if (o[1].status != (cr.status if cr else None))
                     or (o[1].verdict != (cr.verdict if cr else None))]
            cands.sort(key=lambda o: (o[1].status == (cr.status if cr else None),
                                      abs(o[2].row_no - mine_row)))
            if cands:
                oc, ocr, oer = cands[0]
                contrast = {'row_no': oer.row_no, 'tag': oc.tag, 'status': ocr.status,
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
            'operation': ci.operation,
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
                    if q and stated is not None and rule_v == want:
                        alts_m = [x['name'] for x in L.flag_alternatives(mt, port, op, a, b, width, ov_mode)
                                  if x['value'] == stated]
                        missed.append({'scenario_key': sc.scenario_key, 'row_no': er.row_no,
                                       'operation': op, 'stated': q[0]['value'], 'a': a, 'b': b,
                                       'alternatives': alts_m})
                    continue
                alts = [x for x in L.flag_alternatives(mt, port, op, a, b, width, ov_mode)
                        if x['value'] == want]
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
               'pass': 'pass'}


def _num(v, width=None) -> str:
    if v is None:
        return '—'
    if width and width > 8 and v >= 1 << 8:
        return f'{v} (0x{v:0{(width + 3) // 4}X})'
    return str(v)


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
        if stim:
            lines.append("② Applied: " + ', '.join(
                f"{k} = {_num(v, w) if k != 'opcode' else v}" for k, v in stim.items()))
        r = st.get('rule')
        if r:
            sym = _OP_SYMBOL.get(r['op'], r['op'])
            calc = (f"{_num(stim.get('a'), w)} {sym} {_num(stim.get('b'), w)} = {_num(r.get('value'), w)}"
                    if stim.get('a') is not None else '')
            lines.append(f"③ Rule: {r['text']} → {calc}   ({r['source']})")
        o = vals.get('observed')
        agree = (('agrees with the rule' if r.get('value') == o else 'does not agree with the rule')
                 if r and o is not None else '')
        lines.append(f"④ Design output: {tg.get('port')} = {_num(o, w)}  {('— ' + agree) if agree else ''}")
        c = st.get('contrast')
        if c:
            cs = c.get('stimulus') or {}
            c_status = 'passed' if c['status'] == 'pass' else 'failed'
            lines.append(f"⑤ Contrast: row {c['row_no']} (a = {_num(cs.get('a'), w)}, "
                         f"b = {_num(cs.get('b'), w)}) {c_status} — {_VERDICT.get(c.get('verdict'), '')}")
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
                             f" (scenario {x['stated'] if x['stated'] is not None else '—'}, rule {x['rule_value']})")
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
                             f"{alt['name']} ({alt['text']}){also}. The specification seems to use "
                             f"\"{tg['port']}\" in that sense, unlike the rule"
                             f"{' (' + st['rule']['basis'] + ')' if (st.get('rule') or {}).get('basis') else ''}.")
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
        if not L.library_for(mt):
            return []
        width = width or _width_from_steps(s, feature_id, mt)
        ov_mode, ov_basis = _overflow_mode(s, sess.id)
        per: Dict[str, Dict] = {}
        for sc in s.scalars(select(M.Scenario).where(M.Scenario.feature_id == feature_id,
                                                     M.Scenario.is_current)
                            .order_by(M.Scenario.scenario_key)):
            for er in s.scalars(select(M.ExampleRow).where(M.ExampleRow.scenario_id == sc.id,
                                                           M.ExampleRow.is_current)
                                .order_by(M.ExampleRow.row_no)):
                steps = _steps(s, sc.id, er.vals, mt, width)
                op = next((x['values'].get('OP') for x in steps if x['canonical'] == 'ALU.A.OP'), None)
                a = next((x['value'] for x in steps if x['canonical'] == 'ALU.P.A'), None)
                b = next((x['value'] for x in steps if x['canonical'] == 'ALU.P.B'), None)
                for x in steps:
                    if x['role'] != 'Q' or x['port'] not in FLAG_PORTS:
                        continue
                    p = per.setdefault(x['port'], {'port': x['port'], 'rows_stated': 0, 'judged': 0,
                                                   'contradicting': 0, 'examples': [], '_alts': []})
                    p['rows_stated'] += 1
                    stated = L.to_int(x['value'], 1)
                    rule_v = L.flag_value(mt, x['port'], op, a, b, width, ov_mode) if width else None
                    if rule_v is None or stated is None:
                        continue
                    p['judged'] += 1
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
            alts = p.pop('_alts')
            pr = [o for (tag, p_), o in probe.items() if p_ == port]
            p['probe'] = ({'rows': len(pr), 'design_agrees': sum(o['observed'] == o['rule_value'] for o in pr)}
                          if pr else None)
            alternative = None
            for name in sorted({n for x in alts for n in x}):
                if alts and all(name in x for x in alts):
                    alternative = {**L.alternative_by_name(name), 'explains': len(alts)}
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
                        .order_by(M.SimRun.id.desc()).limit(1)).first()
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
                         'text': q.text, 'untested_scenarios': rs.get('untested_scenarios') or [],
                         'subject': rs['subject'], 'compared_by_testbench': rs['compared_by_testbench'],
                         'evidence': rs['evidence'],
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
    with xdb.session_scope(url) as s:
        problems, overview = build_problems(s, sim_run_id, sr, reqs, spec)
    return {'sim_run_id': sim_run_id, 'session_id': session_id, 'feature_id': feature_id,
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


def _inputs(op, stim, w) -> str:
    stim = stim or {}
    return f"{op or 'operation'} with A = {_pv(stim.get('a'), w)}, B = {_pv(stim.get('b'), w)}"


_FLAG_WORD = {'zero': 'zero flag', 'negative': 'negative flag', 'overflow': 'overflow flag'}


def plain_local(st: Dict) -> Dict:
    v, vals, w = st.get('verdict'), st['values'], st['values'].get('width')
    O, B, R = vals.get('observed'), vals.get('expected_bdd'), vals.get('expected_ref')
    title = _inputs(st.get('operation'), st.get('stimulus'), w)
    port = st['target'].get('port') or 'result'
    lines = []
    if v == 'pass':
        lines.append(f'Correct: the design gives {port} = {_pv(O, w)}, as the scenario expects.')
    elif v == 'scenario_wrong':
        lines.append(f'The scenario is wrong, not the design. The scenario expects {_pv(B, w)}, '
                     f'but the correct {port} is {_pv(R, w)} — and that is what the design gives.')
    elif v == 'design_wrong':
        lines.append(f'The design is wrong. It gives {port} = {_pv(O, w)}; the correct value is '
                     f'{_pv(R, w)}, as the scenario expects.')
    elif v == 'both_wrong':
        lines.append(f'Both are wrong: the scenario expects {_pv(B, w)}, the design gives {_pv(O, w)}, '
                     f'the correct value is {_pv(R, w)}.')
    elif v == 'consistent_wrong':
        lines.append(f'The test passed, but scenario and design are wrong in the same way: both say '
                     f'{_pv(O, w)}, the correct value is {_pv(R, w)}.')
    elif v == 'undecided':
        lines.append(f'There is no reference rule for this design type. Expected {_pv(B, w)}, '
                     f'the design gave {_pv(O, w)}.')
    else:
        lines.append('This check was not executed.')
    r = st.get('rule')
    if r and v not in ('pass', None):
        lines.append(f"Why {_pv(R, w)}: {_with_width(r['text'], w)}.")
    c = st.get('contrast')
    if c and v not in ('pass', None):
        lines.append(f"For comparison, {_inputs(st.get('operation'), c.get('stimulus'), w)} "
                     f"{'passed' if c['status'] == 'pass' else 'failed'}.")
    flags = st.get('probed') or []
    bad = [x for x in flags if x['verdict'] != 'pass']
    if flags and not bad:
        lines.append('Flags (' + ', '.join(_FLAG_WORD.get(x['port'], x['port']) for x in flags)
                     + ') are correct.')
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
    if alt:
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
        lines.append(f'Verified: every simulated example of {subj} passed.')
    elif s_ == 'verified_by_probe':
        lines.append(f"Verified: the {_FLAG_WORD.get(subj, subj)} was right in every example "
                     f"(checked by an extra simulation).")
    elif s_ == 'fail':
        kinds = sorted({f['verdict'] for f in st.get('failing', []) if f.get('verdict')})
        who = {'design_wrong': 'the design', 'scenario_wrong': 'the scenario', 'both_wrong': 'both',
               'consistent_wrong': 'both (in the same way)'}
        lines.append(f"Failing in {len(st.get('failing', []))} examples; wrong: "
                     + ', '.join(who.get(k, k) for k in kinds) + '.')
    elif s_ == 'not_tested':
        sc = st.get('untested_scenarios') or []
        lines.append('Not tested: ' + (f"the scenario “{sc[0]['title']}” could not be turned into tests."
                                      if sc else 'no test ran for it.'))
    elif s_ == 'stated_only':
        lines.append(f'Stated in the scenarios, but {subj} was not simulated.')
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
    main_bad_tags = set()
    for ci, cr in main:
        if not ci.tag.startswith('BDD:') or cr.status not in ('pass', 'fail'):
            continue
        total_op[ci.operation or '?'] = total_op.get(ci.operation or '?', 0) + 1
        if cr.verdict in _WHO:
            by_op.setdefault((cr.verdict, ci.operation or '?'), []).append((ci, cr))
            main_bad_tags.add(ci.tag)
    probs: List[Dict] = []

    order = ['design_wrong', 'both_wrong', 'consistent_wrong', 'scenario_wrong']
    for (v, op), items in sorted(by_op.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        ci, cr = items[0]
        n, m = len(items), total_op.get(op, len(items))
        B, R, O = _int(ci.expected_bdd), _int(ci.expected_ref), _int(cr.observed_value)
        ex = _inputs(op, ci.stimulus, width)
        if v == 'design_wrong':
            title = f'The design computes {op} wrong'
            detail = f'{n} of {m} {op} examples give a wrong result. E.g. {ex}: the design gives {_pv(O, width)}, correct is {_pv(R, width)}.'
        elif v == 'both_wrong':
            title = f'{op}: both the design and the scenario are wrong'
            detail = f'{n} of {m} {op} examples. E.g. {ex}: scenario {_pv(B, width)}, design {_pv(O, width)}, correct {_pv(R, width)}.'
        elif v == 'consistent_wrong':
            title = f'{op}: the design repeats a mistake of the scenario'
            detail = (f'{n} of {m} {op} examples pass the test, but both give the wrong value. '
                      f'E.g. {ex}: both say {_pv(O, width)}, correct is {_pv(R, width)}.')
        else:
            title = f'A scenario expects a wrong {op} result' if n == 1 else f'{n} scenario examples expect a wrong {op} result'
            detail = f'E.g. {ex}: the scenario expects {_pv(B, width)}, correct is {_pv(R, width)}. The design is right.'
        probs.append({'severity': 'error' if _WHO[v] == 'design' else 'warning', 'who': _WHO[v],
                      'title': title, 'detail': detail, 'explain': {'kind': 'check', 'tag': ci.tag}})

    # flags seen by the probe (rows whose result is already wrong are covered above)
    probe = _probe_results(s, _probe_run_id(s, run_id))
    fl: Dict[Tuple[str, str, str], List] = {}
    consequences = 0
    for (tag, port), o in probe.items():
        if o['verdict'] == 'pass':
            continue
        if tag in main_bad_tags:
            consequences += 1
            continue
        fl.setdefault((o['verdict'], port, o.get('operation') or '?'), []).append((tag, o))
    for (v, port, op), items in sorted(fl.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 9, kv[0][1], kv[0][2])):
        tag, o = items[0]
        word = _FLAG_WORD.get(port, port)
        ex = _inputs(op, o.get('stimulus'), width)
        obs = o['observed'] if o['observed'] is not None else o['raw']
        if v == 'scenario_wrong':
            title = f'A scenario states a wrong {word} for {op}' if len(items) == 1 else \
                f'{len(items)} scenario examples state a wrong {word} for {op}'
            detail = (f"E.g. {ex}: the scenario says the {word} is {o['stated']}, but it should be "
                      f"{o['rule_value']}. The design is right.")
        else:
            title = f"The design's {word} is wrong for {op}"
            detail = (f"{len(items)} examples. E.g. {ex}: the design's {word} is {obs}, but it should be "
                      f"{o['rule_value']}.")
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
        elif sc['status'] == 'partial':
            n_skip = sum(1 for r in sc.get('rows') or [] if r['status'] == 'skipped')
            probs.append({'severity': 'warning', 'who': 'test',
                          'title': f"Partly tested: “{sc.get('title') or sc['id']}”",
                          'detail': f'{n_skip} of its examples could not be turned into tests.', 'explain': None})
    for r in reqs:
        if r['status'] == 'not_covered':
            probs.append({'severity': 'warning', 'who': 'coverage', 'title': 'A requirement is not tested by any scenario',
                          'detail': f"“{r['text'].rstrip('.')}”", 'explain': {'kind': 'requirement', 'id': r['requirement_id']}})
        elif r['status'] == 'stated_only':
            probs.append({'severity': 'warning', 'who': 'coverage', 'title': 'A requirement was not simulated',
                          'detail': f"“{r['text']}” — the scenarios state it, but nothing simulated "
                                    f"{r.get('subject') or 'it'}.",
                          'explain': {'kind': 'requirement', 'id': r['requirement_id']}})
    if consequences:
        for pr in probs:
            if pr['who'] == 'design' and pr['explain'] and pr['explain'].get('kind') == 'check':
                pr['detail'] += ' The flags of these examples are wrong too, as a consequence.'
                break

    counts = {'design': 0, 'scenario': 0, 'test': 0, 'coverage': 0}
    for pr in probs:
        counts[pr['who']] = counts.get(pr['who'], 0) + 1
    n_checks = sum(total_op.values())
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
