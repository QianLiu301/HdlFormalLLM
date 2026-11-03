"""
Explanation engine v1 (design doc §6; milestone M2). Deterministic: no model
is called. Every explanation is computed from stored data, stored as a
structure with its evidence (D19), and rendered to text on demand (en / zh).

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

ALGORITHM = 'bdx_v1'

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
                       text_rule=render(structure, 'en'), evidence=evidence, auto_checks=auto_checks)
    s.add(ex)
    s.flush()
    return ex


def _out(ex: M.Explanation, lang: str) -> Dict:
    return {'explanation_id': ex.id, 'structure': ex.structure, 'text': render(ex.structure, lang)}


# ----------------------------------------------------------------- local

def _latest_run_with(s, tag: str, session_id: Optional[int]) -> Optional[Tuple[M.CheckItem, M.SimRun]]:
    q = (select(M.CheckItem, M.SimRun)
         .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
         .join(M.SimRun, M.SimRun.id == M.CheckResult.sim_run_id)
         .join(M.Testbench, M.Testbench.id == M.CheckItem.testbench_id)
         .where(M.CheckItem.tag == tag))
    if session_id:
        q = q.where(M.Testbench.session_id == session_id)
    r = s.execute(q.order_by(M.SimRun.id.desc(), M.CheckItem.seq).limit(1)).first()
    return (r[0], r[1]) if r else None


def explain_check(tag: str, sim_run_id: Optional[int] = None, session_id: Optional[int] = None,
                  seq: int = 1, lang: str = 'en', url=None) -> Optional[Dict]:
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
        unverified = []
        for st in steps:
            if st['role'] == 'Q' and st['port'] and st['port'] not in compared:
                # not simulated, but the scenario can still contradict the rule library
                expect = (L.flag_value(mt, st['port'], ci.operation, stim.get('a'), stim.get('b'), width)
                          if width and stim.get('a') is not None else None)
                stated = L.to_int(st['value'], 1)
                unverified.append({'port': st['port'], 'value': st['value'], 'text': st['text'],
                                   'rule_value': expect,
                                   'consistent': None if expect is None or stated is None
                                   else expect == stated})

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
            'rule': ({'op': rule['op'], 'expr': rule['expr'], 'en': rule['en'], 'zh': rule['zh'],
                      'value': rule_val, 'source': f'rule library {L.LIBRARY_VERSION}, hand-written'}
                     if rule else None),
            'unverified': unverified, 'contrast': contrast,
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
        q_t = [{'port': ci.port, 'value': O}]
        ex = _store(s, {'subject_type': 'check_item', 'subject_id': ci.id, 'scope': 'local',
                        'feature_id': tb.feature_id, 'sim_run_id': run.id, 'p_t': p_t, 'q_t': q_t,
                        'mode': 'complete'},
                    structure,
                    selected=[{'scenario_id': sc.id if sc else None,
                               'example_row_id': er.id if er else None, 'overlap': len(p_t)}],
                    evidence=structure['evidence'],
                    auto_checks={'rule_agrees_with_reference': (rule_val == R) if rule_val is not None
                                 and R is not None else None})
        return _out(ex, lang)


# ----------------------------------------------------------------- global

def explain_global(feature_id: int, port: str, value, lang: str = 'en', url=None) -> Dict:
    """In which situations does the specification (one feature version) state port = value?"""
    with xdb.session_scope(url) as s:
        feat = s.get(M.Feature, feature_id)
        sess = s.get(M.Session, feat.session_id)
        design = s.get(M.Design, sess.design_id) if sess.design_id else None
        mt, width = (design.module_type, design.bitwidth) if design else (None, None)
        target_w = width if port == 'result' else 1
        want = L.to_int(value, target_w)
        matches, total_rows = [], 0
        for sc in s.scalars(select(M.Scenario).where(M.Scenario.feature_id == feature_id,
                                                     M.Scenario.is_current)
                            .order_by(M.Scenario.scenario_key)):
            for er in s.scalars(select(M.ExampleRow).where(M.ExampleRow.scenario_id == sc.id,
                                                           M.ExampleRow.is_current)
                                .order_by(M.ExampleRow.row_no)):
                total_rows += 1
                steps = _steps(s, sc.id, er.vals, mt, width)
                q = [x for x in steps if x['role'] == 'Q' and x['port'] == port]
                if not q or q[0]['value_int'] is None or L.to_int(q[0]['value'], target_w) != want:
                    continue
                op = next((x['values'].get('OP') for x in steps if x['canonical'] == 'ALU.A.OP'), None)
                matches.append({'scenario_key': sc.scenario_key, 'row_no': er.row_no, 'operation': op,
                                'P': [x['text'] for x in steps if x['role'] == 'P' and not x['background']],
                                'example_row_id': er.id})
        by_op: Dict[str, int] = {}
        for m in matches:
            by_op[m['operation'] or '?'] = by_op.get(m['operation'] or '?', 0) + 1
        verified = s.scalar(select(M.CheckItem.id).join(M.Testbench)
                            .where(M.Testbench.feature_id == feature_id, M.CheckItem.port == port)
                            .limit(1)) is not None
        flag_rule = (L.library_for(mt) or {}).get('flags', {}).get(port)
        structure = {'kind': 'global', 'version': 1,
                     'target': {'port': port, 'value': value, 'feature_id': feature_id,
                                'feature_title': feat.title},
                     'matches': matches, 'match_count': len(matches), 'rows_total': total_rows,
                     'by_operation': by_op, 'verified_by_testbench': verified,
                     'rule': ({**flag_rule, 'source': f'rule library {L.LIBRARY_VERSION}, hand-written'}
                              if flag_rule else None),
                     'provenance': {'scenario': _call_summary(s, feat.prov_ref)}}
        ex = _store(s, {'subject_type': 'feature', 'subject_id': feature_id, 'scope': 'global',
                        'feature_id': feature_id, 'p_t': [], 'q_t': [{'port': port, 'value': want}]},
                    structure, selected=[{'example_row_id': m['example_row_id']} for m in matches],
                    evidence={'feature_id': feature_id})
        return _out(ex, lang)


# ----------------------------------------------------------------- requirement

def explain_requirement(requirement_id: int, sim_run_id: Optional[int] = None, lang: str = 'en',
                        url=None) -> Optional[Dict]:
    with xdb.session_scope(url) as s:
        q = s.get(M.Requirement, requirement_id)
        if q is None:
            return None
        links = s.execute(select(M.ScenarioRequirement, M.Scenario)
                          .join(M.Scenario, M.Scenario.id == M.ScenarioRequirement.scenario_id)
                          .where(M.ScenarioRequirement.requirement_id == q.id,
                                 M.Scenario.is_current)).all()
        if sim_run_id is None:
            sim_run_id = s.scalar(select(M.SimRun.id).join(M.Testbench)
                                  .where(M.Testbench.session_id == q.session_id)
                                  .order_by(M.SimRun.id.desc()).limit(1))
        scen, failing = [], []
        for link, sc in links:
            counts: Dict[str, int] = {}
            res = s.execute(select(M.CheckItem, M.CheckResult)
                            .join(M.CheckResult, (M.CheckResult.check_id == M.CheckItem.id)
                                  & (M.CheckResult.sim_run_id == sim_run_id))
                            .where(M.CheckItem.scenario_id == sc.id)).all()
            for ci, cr in res:
                k = cr.verdict or cr.status
                counts[k] = counts.get(k, 0) + 1
                if cr.status == 'fail':
                    failing.append({'tag': ci.tag, 'verdict': cr.verdict})
            scen.append({'scenario_key': sc.scenario_key, 'title': sc.title, 'method': link.method,
                         'counts': counts})
        if not scen:
            status = 'not_covered'
        elif failing:
            status = 'fail'
        elif not any(x['counts'] for x in scen):
            status = 'not_tested'
        else:
            status = 'pass'
        structure = {'kind': 'requirement', 'version': 1,
                     'target': {'req_key': q.req_key, 'text': q.text},
                     'status': status, 'scenarios': scen, 'failing': failing,
                     'link_source': 'tag written by the model that generated the scenarios (not reviewed)'}
        ex = _store(s, {'subject_type': 'requirement', 'subject_id': q.id, 'scope': 'global',
                        'sim_run_id': sim_run_id, 'p_t': [], 'q_t': [{'requirement': q.req_key}]},
                    structure, selected=[{'scenario_key': x['scenario_key']} for x in scen],
                    evidence={'sim_run_id': sim_run_id, 'requirement_id': q.id})
        return _out(ex, lang)


# ----------------------------------------------------------------- rendering

_VERDICT = {
    'en': {'pass': 'Design and scenario both agree with the reference model.',
           'scenario_wrong': 'The scenario is wrong, not the design.',
           'design_wrong': 'The design is wrong; the scenario is right.',
           'both_wrong': 'Both the scenario and the design are wrong.',
           'consistent_wrong': 'The test passes, but scenario and design share the same mistake.',
           'undecided': 'There is no reference model for this check, so it cannot be decided who is wrong.',
           None: 'This check was not executed.'},
    'zh': {'pass': '设计和场景都与参考模型一致。',
           'scenario_wrong': '错的是场景，不是设计。',
           'design_wrong': '设计有错，场景是对的。',
           'both_wrong': '场景和设计都有错。',
           'consistent_wrong': '测试通过了，但场景和设计犯了同一个错误。',
           'undecided': '这条检查没有参考模型，无法判定是谁错了。',
           None: '这条检查没有被执行。'},
}

_T = {
    'en': {'why': 'Why did {key} row {row} {st}?', 'passed': 'pass', 'failed': 'fail',
           'local': 'local', 'global': 'global', 'conclusion': 'Conclusion',
           'scenario': 'Scenario', 'applied': 'Applied', 'rule': 'Rule', 'design': 'Design output',
           'agrees': 'agrees with the rule', 'disagrees': 'does not agree with the rule',
           'contrast': 'Contrast', 'row': 'row', 'evidence': 'Evidence', 'check': 'check',
           'sim': 'simulation', 'written_by': 'scenario written by', 'prompt': 'prompt',
           'unverified': 'Not verified', 'unverified_tail': 'stated in the scenario, but the testbench does not compare these',
           'flag_bad': 'and the scenario contradicts the rule library here',
           'differs': 'Note: the scenario writes {sv} for {port}; the testbench applied {av} (same bits as {w}-bit value)',
           'expected': 'expected', 'got': 'got',
           'g_title': 'When does the specification say {port} = {value}?',
           'g_count': '{n} of {total} rows state it ({ops}):',
           'g_none': 'No row of this specification states it.',
           'g_unverified': 'The testbench never compares {port}, so these rows are stated, not verified.',
           'g_rule': 'General rule',
           'r_title': 'Requirement {key}: {text}', 'r_status': 'Status',
           'r_none': 'No scenario claims to cover this requirement.',
           'r_scen': 'Scenarios', 'r_fail': 'Failing checks', 'r_link': 'Links'},
    'zh': {'why': '为什么 {key} 第 {row} 行{st}？', 'passed': '通过了', 'failed': '失败了',
           'local': '局部解释', 'global': '全局解释', 'conclusion': '结论',
           'scenario': '场景写的', 'applied': '实际施加', 'rule': '规则', 'design': '设计输出',
           'agrees': '与规则一致', 'disagrees': '与规则不一致',
           'contrast': '对比', 'row': '第', 'evidence': '依据', 'check': '检查',
           'sim': '仿真', 'written_by': '生成场景的模型', 'prompt': 'prompt',
           'unverified': '未验证', 'unverified_tail': '场景中写了，但 testbench 没有比较这些项',
           'flag_bad': '而且场景在这些项上与规则库矛盾',
           'differs': '注意：场景里 {port} 写的是 {sv}，testbench 实际施加的是 {av}（作为 {w} 位数是同一组比特）',
           'expected': '期望', 'got': '实际',
           'g_title': '规格中什么情况下 {port} = {value}？',
           'g_count': '共 {total} 行中有 {n} 行这样写（{ops}）：',
           'g_none': '这份规格中没有任何一行这样写。',
           'g_unverified': 'testbench 从不比较 {port}，所以这些行只是规格中写了，没有经过验证。',
           'g_rule': '一般规则',
           'r_title': '需求 {key}：{text}', 'r_status': '状态',
           'r_none': '没有任何场景声称覆盖这个需求。',
           'r_scen': '场景', 'r_fail': '失败的检查', 'r_link': '关联来源'},
}
_SHORT = {'en': {'pass': 'pass', 'scenario_wrong': 'scenario wrong', 'design_wrong': 'design wrong',
                'both_wrong': 'both wrong', 'consistent_wrong': 'same mistake in both',
                'undecided': 'undecided', 'not_tested': 'not tested', 'setup': 'setup',
                'fail': 'fail', 'not_covered': 'not covered'},
          'zh': {'pass': '通过', 'scenario_wrong': '场景写错', 'design_wrong': '设计错误',
                 'both_wrong': '两边都错', 'consistent_wrong': '一致地错', 'undecided': '无法判定',
                 'not_tested': '未测试', 'setup': '准备步骤', 'fail': '失败', 'not_covered': '未覆盖'}}
_R_LINK = {'en': 'tag written by the model that generated the scenarios (not reviewed)',
           'zh': '由生成场景的模型自己标注（未经人工审核）'}


def _num(v, width=None) -> str:
    if v is None:
        return '—'
    if width and width > 8 and v >= 1 << 8:
        return f'{v} (0x{v:0{(width + 3) // 4}X})'
    return str(v)


def render(st: Dict, lang: str = 'en') -> str:
    lang = lang if lang in _T else 'en'
    t, V = _T[lang], _VERDICT[lang]
    kind = st.get('kind')
    if kind == 'local':
        tg, vals, w = st['target'], st['values'], st['values'].get('width')
        status = t['passed'] if tg.get('status') == 'pass' else t['failed']
        row_lbl = f"{tg.get('row_no')}"
        lines = [f"❓ {t['why'].format(key=tg.get('scenario_key'), row=row_lbl, st=status)}   [{t['local']}]",
                 '', f"{t['conclusion']}: {V.get(st.get('verdict'), V[None])}", '']
        P = '; '.join(f"{x['keyword']} {x['text']}" for x in st['scenario']['P'] if not x['background'])
        Q = [x for x in st['scenario']['Q'] if x['port'] == tg.get('port')]
        q_txt = f"Then {Q[0]['text']}" if Q else f"Then {tg.get('port')} = {_num(vals.get('expected_bdd'), w)}"
        lines.append(f"① {t['scenario']}: {P}; {q_txt}")
        stim = st.get('stimulus') or {}
        if stim:
            lines.append(f"② {t['applied']}: " + ', '.join(
                f"{k} = {_num(v, w) if k != 'opcode' else v}" for k, v in stim.items()))
        r = st.get('rule')
        if r:
            sym = _OP_SYMBOL.get(r['op'], r['op'])
            calc = (f"{_num(stim.get('a'), w)} {sym} {_num(stim.get('b'), w)} = {_num(r.get('value'), w)}"
                    if stim.get('a') is not None else '')
            lines.append(f"③ {t['rule']}: {r[lang]} → {calc}   ({r['source']})")
        o = vals.get('observed')
        agree = (t['agrees'] if r and r.get('value') == o else t['disagrees']) if r and o is not None else ''
        lines.append(f"④ {t['design']}: {tg.get('port')} = {_num(o, w)}  {('— ' + agree) if agree else ''}")
        c = st.get('contrast')
        if c:
            cs = c.get('stimulus') or {}
            c_status = t['passed'] if c['status'] == 'pass' else t['failed']
            lines.append(f"⑤ {t['contrast']}: {t['row'] if lang == 'en' else '第'} {c['row_no']}"
                         f"{'' if lang == 'en' else ' 行'} (a = {_num(cs.get('a'), w)}, b = {_num(cs.get('b'), w)}) "
                         f"{c_status} — {V.get(c.get('verdict'), '')}")
        for d in st.get('stimulus_differs') or []:
            lines.append(t['differs'].format(port=d['port'], sv=d['scenario'], av=d['applied'], w=w))
        lines.append('')
        pv = (st.get('provenance') or {}).get('scenario')
        ev = st.get('evidence') or {}
        src = (f"; {t['written_by']}: {pv['provider']} {pv['model']} ({t['prompt']} {pv.get('prompt_version') or '?'})"
               if pv else '')
        lines.append(f"🔎 {t['evidence']}: {t['check']} [{tg.get('tag')}], {t['sim']} #{ev.get('sim_run_id')}{src}")
        if st.get('unverified'):
            lines.append(f"⚠ {t['unverified']}: " + ', '.join(x['text'] for x in st['unverified'])
                         + f" — {t['unverified_tail']}")
            bad = [x for x in st['unverified'] if x.get('consistent') is False]
            if bad:
                lines.append(f"   ✗ {t['flag_bad']}: " + ', '.join(
                    f"{x['port']} = {x['value']} ({t['rule']}: {x['rule_value']})" for x in bad))
        return '\n'.join(lines)

    if kind == 'global':
        tg = st['target']
        lines = [f"❓ {t['g_title'].format(port=tg['port'], value=tg['value'])}   [{t['global']}]", '']
        if not st['matches']:
            lines.append(t['g_none'])
        else:
            ops = ', '.join(f'{k} ×{n}' for k, n in st['by_operation'].items())
            lines.append(t['g_count'].format(n=st['match_count'], total=st['rows_total'], ops=ops))
            for m in st['matches'][:20]:
                lines.append(f"  • {m['scenario_key']} r{m['row_no']}: " + '; '.join(m['P']))
            if len(st['matches']) > 20:
                lines.append(f"  … (+{len(st['matches']) - 20})")
        if st.get('rule'):
            lines += ['', f"{t['g_rule']}: {st['rule'][lang]}   ({st['rule']['source']})"]
        if not st.get('verified_by_testbench'):
            lines += ['', '⚠ ' + t['g_unverified'].format(port=tg['port'])]
        return '\n'.join(lines)

    if kind == 'requirement':
        tg = st['target']
        lines = [t['r_title'].format(key=tg['req_key'], text=tg['text']), '',
                 f"{t['r_status']}: {_SHORT[lang].get(st['status'], st['status'])}"]
        if not st['scenarios']:
            lines.append(t['r_none'])
        for x in st['scenarios']:
            cnt = ', '.join(f"{_SHORT[lang].get(k, k)} {n}" for k, n in sorted(x['counts'].items())) or '—'
            lines.append(f"  • {x['scenario_key']} {x['title'] or ''}: {cnt}")
        if st['failing']:
            lines.append(f"{t['r_fail']}: " + ', '.join(
                f"{f['tag']} → {_SHORT[lang].get(f['verdict'], f['verdict'])}" for f in st['failing']))
        lines.append(f"{t['r_link']}: {_R_LINK[lang]}")
        return '\n'.join(lines)
    return ''
