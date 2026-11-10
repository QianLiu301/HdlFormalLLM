"""
Layer 1 -> layer 2 of the canonical-step model (design doc §4): map every
stored scenario step to a canonical step of the library, method `rule`.

Unmatched steps get a step_mapping row with canonical_step_id NULL, so
"how much of what this model wrote could be understood" is a query
(competency question Q18), not a guess.
"""

from typing import Dict, List, Optional

from sqlalchemy import select

from src.xdb import library as L
from src.xdb import models as M

METHOD = 'rule'


def _lib_name(module_type: Optional[str]) -> Optional[str]:
    lib = L.library_for(module_type)
    if lib is None:
        return None
    return next(k for k, v in L.LIBRARIES.items() if v is lib)   # alu_comb -> alu


def ensure_library(s, module_type: Optional[str]) -> Dict[str, int]:
    """Canonical steps of a library in the database; returns key -> canonical_step.id."""
    name = _lib_name(module_type)
    if name is None:
        return {}
    ver = int(L.LIBRARY_VERSION.lstrip('v'))
    have = {c.template: c for c in s.scalars(select(M.CanonicalStep).where(
        M.CanonicalStep.module_type == name, M.CanonicalStep.version == ver))}
    out = {}
    for st in L.library_for(name)['steps']:
        row = have.get(st['template'])
        if row is None:
            row = M.CanonicalStep(module_type=name, role=st['role'], template=st['template'],
                                  params={'key': st['key'],
                                          'names': sorted({g for rx in st['patterns']
                                                           for g in rx.groupindex})},
                                  predicate={'port': st['port'], 'expr': st['expr']}
                                  if st['expr'] else None,
                                  version=ver)
            s.add(row)
            s.flush()
        out[st['key']] = row.id
    return out


def map_scenario(s, scenario_id: int, module_type: Optional[str],
                 ids: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Map the steps of one scenario that have no rule mapping yet."""
    ids = ids if ids is not None else ensure_library(s, module_type)
    done = {m.position for m in s.scalars(select(M.StepMapping).where(
        M.StepMapping.scenario_id == scenario_id, M.StepMapping.method == METHOD))}
    n = {'mapped': 0, 'unmapped': 0}
    for st in s.scalars(select(M.ScenarioStep).where(M.ScenarioStep.scenario_id == scenario_id)
                        .order_by(M.ScenarioStep.position)):
        if st.position in done:
            continue
        hit = L.map_step(module_type, st.surface_text) if ids else None
        if hit:
            canon, params = hit
            s.add(M.StepMapping(scenario_id=scenario_id, position=st.position, method=METHOD,
                                canonical_step_id=ids[canon['key']], params=params,
                                method_ref=f"library {L.LIBRARY_VERSION}:{canon['key']}",
                                confidence=1.0))
            n['mapped'] += 1
        else:
            s.add(M.StepMapping(scenario_id=scenario_id, position=st.position, method=METHOD,
                                canonical_step_id=None, params=None,
                                method_ref=f'library {L.LIBRARY_VERSION}: no match'
                                if ids else 'no library for this module type'))
            n['unmapped'] += 1
    s.flush()
    return n


UNKNOWN_ACTION = 'the tool does not recognise this step, so it cannot tell whether the testbench did it'


def blocked_steps(s, scenario_id: int) -> List[Dict]:
    """Given/When steps of a scenario that the web testbench does not do: steps
    the library marks as not executable (e.g. a write with write enable off),
    steps its tag leaves out (a write in a @read scenario), and action (When)
    steps no rule recognises. A check of such a scenario
    shows what the testbench did, not what the scenario says, so it cannot
    decide whether the design or the scenario is wrong (design doc D29)."""
    rows = s.execute(select(M.ScenarioStep, M.StepMapping, M.CanonicalStep)
                     .outerjoin(M.StepMapping, (M.StepMapping.scenario_id == M.ScenarioStep.scenario_id)
                                & (M.StepMapping.position == M.ScenarioStep.position)
                                & (M.StepMapping.method == METHOD))
                     .outerjoin(M.CanonicalStep, M.CanonicalStep.id == M.StepMapping.canonical_step_id)
                     .where(M.ScenarioStep.scenario_id == scenario_id)
                     .order_by(M.ScenarioStep.position)).all()
    out, in_when, ran = [], False, None
    for st, sm, cs in rows:
        if st.keyword in ('Given', 'When', 'Then'):
            in_when = st.keyword == 'When'
        if st.role != 'P':
            continue
        key = (cs.params or {}).get('key') if cs else None
        why = L.not_executable(key)
        if not why and L.runs_by_tag(key):
            if ran is None:
                ran = {op for op in s.scalars(select(M.CheckItem.operation)
                                              .where(M.CheckItem.scenario_id == scenario_id)) if op}
            why = L.not_run(key, ran)
        if why:
            out.append({'text': f'{st.keyword} {st.surface_text}', 'why': why, 'kind': 'not_executable'})
        elif sm is not None and sm.canonical_step_id is None and in_when:
            out.append({'text': f'{st.keyword} {st.surface_text}', 'why': UNKNOWN_ACTION,
                        'kind': 'unknown_action'})
    return out


def remap_all(s) -> Dict[str, int]:
    """Map every stored scenario that is not mapped yet (e.g. runs stored before M2)."""
    rows = s.execute(select(M.Scenario.id, M.Design.module_type)
                     .join(M.Feature, M.Feature.id == M.Scenario.feature_id)
                     .join(M.Session, M.Session.id == M.Feature.session_id)
                     .join(M.Design, M.Design.id == M.Session.design_id)).all()
    total = {'scenarios': 0, 'mapped': 0, 'unmapped': 0}
    cache: Dict[str, Dict[str, int]] = {}
    for sid, mt in rows:
        if mt not in cache:
            cache[mt] = ensure_library(s, mt)
        n = map_scenario(s, sid, mt, cache[mt])
        total['scenarios'] += 1
        total['mapped'] += n['mapped']
        total['unmapped'] += n['unmapped']
    return total
