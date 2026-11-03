"""
Read side: the v1 competency questions of docs/explainability_db_design.md §3
as functions. Each returns plain dicts so the website can serialise them.
"""

from typing import Dict, List, Optional

from sqlalchemy import func, select, text

from src import xdb
from src.xdb import models as M


def table_counts(url=None) -> Dict[str, int]:
    with xdb.session_scope(url) as s:
        return {t.name: s.scalar(select(func.count()).select_from(t))
                for t in M.Base.metadata.sorted_tables}


def requirement_coverage(session_id: int, url=None) -> List[Dict]:
    """Q1 + Q2: per requirement, the linked scenarios and why it is (not) covered.

    status: pass | fail | weak | not_tested | not_covered, where not_covered
    means no scenario claims the requirement at all.
    """
    sql = text("""
        WITH latest_run AS (
            SELECT DISTINCT ON (t.session_id) r.id, t.session_id
            FROM sim_run r JOIN testbench t ON t.id = r.testbench_id
            WHERE t.session_id = :sid ORDER BY t.session_id, r.id DESC)
        SELECT q.req_key, q.text, sc.scenario_key, sr.method,
               count(cr.check_id)                                   AS checks,
               count(*) FILTER (WHERE cr.status = 'pass')            AS passed,
               count(*) FILTER (WHERE cr.status = 'fail')            AS failed,
               count(*) FILTER (WHERE ci.strength = 'weak')          AS weak
        FROM requirement q
        LEFT JOIN scenario_requirement sr ON sr.requirement_id = q.id
        LEFT JOIN scenario sc   ON sc.id = sr.scenario_id AND sc.is_current
        LEFT JOIN check_item ci ON ci.scenario_id = sc.id
        LEFT JOIN check_result cr ON cr.check_id = ci.id
             AND cr.sim_run_id = (SELECT id FROM latest_run)
        WHERE q.session_id = :sid AND q.is_current
        GROUP BY q.req_key, q.text, sc.scenario_key, sr.method
        ORDER BY q.req_key, sc.scenario_key""")
    out: Dict[str, Dict] = {}
    with xdb.session_scope(url) as s:
        for r in s.execute(sql, {'sid': session_id}).mappings():
            q = out.setdefault(r['req_key'], {'req_key': r['req_key'], 'text': r['text'],
                                              'scenarios': []})
            if r['scenario_key']:
                q['scenarios'].append({k: r[k] for k in
                                       ('scenario_key', 'method', 'checks', 'passed', 'failed', 'weak')})
    for q in out.values():
        sc = q['scenarios']
        if not sc:
            q['status'] = 'not_covered'
        elif any(x['failed'] for x in sc):
            q['status'] = 'fail'
        elif not any(x['checks'] for x in sc):
            q['status'] = 'not_tested'
        elif any(x['weak'] for x in sc):
            q['status'] = 'weak'
        else:
            q['status'] = 'pass'
    return list(out.values())


def scenario_checks(scenario_id: int, url=None) -> List[Dict]:
    """Q3: checks generated from a scenario, with the row values and latest result."""
    sql = text("""
        SELECT ci.tag, ci.strength, er.row_no, er.vals, cr.status, cr.expected,
               cr.observed, cr.message
        FROM check_item ci
        LEFT JOIN example_row er ON er.id = ci.example_row_id
        LEFT JOIN LATERAL (SELECT * FROM check_result c WHERE c.check_id = ci.id
                           ORDER BY c.sim_run_id DESC LIMIT 1) cr ON true
        WHERE ci.scenario_id = :sc ORDER BY er.row_no""")
    with xdb.session_scope(url) as s:
        return [dict(r) for r in s.execute(sql, {'sc': scenario_id}).mappings()]


def scenario_steps(scenario_id: int, url=None) -> Dict[str, List[str]]:
    """The scenario as an implication P -> Q (BDX view)."""
    with xdb.session_scope(url) as s:
        steps = s.scalars(select(M.ScenarioStep).where(M.ScenarioStep.scenario_id == scenario_id)
                          .order_by(M.ScenarioStep.position)).all()
        return {'P': [f'{x.keyword} {x.surface_text}' for x in steps if x.role == 'P'],
                'Q': [f'{x.keyword} {x.surface_text}' for x in steps if x.role == 'Q']}


def provenance(scenario_id: int, url=None) -> Optional[Dict]:
    """Q11 + Q12: who produced a scenario, and the raw LLM exchange if any."""
    with xdb.session_scope(url) as s:
        sc = s.get(M.Scenario, scenario_id)
        if sc is None:
            return None
        out = {'scenario_key': sc.scenario_key, 'prov_kind': sc.prov_kind,
               'prov_ref': sc.prov_ref, 'llm_call': None}
        if sc.prov_ref and sc.prov_ref.startswith('llm_call:'):
            call = s.get(M.LlmCall, int(sc.prov_ref.split(':', 1)[1]))
            if call is not None:
                out['llm_call'] = {'provider': call.provider, 'model_requested': call.model_requested,
                                   'model_served': call.model_served,
                                   'prompt_version': call.prompt_version,
                                   'prompt': call.prompt, 'response': call.response,
                                   'created_at': call.created_at.isoformat()}
        return out


def scenarios_of_session(session_id: int, url=None) -> List[Dict]:
    with xdb.session_scope(url) as s:
        rows = s.execute(select(M.Scenario.id, M.Scenario.scenario_key, M.Scenario.title)
                         .join(M.Feature, M.Feature.id == M.Scenario.feature_id)
                         .where(M.Feature.session_id == session_id, M.Scenario.is_current)
                         .order_by(M.Scenario.scenario_key)).all()
        return [{'id': r.id, 'scenario_key': r.scenario_key, 'title': r.title} for r in rows]
