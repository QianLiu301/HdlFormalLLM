"""
Ingest one website run (Steps 0-4) into the explainability database.

Input is the same set of paths the Step-4 run report uses (requirements,
feature file, DUV, testbench, simulation log), so the database holds exactly
what the report shows, plus provenance:

    requirements  ->  requirement            (origin req_step)
    feature file  ->  feature, scenario, scenario_step, example_row
    @req_N tags   ->  scenario_requirement   (method llm_tag: the model's claim)
    testbench     ->  testbench, check_item  (one per [BDD:...] / [GEN:...] tag)
    sim log       ->  sim_run, check_result
    llm_calls     ->  llm_call               (copied from the SQLite call log)

Idempotent: ingesting the same run again adds nothing; a new simulation of
the same testbench adds one sim_run. Nothing here calls a model.
"""

import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src import xdb
from src.xdb import gherkin
from src.xdb import models as M

try:
    from src import run_report, scenario_results
except ImportError:                                     # pragma: no cover
    import run_report, scenario_results                 # type: ignore

_STATUS = {'pass': 'pass', 'fail': 'fail', 'setup': 'setup',
           'skipped': 'not_tested', 'missing': 'not_tested'}
_EXP_RE = re.compile(r'Expected\s+([^,\s]+),?\s+Got\s+(\S+)', re.I)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(s, kind: str, path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    data = path.read_bytes()
    sha = _sha(data)
    s.execute(pg_insert(M.Artifact).values(sha256=sha, kind=kind, content=data,
                                           orig_name=path.name)
              .on_conflict_do_nothing(index_elements=['sha256']))
    return sha


# ----------------------------------------------------------------- design + ports

_PORT_RE = re.compile(r'\b(input|output|inout)\b\s*(?:wire|reg|logic)?\s*(?:signed\s*)?'
                      r'(\[[^\]]+\])?\s*([A-Za-z_]\w*'
                      # "a, b, input wire c": stop the name list at the next direction keyword
                      r'(?:\s*,\s*(?!(?:input|output|inout)\b)[A-Za-z_]\w*)*)')


def _eval_width(rng: Optional[str], params: Dict[str, int]) -> Optional[int]:
    if not rng:
        return 1
    expr = rng.strip('[]')
    for k, v in params.items():
        expr = re.sub(rf'\b{k}\b', str(v), expr)
    m = re.fullmatch(r'\s*([\d+\-* ()]+)\s*:\s*([\d+\-* ()]+)\s*', expr)
    if not m:
        return None
    try:   # only digits and + - * ( ) reach eval
        return abs(eval(m.group(1)) - eval(m.group(2))) + 1
    except Exception:
        return None


def parse_ports(verilog: str) -> Dict:
    """Module name, parameters and ports of the first module in a Verilog file."""
    src = re.sub(r'//.*?$|/\*.*?\*/', '', verilog, flags=re.S | re.M)
    name = re.search(r'\bmodule\s+(\w+)', src)
    params = {k: int(v) for k, v in re.findall(r'\bparameter\s+(?:\w+\s+)?(\w+)\s*=\s*(\d+)', src)}
    ports = []
    for direction, rng, names in _PORT_RE.findall(src):
        for n in re.split(r'\s*,\s*', names.strip()):
            if n in ('wire', 'reg', 'logic', 'signed') or any(p['name'] == n for p in ports):
                continue
            ports.append({'name': n, 'direction': direction, 'width': _eval_width(rng, params)})
    return {'module': name.group(1) if name else None, 'params': params, 'ports': ports}


def _design(s, module_type: str, verilog: Optional[str], tb_text: str) -> M.Design:
    info = parse_ports(verilog) if verilog else {'module': None, 'params': {}, 'ports': []}
    width = info['params'].get('WIDTH')
    if width is None:
        m = re.search(r'\bparameter\s+WIDTH\s*=\s*(\d+)', tb_text)
        width = int(m.group(1)) if m else None
    name = info['module'] or (f'{module_type}_{width}bit' if width else module_type)
    d = s.scalar(select(M.Design).where(M.Design.source == 'web', M.Design.name == name))
    if d is None:
        d = M.Design(module_type=module_type, bitwidth=width, source='web', name=name)
        s.add(d)
        s.flush()
        for p in info['ports']:
            if p['width'] is not None:
                s.add(M.Port(design_id=d.id, **p))
    return d


# ----------------------------------------------------------------- LLM calls

def _copy_llm_calls(s, session_id: int, run_id: Optional[str]) -> Dict[str, int]:
    """Copy this run's calls from the SQLite call log; returns task_type -> newest id."""
    if not run_id:
        return {}
    try:
        from src.experiment_logger import DB_PATH
    except Exception:
        return {}
    if not Path(DB_PATH).exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id', (run_id,)).fetchall()
    finally:
        conn.close()
    latest: Dict[str, int] = {}
    for r in rows:
        ref = f'sqlite:{r["id"]}'
        call = s.scalar(select(M.LlmCall).where(M.LlmCall.source_ref == ref))
        if call is None:
            extra = json.loads(r['extra']) if r['extra'] else {}
            call = M.LlmCall(
                source_ref=ref, session_id=session_id, task_type=r['task_type'],
                provider=r['provider'], model_requested=r['model'],
                model_served=extra.get('model_served') or extra.get('served_model')
                or extra.get('model_effective'),
                prompt_name=extra.get('prompt_stage') or extra.get('prompt_name'),
                prompt_version=extra.get('prompt_version'),
                system_prompt=r['system_prompt'], prompt=r['prompt'], response=r['response'],
                params=extra or None, latency_ms=r['latency_ms'], success=bool(r['success']),
                error=r['error'])
            s.add(call)
            s.flush()
        if r['task_type']:
            latest[r['task_type']] = call.id
    return latest


# ----------------------------------------------------------------- main entry

def ingest_run(project_root, paths: Dict[str, Optional[str]], run_id: Optional[str] = None,
               url: Optional[str] = None) -> Dict:
    """paths: requirements / bdd / duv / testbench / log, relative to project_root
    (or absolute) and inside output/, exactly as for the run report."""
    report, err = run_report.build(project_root, paths)
    if err:
        return {'ingested': False, 'reason': err}
    root = Path(project_root)
    files = {k: (root / v if v else None) for k, v in report['files'].items()}
    if not files.get('testbench') or not files.get('log'):
        return {'ingested': False, 'reason': 'testbench and simulation log are required'}
    index = scenario_results.load_index(files['testbench'])
    results = report.get('scenario_results')
    if not index or not results:
        return {'ingested': False,
                'reason': 'testbench has no scenario index (generated before scenario tagging)'}

    feature_path = files.get('bdd')
    if feature_path is None and index.get('feature'):
        cand = Path(index['feature'])
        cand = cand if cand.is_absolute() else root / cand
        feature_path = cand if cand.is_file() else None

    tb_text = files['testbench'].read_text(encoding='utf-8', errors='replace')
    duv_text = files['duv'].read_text(encoding='utf-8', errors='replace') if files.get('duv') else None
    feature_text = feature_path.read_text(encoding='utf-8', errors='replace') if feature_path else ''
    record = report.get('requirements_record')
    module_type = index.get('module_type') or (record or {}).get('module_type') or 'other'

    with xdb.session_scope(url) as s:
        sha = {k: _artifact(s, kind, files.get(k)) for k, kind in
               (('requirements', 'requirements'), ('duv', 'duv'),
                ('testbench', 'testbench'), ('log', 'sim_log'))}
        sha['bdd'] = _artifact(s, 'feature', feature_path)

        design = _design(s, module_type, duv_text, tb_text)
        run_key = run_id or f"tb:{sha['testbench'][:24]}"
        sess = s.scalar(select(M.Session).where(M.Session.run_key == run_key))
        if sess is None:
            sess = M.Session(run_key=run_key, run_id=run_id, design_id=design.id)
            s.add(sess)
            s.flush()
        calls = _copy_llm_calls(s, sess.id, run_id)

        # -- requirements (Step 0)
        reqs: Dict[str, M.Requirement] = {}
        for r in (record or {}).get('requirements') or []:
            q = s.scalar(select(M.Requirement).where(
                M.Requirement.session_id == sess.id, M.Requirement.req_key == r['id'],
                M.Requirement.is_current))
            if q is None:
                q = M.Requirement(design_id=design.id, session_id=sess.id, req_key=r['id'],
                                  text=r.get('text', ''), origin='req_step',
                                  prov_kind='llm' if r.get('source') == 'llm' else 'human',
                                  prov_ref=(record or {}).get('analysis_run_id')
                                  or (f"artifact:{sha['requirements']}" if sha['requirements'] else None))
                s.add(q)
                s.flush()
            reqs[r['id']] = q

        # -- feature + scenarios (Step 2)
        feature = None
        if sha['bdd']:
            feature = s.scalar(select(M.Feature).where(M.Feature.session_id == sess.id,
                                                       M.Feature.artifact == sha['bdd']))
            if feature is None:
                bdd_call = calls.get('web_bdd_generation')
                feature = M.Feature(session_id=sess.id, artifact=sha['bdd'],
                                    title=gherkin.feature_title(feature_text),
                                    prov_kind='llm' if bdd_call else 'import',
                                    prov_ref=f'llm_call:{bdd_call}' if bdd_call else feature_path.name)
                s.add(feature)
                s.flush()
        background = gherkin.background_steps(feature_text) if feature_text else []

        scen: Dict[str, M.Scenario] = {}
        rows: Dict[str, M.ExampleRow] = {}
        n_new_scen = 0
        for entry in index.get('catalog') or []:
            if feature is None:
                break
            sc = s.scalar(select(M.Scenario).where(M.Scenario.feature_id == feature.id,
                                                   M.Scenario.scenario_key == entry['id'],
                                                   M.Scenario.is_current))
            span = entry.get('span') or [0, 0]
            block = feature_text[span[0]:span[1]]
            parsed = gherkin.parse_scenario(block)
            if sc is None:
                n_new_scen += 1
                sc = M.Scenario(feature_id=feature.id, scenario_key=entry['id'],
                                title=entry.get('title') or parsed['title'],
                                tags=entry.get('tags') or parsed['tags'],
                                is_outline=parsed['is_outline'], gherkin=block.strip('\n'),
                                prov_kind=feature.prov_kind, prov_ref=feature.prov_ref)
                s.add(sc)
                s.flush()
                pos = 0
                for st in background:
                    pos += 1
                    s.add(M.ScenarioStep(scenario_id=sc.id, position=pos, keyword=st['keyword'],
                                         role=st['role'], surface_text=st['text'],
                                         from_background=True))
                for st in parsed['steps']:
                    pos += 1
                    s.add(M.ScenarioStep(scenario_id=sc.id, position=pos, keyword=st['keyword'],
                                         role=st['role'], surface_text=st['text']))
                for row in parsed['rows']:
                    s.add(M.ExampleRow(scenario_id=sc.id, row_no=row['row_no'], vals=row['vals'],
                                       prov_kind=sc.prov_kind, prov_ref=sc.prov_ref))
                for rid in entry.get('requirements') or []:
                    if rid in reqs:
                        s.add(M.ScenarioRequirement(scenario_id=sc.id, requirement_id=reqs[rid].id,
                                                    method='llm_tag', prov_ref=sc.prov_ref))
                s.flush()
            scen[entry['id']] = sc
            for er in s.scalars(select(M.ExampleRow).where(M.ExampleRow.scenario_id == sc.id,
                                                           M.ExampleRow.is_current)):
                rows[f"{entry['id']}/r{er.row_no}"] = er

        # -- testbench + checks (Step 3)
        tb = s.scalar(select(M.Testbench).where(M.Testbench.session_id == sess.id,
                                                M.Testbench.artifact == sha['testbench']))
        if tb is None:
            tb = M.Testbench(session_id=sess.id, feature_id=feature.id if feature else None,
                             artifact=sha['testbench'], generator='template',
                             oracle_source=index.get('oracle_source'),
                             prov_kind='rule', prov_ref='testbench_generator')
            s.add(tb)
            s.flush()
        trace = index.get('trace') or {}
        weak = trace.get('weak') or {}
        checks: Dict[str, M.CheckItem] = {
            c.tag: c for c in s.scalars(select(M.CheckItem).where(M.CheckItem.testbench_id == tb.id))}

        def _check(tag, **kw):
            if tag not in checks:
                checks[tag] = M.CheckItem(testbench_id=tb.id, tag=tag, **kw)
                s.add(checks[tag])
            return checks[tag]

        for entry in index.get('catalog') or []:
            for rid in entry.get('rows') or []:
                er = rows.get(rid)
                _check(f'BDD:{rid}', scenario_id=scen[entry['id']].id if entry['id'] in scen else None,
                       example_row_id=er.id if er else None,
                       strength='weak' if rid in weak else 'normal', weak_reason=weak.get(rid))
        for g in trace.get('generated') or []:
            _check(f"GEN:{g['id']}", label=g.get('label'))
        s.flush()

        # -- simulation (Step 4)
        dup = s.scalar(select(M.SimRun).where(M.SimRun.testbench_id == tb.id,
                                              M.SimRun.dut_artifact == sha['duv'],
                                              M.SimRun.log_artifact == sha['log']))
        if dup is not None:
            return {'ingested': True, 'new': False, 'session_id': sess.id, 'sim_run_id': dup.id}
        totals = results.get('totals') or {}
        run = M.SimRun(testbench_id=tb.id, dut_artifact=sha['duv'], target='generated_duv',
                       simulator='iverilog', log_artifact=sha['log'],
                       status='ok' if (totals.get('bdd_checks') or totals.get('generated_checks'))
                       else 'no_results')
        s.add(run)
        s.flush()

        n_results = 0
        for sc_res in results.get('scenarios') or []:
            for r in sc_res.get('rows') or []:
                c = checks.get(f"BDD:{r['id']}")
                if c is None:
                    continue
                msg = r.get('message') or r.get('reason')
                m = _EXP_RE.search(msg or '')
                s.add(M.CheckResult(sim_run_id=run.id, check_id=c.id,
                                    status=_STATUS.get(r['status'], 'not_tested'), message=msg,
                                    expected={'value': m.group(1)} if m else None,
                                    observed={'value': m.group(2)} if m else None))
                n_results += 1
        for g in results.get('generated') or []:
            c = checks.get(f"GEN:{g['id']}")
            if c is not None:
                s.add(M.CheckResult(sim_run_id=run.id, check_id=c.id,
                                    status=_STATUS.get(g['status'], 'not_tested'),
                                    message=g.get('message')))
                n_results += 1

        return {'ingested': True, 'new': True, 'session_id': sess.id, 'sim_run_id': run.id,
                'design': design.name, 'requirements': len(reqs), 'scenarios_new': n_new_scen,
                'checks': len(checks), 'check_results': n_results, 'llm_calls': len(calls)}


def ingest_run_safe(project_root, paths, run_id=None, background=True) -> Optional[Dict]:
    """Website hook: no-op without DATABASE_URL, never raises, by default does
    not block the request (Neon may need a moment to wake up)."""
    if not xdb.enabled():
        return None

    def _work():
        try:
            out = ingest_run(project_root, paths, run_id=run_id)
            print(f'🗄️  explainability DB: {out}')
            return out
        except Exception as e:                       # never break the website
            print(f'⚠️  explainability DB ingest failed (non-fatal): {type(e).__name__}: {e}')
            return {'ingested': False, 'reason': f'{type(e).__name__}: {e}'}

    if background:
        threading.Thread(target=_work, name='xdb-ingest', daemon=True).start()
        return {'queued': True}
    return _work()
