"""
Write the website's runs into the explainability database.

Three entry points, one per moment something is generated:

    ingest_duv(...)      Step 1 saved a Verilog design            -> duv
    ingest_feature(...)  Step 2 saved a feature file              -> feature
    ingest_run(...)      Step 4 report (requirements, feature, DUV,
                         testbench, simulation log)               -> everything below

    requirements file -> requirement, decision      (Step 0, read whenever a step used it)
    feature file      -> feature, scenario, scenario_step, example_row
    @req_N tags       -> scenario_requirement       (method llm_tag: the model's claim)
    testbench index   -> testbench, check_item      (one per assertion; v2 index adds
                                                     stimulus, port, B and R values)
    simulation log    -> sim_run, check_result      (with the verdict, see verdict.py)
    SQLite call log   -> llm_call

Every entry point is idempotent: the same file in the same session is stored
once; a new generation in the same session becomes a new version that points
at the previous one (parent_id). Nothing here calls a model.
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
from sqlalchemy.exc import IntegrityError

from src import xdb
from src.xdb import gherkin
from src.xdb import models as M
from src.xdb.verdict import classify, observed_value

try:
    from src import run_report, scenario_results, requirements_store
except ImportError:                                                  # pragma: no cover
    import run_report, scenario_results, requirements_store         # type: ignore

_STATUS = {'pass': 'pass', 'fail': 'fail', 'setup': 'setup',
           'skipped': 'not_tested', 'missing': 'not_tested'}
_EXP_RE = re.compile(r'Expected\s+([^,\s]+),?\s+Got\s+(\S+)', re.I)
# "✗ Test 14: FAIL - Expected 4294967245, Got 4294967246 [BDD:ALU-SUB-001/r7]"
_TEST_RE = re.compile(r'Test\s+(\d+)\s*:\s*(PASS|FAIL)\b(?:.*?Expected\s+([^,\s]+),?\s+Got\s+(\S+))?'
                      r'.*?\[BDD:([^\]\s]+)\]')
_TASK = {'duv': 'web_duv_generation', 'bdd': 'web_bdd_generation'}


# ----------------------------------------------------------------- artifacts

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(s, kind: str, path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    data = path.read_bytes()
    sha = _sha(data)
    s.execute(pg_insert(M.Artifact).values(sha256=sha, kind=kind, content=data,
                                           orig_name=path.name, encoding='identity')
              .on_conflict_do_nothing(index_elements=['sha256']))
    return sha


def artifact_bytes(a: M.Artifact) -> bytes:
    """Content of an artifact, whatever its storage encoding."""
    if a.encoding == 'gzip':
        import gzip
        return gzip.decompress(a.content)
    return a.content


def artifact_text(a: M.Artifact) -> str:
    return artifact_bytes(a).decode('utf-8', errors='replace')


def _safe(project_root, path) -> Optional[Path]:
    p, err = run_report.safe_path(str(path) if path else None, project_root)
    if err:
        raise ValueError(err)
    return p


# ----------------------------------------------------------------- design, ports, session

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


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _design(s, module_type: str, bitwidth: Optional[int], verilog: Optional[str] = None) -> M.Design:
    """Spec-level design: what is being designed (alu_32bit), not a particular Verilog."""
    module_type = (module_type or 'other').strip() or 'other'
    name = f'{module_type}_{bitwidth}bit' if bitwidth else module_type
    d = s.scalar(select(M.Design).where(M.Design.source == 'web', M.Design.name == name))
    if d is None:
        d = M.Design(module_type=module_type, bitwidth=bitwidth, source='web', name=name)
        s.add(d)
        s.flush()
    if verilog and not s.scalar(select(M.Port).where(M.Port.design_id == d.id).limit(1)):
        for p in parse_ports(verilog)['ports']:
            if p['width'] is not None:
                s.add(M.Port(design_id=d.id, **p))
        s.flush()
    return d


def _session(s, run_key: str, run_id: Optional[str], design: M.Design) -> M.Session:
    sess = s.scalar(select(M.Session).where(M.Session.run_key == run_key))
    if sess is None:
        sess = M.Session(run_key=run_key, run_id=run_id, design_id=design.id)
        s.add(sess)
        s.flush()
    elif sess.design_id is None:
        sess.design_id = design.id
    return sess


# ----------------------------------------------------------------- LLM calls + provenance

def _copy_llm_calls(s, session_id: int, run_id: Optional[str]) -> Dict[str, int]:
    """Copy this run's calls from the SQLite call log; returns task_type -> newest id.

    source_ref includes the timestamp: on Render the SQLite file is recreated at
    every deploy, so its row ids alone repeat across deploys.
    """
    if not run_id:
        return {}
    try:
        from src import experiment_logger
        db_path = experiment_logger.DB_PATH
    except Exception:
        return {}
    latest: Dict[str, int] = {}
    for c in s.scalars(select(M.LlmCall).where(M.LlmCall.session_id == session_id)
                       .order_by(M.LlmCall.id)):
        if c.task_type:
            latest[c.task_type] = c.id
    if not Path(db_path).exists():
        return latest
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id', (run_id,)).fetchall()
    finally:
        conn.close()
    for r in rows:
        ref = f'sqlite:{r["created_at"]}:{r["id"]}'
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


def _link_provenance(s, sess: M.Session, calls: Dict[str, int]):
    """A file may be ingested before its call reached the log (streaming writes
    the call when the stream closes). Fill the link in on a later ingest."""
    for model, task in ((M.Duv, _TASK['duv']), (M.Feature, _TASK['bdd'])):
        if task not in calls:
            continue
        for row in s.scalars(select(model).where(model.session_id == sess.id,
                                                 model.prov_kind == 'import')):
            row.prov_kind, row.prov_ref = 'llm', f'llm_call:{calls[task]}'
            if model is M.Duv and row.origin == 'unknown':
                row.origin = 'llm'


# ----------------------------------------------------------------- Step 0

def _requirements(s, sess: M.Session, design: M.Design, record: Optional[Dict],
                  req_sha: Optional[str]) -> Dict[str, M.Requirement]:
    reqs: Dict[str, M.Requirement] = {}
    if not record:
        return reqs
    prov_ref = record.get('analysis_run_id') or (f'artifact:{req_sha}' if req_sha else None)
    for r in record.get('requirements') or []:
        q = s.scalar(select(M.Requirement).where(
            M.Requirement.session_id == sess.id, M.Requirement.req_key == r['id'],
            M.Requirement.is_current))
        if q is None:
            q = M.Requirement(design_id=design.id, session_id=sess.id, req_key=r['id'],
                              text=r.get('text', ''), origin='req_step',
                              prov_kind='llm' if r.get('source') == 'llm' else 'human',
                              prov_ref=prov_ref)
            s.add(q)
            s.flush()
        reqs[r['id']] = q
    questions = {q.get('id'): q for q in record.get('questions') or []}
    for d in record.get('decisions') or []:
        key = d.get('question_id') or d.get('id')
        if not key or s.scalar(select(M.Decision).where(M.Decision.session_id == sess.id,
                                                        M.Decision.question_key == key)):
            continue
        q = questions.get(key, {})
        s.add(M.Decision(session_id=sess.id, question_key=key,
                         question=d.get('question') or q.get('question') or '',
                         options=q.get('options'), default_index=q.get('default'),
                         answer=d.get('answer') or '', answer_source=d.get('source'),
                         why=q.get('why'),
                         # the user confirmed or wrote every answer
                         prov_kind='human', prov_ref=prov_ref))
    s.flush()
    return reqs


def _load_record(project_root, data) -> (Optional[Dict], Optional[Path]):
    path = (data or {}).get('requirements_filepath')
    if not path:
        return None, None
    p = _safe(project_root, path)
    return json.loads(p.read_text(encoding='utf-8')), p


# ----------------------------------------------------------------- Step 1 / Step 2

def _next_version(s, model, sess_id: int, sha: str):
    """(existing row or None, parent id) for a file in a session."""
    existing = s.scalar(select(model).where(model.session_id == sess_id, model.artifact == sha))
    if existing is not None:
        return existing, None
    prev = s.scalar(select(model).where(model.session_id == sess_id)
                    .order_by(model.id.desc()).limit(1))
    return None, (prev.id if prev else None)


def _upsert_duv(s, sess, design, sha, verilog, data, calls, compiles=None,
                feature_id=None) -> M.Duv:
    row, parent = _next_version(s, M.Duv, sess.id, sha)
    if row is not None:
        return row
    call = calls.get(_TASK['duv'])
    spec_first = (data.get('workflow_mode') or 'implementation') == 'specification'
    row = M.Duv(session_id=sess.id, design_id=design.id, artifact=sha,
                module_name=parse_ports(verilog)['module'] if verilog else None,
                origin='llm' if call else 'unknown', parent_id=parent,
                reason='regenerated' if parent else 'generated',
                workflow_mode=data.get('workflow_mode') or 'implementation',
                prompt_has_bdd=bool(spec_first and data.get('bdd_filepath')),
                based_on_feature_id=feature_id if spec_first else None,
                prompt_has_decisions=bool(data.get('requirements_filepath')),
                compiles=compiles,
                prov_kind='llm' if call else 'import',
                prov_ref=f'llm_call:{call}' if call else None)
    s.add(row)
    s.flush()
    return row


def _upsert_feature(s, sess, sha, text, data, calls) -> M.Feature:
    row, parent = _next_version(s, M.Feature, sess.id, sha)
    if row is not None:
        return row
    call = calls.get(_TASK['bdd'])
    row = M.Feature(session_id=sess.id, artifact=sha, title=gherkin.feature_title(text),
                    parent_id=parent, reason='regenerated' if parent else 'generated',
                    workflow_mode=data.get('workflow_mode') or 'implementation',
                    prompt_has_decisions=bool(data.get('requirements_filepath')),
                    prov_kind='llm' if call else 'import',
                    prov_ref=f'llm_call:{call}' if call else None)
    s.add(row)
    s.flush()
    return row


def _retry(fn, *a, **kw):
    """Steps of one run may be ingested concurrently (background threads); the
    loser of a race on a unique key simply runs again and finds the row."""
    try:
        return fn(*a, **kw)
    except IntegrityError:
        return fn(*a, **kw)


def ingest_duv(project_root, path, run_id: Optional[str], data: Dict, meta: Optional[Dict] = None,
               url=None) -> Dict:
    def _do():
        p = _safe(project_root, path)
        verilog = p.read_text(encoding='utf-8', errors='replace')
        with xdb.session_scope(url) as s:
            sha = _artifact(s, 'duv', p)
            # meta holds what Step 1 actually generated (the page's free-text input
            # may override module type / bit width given in the request)
            design = _design(s, (meta or {}).get('module_type') or data.get('module_type'),
                             _int((meta or {}).get('bitwidth') or data.get('bitwidth')), verilog)
            sess = _session(s, run_id or f'duv:{sha[:24]}', run_id, design)
            calls = _copy_llm_calls(s, sess.id, run_id)
            record, rpath = _load_record(project_root, data)
            _requirements(s, sess, design, record, _artifact(s, 'requirements', rpath))
            feat = None
            if (data.get('workflow_mode') or '') == 'specification' and data.get('bdd_filepath'):
                fp = _safe(project_root, data['bdd_filepath'])
                fsha = _artifact(s, 'feature', fp)
                feat = _upsert_feature(s, sess, fsha, fp.read_text(encoding='utf-8', errors='replace'),
                                       data, calls)
            duv = _upsert_duv(s, sess, design, sha, verilog, data, calls,
                              compiles=(meta or {}).get('compiles'),
                              feature_id=feat.id if feat else None)
            _link_provenance(s, sess, calls)
            return {'ingested': True, 'stage': 'duv', 'session_id': sess.id, 'duv_id': duv.id}
    return _retry(_do)


def ingest_feature(project_root, path, run_id: Optional[str], data: Dict, meta: Optional[Dict] = None,
                   url=None) -> Dict:
    def _do():
        p = _safe(project_root, path)
        text = p.read_text(encoding='utf-8', errors='replace')
        with xdb.session_scope(url) as s:
            sha = _artifact(s, 'feature', p)
            design = _design(s, data.get('module_type'), _int(data.get('bitwidth')))
            sess = _session(s, run_id or f'bdd:{sha[:24]}', run_id, design)
            calls = _copy_llm_calls(s, sess.id, run_id)
            record, rpath = _load_record(project_root, data)
            _requirements(s, sess, design, record, _artifact(s, 'requirements', rpath))
            feat = _upsert_feature(s, sess, sha, text, data, calls)
            _link_provenance(s, sess, calls)
            return {'ingested': True, 'stage': 'bdd', 'session_id': sess.id, 'feature_id': feat.id}
    return _retry(_do)


# ----------------------------------------------------------------- Step 4

def _log_by_test(log: str) -> Dict[int, Dict]:
    out = {}
    for line in (log or '').splitlines():
        m = _TEST_RE.search(line)
        if m:
            out[int(m.group(1))] = {'status': m.group(2).lower(), 'expected': m.group(3),
                                    'got': m.group(4), 'row': m.group(5), 'line': line.strip()}
    return out


def ingest_run(project_root, paths: Dict[str, Optional[str]], run_id: Optional[str] = None,
               url: Optional[str] = None, data: Optional[Dict] = None) -> Dict:
    """paths: requirements / bdd / duv / testbench / log, relative to project_root
    (or absolute) and inside output/, exactly as for the run report."""
    return _retry(_ingest_run, project_root, paths, run_id, url, data or {})


def _ingest_run(project_root, paths, run_id, url, data) -> Dict:
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
    tb_text = files['testbench'].read_text(encoding='utf-8', errors='replace')
    log_text = files['log'].read_text(encoding='utf-8', errors='replace')
    duv_text = files['duv'].read_text(encoding='utf-8', errors='replace') if files.get('duv') else None
    feature_text = feature_path.read_text(encoding='utf-8', errors='replace') if feature_path else ''
    record = report.get('requirements_record')
    module_type = index.get('module_type') or (record or {}).get('module_type') or 'other'
    m = re.search(r'\bparameter\s+WIDTH\s*=\s*(\d+)', tb_text)
    bitwidth = _int(data.get('bitwidth')) or (int(m.group(1)) if m else None)

    with xdb.session_scope(url) as s:
        sha = {k: _artifact(s, kind, files.get(k)) for k, kind in
               (('requirements', 'requirements'), ('duv', 'duv'),
                ('testbench', 'testbench'), ('log', 'sim_log'))}
        sha['bdd'] = _artifact(s, 'feature', feature_path)

        design = _design(s, module_type, bitwidth, duv_text)
        sess = _session(s, run_id or f"tb:{sha['testbench'][:24]}", run_id, design)
        calls = _copy_llm_calls(s, sess.id, run_id)
        reqs = _requirements(s, sess, design, record, sha['requirements'])

        feature = _upsert_feature(s, sess, sha['bdd'], feature_text, data, calls) if sha['bdd'] else None
        duv = (_upsert_duv(s, sess, design, sha['duv'], duv_text, data, calls,
                           feature_id=feature.id if feature else None) if sha['duv'] else None)
        _link_provenance(s, sess, calls)
        background = gherkin.background_steps(feature_text) if feature_text else []

        # -- scenarios of this feature version
        scen: Dict[str, M.Scenario] = {}
        rows: Dict[str, M.ExampleRow] = {}
        n_new_scen = 0
        for entry in (index.get('catalog') or []) if feature else []:
            sc = s.scalar(select(M.Scenario).where(M.Scenario.feature_id == feature.id,
                                                   M.Scenario.scenario_key == entry['id'],
                                                   M.Scenario.is_current))
            span = entry.get('span') or [0, 0]
            block = feature_text[span[0]:span[1]]
            if sc is None:
                parsed = gherkin.parse_scenario(block)
                n_new_scen += 1
                sc = M.Scenario(feature_id=feature.id, scenario_key=entry['id'],
                                title=entry.get('title') or parsed['title'],
                                tags=entry.get('tags') or parsed['tags'],
                                is_outline=parsed['is_outline'], gherkin=block.strip('\n'),
                                prov_kind=feature.prov_kind, prov_ref=feature.prov_ref)
                s.add(sc)
                s.flush()
                pos = 0
                for st, bg in [(x, True) for x in background] + [(x, False) for x in parsed['steps']]:
                    pos += 1
                    s.add(M.ScenarioStep(scenario_id=sc.id, position=pos, keyword=st['keyword'],
                                         role=st['role'], surface_text=st['text'],
                                         from_background=bg))
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

        # -- testbench + one check per assertion
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
        detail = trace.get('detail') or {}
        checks: Dict[tuple, M.CheckItem] = {
            (c.tag, c.seq): c for c in s.scalars(select(M.CheckItem)
                                                 .where(M.CheckItem.testbench_id == tb.id))}

        def _check(tag, seq=1, **kw):
            if (tag, seq) not in checks:
                checks[(tag, seq)] = M.CheckItem(testbench_id=tb.id, tag=tag, seq=seq, **kw)
                s.add(checks[(tag, seq)])
            return checks[(tag, seq)]

        for entry in index.get('catalog') or []:
            sc_id = scen[entry['id']].id if entry['id'] in scen else None
            for rid in entry.get('rows') or []:
                er = rows.get(rid)
                common = dict(scenario_id=sc_id, example_row_id=er.id if er else None,
                              strength='weak' if rid in weak else 'normal',
                              weak_reason=weak.get(rid))
                infos = detail.get(rid) or [None]
                for k, info in enumerate(infos, 1):
                    info = info or {}
                    _check(f'BDD:{rid}', k, test_no=info.get('test'), port=info.get('port'),
                           width=info.get('width'), operation=info.get('operation'),
                           stimulus=info.get('stimulus'), expected_bdd=info.get('expected_bdd'),
                           expected_ref=info.get('expected_ref'),
                           expected_used=info.get('expected_used'), oracle=info.get('oracle'),
                           **common)
        for g in trace.get('generated') or []:
            _check(f"GEN:{g['id']}", label=g.get('label'))
        s.flush()

        # -- simulation
        dup = s.scalar(select(M.SimRun).where(M.SimRun.testbench_id == tb.id,
                                              M.SimRun.dut_artifact == sha['duv'],
                                              M.SimRun.log_artifact == sha['log']))
        if dup is not None:
            return {'ingested': True, 'new': False, 'session_id': sess.id, 'sim_run_id': dup.id}
        totals = results.get('totals') or {}
        run = M.SimRun(testbench_id=tb.id, duv_id=duv.id if duv else None,
                       dut_artifact=sha['duv'], target='generated_duv', simulator='iverilog',
                       log_artifact=sha['log'],
                       status='ok' if (totals.get('bdd_checks') or totals.get('generated_checks'))
                       else 'no_results')
        s.add(run)
        s.flush()

        by_test = _log_by_test(log_text)
        row_res = {r['id']: r for sc_res in results.get('scenarios') or [] for r in sc_res.get('rows') or []}
        verdicts: Dict[str, int] = {}
        n_results = 0
        for (tag, seq), c in checks.items():
            if not tag.startswith('BDD:'):
                continue
            rid = tag[4:]
            hit = by_test.get(c.test_no) if c.test_no else None
            if hit is not None:
                status, msg = hit['status'], hit['line']
                got, exp_txt = hit['got'], hit['expected']
            else:
                r = row_res.get(rid)
                if r is None:
                    continue
                status = _STATUS.get(r['status'], 'not_tested')
                if c.test_no and status in ('pass', 'fail'):
                    status = 'not_tested'      # this assertion never reported
                msg = r.get('message') or r.get('reason')
                mm = _EXP_RE.search(msg or '')
                exp_txt, got = (mm.group(1), mm.group(2)) if mm else (None, None)
            o = observed_value(status, got, c.expected_used, c.width)
            v = (classify(o, c.expected_bdd, c.expected_ref, c.width)
                 if status in ('pass', 'fail') else None)
            if v:
                verdicts[v] = verdicts.get(v, 0) + 1
            s.add(M.CheckResult(sim_run_id=run.id, check_id=c.id, status=status, message=msg,
                                expected={c.port or 'value': exp_txt} if exp_txt else None,
                                observed={c.port or 'value': got} if got else None,
                                observed_value=o if isinstance(o, int) else None, verdict=v))
            n_results += 1
        for g in results.get('generated') or []:
            c = checks.get((f"GEN:{g['id']}", 1))
            if c is not None:
                s.add(M.CheckResult(sim_run_id=run.id, check_id=c.id,
                                    status=_STATUS.get(g['status'], 'not_tested'),
                                    message=g.get('message')))
                n_results += 1

        return {'ingested': True, 'new': True, 'session_id': sess.id, 'sim_run_id': run.id,
                'design': design.name, 'duv_id': duv.id if duv else None,
                'requirements': len(reqs), 'scenarios_new': n_new_scen, 'checks': len(checks),
                'check_results': n_results, 'verdicts': verdicts, 'llm_calls': len(calls)}


# ----------------------------------------------------------------- website hooks

def _run_safe(fn, background, label):
    if not xdb.enabled():
        return None

    def _work():
        try:
            out = fn()
            print(f'🗄️  explainability DB ({label}): {out}')
            return out
        except Exception as e:                       # never break the website
            print(f'⚠️  explainability DB {label} failed (non-fatal): {type(e).__name__}: {e}')
            return {'ingested': False, 'reason': f'{type(e).__name__}: {e}'}

    if background:
        threading.Thread(target=_work, name=f'xdb-{label}', daemon=True).start()
        return {'queued': True}
    return _work()


def ingest_run_safe(project_root, paths, run_id=None, data=None, background=True) -> Optional[Dict]:
    """Step 4 hook: no-op without DATABASE_URL, never raises, by default does
    not block the request (Neon may need a moment to wake up)."""
    return _run_safe(lambda: ingest_run(project_root, paths, run_id=run_id, data=data),
                     background, 'run')


def ingest_step_safe(project_root, stage: str, path, run_id=None, data=None, meta=None,
                     background=True) -> Optional[Dict]:
    """Step 1 / Step 2 hook: store the generated file as soon as it is saved."""
    fn = {'duv': ingest_duv, 'bdd': ingest_feature}.get(stage)
    if fn is None or not path:
        return None
    return _run_safe(lambda: fn(project_root, path, run_id, data or {}, meta), background, stage)
