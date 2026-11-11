"""
End-to-end test of the explainability database (src/xdb).

Real pipeline, no LLM: a feature file the website once generated with Gemini
(artifact/ablation/variant_A_original.feature, with @req_N tags added) ->
TestbenchGenerator -> iverilog on an ALU with one injected bug -> ingestion
-> competency queries. The database must agree with run_report, which is
what the Step-4 page shows.

The feature file is a useful witness: two of its expected values are wrong
(SUB row 7 says 50-100 = 4294967245, SRA row 5 says 0x80000000>>>1 =
3221225471), while the design's only bug is SLTU comparing signed. The
verdict must put the blame on the right side for all three.

Needs a PostgreSQL the test may wipe:
    XDB_TEST_URL=postgresql://bdx:bdx@localhost/bdx_test   (default)
Skipped when that database or iverilog is unavailable.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get('XDB_TEST_URL', 'postgresql://bdx:bdx@localhost/bdx_test')

sqlalchemy = pytest.importorskip('sqlalchemy')

from src.xdb.samples import (ALU, CPU_BUGGY, CPU_DUT, CPU_FEATURE, CPU_OK, CPU_RECORD,  # noqa: E402
                             FIXED_ALU, RF_BUGGY, RF_DUT, RF_FEATURE, RF_FIXED, RF_RECORD)



def _db_ok():
    try:
        eng = sqlalchemy.create_engine(URL)
        with eng.connect():
            return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _db_ok(), reason=f'no test database at {URL}'),
    pytest.mark.skipif(shutil.which('iverilog') is None, reason='iverilog not installed'),
]


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    proj = tmp_path_factory.mktemp('proj')
    out = proj / 'output'
    for d in ('bdd', 'dut', 'requirements', 'simulation'):
        (out / d).mkdir(parents=True)

    feat = (ROOT / 'artifact/ablation/variant_A_original.feature').read_text(encoding='utf-8')
    feat = (feat.replace('@add @arithmetic', '@add @arithmetic @req_1')
                .replace('@sltu', '@sltu @req_2', 1)
                # plant one flag contradiction (ADD row 7: 100 + 200 = 300 is not zero);
                # the testbench never compares flags, so it stays byte-identical
                .replace('| 300             | False     |', '| 300             | True      |', 1))
    (out / 'bdd/alu.feature').write_text(feat, encoding='utf-8')
    (out / 'dut/alu_32bit.v').write_text(ALU, encoding='utf-8')
    record = {'module_type': 'alu', 'created_at': '2026-11-03T10:00:00',
              'analysis_run_id': 'web_test_run',
              'requirements': [
                  {'id': 'REQ-1', 'text': 'ADD returns a + b modulo 2^32', 'source': 'llm'},
                  {'id': 'REQ-2', 'text': 'SLTU compares unsigned', 'source': 'llm'},
                  {'id': 'REQ-3', 'text': 'Reset clears outputs', 'source': 'user'}],
              'questions': [{'id': 'Q1', 'question': 'Is the overflow flag set for SUB?',
                             'options': ['yes, signed overflow', 'no, ADD only'], 'default': 0,
                             'why': 'The description only mentions ADD.'}],
              'decisions': [{'id': 'D1', 'question_id': 'Q1',
                             'question': 'Is the overflow flag set for SUB?',
                             'answer': 'yes, signed overflow', 'source': 'default'}]}
    (out / 'requirements/req.json').write_text(json.dumps(record), encoding='utf-8')

    from src.testbench_generator import TestbenchGenerator
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/alu.feature'), {'module_type': 'alu', 'module_name': 'alu_32bit',
                                       'bitwidth': 32})
    assert tb['success'], tb
    tb_path = Path(tb['filepath'])

    vvp = proj / 'sim.vvp'
    subprocess.run(['iverilog', '-g2012', '-o', str(vvp), str(tb_path), str(out / 'dut/alu_32bit.v')],
                   check=True, cwd=proj)
    log = subprocess.run(['vvp', str(vvp)], capture_output=True, text=True, cwd=proj,
                         encoding='utf-8', errors='replace').stdout
    (out / 'simulation/alu.log').write_text(log, encoding='utf-8')

    rel = lambda p: Path(p).relative_to(proj).as_posix()
    paths = {'requirements': 'output/requirements/req.json', 'bdd': 'output/bdd/alu.feature',
             'duv': 'output/dut/alu_32bit.v', 'testbench': rel(tb_path),
             'log': 'output/simulation/alu.log'}
    # two discarded BDD generations of the step-by-step test
    (out / 'bdd/alu_v1.feature').write_text(feat.replace('Verify ADD', 'Check ADD'), encoding='utf-8')
    (out / 'bdd/alu_v2.feature').write_text(feat.replace('Verify SUB', 'Check SUB'), encoding='utf-8')
    return proj, paths


@pytest.fixture(scope='module')
def db(run, tmp_path_factory):
    """Fresh schema via the real migration, plus a SQLite call log with one BDD call."""
    eng = sqlalchemy.create_engine(URL)
    with eng.begin() as c:
        c.exec_driver_sql('DROP SCHEMA public CASCADE; CREATE SCHEMA public;')
    from src.xdb.__main__ import upgrade
    upgrade(URL)

    from src import experiment_logger
    lite = tmp_path_factory.mktemp('lite') / 'calls.db'
    conn = sqlite3.connect(lite)
    conn.executescript(experiment_logger._SCHEMA)
    conn.execute("INSERT INTO llm_calls (created_at, provider, model, task_type, run_id, prompt, "
                 "response, success, extra) VALUES ('2026-11-03', 'GeminiProvider', "
                 "'gemini-2.5-flash', 'web_bdd_generation', 'web_test_run', 'PROMPT', 'FEATURE', 1, "
                 "'{\"prompt_version\": \"v1\", \"model_served\": \"gemini-2.5-flash-001\"}')")
    conn.commit()
    conn.close()
    old = experiment_logger.DB_PATH
    experiment_logger.DB_PATH = str(lite)
    yield URL
    experiment_logger.DB_PATH = old


def test_ingest_matches_run_report(run, db):
    from src import run_report
    from src.xdb.ingest import ingest_run
    from src.xdb.queries import table_counts
    proj, paths = run
    res = ingest_run(proj, paths, run_id='web_test_run', url=db)
    assert res['ingested'] and res['new'], res
    assert res['llm_calls'] == 1

    report, err = run_report.build(proj, paths)
    assert err is None
    sr = report['scenario_results']
    counts = table_counts(db)
    assert counts['scenario'] == sr['totals']['scenarios']
    assert counts['requirement'] == 3
    assert counts['design'] == 1 and counts['port'] == 9
    assert counts['decision'] == 1 and counts['duv'] == 1
    # every row the report lists becomes exactly one check_result
    from sqlalchemy import create_engine, text as _t
    with create_engine(db).connect() as c:
        main_results = c.execute(_t('SELECT count(*) FROM check_result cr JOIN sim_run r '
                                    'ON r.id = cr.sim_run_id WHERE r.probe_of IS NULL')).scalar()
    assert main_results == sum(len(s['rows']) for s in sr['scenarios']) + len(sr['generated'])

    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        main = 'JOIN sim_run r ON r.id = check_result.sim_run_id WHERE r.probe_of IS NULL AND '
        fails = c.execute(text('SELECT count(*) FROM check_result ' + main + "check_result.status='fail'")).scalar()
        passes = c.execute(text('SELECT count(*) FROM check_result ' + main + "check_result.status='pass'")).scalar()
    rep_fail = sum(1 for s in sr['scenarios'] for r in s['rows'] if r['status'] == 'fail')
    rep_pass = sum(1 for s in sr['scenarios'] for r in s['rows'] if r['status'] == 'pass')
    assert (fails, passes) == (rep_fail, rep_pass)
    assert fails > 0 and passes > 0          # the injected SLTU bug is visible


def test_ingest_is_idempotent(run, db):
    from src.xdb.ingest import ingest_run
    from src.xdb.queries import table_counts
    proj, paths = run
    before = table_counts(db)
    res = ingest_run(proj, paths, run_id='web_test_run', url=db)
    assert res['ingested'] and not res['new']
    assert table_counts(db) == before


def test_requirement_coverage_q1_q2(run, db):
    from src.xdb.queries import requirement_coverage
    cov = {r['req_key']: r for r in requirement_coverage(1, url=db)}
    assert cov['REQ-1']['status'] == 'pass'            # ADD is implemented correctly
    assert cov['REQ-2']['status'] == 'fail'            # SLTU carries the bug
    assert cov['REQ-3']['status'] == 'not_covered'     # no scenario claims reset
    assert cov['REQ-1']['scenarios'][0]['method'] == 'llm_tag'


def test_scenario_as_implication_and_rows(run, db):
    from src.xdb.queries import scenario_checks, scenario_steps, scenarios_of_session
    scen = {s['scenario_key']: s for s in scenarios_of_session(1, url=db)}
    add = scen['ALU-ADD-001']
    steps = scenario_steps(add['id'], url=db)
    # Background "Given the ALU is initialized..." + 2 Given + 1 When  -> P
    assert len(steps['P']) == 4 and steps['P'][0].startswith('Given the ALU is initialized')
    assert len(steps['Q']) == 4 and steps['Q'][0].startswith('Then the result should be')
    checks = scenario_checks(add['id'], url=db)
    assert [c['row_no'] for c in checks] == list(range(1, len(checks) + 1))
    assert checks[0]['vals']['A'] == '0' and checks[0]['status'] == 'pass'


def test_failed_check_has_expected_and_observed(run, db):
    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        r = c.execute(text("""SELECT cr.expected, cr.observed, er.vals FROM check_result cr
                              JOIN check_item ci ON ci.id = cr.check_id
                              JOIN example_row er ON er.id = ci.example_row_id
                              WHERE cr.status = 'fail' LIMIT 1""")).mappings().one()
    assert r['expected']['result'] != r['observed']['result']   # keyed by the compared port
    assert 'A' in r['vals']


def test_provenance_q11_q12(run, db):
    from src.xdb.queries import provenance, scenarios_of_session
    sid = scenarios_of_session(1, url=db)[0]['id']
    p = provenance(sid, url=db)
    assert p['prov_kind'] == 'llm'
    assert p['llm_call']['model_served'] == 'gemini-2.5-flash-001'
    assert p['llm_call']['prompt_version'] == 'v1' and p['llm_call']['response'] == 'FEATURE'


def test_disabled_without_database_url(monkeypatch):
    from src import xdb
    from src.xdb.ingest import ingest_run_safe
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.setenv('XDB_DISABLE_DOTENV', '1')
    assert not xdb.enabled()
    assert ingest_run_safe(ROOT, {}, background=False) is None


def test_parse_ports_resolves_parameters():
    from src.xdb.ingest import parse_ports
    info = parse_ports(ALU)
    ports = {p['name']: p for p in info['ports']}
    assert info['module'] == 'alu_32bit'
    assert ports['a']['width'] == 32 and ports['b']['width'] == 32
    assert ports['opcode']['width'] == 4 and ports['zero']['direction'] == 'output'


# ---------------------------------------------------------------- M1: who is wrong

def _verdict_of(db, tag):
    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        return c.execute(text("""SELECT cr.verdict, cr.status, ci.expected_bdd, ci.expected_ref,
                                        cr.observed_value, ci.stimulus
                                 FROM check_result cr JOIN check_item ci ON ci.id = cr.check_id
                                 JOIN sim_run r ON r.id = cr.sim_run_id
                                 WHERE ci.tag = :t AND r.probe_of IS NULL
                                 ORDER BY cr.sim_run_id DESC LIMIT 1"""),
                         {'t': tag}).mappings().one()


def test_who_is_wrong_on_the_real_feature(run, db):
    sub = _verdict_of(db, 'BDD:ALU-SUB-001/r7')
    assert sub['status'] == 'fail' and sub['verdict'] == 'scenario_wrong'
    assert (int(sub['expected_bdd']), int(sub['expected_ref']), int(sub['observed_value'])) == \
        (4294967245, 4294967246, 4294967246)
    assert sub['stimulus'] == {'a': 50, 'b': 100, 'opcode': '0001'}

    sra = _verdict_of(db, 'BDD:ALU-SRA-001/r5')
    assert sra['verdict'] == 'scenario_wrong'

    sltu = _verdict_of(db, 'BDD:ALU-SLTU-001/r4')
    assert sltu['verdict'] == 'design_wrong'
    assert (int(sltu['expected_bdd']), int(sltu['observed_value'])) == (0, 1)


def test_every_failure_gets_a_side(run, db):
    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        rows = c.execute(text('SELECT cr.status, cr.verdict, count(*) n FROM check_result cr '
                              'JOIN sim_run r ON r.id = cr.sim_run_id '
                              'WHERE cr.verdict IS NOT NULL AND r.probe_of IS NULL '
                              'GROUP BY cr.status, cr.verdict')).all()
    by = {(s, v): n for s, v, n in rows}
    assert all(v != 'pass' for (s, v) in by if s == 'fail')
    assert all(v in ('pass', 'consistent_wrong') for (s, v) in by if s == 'pass')
    assert by.get(('fail', 'design_wrong'), 0) >= 1 and by.get(('fail', 'scenario_wrong'), 0) == 2


def test_verdict_table():
    from src.xdb.verdict import classify, observed_value
    assert classify(7, 7, 7) == 'pass'
    assert classify(7, 8, 7) == 'scenario_wrong'
    assert classify(9, 8, 8) == 'design_wrong'
    assert classify(9, 8, 7) == 'both_wrong'
    assert classify(8, 8, 7) == 'consistent_wrong'
    assert classify(8, 8, None) == 'undecided'
    assert classify('x', 8, 8) == 'design_wrong'
    assert classify(4294967295, -1, '0xFFFFFFFF', width=32) == 'pass'
    assert observed_value('pass', None, 5) == 5 and observed_value('fail', '6', 5) == 6
    assert observed_value('not_tested', '6', 5) is None


# ---------------------------------------------------------------- M1: every generation is kept

def test_step_by_step_versions_and_pairing(run, db):
    from sqlalchemy import create_engine, text
    from src.xdb.ingest import ingest_duv, ingest_feature, ingest_run
    proj, paths = run
    data = {'module_type': 'alu', 'bitwidth': 32, 'workflow_mode': 'specification',
            'requirements_filepath': paths['requirements']}
    # spec-first: three BDD generations, the last one is used for the design
    ids = [ingest_feature(proj, f'output/bdd/{n}', 'web_steps', data, url=db)['feature_id']
           for n in ('alu_v1.feature', 'alu_v2.feature', 'alu.feature')]
    assert len(set(ids)) == 3
    again = ingest_feature(proj, 'output/bdd/alu.feature', 'web_steps', data, url=db)
    assert again['feature_id'] == ids[2]                      # same file -> same row
    duv = ingest_duv(proj, paths['duv'], 'web_steps', {**data, 'bdd_filepath': paths['bdd']},
                     meta={'compiles': True}, url=db)
    res = ingest_run(proj, paths, run_id='web_steps', url=db, data=data)
    assert res['new'] and res['duv_id'] == duv['duv_id']

    with create_engine(db).connect() as c:
        feats = c.execute(text("""SELECT f.id, f.parent_id, f.reason FROM feature f
                                  JOIN session s ON s.id = f.session_id
                                  WHERE s.run_key = 'web_steps' ORDER BY f.id""")).all()
        assert [(f.parent_id, f.reason) for f in feats] == \
            [(None, 'generated'), (ids[0], 'regenerated'), (ids[1], 'regenerated')]
        d = c.execute(text("SELECT * FROM duv WHERE id = :i"), {'i': duv['duv_id']}).mappings().one()
        assert d['workflow_mode'] == 'specification' and d['prompt_has_bdd']
        assert d['based_on_feature_id'] == ids[2] and d['compiles'] and d['module_name'] == 'alu_32bit'
        # the exact pair that was simulated together
        pair = c.execute(text("""SELECT t.feature_id, r.duv_id FROM sim_run r
                                 JOIN testbench t ON t.id = r.testbench_id
                                 WHERE r.id = :r"""), {'r': res['sim_run_id']}).one()
        assert tuple(pair) == (ids[2], duv['duv_id'])
        n_dec = c.execute(text("""SELECT count(*) FROM decision d JOIN session s ON s.id = d.session_id
                                  WHERE s.run_key = 'web_steps'""")).scalar()
        assert n_dec == 1


def test_provenance_linked_when_call_arrives_later(run, db):
    import sqlite3
    from sqlalchemy import create_engine, text
    from src import experiment_logger
    from src.xdb.ingest import ingest_duv, ingest_feature
    proj, paths = run
    data = {'module_type': 'alu', 'bitwidth': 32}
    first = ingest_duv(proj, paths['duv'], 'web_late', data, url=db)
    conn = sqlite3.connect(experiment_logger.DB_PATH)
    conn.execute("INSERT INTO llm_calls (created_at, provider, model, task_type, run_id, prompt, "
                 "response, success) VALUES ('2026-11-03T11:00', 'ClaudeProvider', "
                 "'claude-sonnet-5-5', 'web_duv_generation', 'web_late', 'P', 'module ...', 1)")
    conn.commit()
    conn.close()
    ingest_feature(proj, paths['bdd'], 'web_late', data, url=db)
    with create_engine(db).connect() as c:
        d = c.execute(text("SELECT origin, prov_kind, prov_ref FROM duv WHERE id = :i"),
                      {'i': first['duv_id']}).one()
    assert d.origin == 'llm' and d.prov_kind == 'llm' and d.prov_ref.startswith('llm_call:')


def test_storage_usage(db):
    from src.xdb.queries import storage_usage
    u = storage_usage(db)
    assert u['limit_mb'] > 0 and 0 < u['database_mb'] < u['limit_mb'] and u['artifacts_mb'] >= 0


def test_design_that_does_not_compile_is_kept_with_its_error(run, db):
    """A Step-1 design that fails to compile never reaches simulation; the
    stored compiler error is then the only explanation there is."""
    from sqlalchemy import create_engine, text
    from src.xdb.ingest import ingest_duv
    proj, _ = run
    bad = proj / 'output/dut/alu_broken.v'
    bad.write_text('module alu_32bit(input [31:0] a, output [31:0] r);\n'
                   '  assign r = {32{a[31]}};\n  assign r = {31{1\'b0}, a[0]}};\nendmodule\n',
                   encoding='utf-8')
    err = f'{bad}:3: syntax error'
    res = ingest_duv(proj, 'output/dut/alu_broken.v', 'web_broken',
                     {'module_type': 'alu', 'bitwidth': 32},
                     meta={'compiles': False, 'compile_error': err}, url=db)
    with create_engine(db).connect() as c:
        d = c.execute(text('SELECT compiles, compile_error FROM duv WHERE id = :i'),
                      {'i': res['duv_id']}).one()
    assert d.compiles is False and d.compile_error == err


# ---------------------------------------------------------------- M2: library + explanations

def test_rule_library_matches_reference_model():
    import random
    from src.testbench_generator import TestbenchGenerator as G
    from src.xdb.library import ALU_RULES
    rnd = random.Random(7)
    for w in (8, 16, 32):
        for op, r in ALU_RULES.items():
            for _ in range(2000):
                a, b = rnd.getrandbits(w), rnd.getrandbits(w)
                assert r['fn'](a, b, w) == G._reference_result(op, a, b, w), (op, a, b, w)


def test_every_step_of_the_real_feature_is_mapped(run, db):
    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        steps = c.execute(text("""SELECT count(*) FROM scenario_step st JOIN scenario sc ON sc.id = st.scenario_id
                                  JOIN feature f ON f.id = sc.feature_id WHERE f.session_id = 1""")).scalar()
        mapped = c.execute(text("""SELECT count(*) FILTER (WHERE m.canonical_step_id IS NOT NULL),
                                          count(*) FILTER (WHERE m.canonical_step_id IS NULL)
                                   FROM step_mapping m JOIN scenario sc ON sc.id = m.scenario_id
                                   JOIN feature f ON f.id = sc.feature_id WHERE f.session_id = 1""")).one()
    assert steps > 0 and tuple(mapped) == (steps, 0)


def _run1():
    return 1   # first simulation of the module fixture (test_ingest_matches_run_report)


def test_local_explanation_scenario_wrong(run, db):
    from src.xdb.explain import explain_check
    out = explain_check('ALU-SUB-001/r7', sim_run_id=_run1(), url=db)
    st = out['structure']
    assert st['verdict'] == 'scenario_wrong'
    assert st['rule']['op'] == 'SUB' and st['rule']['value'] == st['values']['expected_ref'] == 4294967246
    assert st['values']['observed'] == 4294967246 and st['values']['expected_bdd'] == 4294967245
    assert st['contrast'] and st['contrast']['status'] == 'pass'
    # the testbench compares only result; the flags were observed by the probe instead
    assert st['unverified'] == []
    assert {(x['port'], x['verdict']) for x in st['probed']} == \
        {('zero', 'pass'), ('overflow', 'pass'), ('negative', 'pass')}
    assert 'The scenario is wrong, not the design.' in out['text']
    assert '50 − 100 = 4294967246' in out['text'] and 'agrees with the rule' in out['text']


def test_local_explanation_design_wrong(run, db):
    from src.xdb.explain import explain_check
    out = explain_check('ALU-SLTU-001/r4', sim_run_id=_run1(), url=db)
    assert out['structure']['verdict'] == 'design_wrong'
    assert 'does not agree with the rule' in out['text']
    assert out['structure']['provenance']['scenario']['model'] == 'gemini-2.5-flash-001'


def test_flag_contradiction_is_caught_by_the_probe(run, db):
    from src.xdb.explain import explain_check
    out = explain_check('ALU-ADD-001/r7', sim_run_id=_run1(), url=db)
    st = out['structure']
    assert st['verdict'] == 'pass'                       # the compared result is right ...
    zero = next(x for x in st['probed'] if x['port'] == 'zero')
    # ... the scenario states zero = 1 for 100 + 200; design and rule say 0
    assert (zero['stated'], zero['observed'], zero['rule_value'], zero['verdict']) == ('True', 0, 0, 'scenario_wrong')
    assert 'zero = 0 ✗ scenario wrong' in out['text']


def test_probe_sees_wrong_flags_of_a_wrong_result(run, db):
    """SLTU row 4: the buggy design returns 1, so its zero flag is 0, while the
    correct result 0 needs zero = 1, which is what the scenario states."""
    from src.xdb.explain import explain_check
    st = explain_check('ALU-SLTU-001/r4', sim_run_id=_run1(), url=db)['structure']
    zero = next(x for x in st['probed'] if x['port'] == 'zero')
    assert (zero['observed'], zero['rule_value'], zero['verdict']) == (0, 1, 'design_wrong')
    assert not any(x['port'] == 'overflow' for x in st['probed'])   # no overflow rule for SLTU


def test_probe_run_is_kept_apart_from_the_testbench_run(run, db):
    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        r = c.execute(text("""SELECT r.probe_of, t.generator, r.status, count(cr.check_id)
                              FROM sim_run r JOIN testbench t ON t.id = r.testbench_id
                              LEFT JOIN check_result cr ON cr.sim_run_id = r.id
                              WHERE r.probe_of = 1 GROUP BY 1, 2, 3""")).one()
        # zero + negative for 68 rows, overflow only for the 14 ADD/SUB rows
        # (68: the last SLTU row counts since the parser keeps a final row without a newline)
        assert tuple(r) == (1, 'probe', 'ok', 68 * 2 + 14)
        # the testbench's own pass rate is untouched by the probe
        n = c.execute(text("SELECT count(*) FROM check_result WHERE sim_run_id = 1")).scalar()
        assert n == 68


def test_requirement_subject():
    from src.xdb.explain import requirement_subject
    assert requirement_subject("When opcode is 4'b0000 (ADD), result = (a + b) modulo 2^32.") == 'result'
    assert requirement_subject('The zero output is 1 exactly when the registered result is 0') == 'zero'
    assert requirement_subject('The overflow output is 1 for ADD when the signed sum ...') == 'overflow'
    assert requirement_subject('SLTU compares unsigned') is None


def test_requirement_status_follows_its_subject(run, db):
    from sqlalchemy import create_engine, text
    from src.xdb.explain import explain_requirement
    eng = create_engine(db)
    with eng.begin() as c:
        def add(key, txt, scenarios):
            rid = c.execute(text("INSERT INTO requirement (uid, rev, is_current, design_id, session_id, "
                                 "req_key, text, origin, prov_kind) VALUES (gen_random_uuid(), 1, true, 1, 1, "
                                 ":k, :t, 'user', 'human') RETURNING id"), {'k': key, 't': txt}).scalar()
            for sk in scenarios:
                c.execute(text("INSERT INTO scenario_requirement (scenario_id, requirement_id, method) "
                               "SELECT id, :r, 'human' FROM scenario WHERE scenario_key = :s AND feature_id = 1"),
                          {'r': rid, 's': sk})
            return rid
        zero_all = add('REQ-Z1', 'The zero output is 1 exactly when the result is 0',
                       ['ALU-ADD-001', 'ALU-SLTU-001'])
        zero_sub = add('REQ-Z2', 'The zero output is 1 exactly when the result is 0', ['ALU-SUB-001'])
        res_sub = add('REQ-R1', 'When opcode is SUB, result = (a - b) modulo 2^32', ['ALU-SUB-001'])
    try:
        z1 = explain_requirement(zero_all, sim_run_id=_run1(), url=db)['structure']
        assert (z1['subject'], z1['compared_by_testbench'], z1['status']) == ('zero', False, 'fail')
        assert {(f['via'], f['verdict']) for f in z1['failing']} == \
            {('flag probe', 'scenario_wrong'), ('flag probe', 'design_wrong')}
        z2 = explain_requirement(zero_sub, sim_run_id=_run1(), url=db)
        assert z2['structure']['status'] == 'verified_by_probe'
        assert 'not compared by the testbench; observed by the flag probe' in z2['text']
        r1 = explain_requirement(res_sub, sim_run_id=_run1(), url=db)['structure']
        assert (r1['subject'], r1['status']) == ('result', 'fail')   # SUB row 7: scenario wrong
    finally:
        with eng.begin() as c:
            c.execute(text("DELETE FROM scenario_requirement WHERE requirement_id IN (:a, :b, :c)"),
                      {'a': zero_all, 'b': zero_sub, 'c': res_sub})
            c.execute(text("DELETE FROM requirement WHERE id IN (:a, :b, :c)"),
                      {'a': zero_all, 'b': zero_sub, 'c': res_sub})


def test_global_explanation_overflow(run, db):
    from src.xdb.explain import explain_global
    st = explain_global(1, 'overflow', 'True', url=db)['structure']
    assert st['by_operation'] == {'ADD': 2, 'SUB': 2}
    assert st['verified_by_testbench'] is False and st['rule']['expr'].startswith('overflow')


def test_requirement_explanation(run, db):
    from sqlalchemy import create_engine, text
    from src.xdb.explain import explain_requirement
    with create_engine(db).connect() as c:
        ids = dict(c.execute(text("SELECT req_key, id FROM requirement WHERE session_id = 1")).all())
    r2 = explain_requirement(ids['REQ-2'], sim_run_id=_run1(), url=db)['structure']
    assert r2['status'] == 'fail' and {f['verdict'] for f in r2['failing']} == {'design_wrong'}
    r3 = explain_requirement(ids['REQ-3'], sim_run_id=_run1(), url=db)['structure']
    assert r3['status'] == 'not_covered'


def test_explanations_are_stored_once(run, db):
    from sqlalchemy import create_engine, text
    from src.xdb.explain import explain_check
    a = explain_check('ALU-SRA-001/r5', sim_run_id=_run1(), url=db)
    b = explain_check('ALU-SRA-001/r5', sim_run_id=_run1(), url=db)
    assert a['explanation_id'] == b['explanation_id'] and a['text'] == b['text']
    with create_engine(db).connect() as c:
        row = c.execute(text("SELECT e.structure->>'verdict', e.text_rule, t.scope, t.sim_run_id "
                             "FROM explanation e JOIN explanation_target t ON t.id = e.target_id "
                             "WHERE e.id = :i"), {'i': a['explanation_id']}).one()
    assert row[0] == 'scenario_wrong' and row[1].startswith('❓') and row[2] == 'local' and row[3] == 1


def test_remap_restores_mappings(run, db):
    from sqlalchemy import create_engine, text
    from src import xdb
    from src.xdb.mapping import remap_all
    eng = create_engine(db)
    with eng.begin() as c:
        before = c.execute(text('SELECT count(*) FROM step_mapping')).scalar()
        c.execute(text('DELETE FROM step_mapping WHERE scenario_id IN (SELECT id FROM scenario LIMIT 3)'))
    with xdb.session_scope(db) as s:
        out = remap_all(s)
    with eng.connect() as c:
        assert c.execute(text('SELECT count(*) FROM step_mapping')).scalar() == before
    assert out['mapped'] > 0


CARRY_FEATURE = """Feature: 16-bit ALU Verification
  As a hardware verification engineer
  I want to verify the ALU implementation
  So that I can ensure correct operation

  Background:
    Given the ALU is initialized with 16-bit operands

  @add @arithmetic
  Scenario Outline: Verify ADD operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the ADD operation with opcode 0000
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    Examples:
      | A     | B     | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
      | 0     | 0     | 0               | True      | False    | False         |
      | 65535 | 65535 | 65534           | False     | True     | True          |
      | 65535 | 1     | 0               | True      | True     | False         |
      | 40000 | 30000 | 4464            | False     | True     | False         |
      | 32767 | 1     | 32768           | False     | False    | True          |

  @sub @arithmetic
  Scenario Outline: Verify SUB operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the SUB operation with opcode 0001
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the overflow flag should be <Overflow>
    And the negative flag should be <Negative_Flag>

    Examples:
      | A | B | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
      | 5 | 3 | 2               | False     | False    | False         |
      | 0 | 1 | 65535           | False     | True     | True          |
      | 1 | 2 | 65535           | False     | True     | True          |
"""


def test_global_explanation_detects_carry_meaning_of_overflow(run, db):
    """Reproduces a DeepSeek run: the scenarios use "overflow" for unsigned
    carry/borrow, the design template means signed overflow, and the
    testbench never compares the flag, so the simulation passes."""
    from sqlalchemy import create_engine, text
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.explain import explain_global
    from src.xdb.ingest import ingest_run
    proj, _ = run
    out = proj / 'output'
    (out / 'bdd/carry.feature').write_text(CARRY_FEATURE, encoding='utf-8')
    (out / 'dut/alu_16bit.v').write_text(
        ALU.replace('alu_32bit', 'alu_16bit').replace('WIDTH = 32', 'WIDTH = 16'), encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/carry.feature'),
        {'module_type': 'alu', 'module_name': 'alu_16bit', 'bitwidth': 16},
        output_filename='carry_tb.v')
    assert tb['success'], tb
    vvp = proj / 'carry.vvp'
    subprocess.run(['iverilog', '-g2012', '-o', str(vvp), tb['filepath'], str(out / 'dut/alu_16bit.v')],
                   check=True, cwd=proj)
    log = subprocess.run(['vvp', str(vvp)], capture_output=True, text=True, cwd=proj,
                         encoding='utf-8', errors='replace').stdout
    (out / 'simulation/carry.log').write_text(log, encoding='utf-8')
    res = ingest_run(proj, {'bdd': 'output/bdd/carry.feature', 'duv': 'output/dut/alu_16bit.v',
                            'testbench': str(Path(tb['filepath']).relative_to(proj)),
                            'log': 'output/simulation/carry.log'},
                     run_id='web_carry', url=db, data={'module_type': 'alu', 'bitwidth': 16})
    assert res['verdicts'] == {'pass': 8}                       # the simulation sees nothing wrong
    with create_engine(db).connect() as c:
        fid = c.execute(text("SELECT f.id FROM feature f JOIN session s ON s.id = f.session_id "
                             "WHERE s.run_key = 'web_carry'")).scalar()
    g = explain_global(fid, 'overflow', 'True', url=db)
    rc = g['structure']['rule_check']
    assert (rc['judged'], rc['contradicting']) == (5, 5)
    assert rc['alternative']['name'] == 'unsigned carry/borrow'
    assert [(m['scenario_key'], m['row_no']) for m in rc['missed']] == [('ALU-ADD-001', 5)]
    # 32767 + 1 has no carry, so the omission fits the same alternative meaning
    assert (rc['alternative']['explains_missed'], rc['alternative']['missed_total']) == (1, 1)
    assert 'and so does the one row it omits' in g['text']
    assert 'seems to use "overflow" in that sense' in g['text']



# ---------------------------------------------------------------- A: fixed Step-0 question

def test_fixed_overflow_question_is_added_for_alu():
    from src import requirements_store as R
    a = {'requirements': [], 'questions': [
        {'id': 'Q1', 'question': 'Is overflow signed?', 'options': ['y', 'n'], 'default': 0, 'why': ''},
        {'id': 'Q2', 'question': 'Reset synchronous?', 'options': ['a', 'b'], 'default': 0, 'why': ''}]}
    out = R.add_fixed_questions(a, 'alu')
    assert [q['question'] for q in out['questions']] == ['Reset synchronous?', R.OVERFLOW_QUESTION]
    assert [q['id'] for q in out['questions']] == ['Q1', 'Q2']
    assert out['questions'][1]['options'] == [R.OVERFLOW_SIGNED, R.OVERFLOW_CARRY]
    assert R.add_fixed_questions(a, 'counter') == a
    # the answer survives build_record unchanged, so the rule library can read it back
    rec, err = R.build_record({'module_type': 'alu', 'requirements': [],
                               'questions': [{**out['questions'][1], 'chosen': 1}]})
    assert err is None and rec['decisions'][0]['answer'] == R.OVERFLOW_CARRY


def test_overflow_rule_follows_the_step0_answer(run, db):
    """Same carry-style specification as above, but now Step 0 said overflow
    means carry/borrow: the rows are consistent and nothing is flagged."""
    from sqlalchemy import create_engine, text
    from src import requirements_store as R
    from src.xdb.explain import explain_check, explain_global
    eng = create_engine(db)
    with eng.begin() as c:
        sid, fid = c.execute(text("SELECT s.id, f.id FROM session s JOIN feature f ON f.session_id = s.id "
                                  "WHERE s.run_key = 'web_carry'")).one()
        c.execute(text("INSERT INTO decision (session_id, question_key, question, answer, answer_source, "
                       "prov_kind) VALUES (:s, 'Q1', :q, :a, 'user', 'human')"),
                  {'s': sid, 'q': R.OVERFLOW_QUESTION, 'a': R.OVERFLOW_CARRY})
    g = explain_global(fid, 'overflow', 'True', url=db)
    rc = g['structure']['rule_check']
    assert (rc['judged'], rc['contradicting'], rc['missed_count']) == (5, 0, 0)
    assert rc['alternative'] is None and 'decided in Step 0' in g['text']
    loc = explain_check('ALU-ADD-001/r2', session_id=sid, url=db)['structure']
    # scenario and Step 0 agree (carry); the design computes signed overflow -> the design is wrong now
    ov = next(x for x in loc['probed'] if x['port'] == 'overflow')
    assert (ov['stated'], ov['observed'], ov['rule_value'], ov['verdict']) == ('True', 0, 1, 'design_wrong')
    assert g['structure']['probe'] == {'rows': 8, 'design_agrees': 2,
                                       'verdicts': {'pass': 2, 'design_wrong': 6}}
    with eng.begin() as c:     # leave the session as the earlier test found it
        c.execute(text("DELETE FROM decision WHERE session_id = :s"), {'s': sid})


# ---------------------------------------------------------------- M3: website API

def test_explain_api(run, db, monkeypatch):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    proj, paths = run
    monkeypatch.setattr(main, 'PROJECT_ROOT', proj)
    monkeypatch.setenv('DATABASE_URL', db)
    c = main.app.test_client()
    body = {'testbench_path': paths['testbench'], 'log_file': paths['log']}
    s = c.post('/api/explain/summary', json=body).get_json()
    assert s['enabled'] and s['ready'] and s['design'] == 'alu_32bit'
    assert s['verdicts'] == {'pass': 62, 'design_wrong': 4, 'scenario_wrong': 2}
    assert s['checks']['BDD:ALU-SUB-001/r7']['verdict'] == 'scenario_wrong'
    assert any('zero flag' in x and 'contradicts' in x for x in s['spec_check_text'])
    # P4/P5: what was compared, and what the flag probe saw
    assert s['compared_ports'] == ['result']
    assert s['probe']['status'] == 'ok'
    assert s['probe']['per_port']['zero'] == {'rows': 68, 'design_agrees': 64}
    assert s['probe']['issues']['BDD:ALU-ADD-001/r7'] == [{'port': 'zero', 'verdict': 'scenario_wrong'}]
    # P1: status by subject port
    st = {r['req_key']: (r['status'], r['subject']) for r in s['requirements']}
    assert st == {'REQ-1': ('verified', None), 'REQ-2': ('fail', None), 'REQ-3': ('not_covered', None)}
    e = c.post('/api/explain/check', json={'tag': 'BDD:ALU-SLTU-001/r4',
                                           'sim_run_id': s['sim_run_id']}).get_json()
    assert 'The design is wrong' in e['text']
    g = c.post('/api/explain/global', json={'feature_id': s['feature_id'], 'port': 'zero',
                                            'value': 'True'}).get_json()
    assert g['text'].splitlines()[3].startswith('  • ALU-ADD-001 r7')   # the contradicting row first
    req = next(r['requirement_id'] for r in s['requirements'] if r['req_key'] == 'REQ-3')
    q = c.post('/api/explain/requirement', json={'requirement_id': req}).get_json()
    assert 'not covered' in q['text']
    # a run that is not in the database yet, and no database at all
    nf = c.post('/api/explain/summary', json={'testbench_path': paths['bdd'],
                                              'log_file': paths['log']}).get_json()
    assert nf == {'success': True, 'enabled': True, 'ready': False}
    monkeypatch.delenv('DATABASE_URL')
    monkeypatch.setenv('XDB_DISABLE_DOTENV', '1')
    assert c.post('/api/explain/summary', json=body).get_json() == {'success': True, 'enabled': False}


# ---------------------------------------------------------------- plain problems list

def test_problems_list_in_plain_words(run, db):
    from src.xdb.explain import summarize_run
    proj, paths = run
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    probs = s['problems']
    titles = [p['title'] for p in probs]
    assert titles[0] == 'The design compares SLTU operands as signed numbers'   # most severe first
    assert 'A scenario expects a wrong SUB result' in titles
    assert 'A scenario expects a wrong SRA result' in titles
    assert 'A scenario states a wrong zero flag for ADD' in titles
    assert 'A requirement is not tested by any scenario' in titles         # REQ-3: no reset scenario (D50)
    assert s['overview']['ok'] is False and s['overview']['counts']['design'] == 1
    # no internal ids in what the user reads first
    for p in probs:
        for field in (p['title'], p['detail']):
            assert 'ALU-' not in field and 'BDD:' not in field and 'REQ-' not in field
    sub = next(p for p in probs if 'SUB' in p['title'])
    assert 'A = 50, B = 100' in sub['detail'] and 'The design is right' in sub['detail']


def test_plain_explanation_has_no_ids(run, db):
    from src.xdb.explain import explain_check
    pl = explain_check('ALU-SUB-001/r7', sim_run_id=_run1(), url=db)['plain']
    assert pl['title'] == 'SUB with A = 50, B = 100'
    assert pl['lines'][0].startswith('The scenario is wrong, not the design.')
    assert 'modulo 2^32' in ' '.join(pl['lines'])
    assert not any('ALU-' in l or 'BDD:' in l or '#' in l for l in pl['lines'])


def test_scenario_without_tests_makes_its_requirement_not_tested(run, db):
    """A scenario the generator cannot turn into tests leaves its requirement
    'not tested' -- not 'not compared by the testbench' (the testbench does
    compare result, just not for this scenario)."""
    import json as _json
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.explain import summarize_run
    from src.xdb.ingest import ingest_run
    proj, paths = run
    out = proj / 'output'
    feat = (out / 'bdd/alu.feature').read_text(encoding='utf-8') + (
        '\n  @req_4\n  Scenario: Verify SLT against SLTU on the same operands\n'
        '    Given I have operand A = 4294967295\n    When I perform the SLT operation with opcode 1000\n'
        '    Then the result should be 1\n')
    (out / 'bdd/alu_untested.feature').write_text(feat, encoding='utf-8')
    rec = _json.loads((out / 'requirements/req.json').read_text(encoding='utf-8'))
    rec['requirements'].append({'id': 'REQ-4', 'text': 'For SLT, result is 1 when A < B as signed values.',
                                'source': 'llm'})
    (out / 'requirements/req_untested.json').write_text(_json.dumps(rec), encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/alu_untested.feature'),
        {'module_type': 'alu', 'module_name': 'alu_32bit', 'bitwidth': 32}, output_filename='untested_tb.v')
    vvp = proj / 'untested.vvp'
    subprocess.run(['iverilog', '-g2012', '-o', str(vvp), tb['filepath'], str(out / 'dut/alu_32bit.v')],
                   check=True, cwd=proj)
    log = subprocess.run(['vvp', str(vvp)], capture_output=True, text=True, cwd=proj,
                         encoding='utf-8', errors='replace').stdout
    (out / 'simulation/untested.log').write_text(log, encoding='utf-8')
    p2 = {'requirements': 'output/requirements/req_untested.json', 'bdd': 'output/bdd/alu_untested.feature',
          'duv': paths['duv'], 'testbench': str(Path(tb['filepath']).relative_to(proj)),
          'log': 'output/simulation/untested.log'}
    assert ingest_run(proj, p2, run_id='web_untested', url=db)['new']
    s = summarize_run(proj, {'testbench_path': p2['testbench'], 'log_file': p2['log']}, url=db)
    req4 = next(r for r in s['requirements'] if r['req_key'] == 'REQ-4')
    assert (req4['status'], req4['compared_by_testbench']) == ('not_tested', True)
    nt = next(p for p in s['problems'] if p['who'] == 'test')
    assert nt['title'] == 'Not tested: “Verify SLT against SLTU on the same operands”'
    assert 'without an Examples table' in nt['detail'] and 'For SLT, result is 1' in nt['detail']


# ---------------------------------------------------------------- M4 repair

def test_apply_corrections_edits_only_the_named_cells():
    from src.xdb.repair import apply_corrections
    text = ("Feature: f\n\n  Scenario Outline: A\n    Then the result should be <r>\n"
            "    Examples:\n      | a | r   | zero  |\n      | 1 | 5   | False |\n      | 2 | 6   | True  |\n"
            "    # a comment inside the table block\n"
            "    Examples:\n      | a | r |\n      | 3 | 7 |\n\n  Scenario: B\n    Then x\n")
    span = [text.index('  Scenario Outline'), text.index('  Scenario: B')]
    corr = [{'tag': 'BDD:S1/r2', 'scenario_key': 'S1', 'row_no': 2, 'column': 'zero', 'old': 'True', 'new': 'False'},
            {'tag': 'BDD:S1/r3', 'scenario_key': 'S1', 'row_no': 3, 'column': 'r', 'old': '7', 'new': '1234'},
            # the file says 5, not 9: never overwrite a cell that is not what was simulated
            {'tag': 'BDD:S1/r1', 'scenario_key': 'S1', 'row_no': 1, 'column': 'r', 'old': '9', 'new': '5'}]
    new, applied, skipped = apply_corrections(text, [{'id': 'S1', 'span': span}], corr)
    assert '      | 2 | 6   | False |\n' in new          # column width kept
    assert '      | 3 | 1234 |\n' in new                 # rows are counted across Examples tables
    assert '      | 1 | 5   | False |\n' in new
    assert [c['row_no'] for c in applied] == [2, 3] and [c['row_no'] for c in skipped] == [1]
    assert new.endswith('  Scenario: B\n    Then x\n')


def test_reference_value_keeps_the_cell_style():
    from src.xdb.repair import _fmt_flag, _fmt_result
    assert _fmt_result(0xC0000000, '0x7FFFFFFF', 32) == '0xC0000000'
    assert _fmt_result(0xFFFFFFCE, '-49', 32) == '-50'
    assert _fmt_result(3221225472, '3221225471', 32) == '3221225472'
    assert (_fmt_flag(0, 'True'), _fmt_flag(1, 'false'), _fmt_flag(1, '0')) == ('False', 'true', '1')




def _client(run, db, monkeypatch):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    from src.simulation_runner import WebSimulationRunner
    proj, _ = run
    monkeypatch.setattr(main, 'PROJECT_ROOT', proj)
    monkeypatch.setattr(main, 'simulation_runner', WebSimulationRunner(project_root=str(proj)))
    monkeypatch.setenv('DATABASE_URL', db)
    return main.app.test_client()


def _page_body(paths, **over):
    """What Step 4 of the page sends for the simulation it shows."""
    body = {'testbench_path': paths['testbench'], 'log_file': paths['log'],
            'bdd_filepath': paths['bdd'], 'duv_filepath': paths['duv'],
            'requirements_filepath': paths['requirements'], 'run_id': 'web_test_run',
            'workflow_mode': 'implementation',
            'dut_info': {'module_type': 'alu', 'bitwidth': 32}}
    body.update(over)
    return body


def _session_run_id(db, summary):
    from src import xdb
    from src.xdb import models as M
    with xdb.session_scope(db) as ss:
        return ss.get(M.Session, summary['session_id']).run_id


def _summary_when_ready(c, body, tries=40):
    import time
    for _ in range(tries):
        s = c.post('/api/explain/summary', json=body).get_json()
        if s.get('ready'):
            return s
        time.sleep(0.5)
    raise AssertionError('the new simulation never reached the database')


def test_repair_scenarios_end_to_end(run, db, monkeypatch):
    """SUB r7, SRA r5 (result) and ADD r7 (zero flag) are set to the reference
    values in a NEW feature file; testbench and simulation run again; the new
    run has no scenario errors left and only the design's SLTU bug."""
    from sqlalchemy import select
    from src import xdb
    from src.xdb import models as M
    proj, paths = run
    c = _client(run, db, monkeypatch)
    body = _page_body(paths)
    before = (proj / paths['bdd']).read_text(encoding='utf-8')
    s = _summary_when_ready(c, body)
    rp = s['repairs']
    assert (rp['scenario_values'], rp['design_failures'], rp['design_regeneration']) == (3, 4, True)
    # earlier tests stored discarded generations in this session, so 'regenerated' is right too
    assert rp['feature_reason'] in ('generated', 'regenerated') and rp['duv_reason'] == 'generated'

    r = c.post('/api/repair/scenarios', json=body).get_json()
    assert r['success'], r
    # the new version joins the session of the simulation it repairs (earlier
    # tests stored these same files under several run ids; the newest wins)
    assert r['feature_filename'] == 'alu_fix1.feature' and r['run_id'] == _session_run_id(db, s)
    got = {(x['scenario_key'], x['row_no'], x['port']): (x['old'], x['new']) for x in r['corrections']}
    assert got == {('ALU-SUB-001', 7, 'result'): ('4294967245', '4294967246'),
                   ('ALU-SRA-001', 5, 'result'): ('3221225471', '3221225472'),
                   ('ALU-ADD-001', 7, 'zero'): ('True', 'False')}
    assert r['skipped'] == []
    # in file order, in plain words with the inputs the page shows elsewhere
    assert [x['operation'] for x in r['corrections']] == ['ADD', 'SUB', 'SRA']
    assert r['corrections'][2]['text'] == ('SRA with A = 2147483648 (0x80000000), B = 1: '
                                           'result 3221225471 → 3221225472')
    assert r['corrections'][0]['text'].endswith('zero flag True → False')
    assert (proj / paths['bdd']).read_text(encoding='utf-8') == before     # the old file is kept
    fixed = (proj / r['feature_path']).read_text(encoding='utf-8')
    assert len(fixed) == len(before) and fixed.count('\n') == before.count('\n')
    assert r['simulation']['success'] and r['testbench_path'].endswith('alu_fix1_tb.v')

    # the page then asks for the run report of the new simulation, under the run id the repair returned
    body2 = _page_body(paths, bdd_filepath=r['feature_path'], testbench_path=r['testbench_path'],
                       log_file=r['simulation']['log_file'], run_id=r['run_id'])
    assert c.post('/api/run-report', json=body2).get_json()['success']
    s2 = _summary_when_ready(c, body2)
    assert s2['verdicts'] == {'pass': 64, 'design_wrong': 4}
    assert s2['probe']['issues'].get('BDD:ALU-ADD-001/r7') is None
    assert s2['repairs']['scenario_values'] == 0 and s2['repairs']['design_failures'] == 4
    assert s2['repairs']['feature_reason'] == f"feedback:{r['feedback_id']}"
    assert not any(p['who'] == 'scenario' for p in s2['problems'])

    with xdb.session_scope(db) as ss:
        new = ss.get(M.Feature, r['feature_id'])
        old = ss.get(M.Feature, new.parent_id)
        assert old.artifact != new.artifact and not old.reason.startswith('feedback:')
        fb = ss.get(M.Feedback, r['feedback_id'])
        assert (fb.status, fb.result_ref, fb.subject_id) == ('applied', f'feature:{new.id}', s['sim_run_id'])
        prov = {(sc.scenario_key, er.row_no): er.prov_kind for sc, er in ss.execute(
            select(M.Scenario, M.ExampleRow).join(M.ExampleRow, M.ExampleRow.scenario_id == M.Scenario.id)
            .where(M.Scenario.feature_id == new.id))}
        # only the rows the repair changed are the rules' work
        assert {k for k, v in prov.items() if v == 'rule'} == {('ALU-SUB-001', 7), ('ALU-SRA-001', 5),
                                                               ('ALU-ADD-001', 7)}
        # every other row keeps the provenance of the feature, as before the repair
        assert new.prov_kind == old.prov_kind and new.prov_kind != 'rule'
        assert {v for v in prov.values()} == {'rule', new.prov_kind}

    # nothing left to correct
    again = c.post('/api/repair/scenarios', json=body2).get_json()
    assert not again['success'] and again['error'].startswith('nothing to correct')


def test_repair_refuses_a_feature_that_was_not_simulated(run, db, monkeypatch):
    proj, paths = run
    c = _client(run, db, monkeypatch)
    r = c.post('/api/repair/scenarios', json=_page_body(paths, bdd_filepath='output/bdd/alu_v1.feature')).get_json()
    assert not r['success'] and 'not the one this simulation was generated from' in r['error']


def test_repair_design_with_feedback(run, db, monkeypatch):
    """The design is regenerated with its failing checks in the prompt (fake
    model returns the fixed SLTU); it is stored as the next version of the
    design in the same session, and the rerun has no design errors left."""
    from sqlalchemy import select
    from src import experiment_logger, xdb
    from src.alu_generator import ALUGenerator
    from src.xdb import models as M
    proj, paths = run
    c = _client(run, db, monkeypatch)
    body = _page_body(paths)

    run_id = _session_run_id(db, _summary_when_ready(c, body))
    fb = c.post('/api/repair/design-feedback', json=body).get_json()
    assert fb['success'] and fb['count'] == 4 and fb['operations'] == ['SLTU']
    assert 'SLTU is 1 when A < B as unsigned numbers' in fb['text']
    assert 'result must be 1, the design gave 0' in fb['text'] or 'result must be 0, the design gave 1' in fb['text']
    assert 'BUG: SLTU compares signed' in fb['previous_verilog']

    seen = {}

    class FakeDesignModel:
        def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
            seen['prompt'] = prompt
            return '```verilog\n' + FIXED_ALU + '\n```'
    experiment_logger.instrument_class(FakeDesignModel)
    import main
    for cls in {ALUGenerator, main.ALUGenerator}:            # main may import it under another module name
        monkeypatch.setattr(cls, '_setup_llm', lambda self: FakeDesignModel())

    hw = c.post('/api/generate-hardware', json={
        'module_type': 'alu', 'llm': 'deepseek', 'bitwidth': 32, 'workflow_mode': 'implementation',
        'requirements_filepath': paths['requirements'], 'run_id': 'some_other_run',
        'repair_of': {'testbench_path': paths['testbench'], 'log_file': paths['log']}}).get_json()
    assert hw['success'], hw
    assert hw['run_id'] == run_id                            # continues the repaired run
    assert hw['repair']['failures'] == 4
    p = seen['prompt']
    assert 'IMPORTANT - Repair of a previous version' in p and 'SLTU with a = ' in p
    assert p.index('Confirmed Requirements') < p.index('Repair of a previous version')
    assert 'BUG: SLTU compares signed' in p                  # the previous version is shown

    new_duv = Path(hw['filepath']).relative_to(proj).as_posix()
    rr = c.post('/api/repair/rerun', json=_page_body(paths, duv_filepath=new_duv, run_id=run_id)).get_json()
    assert rr['success'], rr
    body2 = _page_body(paths, duv_filepath=new_duv, testbench_path=rr['testbench_path'],
                       log_file=rr['simulation']['log_file'], run_id=run_id)
    assert c.post('/api/run-report', json=body2).get_json()['success']
    s2 = _summary_when_ready(c, body2)
    assert s2['verdicts'] == {'pass': 66, 'scenario_wrong': 2}
    assert s2['repairs']['design_failures'] == 0
    assert s2['repairs']['duv_reason'] == f"feedback:{hw['repair']['feedback_id']}"

    with xdb.session_scope(db) as ss:
        duv = ss.scalar(select(M.Duv).where(M.Duv.reason == f"feedback:{hw['repair']['feedback_id']}"))
        sess = ss.get(M.Session, duv.session_id)
        assert sess.run_id == run_id and duv.parent_id is not None
        assert duv.prov_kind == 'llm'
        call = ss.get(M.LlmCall, int(duv.prov_ref.split(':')[1]))
        assert call.params['repair_feedback_id'] == hw['repair']['feedback_id']
        fbr = ss.get(M.Feedback, hw['repair']['feedback_id'])
        assert (fbr.status, fbr.result_ref, fbr.proposed['action']) == ('applied', f'duv:{duv.id}',
                                                                         'regenerate_design')


def test_design_repair_only_for_alu(run, db, monkeypatch):
    proj, paths = run
    c = _client(run, db, monkeypatch)
    r = c.post('/api/generate-hardware', json={'module_type': 'counter', 'llm': 'deepseek',
                                               'repair_of': {'testbench_path': paths['testbench'],
                                                             'log_file': paths['log']}})
    assert r.status_code == 400 and 'CPU designs only' in r.get_json()['error']


# ================================================================ register file (M5)
# A stateful design: the testbench runs every Examples row in document order and
# carries the register contents from row to row. RF_FEATURE has one stateful
# scenario mistake (R1 is written 100, then 300 by a later row; the read still
# expects 100); RF_BUGGY does not protect register 0 (writes to it are stored,
# reads of it return them).






def _simulate(proj, tb_path, duv_path, log_rel):
    vvp = proj / (Path(log_rel).stem + '.vvp')
    subprocess.run(['iverilog', '-g2012', '-o', str(vvp), str(tb_path), str(duv_path)], check=True, cwd=proj)
    log = subprocess.run(['vvp', str(vvp)], capture_output=True, text=True, cwd=proj,
                         encoding='utf-8', errors='replace').stdout
    (proj / log_rel).write_text(log, encoding='utf-8')
    return log


@pytest.fixture(scope='module')
def rf_run(db, tmp_path_factory):
    """The register-file sample, simulated on the buggy design and ingested (run id rf_run)."""
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    proj = tmp_path_factory.mktemp('rfproj')
    out = proj / 'output'
    for d in ('bdd', 'dut', 'requirements', 'simulation'):
        (out / d).mkdir(parents=True)
    (out / 'bdd/rf.feature').write_text(RF_FEATURE, encoding='utf-8')
    (out / 'dut/regfile.v').write_text(RF_BUGGY, encoding='utf-8')
    (out / 'requirements/req.json').write_text(json.dumps(RF_RECORD), encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/rf.feature'), {**RF_DUT, 'dut_filepath': str(out / 'dut/regfile.v')})
    assert tb['success'], tb
    _simulate(proj, tb['filepath'], out / 'dut/regfile.v', 'output/simulation/rf.log')
    paths = {'requirements': 'output/requirements/req.json', 'bdd': 'output/bdd/rf.feature',
             'duv': 'output/dut/regfile.v', 'testbench': Path(tb['filepath']).relative_to(proj).as_posix(),
             'log': 'output/simulation/rf.log'}
    res = ingest_run(proj, paths, run_id='rf_run', url=db, data={'bitwidth': 32})
    assert res['ingested'], res
    return proj, paths, res


def test_regfile_replay_matches_the_testbench_model(rf_run):
    """The library's own replay gives the same value for every read as the
    testbench generator's model (two independent implementations)."""
    from src import scenario_results as SR
    from src.xdb import library as L
    proj, paths, _ = rf_run
    detail = SR.load_index(proj / paths['testbench'])['trace']['detail']
    events = [{'tag': f'BDD:{rid}', 'test': d['test'], 'op': d['operation'],
               'addr': d['stimulus'].get('addr'), 'data': d['stimulus'].get('data')}
              for rid, ds in detail.items() for d in ds]
    rep = L.regfile_replay(events, 32)
    reads = {f'BDD:{rid}': ds[0]['expected_ref'] for rid, ds in detail.items() if ds[0]['port'] == 'rdata1'}
    assert reads and {t: rep[t]['value'] for t in reads} == reads
    # causes: a later write, the reset row, register 0, a write in the same row
    assert rep['BDD:RF-READ-001/r1']['cause'] == {'kind': 'write', 'tag': 'BDD:RF-WRITE-001/r4', 'test': 4,
                                                 'value': 300, 'ignored': False}
    assert [w['value'] for w in rep['BDD:RF-READ-001/r1']['earlier']] == [100]
    assert rep['BDD:RF-READ-003/r1']['cause']['kind'] == 'reset'
    assert rep['BDD:RF-READ-002/r1']['cause'] == {'kind': 'x0'}
    assert rep['BDD:RF-WRITE_READ-001/r1']['cause']['tag'] == 'BDD:RF-WRITE_READ-001/r1'
    assert rep['BDD:RF-READ-001/r4']['cause'] == {'kind': 'initial'}


def test_regfile_testbench_is_unchanged_by_the_index(tmp_path):
    """The index records every executed row; the Verilog does not change.
    Pinned by the number of tests and the absence of index text in the file."""
    from src.testbench_generator import TestbenchGenerator
    f = tmp_path / 'rf.feature'
    f.write_text(RF_FEATURE, encoding='utf-8')
    tb = TestbenchGenerator(output_dir=str(tmp_path / 'tb'), debug=False).generate_single(str(f), dict(RF_DUT))
    text = Path(tb['filepath']).read_text(encoding='utf-8')
    assert tb['test_count'] == 15 and 'expected_ref' not in text and text.count('[BDD:') == 9 * 3


def test_regfile_steps_are_all_mapped(rf_run, db):
    from src import xdb
    from src.xdb import models as M
    from sqlalchemy import select, func
    with xdb.session_scope(db) as s:
        n = dict(s.execute(select(M.StepMapping.canonical_step_id.is_(None), func.count())
                           .join(M.Scenario, M.Scenario.id == M.StepMapping.scenario_id)
                           .join(M.Feature, M.Feature.id == M.Scenario.feature_id)
                           .join(M.Session, M.Session.id == M.Feature.session_id)
                           .where(M.Session.run_id == 'rf_run')
                           .group_by(M.StepMapping.canonical_step_id.is_(None))).all())
    assert n.get(True, 0) == 0 and n[False] > 20


def test_regfile_verdicts_problems_and_requirements(rf_run, db):
    from src.xdb.explain import summarize_run
    proj, paths, res = rf_run
    assert res['verdicts'] == {'pass': 7, 'scenario_wrong': 1, 'design_wrong': 1}
    assert res['probe'] == {'status': 'not_applicable'}
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    titles = [p['title'] for p in s['problems']]
    assert titles == ['The design returns wrong values on register reads',
                      'A scenario expects a wrong register value',
                      'A requirement is not tested by any scenario']
    design, scen = s['problems'][0]['detail'], s['problems'][1]['detail']
    assert 'reading R0: the design returns 123, correct is 0' in design
    assert '123 is what example 1 of “Writes to register 0 are ignored” tried to write to register 0' in design
    assert '“register 0 reads as 0 after a write”, fails as well' in design      # the testbench's own check
    assert ('100 was written by example 1 of “Write values into registers”, but example 4 of the same '
            'scenario overwrote it with 300 before this read') in scen
    for p in s['problems']:
        assert 'RF-' not in p['title'] + p['detail'] and 'BDD:' not in p['detail']
    checked = ' '.join(s['overview']['checked'])
    assert checked.startswith('9 register reads were simulated') and 'read port 2' in checked
    st = {r['req_key']: (r['status'], r['subject']) for r in s['requirements']}
    assert st == {'REQ-1': ('fail', 'rdata1'), 'REQ-2': ('fail', 'rdata1'), 'REQ-3': ('verified', 'rdata1'),
                  'REQ-4': ('not_covered', 'rdata2')}
    assert s['repairs']['scenario_values'] == 1 and s['repairs']['design_failures'] == 2
    assert s['spec_check'] == []


def test_regfile_local_explanations(rf_run, db):
    from src.xdb.explain import explain_check, summarize_run
    proj, paths, _ = rf_run
    run = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)['sim_run_id']
    e = explain_check('RF-READ-001/r1', sim_run_id=run, url=db)
    assert e['plain']['title'] == 'Reading R1'
    assert e['plain']['lines'][:2] == [
        'The scenario is wrong, not the design. It expects 100, but R1 holds 300 at this point — and that '
        'is what the design returns.',
        '100 was written by example 1 of “Write values into registers”, but example 4 of the same scenario '
        'overwrote it with 300 before this read.']
    assert e['structure']['history']['cause']['row_no'] == 4
    after_reset = explain_check('RF-READ-003/r1', sim_run_id=run, url=db)['plain']['lines']
    assert after_reset[1] == ('Why 0: the reset in example 1 of “Reset clears every register” cleared it, '
                              'and nothing wrote R1 after that.')
    wr = explain_check('RF-WRITE_READ-001/r1', sim_run_id=run, url=db)['plain']
    assert wr['title'] == 'Writing 42 to R7 and reading it back'
    assert wr['lines'][1] == 'Why 42: this row wrote 42 to R7 just before reading it.'
    x0 = explain_check('RF-READ-002/r1', sim_run_id=run, url=db)
    assert x0['structure']['verdict'] == 'design_wrong'
    assert x0['plain']['lines'][1] == 'Why 0: register 0 always reads 0; writes to it are ignored.'


def test_regfile_repair_scenarios_and_design(rf_run, db, monkeypatch):
    """Correct the stale expected value (100 -> 300), then regenerate the design
    with a fake model returning the fixed register file: no problem is left
    but the uncovered requirement."""
    from src import experiment_logger
    proj, paths, _ = rf_run
    c = _client((proj, paths), db, monkeypatch)
    body = _page_body(paths, run_id='rf_run', dut_info=dict(RF_DUT))
    r = c.post('/api/repair/scenarios', json=body).get_json()
    assert r['success'], r
    assert [(x['column'], x['old'], x['new']) for x in r['corrections']] == [('expected', '100', '300')]
    assert r['corrections'][0]['text'] == 'Reading R1: expected value 100 → 300'
    body2 = _page_body(paths, run_id='rf_run', dut_info=dict(RF_DUT), bdd_filepath=r['feature_path'],
                       testbench_path=r['testbench_path'], log_file=r['simulation']['log_file'])
    assert c.post('/api/run-report', json=body2).get_json()['success']
    s2 = _summary_when_ready(c, body2)
    assert s2['verdicts'] == {'pass': 8, 'design_wrong': 1, 'setup': 6, 'fail': 1}

    fb = c.post('/api/repair/design-feedback', json=body2).get_json()
    assert fb['success'] and fb['count'] == 2
    assert ('- read of R0 at step 12 of the sequence: rdata1 must be 0, the design gave 123. Before it: '
            'step 11 wrote 123 to R0 (must be ignored); register 0 must always read 0') in fb['text']
    assert "the testbench's own check “register 0 reads as 0 after a write” failed" in fb['text']
    assert '- WRITE: a write stores the data at the clock edge; writes to register 0 are ignored' in fb['text']

    class FakeDesignModel:
        def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
            return '```verilog\n' + RF_FIXED + '\n```'
    experiment_logger.instrument_class(FakeDesignModel)
    import main
    monkeypatch.setattr(main.RegFileGenerator, '_setup_llm', lambda self: FakeDesignModel())
    hw = c.post('/api/generate-hardware', json={
        'module_type': 'regfile', 'llm': 'deepseek', 'bitwidth': 32, 'depth': 32,
        'workflow_mode': 'implementation', 'run_id': 'rf_run',
        'repair_of': {'testbench_path': r['testbench_path'], 'log_file': r['simulation']['log_file']}}).get_json()
    assert hw['success'], hw
    new_duv = Path(hw['filepath']).relative_to(proj).as_posix()
    rr = c.post('/api/repair/rerun', json=_page_body(paths, run_id='rf_run', dut_info=dict(RF_DUT),
                                                     bdd_filepath=r['feature_path'], duv_filepath=new_duv)).get_json()
    assert rr['success'], rr
    body3 = _page_body(paths, run_id='rf_run', dut_info=dict(RF_DUT), bdd_filepath=r['feature_path'],
                       duv_filepath=new_duv, testbench_path=rr['testbench_path'],
                       log_file=rr['simulation']['log_file'])
    assert c.post('/api/run-report', json=body3).get_json()['success']
    s3 = _summary_when_ready(c, body3)
    assert s3['verdicts'] == {'pass': 10, 'setup': 6}          # 9 reads + the testbench's own R0 check
    assert [p['who'] for p in s3['problems']] == ['coverage']
    assert s3['repairs']['duv_reason'] == f"feedback:{hw['repair']['feedback_id']}"


# ================================================================ RV32I CPU (M5b)
# Every Examples row runs as its own short program (ADDI loads the operands, the
# instruction runs, SW stores the result). CPU_FEATURE has: two plain scenario
# mistakes (SUB 10-30 expects 20; BLT -5<3 expects "no"), a row that gives x1
# two values, a row that writes x0, a row whose instruction writes another
# register than its rd column (the testbench skips it), an LW with an offset and
# a prompt-format SW row (both checked correctly since the template fix).
# CPU_BUGGY compares SLT/SLTI unsigned.






def _ingest_cpu(db, tmp_path_factory, design, run_id):
    """The CPU sample, simulated on `design` and stored under `run_id`."""
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    proj = tmp_path_factory.mktemp('cpuproj')
    out = proj / 'output'
    for d in ('bdd', 'dut', 'requirements', 'simulation'):
        (out / d).mkdir(parents=True)
    (out / 'bdd/cpu.feature').write_text(CPU_FEATURE, encoding='utf-8')
    (out / 'dut/cpu.v').write_text(design, encoding='utf-8')
    (out / 'requirements/req.json').write_text(json.dumps(CPU_RECORD), encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/cpu.feature'), dict(CPU_DUT))
    assert tb['success'], tb
    _simulate(proj, tb['filepath'], out / 'dut/cpu.v', 'output/simulation/cpu.log')
    paths = {'requirements': 'output/requirements/req.json', 'bdd': 'output/bdd/cpu.feature',
             'duv': 'output/dut/cpu.v', 'testbench': Path(tb['filepath']).relative_to(proj).as_posix(),
             'log': 'output/simulation/cpu.log'}
    res = ingest_run(proj, paths, run_id=run_id, url=db, data={'bitwidth': 32})
    assert res['ingested'], res
    return proj, paths, res


@pytest.fixture(scope='module')
def cpu_run(db, tmp_path_factory):
    """The CPU sample, simulated on the CPU whose SLT compares unsigned (run id cpu_run)."""
    return _ingest_cpu(db, tmp_path_factory, CPU_BUGGY, 'cpu_run')


# A CPU that sends every taken branch to address 0, so the program never gets
# past its first taken branch (trial cpu_v2, 2026-11-04). All examples before it
# are correct; the four branches behind it never ran.
_PC_UPDATE = 'pc <= (is_br && taken) ? pc + imm_b : pc + 4;'
CPU_STRAY = CPU_OK.replace(_PC_UPDATE, "pc <= (is_br && taken) ? 32'd0 : pc + 4;")
CPU_STUCK = CPU_OK.replace(_PC_UPDATE, 'pc <= pc;')


@pytest.fixture(scope='module')
def stray_run(db, tmp_path_factory):
    assert CPU_STRAY != CPU_OK
    return _ingest_cpu(db, tmp_path_factory, CPU_STRAY, 'stray_run')


def test_cpu_examples_behind_a_stray_branch_are_not_judged(stray_run, db):
    from src.xdb.explain import explain_check, summarize_run
    proj, paths, res = stray_run
    assert res['verdicts'] == {'pass': 11, 'scenario_wrong': 3, 'design_wrong': 1}
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    probs = {p['title']: (p['who'], p['detail']) for p in s['problems']}
    assert probs['A BEQ branch sends the program to the wrong place'] == (
        'design', '1 of 1 BEQ examples. E.g. beq x1, x2, 8 with x1 = 10, x2 = 10: the branch should be taken, '
                  'but after it the program did not arrive where it should.')
    who, detail = probs['The program never reached 4 examples']
    assert who == 'design' and 'The last example the program reached is beq x1, x2, 8 with x1 = 10, x2 = 10.' in detail
    assert not any('decides' in t or 'BLT' in t for t in probs)   # not reached: says nothing about BLT
    assert s['repairs']['design_failures'] == 5

    run = s['sim_run_id']
    taken = explain_check('CPU-BRANCH-001/r1', sim_run_id=run, url=db)
    assert taken['structure']['verdict'] == 'design_wrong'
    assert taken['plain']['lines'][0] == ('The design is wrong. The branch should be taken, but after it the '
                                          'program did not arrive where it should, so the branch sent it '
                                          'somewhere else.')
    behind = explain_check('CPU-BRANCH-001/r2', sim_run_id=run, url=db)
    assert behind['structure']['verdict'] is None and behind['structure']['target']['status'] == 'not_tested'
    assert behind['plain']['lines'][0].startswith('The program never reached this example')


def test_cpu_repair_feedback_names_the_stray_branch(stray_run, db):
    from src.xdb.repair import design_feedback
    proj, paths, _ = stray_run
    fb, err = design_feedback(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    assert err is None and fb['count'] == 2 and fb['operations'] == ['BEQ']
    assert 'the branch must be taken, and the program must then continue with the instruction 8 bytes' in fb['text']
    assert '4 further example(s) were never reached' in fb['text']
    assert 'must be not taken' not in fb['text']


def test_cpu_program_that_never_starts_can_still_be_repaired(db, tmp_path_factory):
    from src.xdb.explain import summarize_run
    from src.xdb.repair import design_feedback
    assert CPU_STUCK != CPU_OK
    proj, paths, res = _ingest_cpu(db, tmp_path_factory, CPU_STUCK, 'stuck_run')
    assert res['verdicts'] == {}
    q = {'testbench_path': paths['testbench'], 'log_file': paths['log']}
    s = summarize_run(proj, q, url=db)
    assert s['problems'][0]['title'] == 'The program never reached 19 examples'
    assert s['problems'][0]['detail'].startswith('Not even the first example ran')
    assert s['repairs']['design_failures'] == 19
    fb, err = design_feedback(proj, q, url=db)
    assert err is None and '19 further example(s) were never reached' in fb['text']


def test_cpu_rules_match_the_testbench_reference():
    """Library RV32I semantics vs. TestbenchGenerator._rv32i_reference / _branch_taken."""
    import random
    from src.testbench_generator import TestbenchGenerator as T
    from src.xdb import library as L
    rnd = random.Random(7)
    corners = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 2047, 0xFFFFF800]
    for op in ('add', 'sub', 'and', 'or', 'xor', 'slt', 'addi', 'andi', 'ori', 'xori', 'slti'):
        for a in corners + [rnd.getrandbits(32) for _ in range(40)]:
            for b in corners[:4] + [rnd.getrandbits(32)]:
                imm = L._s32(b) % 4096 - 2048 if op.endswith('i') else None
                lib = L.CPU_RULES[op.upper()]['cfn'](a, (imm & 0xFFFFFFFF) if imm is not None else b)
                assert lib == T._rv32i_reference(op, a, b, imm), (op, a, b, imm)
    for op in ('beq', 'bne', 'blt', 'bge'):
        for a in corners:
            for b in corners:
                assert L.CPU_RULES[op.upper()]['cfn'](a, b) == T._branch_taken(op, a, b)


def test_cpu_reference_judges_a_row_as_executed():
    from src.xdb.library import cpu_reference as ref
    row = lambda ins, **kw: {'instruction': ins, 'rs1': 1, 'rs1_value': 10, 'rs2': 2, 'rs2_value': 20,
                             'rd': 3, 'path': 'register', **kw}
    assert ref(row('add x3, x1, x2'))['value'] == 30
    same = ref(row('add x3, x1, x1', rs2=1, rs2_value=7))
    assert same['value'] == 14 and [n['kind'] for n in same['notes']] == ['same_register']
    x0 = ref(row('addi x0, x1, 5', rd=0))
    assert x0['value'] == 0 and [n['kind'] for n in x0['notes']] == ['x0_dest']
    assert ref(row('add x3, x1, x0', rs2=0, rs2_value=5))['notes'][0]['kind'] == 'x0_value'
    kinds = lambda r: [(i['kind'], i['fault']) for i in r['issues']]
    assert kinds(ref(row('lw x3, 4(x1)', mem_value=5))) == [('offset', 'test')]      # testbench before the fix
    assert ref(row('lw x3, 4(x1)', mem_value=5, offset_applied=True))['issues'] == []
    assert kinds(ref(row('sw x2, 0(x1)', rd=0))) == [('store_checked_as_register', 'test')]
    assert ref(row('sw x2, 0(x1)', rd=None, path='memory'))['value'] == 20
    assert kinds(ref(row('add x5, x1, x2'))) == [('rd_mismatch', 'scenario')]
    assert kinds(ref(row('add x3, x4, x2'))) == [('unset_register', 'scenario')]
    assert kinds(ref(row('add x3, x30, x2'))) == [('reserved', 'test'), ('unset_register', 'scenario')]
    assert ref(row('blt x1, x2, 8', rs1_value=-5, rs2_value=3))['value'] == 1
    assert ref(row('lw x3, 4(x1)', mem_value=5))['value'] is None


def test_cpu_steps_are_all_mapped(cpu_run, db):
    from sqlalchemy import func, select
    from src import xdb
    from src.xdb import models as M
    with xdb.session_scope(db) as s:
        n = dict(s.execute(select(M.StepMapping.canonical_step_id.is_(None), func.count())
                           .join(M.Scenario, M.Scenario.id == M.StepMapping.scenario_id)
                           .join(M.Feature, M.Feature.id == M.Scenario.feature_id)
                           .join(M.Session, M.Session.id == M.Feature.session_id)
                           .where(M.Session.run_id == 'cpu_run')
                           .group_by(M.StepMapping.canonical_step_id.is_(None))).all())
    assert n.get(True, 0) == 0 and n[False] > 25


def test_cpu_verdicts_problems_and_requirements(cpu_run, db):
    from src.xdb.explain import summarize_run
    proj, paths, res = cpu_run
    assert res['verdicts'] == {'pass': 13, 'scenario_wrong': 4, 'design_wrong': 2}
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    probs = {p['title']: (p['who'], p['detail']) for p in s['problems']}
    assert list(probs)[:2] == ['The design executes SLT wrong', 'The design executes SLTI wrong']
    assert probs['The design executes SLT wrong'][1].endswith(
        'the design gives 0, correct is 1. The design seems to compare the operands as unsigned numbers.')
    assert 'the scenario expects 20, correct is -20' in probs['A scenario expects a wrong SUB result'][1]
    assert 'the scenario says not taken, but it is taken' in probs['A scenario expects the wrong BLT decision'][1]
    assert 'gives x1 two values, 5 and 7' in probs['A scenario expects a wrong ADD result'][1]
    assert 'writes x0, which always stays 0' in probs['A scenario expects a wrong ADDI result'][1]
    who, detail = probs['Partly tested: “Instructions that write a register”']
    assert who == 'test' and detail == ('1 of its examples could not be turned into tests (e.g. the '
                                        'instruction uses other registers than the rs1/rs2/rd columns).')
    for title, (_, detail) in probs.items():
        assert 'CPU-' not in title + detail and 'BDD:' not in detail
    assert s['overview']['checked'][0].startswith('19 instructions were simulated')
    st = {r['req_key']: (r['status'], r['subject']) for r in s['requirements']}
    assert st == {'REQ-1': ('fail', None), 'REQ-2': ('fail', None), 'REQ-3': ('verified', None),
                  'REQ-4': ('fail', 'taken')}
    assert next(r for r in s['requirements'] if r['req_key'] == 'REQ-3')['unchecked'] == 0
    assert s['repairs']['scenario_values'] == 3 and s['repairs']['design_failures'] == 2


def test_cpu_local_explanations(cpu_run, db):
    from src.xdb.explain import explain_check, summarize_run
    proj, paths, _ = cpu_run
    run = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)['sim_run_id']
    sub = explain_check('CPU-ARITHMETIC-001/r2', sim_run_id=run, url=db)['plain']
    assert sub['title'] == 'sub x3, x1, x2 with x1 = 10, x2 = 30'
    assert sub['lines'][1] == 'Why -20: SUB writes rs1 − rs2 (modulo 2^32) to rd; here 10 − 30 = -20.'
    slt = explain_check('CPU-COMPARE-001/r1', sim_run_id=run, url=db)
    assert slt['structure']['verdict'] == 'design_wrong'
    assert slt['plain']['lines'][0] == ('The design is wrong: x3 should be 1, as the scenario expects, '
                                        'but the design gives 0.')
    assert 'The design seems to compare the operands as unsigned numbers.' in slt['plain']['lines']
    br = explain_check('CPU-BRANCH-001/r4', sim_run_id=run, url=db)['plain']['lines']
    assert br[0].startswith('The scenario is wrong, not the design. It says the branch is not taken, but it is taken')
    assert br[1] == 'Why taken: BLT is taken when rs1 < rs2 as signed numbers; here -5 < 3 is true.'
    sw = explain_check('CPU-MEMORY-001/r3', sim_run_id=run, url=db)
    assert sw['structure']['verdict'] == 'pass'               # checked in memory since the template fix
    assert sw['plain']['lines'] == ['Correct: the stored word is 1980, as the scenario expects.']
    assert sw['structure']['stimulus']['tb_reference'] == 1980
    lw = explain_check('CPU-MEMORY-001/r2', sim_run_id=run, url=db)
    assert lw['structure']['verdict'] == 'pass' and lw['structure']['cpu']['issues'] == []


def test_cpu_repair_scenarios_and_design(cpu_run, db, monkeypatch):
    from src import experiment_logger
    proj, paths, _ = cpu_run
    c = _client((proj, paths), db, monkeypatch)
    body = _page_body(paths, run_id='cpu_run', dut_info=dict(CPU_DUT))
    r = c.post('/api/repair/scenarios', json=body).get_json()
    assert r['success'], r
    # the row that gives x1 two values is left alone: a new expected value would hide the contradiction
    assert [(x['operation'], x['old'], x['new']) for x in r['corrections']] == [
        ('SUB', '20', '-20'), ('ADDI', '15', '0'), ('BLT', 'no', 'yes')]
    assert r['corrections'][2]['text'] == 'blt x1, x2, 8 with x1 = -5, x2 = 3: branch taken no → yes'
    body2 = _page_body(paths, run_id='cpu_run', dut_info=dict(CPU_DUT), bdd_filepath=r['feature_path'],
                       testbench_path=r['testbench_path'], log_file=r['simulation']['log_file'])
    assert c.post('/api/run-report', json=body2).get_json()['success']
    s2 = _summary_when_ready(c, body2)
    assert s2['verdicts'] == {'pass': 16, 'scenario_wrong': 1, 'design_wrong': 2, 'not_tested': 1}

    fb = c.post('/api/repair/design-feedback', json=body2).get_json()
    assert fb['success'] and fb['count'] == 2 and fb['operations'] == ['SLT', 'SLTI']
    assert '- slt x3, x1, x2 with x1 = -1, x2 = 1: x3 must be 1, the design gave 0' in fb['text']
    assert '- SLTI: SLTI writes 1 to rd if rs1 < imm as signed numbers, else 0' in fb['text']

    class FakeDesignModel:
        def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
            return '```verilog\n' + CPU_OK + '\n```'
    experiment_logger.instrument_class(FakeDesignModel)
    import main
    monkeypatch.setattr(main.CPUGenerator, '_setup_llm', lambda self: FakeDesignModel())
    hw = c.post('/api/generate-hardware', json={
        'module_type': 'cpu', 'llm': 'deepseek', 'bitwidth': 32, 'pipeline_stages': 5,
        'workflow_mode': 'implementation', 'run_id': 'cpu_run',
        'repair_of': {'testbench_path': r['testbench_path'], 'log_file': r['simulation']['log_file']}}).get_json()
    assert hw['success'], hw
    new_duv = Path(hw['filepath']).relative_to(proj).as_posix()
    rr = c.post('/api/repair/rerun', json=_page_body(paths, run_id='cpu_run', dut_info=dict(CPU_DUT),
                                                     bdd_filepath=r['feature_path'], duv_filepath=new_duv)).get_json()
    assert rr['success'], rr
    body3 = _page_body(paths, run_id='cpu_run', dut_info=dict(CPU_DUT), bdd_filepath=r['feature_path'],
                       duv_filepath=new_duv, testbench_path=rr['testbench_path'],
                       log_file=rr['simulation']['log_file'])
    assert c.post('/api/run-report', json=body3).get_json()['success']
    s3 = _summary_when_ready(c, body3)
    assert s3['verdicts'] == {'pass': 18, 'scenario_wrong': 1, 'not_tested': 1}
    # left: the row that gives x1 two values, and the row the testbench skips
    assert sorted(p['who'] for p in s3['problems']) == ['scenario', 'test']


# ================================================================ compile-error repair (D30)

CPU_NOT_COMPILING = CPU_OK.replace('endmodule', "    wire bad_wire;\n    always @(*) bad_wire = 1'b0;\nendmodule", 1)


class _ScriptedModel:
    """Answers the first design prompt with `first`, a repair prompt with `fixed`;
    keeps every prompt it was given."""
    def __init__(self, first, fixed):
        self.first, self.fixed, self.prompts = first, fixed, []

    def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None):
        self.prompts.append(prompt)
        code = self.fixed if 'Repair of a previous version' in prompt else self.first
        return '```verilog\n' + code + '\n```'


def _duv_when_stored(db, path, tries=40):
    import hashlib
    import time
    from src import xdb
    from src.xdb import models as M
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for _ in range(tries):
        with xdb.session_scope(db) as s:
            d = s.scalar(sqlalchemy.select(M.Duv).where(M.Duv.artifact == sha))
            if d is not None:
                return {'id': d.id, 'compiles': d.compiles, 'reason': d.reason}
        time.sleep(0.25)
    raise AssertionError('the design never reached the database')


def test_compile_fix_sends_the_compiler_errors_back(cpu_run, db, monkeypatch):
    """Step 1 made a design that does not compile: its iverilog errors and its
    code go back to the model; the new version compiles and is stored as
    the result of that feedback."""
    from src import experiment_logger
    from src import xdb
    from src.xdb import models as M
    proj, paths, _ = cpu_run
    c = _client((proj, paths), db, monkeypatch)
    import main
    model = _ScriptedModel(CPU_NOT_COMPILING, CPU_OK)
    experiment_logger.instrument_class(_ScriptedModel)
    monkeypatch.setattr(main.CPUGenerator, '_setup_llm', lambda self: model)
    req = {'module_type': 'cpu', 'llm': 'deepseek', 'bitwidth': 32, 'pipeline_stages': 5,
           'workflow_mode': 'implementation'}
    hw1 = c.post('/api/generate-hardware', json=req).get_json()
    assert hw1['success'] and hw1['duv_compile']['ok'] is False, hw1
    broken = Path(hw1['filepath']).relative_to(proj).as_posix()
    first = _duv_when_stored(db, hw1['filepath'])
    assert first['compiles'] is False

    hw2 = c.post('/api/generate-hardware', json={**req, 'run_id': hw1['run_id'],
                                                  'compile_fix_of': {'duv_filepath': broken}}).get_json()
    assert hw2['success'], hw2
    assert hw2['duv_compile']['ok'] is True and hw2['repair'] is None
    cf = hw2['compile_fix']
    assert cf['kind'] == 'compile' and cf['errors'] >= 1 and cf['feedback_id']
    assert any(ln.startswith('design line ') and 'l-value' in ln for ln in cf['error_lines']), cf
    prompt = model.prompts[-1]
    assert 'does not compile with Icarus Verilog' in prompt
    assert 'iverilog -g2012 reports:' in prompt and 'bad_wire' in prompt      # errors + previous code
    assert str(proj) not in prompt                                         # no server paths
    second = _duv_when_stored(db, hw2['filepath'])
    assert second['compiles'] is True and second['reason'] == f"feedback:{cf['feedback_id']}"
    with xdb.session_scope(db) as s:
        fb = s.get(M.Feedback, cf['feedback_id'])
        assert (fb.subject_type, fb.subject_id, fb.status) == ('duv', first['id'], 'applied')
        assert fb.proposed['action'] == 'fix_compile_error' and fb.proposed['kind'] == 'compile'
        assert fb.result_ref == f"duv:{second['id']}"

    # nothing left to fix
    r = c.post('/api/generate-hardware', json={**req, 'compile_fix_of': {
        'duv_filepath': Path(hw2['filepath']).relative_to(proj).as_posix()}})
    assert r.status_code == 400 and r.get_json()['error'] == 'the design compiles: nothing to fix'


def test_compile_fix_of_ports_the_testbench_cannot_connect(cpu_run, db, monkeypatch):
    """The design compiles alone, but its ports are named differently from the
    ones the testbench connects: the model is given the testbench's interface."""
    from src import experiment_logger
    proj, paths, _ = cpu_run
    c = _client((proj, paths), db, monkeypatch)
    import main
    renamed = proj / 'output/dut/cpu_renamed.v'
    renamed.write_text(CPU_OK.replace('rst_n', 'reset_n'), encoding='utf-8')
    model = _ScriptedModel(CPU_OK, CPU_OK)
    experiment_logger.instrument_class(_ScriptedModel)
    monkeypatch.setattr(main.CPUGenerator, '_setup_llm', lambda self: model)
    hw = c.post('/api/generate-hardware', json={
        'module_type': 'cpu', 'llm': 'deepseek', 'bitwidth': 32, 'pipeline_stages': 5,
        'workflow_mode': 'implementation', 'run_id': 'cpu_run',
        'compile_fix_of': {'duv_filepath': 'output/dut/cpu_renamed.v',
                           'testbench_path': paths['testbench']}}).get_json()
    assert hw['success'], hw
    assert hw['compile_fix']['kind'] == 'interface'
    assert hw['compile_fix']['feedback_id'] is None             # this file never came from Step 1
    assert any('rst_n' in ln for ln in hw['compile_fix']['error_lines']), hw['compile_fix']
    prompt = model.prompts[-1]
    assert 'the testbench cannot connect to it' in prompt
    assert 'The testbench instantiates the design as module `riscv_cpu` with the ports clk, rst_n,' in prompt

    # a design whose ports match: nothing to fix
    r = c.post('/api/generate-hardware', json={
        'module_type': 'cpu', 'llm': 'deepseek', 'compile_fix_of': {
            'duv_filepath': paths['duv'], 'testbench_path': paths['testbench']}})
    assert r.status_code == 400 and 'nothing to fix' in r.get_json()['error']
    # paths outside the project are refused
    r = c.post('/api/generate-hardware', json={
        'module_type': 'cpu', 'llm': 'deepseek', 'compile_fix_of': {'duv_filepath': '../../etc/passwd'}})
    assert r.status_code == 400


def test_ingestion_retries_a_deadlock_victim():
    """Two background ingestions of one run can deadlock; PostgreSQL aborts one
    of them, which runs again. Other database errors are not retried."""
    from sqlalchemy.exc import OperationalError
    from src.xdb.ingest import _retry

    class _PgError(Exception):
        def __init__(self, code):
            self.pgcode = code

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OperationalError('INSERT', {}, _PgError('40P01'))
        return 'stored'
    assert _retry(flaky) == 'stored' and len(calls) == 3

    calls.clear()

    def down():
        calls.append(1)
        raise OperationalError('connect', {}, _PgError('08006'))   # connection failure
    with pytest.raises(OperationalError):
        _retry(down)
    assert len(calls) == 1


# ================================================================ trial runner (R1)

def test_trial_runner_dry_run(rf_run, db, monkeypatch, tmp_path):
    """The batch runner drives the web flow end to end with the mock model:
    Step 0-4, explanations, then both repairs, and reports what it saw."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    from src.simulation_runner import WebSimulationRunner
    from src.xdb import trial as TR
    from src import xdb
    from src.xdb import models as M
    proj = tmp_path / 'proj'
    (proj / 'output').mkdir(parents=True)
    monkeypatch.setattr(main, 'PROJECT_ROOT', proj)
    monkeypatch.setattr(main, 'simulation_runner', WebSimulationRunner(project_root=str(proj)))
    monkeypatch.setenv('DATABASE_URL', db)
    TR.install_mock()
    t = TR.Trial('pytest', step0=True, repair_rounds=2, out_root=tmp_path / 'trials',
                 app=main.app, project_root=proj)
    rec = t.run_one('mock', 'regfile', 1)
    assert rec['stage'] == 'done' and rec['error'] is None, rec.get('error')
    assert rec['step0']['ok'] and rec['step0']['requirements'] == 6     # the standard list, no model
    v = rec['initial']['verdicts']
    assert (v['pass'], v['design_wrong'], v['scenario_wrong']) == (7, 1, 1)
    assert rec['initial']['steps_unmapped'] == 0 and rec['initial']['steps_mapped'] > 20
    assert [(r['action'], r['ok']) for r in rec['rounds']] == [('scenarios', True), ('design', True)]
    assert rec['final']['all_clear'] and rec['final']['verdicts']['pass'] == 9
    assert rec['llm_calls'] >= 3                              # design, scenarios, repair
    summary = TR.write_outputs(t.out, [rec])
    assert '| mock | regfile | 1 | 1 | 9 | 78% | 11% | 11% | 0% | 100% | 0 | 0/1 | 1/1 | 1 | — |' in summary
    assert (t.out / 'runs.csv').read_text(encoding='utf-8').startswith('llm,design,rep,stage')
    with xdb.session_scope(db) as s:
        assert s.get(M.Session, rec['initial']['session_id']).note == 'trial:pytest'


@pytest.mark.parametrize('fault', ['compile', 'interface'])
def test_trial_runner_fixes_compile_errors(rf_run, db, monkeypatch, tmp_path, fault):
    """--mock-fault: the first design does not compile (or names its ports
    differently from the testbench); the runner sends the errors back once and
    goes on with the fixed design."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    from src.simulation_runner import WebSimulationRunner
    from src.xdb import trial as TR
    proj = tmp_path / 'proj'
    (proj / 'output').mkdir(parents=True)
    monkeypatch.setattr(main, 'PROJECT_ROOT', proj)
    monkeypatch.setattr(main, 'simulation_runner', WebSimulationRunner(project_root=str(proj)))
    monkeypatch.setenv('DATABASE_URL', db)
    TR.install_mock()
    monkeypatch.setattr(TR.MockLLM, 'first_fault', fault)
    monkeypatch.setattr(TR.MockLLM, '_faulted', set())
    t = TR.Trial(f'pytest_{fault}', step0=False, repair_rounds=1, out_root=tmp_path / 'trials',
                 app=main.app, project_root=proj)
    rec = t.run_one('mock', 'regfile', 1)
    assert rec['stage'] == 'done' and rec['error'] is None, rec.get('error')
    fixes = [r for r in rec['rounds'] if r['action'] in ('compile', 'interface')]
    assert [(r['action'], r['ok']) for r in fixes] == [(fault, True)]
    assert fixes[0]['errors_sent'] >= 1
    flat = TR._flat(rec)
    assert (flat['compile_fixes'], flat['compile_fixes_ok']) == (1, 1)
    assert rec['initial']['verdicts']['pass'] > 0          # the fixed design was simulated


# ================================================================ steps the testbench does not do (D29)
# Phrasings from the first real trial (data/trials/first): every register-file run
# blamed on the scenario had such a step; the testbench ran it as a plain write/read.

TRIAL_STEPS = {
    'I attempt to write <data> to register <addr> with write enable deasserted': ('RF.A.WRITE_DISABLED', False),
    'I disable write and attempt to write <data> to register <addr>': ('RF.A.WRITE_DISABLED', False),
    'I simultaneously write <data> to register <addr> and read register <addr>': ('RF.A.SAME_CYCLE', False),
    'I write <data> to register <addr> while reading register <addr> on port 1': ('RF.A.SAME_CYCLE', False),
    'I read register <addr> on port 2': ('RF.A.READ_PORT2', False),
    'register <addr> holds value <initial_data>': ('RF.P.PRELOAD', False),
    'reset is active': ('RF.P.IN_RESET', False),
    'I read register <addr> on read port A': ('RF.A.READ', True),
    'I write <data> to register <addr> with write enable asserted on the rising clock edge': ('RF.A.WRITE', True),
    'register 0 is hardwired to zero (REQ-7, REQ-8)': ('RF.P.STATE', True),
    'the register file holds values from earlier writes': ('RF.P.STATE', True),
    'the value in register <addr> should remain <expected>': ('RF.Q.RDATA', True),
    'the value should still be <expected>': ('RF.Q.RDATA', True),
}


def test_trial_phrasings_are_classified():
    from src.xdb import library as L
    for text, (key, executable) in TRIAL_STEPS.items():
        step, _ = L.map_step('regfile', text)
        assert (step['key'], step.get('executable', True)) == (key, executable), text
    assert L.map_step('alu', 'I perform the <Operation> operation with opcode <Opcode>')[0]['key'] == 'ALU.A.OP'
    assert L.map_step('cpu', 'memory at that address should contain <expected>')[0]['key'] == 'CPU.Q.MEM'


RF_SAME_CYCLE = RF_FEATURE + """
  @write_read @req_1
  Scenario Outline: Read during write returns the old value
    Given the register file is ready
    When I simultaneously write <data> to register <addr> and read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | data | expected |
      | 12   | 99   | 7        |
"""


def test_scenario_the_testbench_cannot_run_decides_nothing(rf_run, db):
    """R12 holds 7; the scenario reads it in the same cycle as writing 99 and
    expects the old value 7, which is right. The testbench writes first and
    reads 99. Without D29 that was 'scenario wrong' and "Correct the
    scenarios" rewrote the right value 7 to 99."""
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    from src.xdb.explain import explain_check, summarize_run
    from src.xdb import repair as XR
    from src import xdb
    proj, _, _ = rf_run
    out = proj / 'output'
    (out / 'bdd/rf_same_cycle.feature').write_text(RF_SAME_CYCLE, encoding='utf-8')
    (out / 'dut/regfile_ok.v').write_text(RF_FIXED, encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/rf_same_cycle.feature'), {**RF_DUT, 'dut_filepath': str(out / 'dut/regfile_ok.v')},
        output_filename='rf_same_cycle_tb.v')
    _simulate(proj, tb['filepath'], out / 'dut/regfile_ok.v', 'output/simulation/rf_same_cycle.log')
    paths = {'requirements': 'output/requirements/req.json', 'bdd': 'output/bdd/rf_same_cycle.feature',
             'duv': 'output/dut/regfile_ok.v', 'testbench': Path(tb['filepath']).relative_to(proj).as_posix(),
             'log': 'output/simulation/rf_same_cycle.log'}
    res = ingest_run(proj, paths, run_id='rf_same_cycle', url=db, data={'bitwidth': 32})
    # the fixed design passes everything else; the stale R1 read is the scenario's real mistake
    assert res['verdicts'] == {'pass': 8, 'scenario_wrong': 1, 'undecided': 1}
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    blocked = next(p for p in s['problems'] if p['who'] == 'test')
    assert blocked['title'] == 'The testbench cannot run one example as the scenario describes it'
    assert ('“Read during write returns the old value”: the scenario says “When I simultaneously write <data> to '
            'register <addr> and read register <addr>”, but the testbench reads only after a write has taken effect, '
            'never in the same clock cycle.') in blocked['detail']
    e = explain_check(blocked['explain']['tag'], sim_run_id=s['sim_run_id'], url=db)
    assert e['plain']['lines'][0].startswith('This example cannot be checked as written: the scenario says')
    with xdb.session_scope(db) as ss:
        fixes = XR.scenario_corrections(ss, s['sim_run_id'])
    assert [(c['scenario_key'], c['old'], c['new']) for c in fixes] == [('RF-READ-001', '100', '300')]


# Real scenarios from one trial (box_after3, 2026-11-06). The first passed on a reset
# register, because its tag left the write out. The second is a plain write and
# read, which the model also linked to the same-cycle requirement.
RF_SAME_CYCLE_CLAIMS = RF_FEATURE + """
  @read @req_5
  Scenario Outline: Read of register written in same cycle returns old value
    Given the register file is ready
    When I write <data> to register <addr>
    And I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | data | expected |
      | 20   | 99   | 0        |
      | 21   | 50   | 0        |

  @write_read @req_1 @req_5
  Scenario Outline: Write and immediately read back
    Given the register file is ready
    When I write <data> to register <addr>
    And I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | data | expected |
      | 22   | 42   | 42       |
"""
SAME_CYCLE_REQ = 'A read of the register being written in the same cycle returns the old value.'


@pytest.fixture(scope='module')
def rf_claims(rf_run, db):
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.explain import summarize_run
    from src.xdb.ingest import ingest_run
    proj, _, _ = rf_run
    out = proj / 'output'
    record = {**RF_RECORD, 'requirements': RF_RECORD['requirements'] + [
        {'id': 'REQ-5', 'text': SAME_CYCLE_REQ, 'source': 'template'}]}
    (out / 'requirements/claims.json').write_text(json.dumps(record), encoding='utf-8')
    (out / 'bdd/rf_claims.feature').write_text(RF_SAME_CYCLE_CLAIMS, encoding='utf-8')
    (out / 'dut/regfile_ok.v').write_text(RF_FIXED, encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/rf_claims.feature'), {**RF_DUT, 'dut_filepath': str(out / 'dut/regfile_ok.v')},
        output_filename='rf_claims_tb.v')
    _simulate(proj, tb['filepath'], out / 'dut/regfile_ok.v', 'output/simulation/rf_claims.log')
    paths = {'requirements': 'output/requirements/claims.json', 'bdd': 'output/bdd/rf_claims.feature',
             'duv': 'output/dut/regfile_ok.v', 'testbench': Path(tb['filepath']).relative_to(proj).as_posix(),
             'log': 'output/simulation/rf_claims.log'}
    res = ingest_run(proj, paths, run_id='rf_claims', url=db, data={'bitwidth': 32})
    return res, summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)


def test_a_write_the_tag_leaves_out_decides_nothing(rf_claims):
    """Both rows of the @read scenario read a register nobody wrote, so they passed."""
    res, s = rf_claims
    assert res['verdicts']['undecided'] == 2
    blocked = next(p for p in s['problems'] if p['title'].startswith('The testbench cannot run'))
    assert ("“When I write <data> to register <addr>”, but the scenario's tag makes the testbench "
            "leave this step out.") in blocked['detail']


def test_a_passing_scenario_does_not_verify_what_the_testbench_cannot_show(rf_claims, db):
    from src.xdb.explain import explain_requirement
    _, s = rf_claims
    by_key = {r['req_key']: r for r in s['requirements']}
    assert by_key['REQ-5']['status'] == 'not_tested' and by_key['REQ-5']['evidence'] is None
    assert by_key['REQ-1']['status'] != 'not_tested'       # the same passing row still counts for this one
    card = next(p for p in s['problems'] if p['title'] == 'The testbench cannot test a requirement')
    assert card['detail'] == (f"“{SAME_CYCLE_REQ.rstrip('.')}”: the testbench reads only after a write has "
                              f"taken effect, never in the same clock cycle.")
    lines = explain_requirement(by_key['REQ-5']['requirement_id'], url=db)['plain']['lines']
    assert lines[0] == ('Not tested: the testbench reads only after a write has taken effect, never in the '
                        'same clock cycle, so no scenario can show this.')


def test_only_the_same_cycle_requirement_is_out_of_reach():
    from src import design_templates as T
    from src.xdb import library as L
    rf = T._regfile_requirements(32, 32, 5)
    assert [t for t in rf if L.cannot_test('regfile', t)] == [SAME_CYCLE_REQ]
    assert L.cannot_test('regfile', 'Concurrent read of the register being written returns old value')
    # the same words on another design: a counter's zero flag can be checked in its cycle
    assert not any(L.cannot_test('counter', t) for t in T._counter_requirements(32, 32, 5))
    # D50: an ALU scenario can assert reset now; only a combinational ALU has none to test
    reset = 'On reset (rst = 1), result, zero, overflow and negative are 0 after the next rising clock edge.'
    assert not any(L.cannot_test('alu', t) for t in T._alu_requirements(32, 32, 5))
    assert L.cannot_test('alu_comb', reset) == 'the design has no reset input'
    assert L.not_run('RF.A.WRITE', {'READ'}) == L.TAG_SKIPS_STEP
    assert L.not_run('RF.A.WRITE', {'WRITE_READ'}) is None and L.not_run('RF.A.WRITE', set()) is None


# ================================================================ reading a card (D37)
# The fixed example of the explanation standard: a real run (simulation #106,
# 2026-11-05), a 16-bit ALU from qwen3-coder-next. The design has no $signed at
# all, so SLT compares unsigned and SRA fills with zeros, and one scenario swaps
# SLT and SLTU. The numbers were always right; these tests hold how they read.

def _ingest_alu(db, tmp_path_factory, folder, module, width, run_id, edit=None, edit_design=None):
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    proj = tmp_path_factory.mktemp(folder)
    out = proj / 'output'
    for d in ('bdd', 'dut', 'simulation'):
        (out / d).mkdir(parents=True)
    src = ROOT / 'tests/data' / folder
    feature = (src / 'alu.feature').read_text(encoding='utf-8')
    (out / 'bdd/alu.feature').write_text(edit(feature) if edit else feature, encoding='utf-8')
    verilog = (src / 'alu.v').read_text(encoding='utf-8')
    (out / 'dut/alu.v').write_text(edit_design(verilog) if edit_design else verilog, encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/alu.feature'), {'module_type': 'alu', 'module_name': module, 'bitwidth': width})
    assert tb['success'], tb
    _simulate(proj, tb['filepath'], out / 'dut/alu.v', 'output/simulation/alu.log')
    paths = {'bdd': 'output/bdd/alu.feature', 'duv': 'output/dut/alu.v',
             'testbench': Path(tb['filepath']).relative_to(proj).as_posix(), 'log': 'output/simulation/alu.log'}
    if (src / 'requirements.json').exists():      # a run with Step 0 answered
        (out / 'requirements').mkdir()
        (out / 'requirements/req.json').write_text((src / 'requirements.json').read_text(encoding='utf-8'),
                                                   encoding='utf-8')
        paths['requirements'] = 'output/requirements/req.json'
    res = ingest_run(proj, paths, run_id=run_id, url=db, data={'module_type': 'alu', 'bitwidth': width})
    assert res['ingested'], res
    from src.xdb.explain import summarize_run
    return proj, paths, res, summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)


@pytest.fixture(scope='module')
def unsigned_run(db, tmp_path_factory):
    return _ingest_alu(db, tmp_path_factory, 'unsigned_alu', 'alu_16bit', 16, 'unsigned_run')


# The second fixed example: simulation #109, the same fault at 8 bits, with scenarios
# whose results are all right. Checked against every input: SLT and SRA are its only bugs.
@pytest.fixture(scope='module')
def unsigned_run_8(db, tmp_path_factory):
    return _ingest_alu(db, tmp_path_factory, 'unsigned_alu_8', 'alu_8bit', 8, 'unsigned_run_8')


def test_an_8_bit_run_reads_like_a_16_bit_run(unsigned_run_8):
    res, s = unsigned_run_8[2], unsigned_run_8[3]
    assert res['verdicts'] == {'pass': 45, 'design_wrong': 5}
    probs = s['problems']
    assert [p['title'] for p in probs] == ['The design compares SLT operands as unsigned numbers',
                                           'The design shifts SRA with zeros instead of copying the sign bit',
                                           'The scenarios mix two meanings of overflow']
    assert s['overview']['counts'] == {'design': 2, 'scenario': 1, 'test': 0, 'coverage': 0, 'limit': 0}


def test_each_design_card_names_its_own_consequence(unsigned_run_8):
    slt, sra = unsigned_run_8[3]['problems'][:2]
    assert slt['detail'].endswith('The zero flag of these examples is wrong too, as a consequence.')
    assert sra['detail'].endswith('The negative flag of these examples is wrong too, as a consequence.')


def test_one_overflow_card_says_what_it_relies_on(unsigned_run_8):
    card = unsigned_run_8[3]['problems'][2]
    assert card['who'] == 'scenario' and card['explain'] == {'kind': 'global', 'port': 'overflow'}
    # 127 + 1 states overflow and only signed overflow gives that; the other three follow carry/borrow
    assert card['detail'].startswith('Where the two meanings differ, 3 examples, in ADD and SUB, follow unsigned '
                                     'carry/borrow and 1 example follows signed overflow.')
    assert ("Overflow is judged as signed overflow, the design template's definition, because Step 0 did not "
            "decide it. The design is right.") in card['detail']


def test_scenarios_that_keep_to_one_other_meaning_are_called_that(db, tmp_path_factory):
    fixed = lambda text: re.sub(r'(\| *127 *\| *1 *\|[^\n]*?\| *)True', r'\1False', text, count=1)
    s = _ingest_alu(db, tmp_path_factory, 'unsigned_alu_8', 'alu_8bit', 8, 'one_meaning', edit=fixed)[3]
    card = next(p for p in s['problems'] if 'overflow' in p['title'])
    assert card['title'] == 'The scenarios treat overflow as unsigned carry/borrow'
    assert card['detail'].startswith('All 4 wrong overflow flags, in ADD and SUB, fit that meaning.')


def test_a_signed_result_is_shown_as_signed_too(unsigned_run_8, db):
    from src.xdb.explain import explain_check
    sra = unsigned_run_8[3]['problems'][1]['detail']
    assert 'SRA with A = 0xFF (-1 as signed), B = 1: the design gives 127, correct is 0xFF (-1 as signed).' in sra
    x = explain_check('ALU-SRA-001/r3', sim_run_id=unsigned_run_8[3]['sim_run_id'], url=db)
    assert x['plain']['lines'][0] == ('The design is wrong. It gives result = 127; the correct value is '
                                      '0xFF (-1 as signed), as the scenario expects.')
    assert x['plain']['lines'][1].startswith('Why 0xFF (-1 as signed): ')
    y = explain_check('ALU-SLT-001/r3', sim_run_id=unsigned_run_8[3]['sim_run_id'], url=db)
    assert '(a = 0xFF (-1 as signed), b = 0xFF (-1 as signed)) passed' in y['text']


def test_details_list_every_flag_the_probe_saw(unsigned_run_8, db):
    """The scenario states only zero, the card says negative is wrong: Details must say so too."""
    from src.xdb.explain import explain_check
    run_id = unsigned_run_8[3]['sim_run_id']
    x = explain_check('ALU-SRA-001/r3', sim_run_id=run_id, url=db)
    flags = {f['port']: f for f in x['structure']['probed']}
    assert list(flags) == ['zero', 'negative']                  # the rule defines no overflow for SRA
    assert (flags['negative']['stated'], flags['negative']['observed'], flags['negative']['rule_value'],
            flags['negative']['verdict']) == (None, 0, 1, 'design_wrong')
    assert x['plain']['lines'][-3:] == [
        "The design's negative flag is 0, but it should be 1 — a consequence of the wrong result.",
        'The zero flag is correct.',
        'The scenario does not state the negative flag.']
    assert 'negative = 0 ✗ design wrong (scenario not stated, rule 1)' in x['text']
    ok = explain_check('ALU-SRA-001/r1', sim_run_id=run_id, url=db)
    assert ok['plain']['lines'][-2:] == ['The zero and negative flags are correct.',
                                         'The scenario does not state the negative flag.']


# The third fixed example: simulation #111, 32 bits, Step 0 answered. Its shifter has no stage
# for bit 4 of B. A separate simulation showed that SRA has the fault too; its examples cannot.
@pytest.fixture(scope='module')
def shift_run(db, tmp_path_factory):
    return _ingest_alu(db, tmp_path_factory, 'shift_alu', 'alu_32bit', 32, 'shift_run')


def test_the_verdicts_of_the_shift_run_stay(shift_run):
    assert shift_run[2]['verdicts'] == {'pass': 42, 'design_wrong': 6, 'scenario_wrong': 2}


def test_shifts_that_lose_the_same_bit_are_one_card(shift_run):
    probs = shift_run[3]['problems']
    assert [p['title'] for p in probs] == ['The design ignores bit 4 of the shift amount',
                                           'The design compares SLT operands as unsigned numbers',
                                           '2 scenario examples expect a wrong AND result',
                                           'The scenarios mix two meanings of overflow',
                                           'A scenario states a wrong negative flag for SUB',
                                           'A requirement is not tested by any scenario']   # reset: no scenario
    assert 'ops' not in probs[0]
    assert probs[0]['detail'] == (
        '2 of 10 SLL and SRL examples give a wrong result. E.g. SLL with A = 1, B = 31: the design gives '
        '32768 (0x00008000), correct is 2147483648 (0x80000000). All 10 SLL and SRL examples fit this, so '
        'bit 4 of B must shift by 16 more positions. The 5 SRA examples pass, but none of them can show this '
        'fault, so SRA may have it too: SRA with A = 0x80000000 (-2147483648 as signed), B = 16 would show it, '
        'correct is 0xFFFF8000 (-32768 as signed), with this fault 0x80000000 (-2147483648 as signed). '
        'The negative flag of these examples is wrong too, as a consequence.')


def test_details_of_a_lost_bit_name_the_cause(shift_run, db):
    from src.xdb.explain import explain_check
    x = explain_check('ALU-SRL-001/r3', sim_run_id=shift_run[3]['sim_run_id'], url=db)
    assert x['structure']['cause']['bit'] == 4
    assert ('Cause: the design ignores bit 4 of the shift amount. That reproduces all 5 SRL examples, so '
            'bit 4 of B must shift by 16 more positions.') in x['plain']['lines']
    assert any(ln.endswith('passed: the fault does not show on these values.') for ln in x['plain']['lines'])


def test_mixed_overflow_meanings_are_not_called_one_meaning(shift_run, db):
    from src.xdb.explain import explain_global
    s = shift_run[3]
    card = next(p for p in s['problems'] if 'overflow' in p['title'])
    assert card['detail'].startswith('Where the two meanings differ, 2 examples, in ADD and SUB, follow unsigned '
                                     'carry/borrow and 4 examples follow signed overflow. E.g. ADD with A = 4294967295')
    assert 'Overflow is judged as signed overflow, as decided in Step 0. The design is right.' in card['detail']
    g = explain_global(s['feature_id'], 'overflow', 'True', url=db)
    alt = g['structure']['rule_check']['alternative']
    assert (alt['follows_other'], alt['follows_rule']) == (2, 4)
    assert any(ln.startswith('The scenarios mix two meanings of “overflow”: where they differ, 2 examples')
               for ln in g['plain']['lines'])
    assert 'so the specification mixes both.' in g['text']


def test_a_checked_flag_requirement_is_not_called_untested(shift_run, db):
    from src.xdb.explain import explain_requirement
    s = shift_run[3]
    st = {r['req_key']: r for r in s['requirements']}
    assert st['REQ-11']['status'] == 'not_covered'     # reset: nothing in the run touched it
    r12 = st['REQ-12']
    assert (r12['status'], r12['probe_rows'], r12['probe_right']) == ('checked_unlinked', 50, 46)
    assert [p['detail'] for p in s['problems'] if p['who'] == 'coverage'] == [f"“{st['REQ-11']['text'].rstrip('.')}”"]
    x = explain_requirement(r12['requirement_id'], sim_run_id=s['sim_run_id'], url=db)
    assert x['plain']['lines'][0] == ('No scenario is linked to it, but the extra simulation checked the zero flag '
                                      'in all 50 examples: 46 right, 4 wrong.')


def test_the_model_call_of_step_0_reaches_the_database(db, tmp_path_factory):
    from sqlalchemy import create_engine, text
    from src import experiment_logger
    record = json.loads((ROOT / 'tests/data/shift_alu/requirements.json').read_text(encoding='utf-8'))
    conn = sqlite3.connect(experiment_logger.DB_PATH)
    conn.execute("INSERT INTO llm_calls (created_at, provider, model, task_type, run_id, prompt, response, success, "
                 "extra) VALUES ('2026-11-05T18:33:08Z', 'QwenProvider3', 'qwen3-coder-next', "
                 "'web_requirements_analysis', ?, 'PROMPT', 'REPLY', 1, '{\"prompt_template_version\": \"v2\"}')",
                 (record['analysis_run_id'],))
    conn.commit()
    conn.close()
    _ingest_alu(db, tmp_path_factory, 'shift_alu', 'alu_32bit', 32, 'step0_call')     # Step 0 has another run id
    with create_engine(db).connect() as c:
        call, version = c.execute(text("SELECT l.id, l.prompt_version FROM llm_call l JOIN session s "
                                       "ON s.id = l.session_id WHERE l.task_type = 'web_requirements_analysis' "
                                       "AND s.run_id = 'step0_call'")).one()
        refs = c.execute(text("SELECT DISTINCT r.prov_ref FROM requirement r JOIN session s ON s.id = r.session_id "
                              "WHERE s.run_id = 'step0_call'")).scalars().all()
    assert refs == [f'llm_call:{call}'] and version == 'v2'


def test_a_wrong_flag_on_a_wrong_scenario_row_stays_on_its_card(shift_run):
    card = next(p for p in shift_run[3]['problems'] if 'wrong AND result' in p['title'])
    assert card['detail'].endswith('The design is right. In 1 of them the negative flag is stated wrongly too.')


@pytest.fixture(scope='module')
def flag_run(db, tmp_path_factory):
    """The shift run with two planted flag faults: the scenario states a wrong negative flag on an
    SLT row the design fails, and the design gets the zero flag wrong on AND rows it computes right."""
    def scenario(text):
        new, n = re.subn(r'(\| -1 +\| 0 +\| 1000 +\| 1 +\| False +\| )False', r'\1True', text)
        assert n == 1
        return new

    def design(text):
        old = 'zero <= (temp_result == 32\'b0);'
        assert text.count(old) == 1, text.count(old)
        return text.replace(old, "zero <= (temp_result == 32'b0) ^ (opcode == 4'b0010 && a[1]);")

    return _ingest_alu(db, tmp_path_factory, 'shift_alu', 'alu_32bit', 32, 'flag_run', edit=scenario,
                       edit_design=design)[3]


def test_a_scenario_flag_is_not_the_design_s_consequence(flag_run):
    titles = [p['title'] for p in flag_run['problems']]
    assert 'A scenario states a wrong negative flag for SLT' in titles
    card = next(p for p in flag_run['problems'] if p['title'] == 'A scenario states a wrong negative flag for SLT')
    assert card['detail'].endswith("The design's negative flag is right, though its result is wrong.")
    slt = next(p for p in flag_run['problems'] if p['title'].startswith('The design compares SLT'))
    assert 'negative flag' not in slt['detail']         # the design's negative flag is right
    assert 'The zero flag of these examples is wrong too' in slt['detail']


def test_a_design_flag_is_not_lost_on_a_row_the_scenario_got_wrong(flag_run):
    card = next(p for p in flag_run['problems'] if p['title'] == "The design's zero flag is wrong for AND")
    assert card['who'] == 'design' and card['severity'] == 'error'


def test_a_lost_bit_needs_one_bit_that_fits_every_row():
    from src.xdb import library as L
    assert L.alu_mixup('SLL', [(1, 31, 1 << 15), (1, 1, 2)], 32)['bit'] == 4
    assert L.alu_mixup('SRA', [(0x80000000, 16, 0x80000000)], 32)['bit'] == 4
    assert L.alu_mixup('SLL', [(1, 31, 1 << 15), (1, 16, 1 << 16)], 32) is None     # bit 4 works here
    assert L.alu_mixup('SLL', [(0, 31, 1)], 32) is None                              # no bit explains it
    assert L.alu_mixup('SLL', [(1, 3, 1)], 32) is None                               # bit 0 or bit 1: unknown
    assert L.alu_mixup('SLL', [(1, 31, 1 << 31)], 32) is None                        # nothing is wrong
    assert L.alu_shift_hidden('SRA', [(0xFFFFFFFF, 31, 0xFFFFFFFF)], 4, 32)
    assert not L.alu_shift_hidden('SRA', [(0x80000000, 16, 0xFFFF8000)], 4, 32)
    assert L.alu_shift_witness('SRL', 4, 32) == (0x80000000, 16, 0x8000, 0x80000000)


def test_a_run_has_the_same_verdicts_as_before_the_wording_changed(unsigned_run):
    assert unsigned_run[2]['verdicts'] == {'pass': 44, 'design_wrong': 4, 'consistent_wrong': 5, 'scenario_wrong': 3}


def test_one_card_per_root_cause_with_the_true_count(unsigned_run):
    s = unsigned_run[3]
    design = [p for p in s['problems'] if p['who'] == 'design']
    assert [p['title'] for p in design] == ['The design compares SLT operands as unsigned numbers',
                                            'The design shifts SRA with zeros instead of copying the sign bit']
    assert s['overview']['counts']['design'] == 2
    slt = design[0]['detail']
    assert slt.startswith('7 of 8 SLT examples give a wrong result. 5 of them pass the test, because the '
                          'scenario makes the same mistake.')
    assert 'All 8 SLT examples fit this, so the comparison needs $signed on both operands.' in slt
    assert not any('repeats a mistake' in p['title'] for p in s['problems'])


def test_a_scenario_mistake_the_design_repeats_gets_its_own_card(unsigned_run):
    s = unsigned_run[3]
    card = next(p for p in s['problems'] if p['title'] == '2 scenario examples expect a wrong SLT result')
    assert card['who'] == 'scenario'
    assert card['detail'] == ('E.g. SLT with A = 32767 (0x7FFF), B = 0x8000 (-32768 as signed): the scenario expects 1, '
                              'correct is 0. The design gives the same wrong value, so the test passes.')
    assert card['explain']['tag'] == 'BDD:ALU-SLT-001/r4'
    assert s['overview']['counts'] == {'design': 2, 'scenario': 5, 'test': 0, 'coverage': 0, 'limit': 0}


def test_a_scenario_card_keeps_saying_who_is_right():
    from src.xdb.explain import _with_repeats
    text = 'E.g. x: the scenario expects 1, correct is 0. The design is right.'
    assert _with_repeats(text, 2, 0) == text
    assert _with_repeats(text, 0, 2).endswith('The design gives the same wrong value, so the test passes.')
    assert _with_repeats(text, 2, 1).endswith('The design is right. In 1 of them the design gives the same wrong '
                                              'value, so the test passes.')


def test_a_signed_operand_is_shown_as_signed(unsigned_run):
    slt = unsigned_run[3]['problems'][0]['detail']
    assert 'E.g. SLT with A = 0xFFFF (-1 as signed), B = 0: the design gives 0, correct is 1.' in slt
    sra = unsigned_run[3]['problems'][1]['detail']
    assert 'SRA with A = 0x8000 (-32768 as signed), B = 15' in sra


def test_a_swapped_scenario_is_named_once(unsigned_run):
    probs = {p['title']: p for p in unsigned_run[3]['problems']}
    card = probs['A scenario swaps SLT and SLTU']
    assert card['who'] == 'scenario' and 'all 6 examples' in card['detail']
    assert 'E.g. SLTU with A = 65535 (0xFFFF), B = 1: the scenario expects 1, correct is 0.' in card['detail']
    assert not any('wrong SLTU result' in t for t in probs)


def test_the_details_show_the_working_and_the_cause(unsigned_run, db):
    from src.xdb.explain import explain_check
    x = explain_check('ALU-SLT-001/r2', sim_run_id=unsigned_run[3]['sim_run_id'], url=db)
    assert x['plain']['title'] == 'SLT with A = 0xFFFF (-1 as signed), B = 0'
    assert x['plain']['lines'][:4] == [
        'The design is wrong. It gives result = 0; the correct value is 1, as the scenario expects.',
        'Why 1: SLT is 1 when A < B as signed numbers, else 0; here -1 < 0 is true.',
        'Cause: the design compares SLT operands as unsigned numbers. That reproduces all 8 SLT examples, '
        'so the comparison needs $signed on both operands.',
        'For comparison, SLT with A = 0, B = 0 passed: the fault does not show on these values.']
    assert '-1 <ₛ 0 = 1' in x['text'] and 'a = 0xFFFF (-1 as signed)' in x['text']


def test_a_cause_needs_every_example(unsigned_run):
    from src.xdb import library as L
    rows = [(0, 0, 0), (65535, 0, 0), (0, 65535, 1)]
    assert L.alu_mixup('SLT', rows, 16)['rows'] == 3
    assert L.alu_mixup('SLT', rows + [(5, 3, 1)], 16) is None      # one row no mix-up explains: cause unknown
    assert L.alu_mixup('SLT', [(1, 2, 1)], 16) is None             # nothing is wrong


def test_the_same_files_stored_without_their_scenarios_do_not_replace_the_run(unsigned_run, db):
    """Asking for a report with only the testbench and the log stores a second,
    thin record of the same files. The page must keep reading the full one."""
    from src.xdb import models as M, repair
    from src import xdb
    from src.xdb.explain import summarize_run
    from src.xdb.ingest import ingest_run
    proj, paths, _, full = unsigned_run
    thin = ingest_run(proj, {'testbench': paths['testbench'], 'log': paths['log']}, url=db, data={})
    assert thin['ingested'] and thin['sim_run_id'] != full['sim_run_id']
    q = {'testbench_path': paths['testbench'], 'log_file': paths['log']}
    assert summarize_run(proj, q, url=db)['sim_run_id'] == full['sim_run_id']
    with xdb.session_scope(db) as s:
        assert repair.find_run(s, proj, q)[0].id == full['sim_run_id']


# ================================================================ local fixes and chat (D46)

def test_a_card_shows_its_lines_and_a_local_fix_is_checked_kept_and_recorded(run, db):
    """The sample ALU's SLTU compares signed (line 23). The fix may touch line 23 only."""
    from src.xdb import fix as XF
    from src.xdb.ingest import ingest_run
    from src.xdb.explain import summarize_run
    from src import xdb
    from src.xdb import models as M
    proj, paths = run
    ingest_run(proj, paths, run_id='web_test_run', url=db, data={'bitwidth': 32})
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    card = next(p for p in s['problems'] if p.get('fix') and 'design' in p['fix']['kinds'])
    assert card['fix']['ops'] == ['SLTU'] and [ln['no'] for ln in card['fix']['lines']] == [23]

    prompts = []

    def model(system, prompt):
        prompts.append(prompt)
        return '{"changes": [{"line": 23, "new": "            4\'b1001: result = (a < b) ? 1 : 0;"}], "why": "unsigned"}'
    out, err = XF.propose(s['sim_run_id'], [{'ops': ['SLTU']}], model, note='keep the style', url=db)
    assert err is None and out['accepted'], out
    assert 'Lines you may change: 23.' in prompts[0] and 'The engineer adds: keep the style' in prompts[0]
    assert 'SLTU with A =' in prompts[0]                                  # the failing examples
    assert out['summary']['scenario_right_after'] > out['summary']['scenario_right_before']
    assert '-            4\'b1001: result = ($signed(a)' in out['diff']

    kept, err = XF.keep(proj, out['feedback_id'], paths['duv'], url=db)
    assert err is None and kept['design_filename'] == 'alu_32bit_fix1.v'
    assert (proj / kept['design_path']).read_text(encoding='utf-8').count('(a < b) ? 1 : 0') == 1
    assert XF.keep(proj, out['feedback_id'], paths['duv'], url=db)[1] == 'this fix is already applied'
    with xdb.session_scope(db) as ss:
        fb = ss.get(M.Feedback, out['feedback_id'])
        assert fb.status == 'applied' and fb.proposed['human_note'] and fb.proposed['groups'][0]['lines'] == [23]

    # a model that answers with a special case for one example: rejected, design unchanged, recorded
    out, _ = XF.propose(s['sim_run_id'], [{'ops': ['SLTU']}],
                        lambda sy, p: '{"changes": [{"line": 23, "new": "            4\'b1001: result = '
                                      '(a == 1 && b == 32\'hFFFFFFFF) ? 1 : ($signed(a) < $signed(b)) ? 1 : 0;"}]}',
                        url=db)
    assert not out['accepted'] and out['diff'] == ''
    with xdb.session_scope(db) as ss:
        assert ss.get(M.Feedback, out['feedback_id']).status == 'rejected'


def test_the_chat_under_a_card_is_grounded_stored_and_can_propose_a_change(run, db):
    from src.xdb import fix as XF
    from src.xdb.explain import summarize_run
    from src import xdb
    from src.xdb import models as M
    proj, paths = run
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    card = next(p for p in s['problems'] if p.get('fix') and 'design' in p['fix']['kinds'])
    seen = []

    def model(system, prompt):
        seen.append(prompt)
        return ('Line 23 compares $signed(a) < $signed(b), so SLTU behaves like SLT.\n'
                '```json\n{"changes": [{"line": 23, "new": "            4\'b1001: result = (a < b) ? 1 : 0;"}], '
                '"why": "compare unsigned"}\n```')
    first, err = XF.chat(s['sim_run_id'], {'title': card['title'], 'detail': card['detail'],
                                           'ops': card['fix']['ops']}, 'Why is SLTU wrong?', model, url=db)
    assert err is None and first['answer'].startswith('Line 23') and '```' not in first['answer']
    assert first['proposal']['changes'][0]['line'] == 23
    assert '  23 | ' in seen[0] and 'SLTU with A =' in seen[0]       # the facts it may use
    second, _ = XF.chat(s['sim_run_id'], {'title': card['title'], 'ops': ['SLTU']}, 'And how do I fix it?',
                        model, thread=first['thread'], url=db)
    assert 'Engineer: Why is SLTU wrong?' in seen[1]                  # the earlier turn is in the prompt
    with xdb.session_scope(db) as ss:
        turns = [t.q_t[0]['turn'] for t in ss.scalars(
            __import__('sqlalchemy').select(M.ExplanationTarget).where(M.ExplanationTarget.mode == 'chat'))
            if t.q_t and t.q_t[0].get('thread') == first['thread']]
    assert turns == [1, 2]
    # the proposal from the chat is tested like any other fix, without a model call
    out, _ = XF.propose(s['sim_run_id'], [{'ops': ['SLTU']}], None, proposal=first['proposal'], url=db)
    assert out['accepted']


def test_step3_again_keeps_the_earlier_testbench_so_its_run_still_opens(run, db):
    """D47: the second testbench for the same feature is numbered; the first one stays,
    so the simulation made with it is still found from its files."""
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    from src.xdb.explain import summarize_run
    proj, paths = run
    ingest_run(proj, paths, run_id='web_test_run', url=db, data={'bitwidth': 32})
    first = (proj / paths['testbench']).read_bytes()
    again = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(proj / paths['bdd']), {'module_type': 'alu', 'module_name': 'alu_32bit', 'bitwidth': 32})
    import re as _re
    assert again['success'] and _re.search(r'_tb_\d+\.v$', Path(again['filepath']).name)
    assert Path(again['filepath']).name != Path(paths['testbench']).name
    assert Path(again['filepath']).with_name(Path(again['filepath']).stem + '.scenarios.json').exists()
    assert (proj / paths['testbench']).read_bytes() == first
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    assert s is not None and s['sim_run_id']
    Path(again['filepath']).unlink()          # leave the module's project as the other tests expect


# ================================================================ reset in the ALU testbench (D50)

RESET_FEATURE = """Feature: 32-bit ALU Verification

  @add @req_1
  Scenario Outline: Verify ADD operation
    Given I have operand A = <A>
    And I have operand B = <B>
    When I perform the ADD operation with opcode 0000
    Then the result should be <Expected_Result>

    Examples:
      | A | B | Opcode | Expected_Result |
      | 1 | 2 | 0000   | 3               |

  @reset @req_2
  Scenario Outline: Reset clears the outputs
    Given I have operand A = <A>
    And I have operand B = <B>
    When reset is asserted
    Then the result should be <Expected_Result>
    And the zero flag should be <Zero_Flag>
    And the negative flag should be <Negative_Flag>
    And the overflow flag should be <Overflow_Flag>

    Examples:
      | A   | B  | Opcode | Expected_Result | Zero_Flag | Negative_Flag | Overflow_Flag |
      | 100 | 27 | 0000   | 0               | False     | False         | False         |
      | -1  | 0  | 0000   | 0               | False     | False         | False         |
"""


QWEN_RESET = RESET_FEATURE.split('  @reset')[0] + '''  @reset @req_2
  Scenario Outline: Reset clears the outputs
    Given I have operand A = <A>
    And I have operand B = <B>
    When reset is asserted
    Then the result should be 0
    And the zero flag should be False

    Examples:
      | A   | B  | Opcode |
      | 100 | 27 | 0000   |
      | -1  | 0  | 0000   |
'''


NO_VALUE_RESET = QWEN_RESET.replace('Then the result should be 0', 'Then the outputs are cleared')


@pytest.mark.parametrize('broken, feature', [(False, RESET_FEATURE), (True, RESET_FEATURE), (False, QWEN_RESET),
                                             (False, NO_VALUE_RESET)])
def test_a_reset_scenario_is_run_and_judged(db, tmp_path, broken, feature):
    """The real design of session 109 after its fixes clears every output on reset.
    Broken: its reset leaves the result alone, which the reset rows must show."""
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    from src.xdb.explain import summarize_run
    design = (ROOT / 'tests/data/qwen_alu_32/alu_fixed.v').read_text(encoding='utf-8')
    if broken:
        design = design.replace("            result <= 32'b0;\n", "            result <= temp_result;\n", 1)
    proj = tmp_path / 'proj'
    out = proj / 'output'
    for d in ('bdd', 'dut', 'requirements', 'simulation'):
        (out / d).mkdir(parents=True)
    (out / 'bdd/alu.feature').write_text(feature, encoding='utf-8')   # QWEN_RESET: values in the steps, as qwen wrote them
    (out / 'dut/alu_32bit.v').write_text(design, encoding='utf-8')
    record = {'module_type': 'alu', 'created_at': '2026-11-10T10:00:00', 'questions': [], 'decisions': [],
              'requirements': [{'id': 'REQ-1', 'text': 'ADD (opcode 0000): result = A + B.', 'source': 'template'},
                               {'id': 'REQ-2', 'text': 'On reset (rst = 1), result, zero, overflow and negative '
                                                       'are 0 after the next rising clock edge.', 'source': 'template'}]}
    (out / 'requirements/req.json').write_text(json.dumps(record), encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/alu.feature'), {'module_type': 'alu', 'module_name': 'alu_32bit', 'bitwidth': 32,
                                       'dut_filepath': str(out / 'dut/alu_32bit.v')})
    assert tb['success'] and tb['test_count'] == (1 if feature is NO_VALUE_RESET else 3)
    text = Path(tb['filepath']).read_text(encoding='utf-8')
    assert ('@(posedge clk);' in text) == (feature is not NO_VALUE_RESET)
    log = _simulate(proj, tb['filepath'], out / 'dut/alu_32bit.v', 'output/simulation/alu.log')
    assert ('FAILED' in log or 'FAIL -' in log) == broken
    paths = {'requirements': 'output/requirements/req.json', 'bdd': 'output/bdd/alu.feature',
             'duv': 'output/dut/alu_32bit.v', 'testbench': Path(tb['filepath']).relative_to(proj).as_posix(),
             'log': 'output/simulation/alu.log'}
    res = ingest_run(proj, paths, run_id=f'reset_{broken}_{id(feature)}', url=db, data={'bitwidth': 32})
    s = summarize_run(proj, {'testbench_path': paths['testbench'], 'log_file': paths['log']}, url=db)
    req = {r['req_key']: r for r in s['requirements']}
    if feature is NO_VALUE_RESET:
        # the reset rows state no value, so no check ran: the requirement is not verified (D50)
        assert req['REQ-2']['status'] == 'not_tested'
    elif broken:
        assert res['verdicts'].get('design_wrong') == 2
        assert s['problems'][0]['title'] == 'The design does not clear its result on reset'
        assert req['REQ-2']['status'] == 'fail'
    else:
        assert res['verdicts'] == {'pass': 3}
        assert s['overview']['ok'] and req['REQ-2']['status'] in ('verified', 'verified_by_probe')
        assert not [p for p in s['problems'] if p['who'] != 'limit']
