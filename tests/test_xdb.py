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
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get('XDB_TEST_URL', 'postgresql://bdx:bdx@localhost/bdx_test')

sqlalchemy = pytest.importorskip('sqlalchemy')

ALU = """
module alu_32bit #(parameter WIDTH = 32) (
    input  wire clk, input wire rst,
    input  wire [WIDTH-1:0] a, b,
    input  wire [3:0] opcode,
    output reg  [WIDTH-1:0] result,
    output wire zero, output reg overflow, output wire negative
);
    always @(*) begin
        overflow = 1'b0;
        case (opcode)
            4'b0000: begin result = a + b;
                     overflow = (a[WIDTH-1] == b[WIDTH-1]) && (result[WIDTH-1] != a[WIDTH-1]); end
            4'b0001: begin result = a - b;
                     overflow = (a[WIDTH-1] != b[WIDTH-1]) && (result[WIDTH-1] != a[WIDTH-1]); end
            4'b0010: result = a & b;
            4'b0011: result = a | b;
            4'b0100: result = a ^ b;
            4'b0101: result = a << b[4:0];
            4'b0110: result = a >> b[4:0];
            4'b0111: result = $signed(a) >>> b[4:0];
            4'b1000: result = ($signed(a) < $signed(b)) ? 1 : 0;
            4'b1001: result = ($signed(a) < $signed(b)) ? 1 : 0;   // BUG: SLTU compares signed
            default: result = 0;
        endcase
    end
    assign zero = (result == 0);
    assign negative = result[WIDTH-1];
endmodule
"""


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
                .replace('@sltu', '@sltu @req_2', 1))
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
    assert counts['check_result'] == sum(len(s['rows']) for s in sr['scenarios']) + len(sr['generated'])

    from sqlalchemy import create_engine, text
    with create_engine(db).connect() as c:
        fails = c.execute(text("SELECT count(*) FROM check_result WHERE status='fail'")).scalar()
        passes = c.execute(text("SELECT count(*) FROM check_result WHERE status='pass'")).scalar()
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
                                 WHERE ci.tag = :t ORDER BY cr.sim_run_id DESC LIMIT 1"""),
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
        rows = c.execute(text("SELECT status, verdict, count(*) n FROM check_result "
                              "WHERE verdict IS NOT NULL GROUP BY status, verdict")).all()
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
