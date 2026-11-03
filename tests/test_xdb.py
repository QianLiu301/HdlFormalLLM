"""
End-to-end test of the explainability database (src/xdb).

Real pipeline, no LLM: a feature file the website once generated
(artifact/ablation/variant_A_original.feature, with @req_N tags added) ->
TestbenchGenerator -> iverilog on an ALU with one injected bug -> ingestion
-> competency queries. The database must agree with run_report, which is
what the Step-4 page shows.

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
              'questions': [], 'decisions': []}
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
    assert r['expected']['value'] != r['observed']['value']
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
