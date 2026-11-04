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
        # zero + negative for 67 rows, overflow only for the 14 ADD/SUB rows
        assert tuple(r) == (1, 'probe', 'ok', 67 * 2 + 14)
        # the testbench's own pass rate is untouched by the probe
        n = c.execute(text("SELECT count(*) FROM check_result WHERE sim_run_id = 1")).scalar()
        assert n == 67


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
    assert s['verdicts'] == {'pass': 62, 'design_wrong': 3, 'scenario_wrong': 2}
    assert s['checks']['BDD:ALU-SUB-001/r7']['verdict'] == 'scenario_wrong'
    assert any('zero flag' in x and 'contradicts' in x for x in s['spec_check_text'])
    # P4/P5: what was compared, and what the flag probe saw
    assert s['compared_ports'] == ['result']
    assert s['probe']['status'] == 'ok'
    assert s['probe']['per_port']['zero'] == {'rows': 67, 'design_agrees': 64}
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
    assert titles[0] == 'The design computes SLTU wrong'                  # most severe first
    assert 'A scenario expects a wrong SUB result' in titles
    assert 'A scenario expects a wrong SRA result' in titles
    assert 'A scenario states a wrong zero flag for ADD' in titles
    assert 'A requirement is not tested by any scenario' in titles         # REQ-3
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


FIXED_ALU = ALU.replace("4'b1001: result = ($signed(a) < $signed(b)) ? 1 : 0;   // BUG: SLTU compares signed",
                        "4'b1001: result = (a < b) ? 1 : 0;")


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
    assert (rp['scenario_values'], rp['design_failures'], rp['design_regeneration']) == (3, 3, True)
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
    assert s2['verdicts'] == {'pass': 64, 'design_wrong': 3}
    assert s2['probe']['issues'].get('BDD:ALU-ADD-001/r7') is None
    assert s2['repairs']['scenario_values'] == 0 and s2['repairs']['design_failures'] == 3
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
    assert fb['success'] and fb['count'] == 3 and fb['operations'] == ['SLTU']
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
    assert hw['repair']['failures'] == 3
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
    assert s2['verdicts'] == {'pass': 65, 'scenario_wrong': 2}
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

RF_FEATURE = """Feature: 32x32-bit Register File Verification
  As a hardware verification engineer
  I want to verify the Register File implementation
  So that I can ensure correct read/write behavior

  Background:
    Given the Register File has 32 registers of 32-bit width

  @write @req_1
  Scenario Outline: Write values into registers
    Given the register file is ready
    When I write <data> to register <addr>
    Then the write is accepted

    Examples:
      | addr | data       |
      | 1    | 100        |
      | 2    | 255        |
      | 5    | 4294967295 |
      | 1    | 300        |

  @read @req_1
  Scenario Outline: Read back values written earlier
    Given the register file holds earlier writes
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected   |
      | 1    | 100        |
      | 2    | 255        |
      | 5    | 4294967295 |
      | 9    | 0          |

  @write_read @req_1
  Scenario Outline: Write and immediately read back
    Given the register file is ready
    When I write <data> to register <addr>
    And I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | data | expected |
      | 7    | 42   | 42       |
      | 12   | 7    | 7        |

  @write @req_2
  Scenario Outline: Writes to register 0 are ignored
    Given the register file is ready
    When I write <data> to register <addr>
    Then the write is discarded

    Examples:
      | addr | data |
      | 0    | 123  |

  @read @req_2
  Scenario Outline: Register 0 always reads as zero
    Given register 0 is hardwired
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected |
      | 0    | 0        |

  @reset @req_3
  Scenario Outline: Reset clears every register
    Given registers hold values
    When reset is asserted
    Then register <addr> is cleared

    Examples:
      | addr |
      | 1    |

  @read @req_3
  Scenario Outline: Everything reads zero after reset
    Given reset has completed
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected |
      | 1    | 0        |
      | 7    | 0        |
"""

RF_BUGGY = """module regfile #(parameter WIDTH = 32, parameter DEPTH = 32, parameter ADDR_WIDTH = 5) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [ADDR_WIDTH-1:0] raddr1,
    output wire [WIDTH-1:0]      rdata1,
    input  wire [ADDR_WIDTH-1:0] raddr2,
    output wire [WIDTH-1:0]      rdata2,
    input  wire                  wen,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [WIDTH-1:0]      wdata
);
    reg [WIDTH-1:0] registers [0:DEPTH-1];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) registers[i] <= 0;
        end else if (wen) begin
            registers[waddr] <= wdata;          // BUG: register 0 is not protected
        end
    end
    assign rdata1 = registers[raddr1];          // BUG: register 0 is not hardwired
    assign rdata2 = (raddr2 == 0) ? 0 : registers[raddr2];
endmodule
"""

RF_FIXED = (RF_BUGGY.replace("registers[waddr] <= wdata;          // BUG: register 0 is not protected",
                             "if (waddr != 0) registers[waddr] <= wdata;")
            .replace("assign rdata1 = registers[raddr1];          // BUG: register 0 is not hardwired",
                     "assign rdata1 = (raddr1 == 0) ? 0 : registers[raddr1];"))

RF_RECORD = {'module_type': 'regfile', 'created_at': '2026-11-03T12:00:00', 'questions': [], 'decisions': [],
             'requirements': [
                 {'id': 'REQ-1', 'text': 'A read returns the value last written to that register.', 'source': 'llm'},
                 {'id': 'REQ-2', 'text': 'Register 0 always reads as zero; writes to it are ignored.', 'source': 'llm'},
                 {'id': 'REQ-3', 'text': 'Reset clears every register.', 'source': 'llm'},
                 {'id': 'REQ-4', 'text': 'Both read ports can read any register at the same time.', 'source': 'llm'}]}
RF_DUT = {'module_type': 'regfile', 'module_name': 'regfile', 'bitwidth': 32, 'depth': 32}


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

CPU_FEATURE = """Feature: RISC-V CPU Verification (5-stage pipeline)
  As a hardware verification engineer
  I want to verify the RISC-V CPU implementation
  So that I can ensure correct instruction execution

  Background:
    Given the CPU is a 5-stage pipeline RV32I processor
    And all registers are initialized to 0

  @arithmetic @req_1
  Scenario Outline: Instructions that write a register
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction      | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | add x3, x1, x2   | 1   | 10        | 2   | 20        | 3  | 30       |
      | sub x3, x1, x2   | 1   | 10        | 2   | 30        | 3  | 20       |
      | addi x3, x1, 100 | 1   | 50        | 0   | 0         | 3  | 150      |
      | add x3, x1, x1   | 1   | 5         | 1   | 7         | 3  | 12       |
      | addi x0, x1, 5   | 1   | 10        | 0   | 0         | 0  | 15       |
      | add x5, x1, x2   | 1   | 10        | 2   | 20        | 3  | 30       |

  @logical @req_1
  Scenario Outline: Bitwise instructions
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction     | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | and x3, x1, x2  | 1   | 0xFF      | 2   | 0x0F      | 3  | 0x0F     |
      | or x3, x1, x2   | 1   | 0xF0      | 2   | 0x0F      | 3  | 0xFF     |
      | xori x3, x1, 15 | 1   | 0xFF      | 0   | 0         | 3  | 0xF0     |

  @compare @req_2
  Scenario Outline: Signed comparison
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction     | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | slt x3, x1, x2  | 1   | -1        | 2   | 1         | 3  | 1        |
      | slt x3, x1, x2  | 1   | 5         | 2   | 3         | 3  | 0        |
      | slti x3, x1, 0  | 1   | -1        | 0   | 0         | 3  | 1        |

  @memory @req_3
  Scenario Outline: Load and store, one instruction per row
    Given register x<rs1> contains address <rs1_value>
    And memory at that address contains <mem_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction    | rs1 | rs1_value | rs2 | rs2_value | rd | mem_value | expected |
      | lw x3, 0(x1)   | 1   | 0x100     | 0   | 0         | 3  | 0x1234    | 0x1234   |
      | lw x4, 4(x1)   | 1   | 0x180     | 0   | 0         | 4  | 0x55      | 0x55     |
      | sw x2, 0(x1)   | 1   | 0x140     | 2   | 0x7BC     | 0  | 0         | 0x7BC    |

  @branch @req_4
  Scenario Outline: Branch taken or not taken
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then the branch is taken should be <taken>

    Examples:
      | instruction    | rs1 | rs1_value | rs2 | rs2_value | taken |
      | beq x1, x2, 8  | 1   | 10        | 2   | 10        | yes   |
      | beq x1, x2, 8  | 1   | 10        | 2   | 20        | no    |
      | bne x1, x2, 8  | 1   | 10        | 2   | 20        | yes   |
      | blt x1, x2, 8  | 1   | -5        | 2   | 3         | no    |
      | bge x1, x2, 8  | 1   | -5        | 2   | 3         | no    |
"""

CPU_OK = """// Minimal RV32I subset CPU for the explainability tests: one instruction per
// cycle, a load takes two (the testbench's data memory reads synchronously).
module riscv_cpu (
    input  wire        clk,
    input  wire        rst_n,
    output wire [31:0] imem_addr,
    input  wire [31:0] imem_data,
    output wire [31:0] dmem_addr,
    output wire [31:0] dmem_wdata,
    output wire        dmem_wen,
    output wire [3:0]  dmem_be,
    input  wire [31:0] dmem_rdata,
    output wire [31:0] debug_pc,
    output wire [31:0] debug_inst
);
    reg [31:0] pc;
    reg [31:0] x [0:31];
    reg        load_wait;
    integer k;

    wire [31:0] inst = imem_data;
    wire [6:0]  opc  = inst[6:0];
    wire [4:0]  rd   = inst[11:7];
    wire [2:0]  f3   = inst[14:12];
    wire [4:0]  rs1  = inst[19:15];
    wire [4:0]  rs2  = inst[24:20];
    wire [6:0]  f7   = inst[31:25];
    wire [31:0] a    = (rs1 == 0) ? 32'd0 : x[rs1];
    wire [31:0] b    = (rs2 == 0) ? 32'd0 : x[rs2];
    wire [31:0] imm_i = {{20{inst[31]}}, inst[31:20]};
    wire [31:0] imm_s = {{20{inst[31]}}, inst[31:25], inst[11:7]};
    wire [31:0] imm_b = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0};

    wire is_r = opc == 7'b0110011, is_i = opc == 7'b0010011, is_lw = opc == 7'b0000011,
         is_sw = opc == 7'b0100011, is_br = opc == 7'b1100011;
    wire [31:0] op2 = is_r ? b : imm_i;

    reg [31:0] alu;
    always @(*) begin
        case (f3)
            3'b000: alu = (is_r && f7[5]) ? a - op2 : a + op2;
            3'b010: alu = ($signed(a) < $signed(op2)) ? 32'd1 : 32'd0;     // SLT / SLTI
            3'b100: alu = a ^ op2;
            3'b110: alu = a | op2;
            3'b111: alu = a & op2;
            default: alu = 32'd0;
        endcase
    end

    reg taken;
    always @(*) begin
        case (f3)
            3'b000: taken = a == b;
            3'b001: taken = a != b;
            3'b100: taken = $signed(a) < $signed(b);
            3'b101: taken = $signed(a) >= $signed(b);
            default: taken = 1'b0;
        endcase
    end

    assign imem_addr  = pc;
    assign dmem_addr  = is_sw ? a + imm_s : a + imm_i;
    assign dmem_wdata = b;
    assign dmem_wen   = is_sw & rst_n;
    assign dmem_be    = 4'b1111;
    assign debug_pc   = pc;
    assign debug_inst = inst;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; load_wait <= 0;
            for (k = 0; k < 32; k = k + 1) x[k] <= 0;
        end else if (is_lw && !load_wait) begin
            load_wait <= 1;                                   // address out, data next cycle
        end else begin
            load_wait <= 0;
            if ((is_r || is_i) && rd != 0) x[rd] <= alu;
            if (is_lw && rd != 0) x[rd] <= dmem_rdata;
            pc <= (is_br && taken) ? pc + imm_b : pc + 4;
        end
    end
endmodule
"""

CPU_BUGGY = CPU_OK.replace("3'b010: alu = ($signed(a) < $signed(op2)) ? 32'd1 : 32'd0;     // SLT / SLTI",
                           "3'b010: alu = (a < op2) ? 32'd1 : 32'd0;     // BUG: SLT compares unsigned")

CPU_RECORD = {'module_type': 'cpu', 'created_at': '2026-11-03T14:00:00', 'questions': [], 'decisions': [],
              'requirements': [
                  {'id': 'REQ-1', 'text': 'Arithmetic and logical instructions write their result to rd.',
                   'source': 'llm'},
                  {'id': 'REQ-2', 'text': 'SLT and SLTI compare signed values.', 'source': 'llm'},
                  {'id': 'REQ-3', 'text': 'LW loads a word from memory; SW stores a word to memory.',
                   'source': 'llm'},
                  {'id': 'REQ-4', 'text': 'Branches are taken exactly when their condition holds.',
                   'source': 'llm'}]}
CPU_DUT = {'module_type': 'cpu', 'module_name': 'riscv_cpu', 'bitwidth': 32, 'pipeline_stages': 5}


@pytest.fixture(scope='module')
def cpu_run(db, tmp_path_factory):
    """The CPU sample, simulated on the CPU whose SLT compares unsigned (run id cpu_run)."""
    from src.testbench_generator import TestbenchGenerator
    from src.xdb.ingest import ingest_run
    proj = tmp_path_factory.mktemp('cpuproj')
    out = proj / 'output'
    for d in ('bdd', 'dut', 'requirements', 'simulation'):
        (out / d).mkdir(parents=True)
    (out / 'bdd/cpu.feature').write_text(CPU_FEATURE, encoding='utf-8')
    (out / 'dut/cpu.v').write_text(CPU_BUGGY, encoding='utf-8')
    (out / 'requirements/req.json').write_text(json.dumps(CPU_RECORD), encoding='utf-8')
    tb = TestbenchGenerator(project_root=str(proj), debug=False).generate_single(
        str(out / 'bdd/cpu.feature'), dict(CPU_DUT))
    assert tb['success'], tb
    _simulate(proj, tb['filepath'], out / 'dut/cpu.v', 'output/simulation/cpu.log')
    paths = {'requirements': 'output/requirements/req.json', 'bdd': 'output/bdd/cpu.feature',
             'duv': 'output/dut/cpu.v', 'testbench': Path(tb['filepath']).relative_to(proj).as_posix(),
             'log': 'output/simulation/cpu.log'}
    res = ingest_run(proj, paths, run_id='cpu_run', url=db, data={'bitwidth': 32})
    assert res['ingested'], res
    return proj, paths, res


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
