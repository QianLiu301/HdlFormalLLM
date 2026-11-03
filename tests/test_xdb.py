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
