"""
Batch trial runner: the website's flow, without the browser, many times.

For every (model, design type, repetition) it does what a user does on the
page: Step 0 (requirements, defaults accepted), Step 1 (design), Step 2
(scenarios), Step 3 (testbench), Step 4 (simulation), then reads the
explanation summary and applies the repairs the Problems card offers
(correct the scenarios, regenerate the design) for a few rounds. Everything
goes through the same Flask endpoints the page calls and is stored in the
explainability database like any web run; each run's session is marked
`trial:<tag>`.

    python -m src.xdb.trial run --llms deepseek,gemini --designs alu,regfile,cpu --reps 2
    python -m src.xdb.trial run --llms mock --reps 1          # dry run, no API keys
    python -m src.xdb.trial summary --tag <tag>

Output in output/trials/<tag>/: runs.jsonl (one full record per run),
runs.csv (one flat row per run), summary.md (per model x design type), and
logs/ (the server output of each run). Needs DATABASE_URL and iverilog.

The point is to find what does not work yet with real model output: steps
the rule library cannot read, checks nobody can decide, explanations that
miss, repairs that do not converge.
"""

import argparse
import contextlib
import csv
import io
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]

DESIGNS = ('alu', 'regfile', 'cpu')
BITWIDTH = 32
DEPTH = 32
STAGES = 5

# What the page pre-fills for Step 0 and Step 2 (static/bdd_generator.html,
# BDD_TEMPLATES), so a trial asks the models what a user who accepts the
# defaults would ask.
DESCRIPTIONS = {
    'alu': lambda bw: f"""{bw}-bit ALU with 4-bit opcode selecting:
- ADD  (opcode 0000): A + B
- SUB  (opcode 0001): A - B
- AND  (opcode 0010): A & B
- OR   (opcode 0011): A | B
- XOR  (opcode 0100): A ^ B
- SLL  (opcode 0101): A << B, shift amount is the low {bw.bit_length() - 1} bits of B
- SRL  (opcode 0110): A >> B, zero-filled
- SRA  (opcode 0111): A >>> B, sign-extended
- SLT  (opcode 1000): 1 if A < B as signed values, else 0
- SLTU (opcode 1001): 1 if A < B as unsigned values, else 0
- Zero flag: high when result is 0, for every operation
- Overflow flag: signed overflow, meaningful for ADD and SUB
- Boundary value tests (0, 1, max, min, all-ones)
- Contrast SRL against SRA on a negative operand
- Contrast SLT against SLTU on the same operand pair""",
    'regfile': lambda bw: f"""{DEPTH}x{bw}-bit Register File with:
- Dual read ports
- Single write port
- x0 always returns 0
- Concurrent read/write test""",
    'cpu': lambda bw: f"""{STAGES}-stage RISC-V CPU (RV32I subset) with:
- R-type: ADD, SUB, AND, OR, XOR, SLT
- I-type: ADDI, ANDI, ORI, XORI, SLTI, LW
- S-type: SW
- B-type: BEQ, BNE, BLT, BGE
- J-type: JAL, JALR
- Forwarding: EX/MEM and MEM/WB to EX
- Load-use hazard: stall one cycle after LW
- Control hazard: flush the pipeline on a taken branch or jump
- Signed comparison: contrast SLT against unsigned interpretation
- x0 stays zero on any write""",
}

VERDICTS = ('pass', 'design_wrong', 'scenario_wrong', 'both_wrong', 'consistent_wrong', 'undecided')


# ----------------------------------------------------------------- mock model

class MockLLM:
    """Answers every prompt of the flow with the canned samples
    (src/xdb/samples.py): a design with a known bug, scenarios with known
    mistakes, the fixed design when asked for a repair. For dry runs only.

    first_fault = 'compile' | 'interface' makes the first design of each type
    not compile, or use other port names, so the compile fixes run too."""
    model = 'mock'
    first_fault: Optional[str] = None
    _faulted: set = set()

    @classmethod
    def _fault(cls, kind: str, design: str) -> str:
        if not cls.first_fault or kind in cls._faulted:
            return design
        cls._faulted.add(kind)
        if cls.first_fault == 'compile':
            # a wire driven from an always block: Icarus rejects it
            # ('not a valid l-value'), the error class seen in real runs
            return design.replace('endmodule', "    wire bad_wire;\n    always @(*) bad_wire = 1'b0;\nendmodule", 1)
        # other port names: the testbench cannot connect
        return design.replace('wen', 'we_n').replace('rst_n', 'reset_n').replace('opcode', 'op')

    @staticmethod
    def _kind(prompt: str) -> str:
        # only the CPU prompts name RV32I; the register-file prompt also says RISC-V (x0)
        if 'RV32I' in prompt:
            return 'cpu'
        if 'Register File' in prompt:
            return 'regfile'
        return 'alu'

    def _call_api(self, prompt, max_tokens=None, system_prompt=None, sampling=None, **kw):
        from src.xdb import samples as S
        kind = self._kind(prompt)
        if 'Repair of a previous version' in prompt:
            return '```verilog\n' + {'alu': S.FIXED_ALU, 'regfile': S.RF_FIXED, 'cpu': S.CPU_OK}[kind] + '\n```'
        if 'answer with JSON only' in (system_prompt or '') or '"questions"' in prompt:
            rec = {'alu': S.ALU_RECORD, 'regfile': S.RF_RECORD, 'cpu': S.CPU_RECORD}[kind]
            return json.dumps({'requirements': [r['text'] for r in rec['requirements']], 'questions': []})
        if 'Gherkin' in prompt:
            return {'alu': S.alu_feature(), 'regfile': S.RF_FEATURE, 'cpu': S.CPU_FEATURE}[kind]
        return '```verilog\n' + self._fault(kind, {'alu': S.ALU, 'regfile': S.RF_BUGGY, 'cpu': S.CPU_BUGGY}[kind]) \
            + '\n```'


def install_mock():
    """Make LLMFactory.create_provider('mock') return MockLLM (both import paths)."""
    import importlib
    from src import experiment_logger
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))          # main and some generators import 'llm_providers'
    if not getattr(MockLLM._call_api, '_exp_logged', False):
        experiment_logger.instrument_class(MockLLM)   # calls are logged like real ones
    for name in ('llm_providers', 'src.llm_providers'):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        factory = mod.LLMFactory
        orig = factory.__dict__['create_provider'].__func__ \
            if isinstance(factory.__dict__.get('create_provider'), staticmethod) else factory.create_provider
        if getattr(orig, '_trial_mock', False):
            continue

        def create(provider_type, *a, _orig=orig, **kw):
            if str(provider_type).lower() == 'mock':
                return MockLLM()
            return _orig(provider_type, *a, **kw)
        create._trial_mock = True
        factory.create_provider = staticmethod(create)


# ----------------------------------------------------------------- one run

class Trial:
    def __init__(self, tag: str, step0: bool = True, repair_rounds: int = 2, wait: float = 120.0,
                 compile_fix: int = 2,
                 out_root: Optional[Path] = None, app=None, project_root: Optional[Path] = None):
        self.tag = tag
        self.step0 = step0
        self.repair_rounds = repair_rounds
        self.compile_fix = compile_fix
        self.wait = wait
        self.out = (out_root or ROOT / 'output' / 'trials') / tag
        (self.out / 'logs').mkdir(parents=True, exist_ok=True)
        if app is None:
            with contextlib.redirect_stdout(io.StringIO()):
                import main
            app, project_root = main.app, main.PROJECT_ROOT
        self.client = app.test_client()
        self.root = Path(project_root or ROOT)
        self._log = None
        self._files: Optional[List[str]] = None

    # -- plumbing
    def _rel(self, path) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def _post(self, path: str, body: Dict) -> Dict:
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            r = self.client.post(path, json=body)
        if self._log:
            self._log.write(f'\n===== POST {path} ({time.time() - t0:.1f}s, HTTP {r.status_code})\n')
            self._log.write(buf.getvalue())
        try:
            data = r.get_json() or {}
        except Exception:
            data = {}
        data.setdefault('success', r.status_code < 400)
        return data

    def _summary(self, body: Dict) -> Optional[Dict]:
        """Ask for the run report (stores the run), then wait for the summary."""
        if self._files is not None:
            self._files += [body.get(k) for k in ('requirements_filepath', 'bdd_filepath', 'duv_filepath',
                                                  'testbench_path', 'log_file') if body.get(k)]
        rep = self._post('/api/run-report', body)
        if not rep.get('success'):
            return {'error': rep.get('error') or 'run report failed'}
        deadline = time.time() + self.wait
        while time.time() < deadline:
            s = self._post('/api/explain/summary', body)
            if not s.get('enabled'):
                return {'error': 'explainability database is not configured (DATABASE_URL)'}
            if s.get('ready'):
                return s
            time.sleep(1.0)
        return {'error': f'the run was not stored within {self.wait:.0f} s'}

    @staticmethod
    def _facts(s: Dict) -> Dict:
        """The numbers one summary contributes to the trial record."""
        if not s or s.get('error'):
            return {'error': (s or {}).get('error')}
        checks = s.get('checks') or {}
        bdd = [c for tag, c in checks.items() if tag.startswith('BDD:')]     # the scenarios' checks only
        v: Dict[str, int] = {}
        for c in bdd:
            if c.get('verdict'):
                v[c['verdict']] = v.get(c['verdict'], 0) + 1
        probs = s.get('problems') or []
        return {'sim_run_id': s.get('sim_run_id'), 'session_id': s.get('session_id'),
                'feature_id': s.get('feature_id'),
                'verdicts': {k: v.get(k, 0) for k in VERDICTS},
                'not_tested': sum(1 for c in bdd if c.get('status') == 'not_tested'),
                'setup': sum(1 for c in bdd if c.get('status') == 'setup'),
                # checks the testbench adds by itself (e.g. register 0 after a write)
                'generated_failed': sum(1 for tag, c in checks.items()
                                        if tag.startswith('GEN:') and c.get('status') == 'fail'),
                'problems': [{'who': p['who'], 'title': p['title']} for p in probs],
                'problem_counts': (s.get('overview') or {}).get('counts'),
                # clear = something was checked and neither side is wrong (a run without a
                # single check is not clear, it is untested)
                'all_clear': bool(v) and not any(p['who'] in ('design', 'scenario') for p in probs),
                'requirements': {r['req_key']: r['status'] for r in s.get('requirements') or []},
                'repairs': s.get('repairs')}

    def _mapping(self, feature_id: Optional[int]) -> Dict:
        """How many scenario steps the rule library could read (canonical steps)."""
        if not feature_id:
            return {}
        from sqlalchemy import func, select
        from src import xdb
        from src.xdb import models as M
        with xdb.session_scope() as s:
            rows = dict(s.execute(select(M.StepMapping.canonical_step_id.is_(None), func.count())
                                  .join(M.Scenario, M.Scenario.id == M.StepMapping.scenario_id)
                                  .where(M.Scenario.feature_id == feature_id)
                                  .group_by(M.StepMapping.canonical_step_id.is_(None))).all())
            unmapped = [t for (t,) in s.execute(
                select(M.ScenarioStep.surface_text)
                .join(M.StepMapping, (M.StepMapping.scenario_id == M.ScenarioStep.scenario_id)
                      & (M.StepMapping.position == M.ScenarioStep.position))
                .join(M.Scenario, M.Scenario.id == M.ScenarioStep.scenario_id)
                .where(M.Scenario.feature_id == feature_id, M.StepMapping.canonical_step_id.is_(None))
                .distinct().limit(20))]
        return {'steps_mapped': rows.get(False, 0), 'steps_unmapped': rows.get(True, 0),
                'unmapped_examples': unmapped}

    def _mark_session(self, session_id: Optional[int]):
        if not session_id:
            return
        from src import xdb
        from src.xdb import models as M
        with xdb.session_scope() as s:
            sess = s.get(M.Session, session_id)
            if sess is not None and not (sess.note or '').startswith('trial:'):
                sess.note = f'trial:{self.tag}'

    @staticmethod
    def _llm_usage(run_ids: List[str]) -> Dict:
        from src import experiment_logger
        ids = [r for r in dict.fromkeys(run_ids) if r]
        if not ids or not Path(experiment_logger.DB_PATH).exists():
            return {}
        conn = sqlite3.connect(experiment_logger.DB_PATH)
        try:
            q = ','.join('?' * len(ids))
            n, ok, pc, rc, ms = conn.execute(
                f'SELECT COUNT(*), SUM(success), SUM(prompt_chars), SUM(response_chars), SUM(latency_ms) '
                f'FROM llm_calls WHERE run_id IN ({q})', ids).fetchone()
        finally:
            conn.close()
        return {'llm_calls': n or 0, 'llm_calls_ok': ok or 0, 'prompt_chars': pc or 0,
                'response_chars': rc or 0, 'llm_seconds': round((ms or 0) / 1000, 1)}

    # -- the flow
    def run_one(self, llm: str, design: str, rep: int, model: Optional[str] = None) -> Dict:
        name = f'{llm}_{design}_{rep}'
        rec: Dict = {'tag': self.tag, 'llm': llm, 'model': model, 'design': design, 'rep': rep,
                     'started': datetime.now().isoformat(timespec='seconds'), 'stage': 'start',
                     'error': None, 'rounds': []}
        cfg = {'module_type': design, 'bitwidth': BITWIDTH, 'depth': DEPTH, 'pipeline_stages': STAGES}
        desc = DESCRIPTIONS[design](BITWIDTH)
        run_ids: List[str] = []
        t_start = time.time()
        self._files = []
        with open(self.out / 'logs' / f'{name}.log', 'w', encoding='utf-8') as self._log:
            try:
                self._flow(rec, llm, model, design, cfg, desc, run_ids)
            except Exception as e:                       # a broken run must not stop the trial
                import traceback
                rec['error'] = f'{type(e).__name__}: {e}'
                self._log.write(traceback.format_exc())
            finally:
                self._log = None
        rec['seconds'] = round(time.time() - t_start, 1)
        rec['files'] = [f for f in dict.fromkeys(self._files) if f]
        for k in ('duv', 'bdd'):                          # also when the run stopped before simulation
            f = (rec.get(k) or {}).get('file')
            if f and f not in rec['files']:
                rec['files'].append(f)
        self._files = None
        rec.update(self._llm_usage(run_ids))
        return rec

    def _flow(self, rec, llm, model, design, cfg, desc, run_ids):
        common = {'llm': llm, 'model': model, 'workflow_mode': 'implementation', **cfg}
        # Step 0: requirements, every default answer accepted
        req_path = None
        if self.step0:
            rec['stage'] = 'step0'
            a = self._post('/api/requirements/analyze', {**common, 'description': desc})
            if a.get('success'):
                qs = [{**q, 'chosen': q.get('default', 0)} for q in a.get('questions') or []]
                sv = self._post('/api/requirements/save', {
                    **common, 'config': cfg, 'description': desc, 'run_id': a.get('run_id'),
                    'accepted_all_defaults': True,
                    'requirements': [{'text': r['text'], 'source': 'llm'} for r in a.get('requirements') or []],
                    'questions': qs})
                req_path = sv.get('filepath') if sv.get('success') else None
                rec['step0'] = {'ok': bool(req_path), 'requirements': len(a.get('requirements') or []),
                                'questions': len(qs), 'json_repaired': a.get('json_repaired'),
                                'retried': a.get('retried'), 'error': sv.get('error')}
            else:
                rec['step0'] = {'ok': False, 'error': a.get('error')}
            run_ids.append(a.get('run_id'))
        extra = {'requirements_filepath': req_path} if req_path else {}

        # Step 1: design
        rec['stage'] = 'duv'
        hw = self._post('/api/generate-hardware', {**common, **extra, 'input': ''})
        run_id = hw.get('run_id')
        run_ids.append(run_id)
        dc = hw.get('duv_compile') or {}
        rec['duv'] = {'ok': bool(hw.get('success')), 'error': hw.get('error'),
                      'file': self._rel(hw.get('filepath')), 'compiles': dc.get('ok'),
                      'compile_error': dc.get('error'),
                      'model_served': (hw.get('call_meta') or {}).get('model_effective')}
        if not hw.get('success'):
            rec['error'] = f"design: {hw.get('error')}"
            return
        duv = rec['duv']['file']
        # a design that does not compile goes back to the model with the compiler's messages
        k = 0
        while dc.get('ok') is False and k < self.compile_fix:
            k += 1
            fx = self._post('/api/generate-hardware', {**common, **extra, 'input': '', 'run_id': run_id,
                                                       'compile_fix_of': {'duv_filepath': duv}})
            run_ids.append(fx.get('run_id'))
            step = {'round': k, 'action': 'compile', 'ok': bool(fx.get('success')), 'error': fx.get('error'),
                    'errors_sent': (fx.get('compile_fix') or {}).get('errors')}
            if fx.get('success'):
                dc = fx.get('duv_compile') or {}
                duv = self._rel(fx.get('filepath'))
                step.update(ok=dc.get('ok') is not False, compile_error=dc.get('error'))
            rec['rounds'].append(step)
            if not fx.get('success'):
                break
        rec['duv'].update(compile_fixes=k, compiles_in_the_end=dc.get('ok'), final_file=duv)
        if dc.get('ok') is False:
            rec['error'] = 'design does not compile'
            return

        # Step 2: scenarios
        rec['stage'] = 'bdd'
        bd = self._post('/api/generate', {**common, **extra, 'run_id': run_id, 'input': desc})
        run_ids.append(bd.get('run_id'))
        rec['bdd'] = {'ok': bool(bd.get('success')), 'error': bd.get('error'),
                      'file': self._rel(bd.get('filepath'))}
        if not bd.get('success'):
            rec['error'] = f"scenarios: {bd.get('error')}"
            return
        feature = rec['bdd']['file']
        dut_info = {k: cfg[k] for k in ('module_type', 'bitwidth', 'depth', 'pipeline_stages')}

        # Step 3 + 4
        rec['stage'] = 'tb'
        tb = self._post('/api/generate-testbench', {'bdd_filepath': feature, 'dut_filepath': duv,
                                                    'dut_info': dut_info, 'run_id': run_id,
                                                    'workflow_mode': 'implementation'})
        rec['tb'] = {'ok': bool(tb.get('success')), 'error': tb.get('error'),
                     'test_count': tb.get('test_count'), 'parse_stats': tb.get('parse_stats'),
                     'expected_values': {k: v for k, v in (tb.get('expected_values') or {}).items()
                                         if k != 'skipped'}}
        if not tb.get('success'):
            rec['error'] = f"testbench: {tb.get('error')}"
            return
        tb_path = self._rel(tb.get('filepath'))
        rec['stage'] = 'sim'
        sim = self._post('/api/run-simulation', {'testbench_path': tb_path, 'dut_path': duv,
                                                 'run_id': run_id, 'workflow_mode': 'implementation'})
        rec['sim'] = {'ok': bool(sim.get('success')), 'error': sim.get('error') or sim.get('errors'),
                      'total': sim.get('total_tests'), 'passed': sim.get('passed_tests')}
        # the testbench cannot connect to the design (e.g. other port names): the model
        # gets the compiler's messages and the ports the testbench uses
        k = 0
        while not sim.get('success') and 'Compilation failed' in str(rec['sim']['error']) and k < self.compile_fix:
            k += 1
            fx = self._post('/api/generate-hardware', {**common, **extra, 'input': '', 'run_id': run_id,
                                                       'compile_fix_of': {'duv_filepath': duv,
                                                                          'testbench_path': tb_path}})
            run_ids.append(fx.get('run_id'))
            step = {'round': k, 'action': 'interface', 'ok': False, 'error': fx.get('error'),
                    'errors_sent': (fx.get('compile_fix') or {}).get('errors')}
            if fx.get('success') and (fx.get('duv_compile') or {}).get('ok') is not False:
                duv = self._rel(fx.get('filepath'))
                rr = self._post('/api/repair/rerun', {'bdd_filepath': feature, 'duv_filepath': duv,
                                                      'run_id': run_id, 'workflow_mode': 'implementation',
                                                      'dut_info': dut_info})
                if rr.get('success'):
                    tb_path, sim = rr['testbench_path'], rr.get('simulation') or {}
                    rec['sim'] = {'ok': bool(sim.get('success')), 'error': sim.get('error') or sim.get('errors'),
                                  'total': sim.get('total_tests'), 'passed': sim.get('passed_tests')}
                    step['ok'] = bool(sim.get('success'))
                else:
                    step['error'] = rr.get('error')
            rec['rounds'].append(step)
            if not fx.get('success'):
                break
        if not sim.get('success'):
            rec['error'] = f"simulation: {rec['sim']['error']}"
            return

        rec['stage'] = 'explain'
        page = {**extra, 'bdd_filepath': feature, 'duv_filepath': duv, 'testbench_path': tb_path,
                'log_file': sim.get('log_file'), 'run_id': run_id, 'workflow_mode': 'implementation',
                'dut_info': dut_info}
        s = self._summary(page)
        rec['initial'] = self._facts(s)
        if s.get('error'):
            rec['error'] = f"explanations: {s['error']}"
            return
        self._mark_session(s.get('session_id'))
        rec['initial'].update(self._mapping(s.get('feature_id')))

        # Repairs, as offered by the Problems card
        rec['stage'] = 'repair'
        for k in range(1, self.repair_rounds + 1):
            r = (s.get('repairs') or {})
            did = False
            if r.get('scenario_values'):
                fix = self._post('/api/repair/scenarios', page)
                step = {'round': k, 'action': 'scenarios', 'ok': bool(fix.get('success')),
                        'error': fix.get('error'), 'changed': len(fix.get('corrections') or [])}
                if fix.get('success'):
                    page = {**page, 'bdd_filepath': fix['feature_path'], 'testbench_path': fix['testbench_path'],
                            'log_file': (fix.get('simulation') or {}).get('log_file'),
                            'run_id': fix.get('run_id') or page['run_id']}
                    s = self._summary(page)
                    step['after'] = self._facts(s)
                    did = True
                rec['rounds'].append(step)
                if s.get('error'):
                    break
                r = s.get('repairs') or {}
            if r.get('design_failures') and r.get('design_regeneration'):
                hw = self._post('/api/generate-hardware', {
                    **common, **extra, 'input': '', 'run_id': page['run_id'],
                    'repair_of': {'testbench_path': page['testbench_path'], 'log_file': page['log_file']}})
                run_ids.append(hw.get('run_id'))
                dc = hw.get('duv_compile') or {}
                step = {'round': k, 'action': 'design', 'ok': bool(hw.get('success')) and dc.get('ok') is not False,
                        'error': hw.get('error') or (dc.get('error') if dc.get('ok') is False else None),
                        'failures_sent': (hw.get('repair') or {}).get('failures')}
                if step['ok']:
                    page = {**page, 'duv_filepath': self._rel(hw.get('filepath')), 'run_id': hw.get('run_id') or page['run_id']}
                    rr = self._post('/api/repair/rerun', page)
                    if rr.get('success'):
                        page = {**page, 'testbench_path': rr['testbench_path'],
                                'log_file': (rr.get('simulation') or {}).get('log_file')}
                        s = self._summary(page)
                        step['after'] = self._facts(s)
                        did = True
                    else:
                        step.update(ok=False, error=f"rerun: {rr.get('error')}")
                rec['rounds'].append(step)
                if s.get('error'):
                    break
            if not did:
                break
        rec['final'] = self._facts(s)
        rec['stage'] = 'done'


# ----------------------------------------------------------------- reporting

# A run that stopped because the provider did not answer (rate limit, quota,
# timeout, network, no balance) says nothing about the model; it is retried
# when the same trial command runs again, and the summary counts it apart.
_API_FAILURE = re.compile(r'The model call failed|\b429\b|Too Many Requests|RESOURCE_EXHAUSTED|quota|rate.?limit'
                          r'|Timeout|timed out|HTTP 5\d\d|\b50[234]\b|ConnectionError|Max retries exceeded'
                          r'|Insufficient Balance|\b402\b', re.I)


def failure_kind(rec: Dict) -> Optional[str]:
    """None (the run finished), 'api' (the provider did not answer), 'tool'
    (an exception in this code) or 'model' (the answer was unusable)."""
    err = rec.get('error')
    if not err:
        return None
    if _API_FAILURE.search(err):
        return 'api'
    if re.match(r'^[A-Z]\w*(Error|Exception)\b', err):
        return 'tool'
    return 'model'


def _flat(rec: Dict) -> Dict:
    i, f = rec.get('initial') or {}, rec.get('final') or {}
    iv, fv = i.get('verdicts') or {}, f.get('verdicts') or {}
    row = {'llm': rec['llm'], 'design': rec['design'], 'rep': rec['rep'], 'stage': rec['stage'],
           'error': (rec.get('error') or '')[:200], 'failure': failure_kind(rec) or '',
           'seconds': rec.get('seconds'),
           'step0_ok': (rec.get('step0') or {}).get('ok'),
           'duv_compiles': (rec.get('duv') or {}).get('compiles'),
           'test_count': (rec.get('tb') or {}).get('test_count'),
           'skipped_no_expected': ((rec.get('tb') or {}).get('expected_values') or {}).get('skipped_no_expected'),
           'scenarios_unmapped': ((rec.get('tb') or {}).get('parse_stats') or {}).get('scenarios_unmapped'),
           'steps_mapped': i.get('steps_mapped'), 'steps_unmapped': i.get('steps_unmapped'),
           'not_tested': i.get('not_tested')}
    for k in VERDICTS:
        row[f'initial_{k}'] = iv.get(k)
    for k in VERDICTS:
        row[f'final_{k}'] = fv.get(k)
    row.update({'initial_all_clear': i.get('all_clear'), 'final_all_clear': f.get('all_clear'),
                'scenario_repairs': sum(1 for r in rec.get('rounds') or [] if r['action'] == 'scenarios'),
                'compile_fixes': sum(1 for r in rec.get('rounds') or [] if r['action'] in ('compile', 'interface')),
                'compile_fixes_ok': sum(1 for r in rec.get('rounds') or []
                                        if r['action'] in ('compile', 'interface') and r['ok']),
                'design_repairs': sum(1 for r in rec.get('rounds') or [] if r['action'] == 'design'),
                'repair_failures': sum(1 for r in rec.get('rounds') or [] if not r['ok']),
                'llm_calls': rec.get('llm_calls'), 'prompt_chars': rec.get('prompt_chars'),
                'response_chars': rec.get('response_chars'), 'llm_seconds': rec.get('llm_seconds')})
    return row


def summarize(records: List[Dict]) -> str:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in records:
        groups.setdefault((r['llm'], r['design']), []).append(r)
    lines = ['| model | design | runs | reached explanations | checks | pass | design wrong | scenario wrong '
             '| other | steps read | skipped rows | clear at start | clear at end | design repairs '
             '| compile fixes | requirements | errors |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (llm, design), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get('initial') and not (r['initial'] or {}).get('error')]
        tot = {k: sum((r['initial']['verdicts'] or {}).get(k, 0) for r in ok) for k in VERDICTS}
        n = sum(tot.values())
        pct = lambda x: f'{100 * x / n:.0f}%' if n else '—'
        mapped = sum(r['initial'].get('steps_mapped') or 0 for r in ok)
        unmapped = sum(r['initial'].get('steps_unmapped') or 0 for r in ok)
        skipped = sum(r['initial'].get('not_tested') or 0 for r in ok)    # rows that became no check
        clear0 = sum(1 for r in ok if r['initial'].get('all_clear'))
        clear1 = sum(1 for r in ok if (r.get('final') or {}).get('all_clear'))
        drep = sum(1 for r in ok for x in r.get('rounds') or [] if x['action'] == 'design')
        cfix = [x for r in rs for x in r.get('rounds') or [] if x['action'] in ('compile', 'interface')]
        cfix_txt = f"{sum(1 for x in cfix if x['ok'])}/{len(cfix)}" if cfix else '—'
        stops: Dict[str, int] = {}
        for r in rs:
            if r.get('error'):
                k = (r['error'].split(':')[0] or 'error') + (' (API)' if failure_kind(r) == 'api' else '')
                stops[k] = stops.get(k, 0) + 1
        errors = [f'{k} ×{v}' for k, v in sorted(stops.items())]
        tried = [r for r in rs if r.get('step0')]
        req_txt = f"{sum(1 for r in tried if r['step0'].get('ok'))}/{len(tried)}" if tried else '—'
        lines.append(f"| {llm} | {design} | {len(rs)} | {len(ok)} | {n} | {pct(tot['pass'])} "
                     f"| {pct(tot['design_wrong'])} | {pct(tot['scenario_wrong'])} "
                     f"| {pct(tot['both_wrong'] + tot['consistent_wrong'] + tot['undecided'])} "
                     f"| {f'{100 * mapped / (mapped + unmapped):.0f}%' if mapped + unmapped else '—'} "
                     f"| {skipped} | {clear0}/{len(ok)} | {clear1}/{len(ok)} | {drep} | {cfix_txt} "
                     f"| {req_txt} | {', '.join(e for e in errors if e) or '—'} |")
    unread = sorted({t for r in records for t in (r.get('initial') or {}).get('unmapped_examples') or []})
    out = ['# Trial summary', '', *lines]
    lost = [r for r in records if failure_kind(r) == 'api']
    if lost:
        out += ['', f'{len(lost)} run(s) stopped because the provider did not answer (marked "(API)" under '
                    'errors: rate limit, quota, timeout, network, balance). They say nothing about the model; '
                    'running the same trial command again retries them.']
        out += [f"- {r['llm']} {r['design']} #{r['rep']}: {r['error'][:160]}" for r in lost]
    no_req = [r for r in records if r.get('step0') and not r['step0'].get('ok')]
    if no_req:
        out += ['', f'{len(no_req)} run(s) have no requirements because Step 0 failed: the design and the '
                    'scenarios were generated without them, so these runs are not comparable to the others.']
        out += [f"- {r['llm']} {r['design']} #{r['rep']}: {(r['step0'].get('error') or 'no reason given')[:160]}"
                for r in no_req]
    bad_calls = sum((r.get('llm_calls') or 0) - (r.get('llm_calls_ok') or 0) for r in records)
    if bad_calls:
        out += ['', f'{bad_calls} model call(s) failed or came back empty. An empty answer caused by the output '
                    'budget is retried once with twice the budget.']
    if unread:
        out += ['', 'Scenario steps the rule library could not read (up to 20 per run):', '']
        out += [f'- `{t}`' for t in unread[:60]]
    probs: Dict[str, int] = {}
    for r in records:
        for p in (r.get('initial') or {}).get('problems') or []:
            probs[f"[{p['who']}] {re.sub(r'[0-9]+', 'N', p['title'])}"] = probs.get(
                f"[{p['who']}] {re.sub(r'[0-9]+', 'N', p['title'])}", 0) + 1
    if probs:
        out += ['', 'Problems shown at the first simulation (how many runs):', '']
        out += [f'- {k}: {v}' for k, v in sorted(probs.items(), key=lambda kv: -kv[1])]
    return '\n'.join(out) + '\n'


def write_outputs(out: Path, records: List[Dict]) -> str:
    # these files are meant to be shared (data/trials/ in git): no credential
    # may be in them, e.g. a key inside a request URL quoted by an error message
    from src.redact import redact_obj
    records = redact_obj(records)
    with open(out / 'runs.jsonl', 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rows = [_flat(r) for r in records]
    if rows:
        with open(out / 'runs.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    text = summarize(records)
    (out / 'summary.md').write_text(text, encoding='utf-8')
    return text


def export(out: Path, dest: Path, root: Path = ROOT) -> int:
    """Copy every file of every run into dest/<model>_<design>_<rep>/ (to share a trial).

    Runs recorded before 2026-11-05 have no file list; for them the design and
    feature files are taken, with the feature's corrected versions and the
    testbenches and logs named after them."""
    import shutil
    n = 0
    for r in load(out):
        files = list(r.get('files') or [])
        if not files:
            for k in ('duv', 'bdd'):
                f = (r.get(k) or {}).get('file')
                if f:
                    files.append(f)
            bdd = (r.get('bdd') or {}).get('file')
            if bdd:
                stem = Path(bdd).stem
                files += [p.relative_to(root).as_posix() for p in (root / bdd).parent.glob(f'{stem}_fix*.feature')]
                for tb in (root / 'output' / 'testbench').rglob(f'{stem}*_tb.*'):
                    files.append(tb.relative_to(root).as_posix())
                    files += [lg.relative_to(root).as_posix()
                              for lg in (root / 'output' / 'results').glob(f'{Path(tb.name).stem.split(".")[0]}_*.log')]
        # each testbench's scenario index (what every assertion checks)
        files += [str(Path(f).with_name(Path(f).stem + '.scenarios.json')) for f in files if f.endswith('_tb.v')]
        target = dest / f"{r['llm']}_{r['design']}_{r['rep']}"
        for f in dict.fromkeys(files):
            src = root / f
            if src.is_file():
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target / src.name)
                n += 1
        if (out / 'logs' / f"{target.name}.log").is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / 'logs' / f"{target.name}.log", target / 'server.log')
            n += 1
    return n


def load(out: Path) -> List[Dict]:
    p = out / 'runs.jsonl'
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.exists() else []


# ----------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m src.xdb.trial', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run the flow for models x design types x repetitions')
    r.add_argument('--llms', required=True, help="comma-separated providers, e.g. deepseek,gemini; 'mock' = no API")
    r.add_argument('--designs', default=','.join(DESIGNS), help='alu,regfile,cpu (default: all three)')
    r.add_argument('--reps', type=int, default=1)
    r.add_argument('--repair', type=int, default=2, help='repair rounds per run (0 = none)')
    r.add_argument('--mock-fault', choices=('compile', 'interface'), default=None,
                   help="with --llms mock: the first design of each type does not compile ('compile') "
                        "or uses other port names ('interface')")
    r.add_argument('--compile-fix', type=int, default=2,
                   help='attempts to let the model fix a design that does not compile (0 = none)')
    r.add_argument('--no-step0', action='store_true', help='skip the requirements step')
    r.add_argument('--model', action='append', default=[], metavar='LLM=MODEL',
                   help='model for a provider, e.g. gemini=gemini-2.5-flash (repeatable)')
    r.add_argument('--tag', default=None, help='name of this trial (default: date and time)')
    s = sub.add_parser('summary', help='print the summary of an earlier trial')
    s.add_argument('--tag', required=True)
    e = sub.add_parser('export', help='copy the files of every run of a trial into one folder')
    e.add_argument('--tag', required=True)
    e.add_argument('--to', required=True, help='destination folder, e.g. data/trials/<tag>/files')
    args = ap.parse_args(argv)

    if args.cmd == 'export':
        out = ROOT / 'output' / 'trials' / args.tag
        if not load(out):
            sys.exit(f'no runs found in {out}')
        n = export(out, ROOT / args.to)
        print(f'copied {n} files to {args.to}')
        return

    if args.cmd == 'summary':
        out = ROOT / 'output' / 'trials' / args.tag
        recs = load(out)
        if not recs:
            sys.exit(f'no runs found in {out}')
        print(write_outputs(out, recs))
        return

    from src import xdb
    if not xdb.enabled():
        sys.exit('DATABASE_URL is not set (environment or .env): the trial stores every run in the '
                 'explainability database. See docs/explainability_db_setup.md.')
    llms = [x.strip().lower() for x in args.llms.split(',') if x.strip()]
    designs = [x.strip().lower() for x in args.designs.split(',') if x.strip()]
    bad = [d for d in designs if d not in DESIGNS]
    if bad:
        sys.exit(f'unknown design type(s): {", ".join(bad)} (choose from {", ".join(DESIGNS)})')
    models = dict(m.split('=', 1) for m in args.model if '=' in m)
    tag = args.tag or datetime.now().strftime('trial_%Y%m%d_%H%M%S')
    trial = Trial(tag, step0=not args.no_step0, repair_rounds=args.repair, compile_fix=args.compile_fix)
    if 'mock' in llms:
        install_mock()            # after the app import: main loads llm_providers by its short name
        MockLLM.first_fault = args.mock_fault
    records = load(trial.out)
    done = {(r['llm'], r['design'], r['rep']) for r in records if failure_kind(r) != 'api'}
    retry = sum(1 for r in records if failure_kind(r) == 'api')
    plan = [(l, d, k) for l in llms for d in designs for k in range(1, args.reps + 1) if (l, d, k) not in done]
    say = lambda s, end='\n': print(s, end=end, file=sys.__stdout__, flush=True)
    say(f'trial {tag}: {len(plan)} runs ({len(done)} already done'
        + (f', {retry} retried after an API failure' if retry else '') + f') -> {trial.out}')
    # everything the server and its background threads print goes to server.log
    with open(trial.out / 'server.log', 'a', encoding='utf-8') as server_log, \
            contextlib.redirect_stdout(server_log):
        for n, (llm, design, rep) in enumerate(plan, 1):
            say(f'[{n}/{len(plan)}] {llm} {design} #{rep} ... ', end='')
            rec = trial.run_one(llm, design, rep, model=models.get(llm))
            records = [r for r in records if (r['llm'], r['design'], r['rep']) != (llm, design, rep)]
            records.append(rec)
            write_outputs(trial.out, records)             # keep results if the trial is interrupted
            v = (rec.get('initial') or {}).get('verdicts') or {}
            status = (f"{rec['error']}" if rec.get('error') else
                      f"{sum(v.values())} checks, design wrong {v.get('design_wrong', 0)}, scenario wrong "
                      f"{v.get('scenario_wrong', 0)}; after repair: "
                      f"{'clear' if (rec.get('final') or {}).get('all_clear') else 'problems left'}")
            from src.redact import redact_secrets
            say(f'{redact_secrets(status)} ({rec.get("seconds")} s)')
        time.sleep(2)                                     # let the last background writes finish
    say('')
    say(write_outputs(trial.out, records))


if __name__ == '__main__':
    main()
