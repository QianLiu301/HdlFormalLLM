"""
Local fixes and per-problem chat for ALU designs (D46).

A problem card names an operation that the design computes wrong. Here:

    locate(verilog, op, port)   the lines of the design that compute it: the case
                                item of its opcode and the lines it reads from
    attach_code(...)            adds those lines to each design card; cards that
                                point at the same lines are fixed together
    propose(...)                asks a model to rewrite only those lines, then
                                checks the result before anything is saved:
                                  1. only the given lines changed
                                  2. the design compiles
                                  3. the same testbench: no check that matched the
                                     reference before misses it now, and at least
                                     one that missed matches now
                                  4. random inputs the model never saw: no
                                     operation gets more wrong than before
    keep(...) / discard(...)    the user decides; keep stores a new design version
    chat(...)                   a conversation per card, grounded in the card's
                                facts (examples with O/B/R, the code lines, the
                                rules); a code change it proposes goes through
                                propose() like any other

The verified facts and the model's words stay apart: the card text is computed
by rules (explain.py); what the model writes is shown and stored as its answer.
"""

import difflib
import json
import random
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sqlalchemy import select

from src import xdb
from src.xdb import library as L
from src.xdb import models as M

ALU_OPS = ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'SLL', 'SRL', 'SRA', 'SLT', 'SLTU']
_FLAGS = ('zero', 'negative', 'overflow')
_KEYWORDS = {'begin', 'end', 'if', 'else', 'case', 'casez', 'casex', 'endcase', 'assign', 'wire', 'reg',
             'signed', 'unsigned', 'always', 'posedge', 'negedge', 'or', 'and', 'not', 'default', 'integer',
             'logic', 'for', 'module', 'endmodule', 'input', 'output', 'localparam', 'parameter'}
_INPUTS = {'a', 'b', 'opcode', 'clk', 'rst', 'rst_n', 'reset'}
_DESIGN_FAULT = ('design_wrong', 'both_wrong', 'consistent_wrong')
MAX_LINES = 30

Caller = Callable[[str, str], str]      # (system prompt, prompt) -> model text


# ----------------------------------------------------------------- locate

def _code(line: str) -> str:
    return re.sub(r'//.*', '', line)


def _literal(tok: str, params: Dict[str, int]) -> Optional[int]:
    tok = tok.strip()
    m = re.fullmatch(r"\d*'([bBhHdDoO])([0-9a-fA-F_xXzZ]+)", tok)
    if m:
        base = {'b': 2, 'h': 16, 'd': 10, 'o': 8}[m.group(1).lower()]
        try:
            return int(m.group(2).replace('_', ''), base)
        except ValueError:
            return None
    if re.fullmatch(r'\d+', tok):
        return int(tok)
    return params.get(tok)


def _params(lines: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ln in lines:
        c = _code(ln)
        if re.search(r'\b(localparam|parameter)\b', c):
            for name, val in re.findall(r"(\w+)\s*=\s*([\w']+)", c):
                v = _literal(val, out)
                if v is not None:
                    out[name] = v
    return out


def _block_end(lines: List[str], i: int) -> int:
    """Last line of a case item that opens a begin ... end block."""
    depth = 0
    for j in range(i, len(lines)):
        c = _code(lines[j])
        depth += len(re.findall(r'\bbegin\b', c)) - len(re.findall(r'\bend\b', c))
        if depth <= 0:
            return j
    return i


def _assigning(lines: List[str], name: str) -> List[int]:
    rx = re.compile(r'^\s*(?:assign\s+)?' + re.escape(name) + r'\s*(?:\[[^\]]*\])?\s*<?=(?!=)')
    decl = re.compile(r'^\s*(?:wire|reg)\b[^;]*\b' + re.escape(name) + r'\s*=(?!=)')
    hits = []
    for i, ln in enumerate(lines):
        c = _code(ln)
        if rx.search(c) or decl.search(c):
            hits.append(i)
    return hits


def _reads(line: str, ports: set) -> List[str]:
    c = _code(line)
    rhs = c.split('=', 1)[1] if re.search(r'<?=(?!=)', c) else c
    rhs = re.sub(r"\d*'[bBhHdDoO][0-9a-fA-F_xXzZ]+", ' ', rhs)
    return [t for t in re.findall(r'\b[A-Za-z_]\w*\b', rhs)
            if t not in _KEYWORDS and t not in ports and not t.startswith('$')]


def locate(verilog: str, op: Optional[str] = None, port: Optional[str] = None) -> List[Dict]:
    """[{no, text}] of the lines that compute this ALU operation (and/or flag)."""
    lines = verilog.splitlines()
    params = _params(lines)
    found: set = set()
    if op in ALU_OPS:
        k = ALU_OPS.index(op)
        for i, ln in enumerate(lines):
            m = re.match(r"^\s*([\w']+(?:\s*,\s*[\w']+)*)\s*:(?!:)", _code(ln))
            if not m or m.group(1).strip() == 'default':
                continue
            if any(_literal(t, params) == k for t in m.group(1).split(',')):
                end = _block_end(lines, i) if re.search(r'\bbegin\b', _code(ln)) else i
                found.update(range(i, end + 1))
    if port in _FLAGS:
        for i in _assigning(lines, port):
            found.add(i)
    ports = _INPUTS | set(_FLAGS) | {'result'} | set(params)
    # What these lines assign is their output, not a cause (temp_result in every case item);
    # a signal assigned in many places is such an output too, and following it would pull in
    # every other operation.
    seen = {m.group(1) for i in found
            for m in [re.match(r'^\s*(?:assign\s+)?(\w+)\s*(?:\[[^\]]*\])?\s*<?=(?!=)', _code(lines[i]))] if m}
    seen |= {n for n in set(re.findall(r'\b[A-Za-z_]\w*\b', _code(verilog)))
             if n not in _KEYWORDS and len(_assigning(lines, n)) > 3}
    frontier = list(found)
    traced = set()
    for _ in range(8):                       # follow the signals these lines read back to the inputs
        nxt = []
        for i in frontier:
            for name in _reads(lines[i], ports):
                if name in seen:
                    continue
                seen.add(name)
                hits = _assigning(lines, name)
                if hits:
                    traced.add(name)
                for j in hits:
                    if j not in found and len(found) < MAX_LINES:
                        found.add(j)
                        nxt.append(j)
        frontier = nxt
    # their declarations too: a fix often has to make an intermediate signal signed
    for name in traced:
        rx = re.compile(r'^\s*(?:reg|wire|logic)\b[^;=]*\b' + re.escape(name) + r'\b[^=]*;')
        for i, ln in enumerate(lines):
            if rx.search(_code(ln)) and len(found) < MAX_LINES:
                found.add(i)
    return [{'no': i + 1, 'text': lines[i]} for i in sorted(found)]


def _flag_port(title: str) -> Optional[str]:
    m = re.search(r'\b(zero|negative|overflow)\b (?:flag|is)', title or '', re.I)
    return m.group(1).lower() if m else None


def attach_code(s, run_id: int, problems: List[Dict]) -> None:
    """Add the located lines and what can be fixed to each ALU problem card."""
    from src.xdb.ingest import artifact_text
    run = s.get(M.SimRun, run_id)
    tb = s.get(M.Testbench, run.testbench_id)
    sess = s.get(M.Session, tb.session_id)
    design = s.get(M.Design, sess.design_id) if sess.design_id else None
    if not design or design.module_type not in ('alu', 'alu_comb') or not run.dut_artifact:
        return
    art = s.get(M.Artifact, run.dut_artifact)
    if art is None:
        return
    verilog = artifact_text(art)
    ops_of = {ci.tag: ci.operation for ci in s.scalars(select(M.CheckItem)
                                                       .where(M.CheckItem.testbench_id == tb.id))}
    groups: Dict[str, int] = {}
    for i, p in enumerate(problems):
        if p.get('who') not in ('design', 'scenario'):
            continue
        tag = (p.get('explain') or {}).get('tag')
        ops = [o for o in (p.get('ops') or [ops_of.get(tag)]) if o]
        port = _flag_port(p.get('title') or '')
        if 'meanings of overflow' in (p.get('title') or ''):
            ops, port = ['ADD', 'SUB'], 'overflow'           # its examples are in both
        if not ops:
            continue
        both = 'both the design and the scenario' in (p.get('title') or '')
        kinds = (['design', 'scenario'] if both else [p['who']])
        fix = {'kinds': kinds, 'ops': ops, 'port': port}
        if 'design' in kinds:
            found: Dict[int, Dict] = {}
            for op in ops:
                for ln in locate(verilog, op, port):
                    found[ln['no']] = ln
            fix['lines'] = [found[n] for n in sorted(found)]
            key = ','.join(str(n) for n in sorted(found))
            if key and key in groups:
                fix['same_as'] = groups[key]
                problems[groups[key]]['fix']['covers'].append(i)
            elif key:
                groups[key] = i
                fix['covers'] = [i]
        p['fix'] = fix


# ----------------------------------------------------------------- views (D52)

def feature_row_lines(text: str, catalog: List[Dict]) -> Dict[str, int]:
    """'BDD:<scenario>/r<n>' -> line number (1-based) of that Examples row in the feature."""
    out: Dict[str, int] = {}
    for entry in catalog or []:
        span = entry.get('span') or []
        if len(span) != 2 or not entry.get('id'):
            continue
        start, end = int(span[0]), int(span[1])
        offset, row_no = start, 0
        in_table, header_seen = False, False
        for ln in text[start:end].splitlines(True):
            stripped = ln.strip()
            if re.match(r'Examples?\s*:', stripped):
                in_table, header_seen = True, False
            elif in_table and stripped.startswith('|'):
                if not header_seen:
                    header_seen = True
                else:
                    row_no += 1
                    out[f"BDD:{entry['id']}/r{row_no}"] = text.count('\n', 0, offset) + 1
            elif in_table and stripped:
                in_table = False
            offset += len(ln)
    return out


_SCEN_FAULT = ('scenario_wrong', 'both_wrong', 'consistent_wrong')


def _dec_hex(v, w):
    """The way the cards write a value: 2684354560 (0xA0000000)."""
    v = int(v) % (1 << w)
    return f'{v} (0x{v:X})'


def attach_views(s, run_id: int, problems: List[Dict], catalog: List[Dict]) -> Optional[Dict]:
    """The design and the feature as simulated, and for each card its marks in them:
    {'file': 'design'|'feature', 'line', 'kind': 'design'|'scenario'|'example', 'note'}."""
    from src.xdb.ingest import artifact_text
    from src.xdb.explain import _probe_results, _probe_run_id
    run = s.get(M.SimRun, run_id)
    tb = s.get(M.Testbench, run.testbench_id)
    sess = s.get(M.Session, tb.session_id)
    design = s.get(M.Design, sess.design_id) if sess.design_id else None
    w = (design.bitwidth if design else None) or 32
    dut = s.get(M.Artifact, run.dut_artifact) if run.dut_artifact else None
    feat_row = s.get(M.Feature, tb.feature_id) if tb.feature_id else None
    feat = s.get(M.Artifact, feat_row.artifact) if feat_row else None
    files = {}
    if dut is not None:
        files['design'] = {'name': dut.orig_name, 'text': artifact_text(dut)}
    if feat is not None:
        files['feature'] = {'name': feat.orig_name, 'text': artifact_text(feat)}
    if not files:
        return None
    rows = feature_row_lines(files.get('feature', {}).get('text', ''), catalog)
    main = [(ci, cr) for ci, cr in s.execute(select(M.CheckItem, M.CheckResult)
                                             .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                                             .where(M.CheckResult.sim_run_id == run_id, M.CheckItem.seq == 1))
            if ci.tag.startswith('BDD:')]
    probe = _probe_results(s, _probe_run_id(s, run_id))
    for p in problems:
        f = p.get('fix') or {}
        marks: List[Dict] = [{'file': 'design', 'line': ln['no'], 'kind': 'design'} for ln in f.get('lines') or []]
        ops, port, kinds = set(f.get('ops') or []), f.get('port'), f.get('kinds') or [p.get('who')]
        tag = (p.get('explain') or {}).get('tag')
        if not ops and tag:
            ops = {ci.operation for ci, _ in main if ci.tag == tag}
        if port:
            for (t, pt), o in probe.items():
                if pt == port and o.get('operation') in ops and o['verdict'] in (_SCEN_FAULT if 'scenario' in kinds
                                                                                 else _DESIGN_FAULT) and t in rows:
                    marks.append({'file': 'feature', 'line': rows[t], 'kind': 'scenario' if 'scenario' in kinds else 'example',
                                  'note': f"{port} should be {o['rule_value']}"
                                          + (f", the design gives {o['observed']}" if 'design' in kinds else '')})
        else:
            for ci, cr in main:
                if ci.operation not in ops or ci.tag not in rows or (ci.port or 'result') != 'result':
                    continue
                if 'scenario' in kinds and cr.verdict in _SCEN_FAULT and ci.expected_ref is not None:
                    marks.append({'file': 'feature', 'line': rows[ci.tag], 'kind': 'scenario',
                                  'note': f'correct: {_dec_hex(ci.expected_ref, ci.width or w)}'})
                elif 'design' in kinds and cr.verdict in _DESIGN_FAULT and ci.expected_ref is not None:
                    got = _dec_hex(cr.observed_value, ci.width or w) if cr.observed_value is not None else 'x'
                    marks.append({'file': 'feature', 'line': rows[ci.tag], 'kind': 'example',
                                  'note': f'the design gives {got}, correct: {_dec_hex(ci.expected_ref, ci.width or w)}'})
        seen = set()
        p['marks'] = [m for m in marks if (m['file'], m['line']) not in seen and not seen.add((m['file'], m['line']))]
    return files


# ----------------------------------------------------------------- simulate

def _run(cmd: List[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=timeout)


def compiles(verilog: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'design.v').write_text(verilog, encoding='utf-8')
        r = _run(['iverilog', '-g2012', '-o', 'x.vvp', 'design.v'], Path(d))
        return r.returncode == 0, (r.stderr or r.stdout or '').strip()[:800]


def simulate(tb_text: str, verilog: str) -> Tuple[Optional[str], str]:
    """(log, error) of the testbench with this design."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'tb.v').write_text(tb_text, encoding='utf-8')
        Path(d, 'design.v').write_text(verilog, encoding='utf-8')
        r = _run(['iverilog', '-g2012', '-o', 'sim.vvp', 'tb.v', 'design.v'], Path(d))
        if r.returncode != 0:
            return None, (r.stderr or r.stdout or '').strip()[:800]
        r = _run(['vvp', '-n', 'sim.vvp'], Path(d))
        return r.stdout, ''


_TEST = re.compile(r'Test (\d+): (PASS|FAIL)(?: - Expected\s+(-?\d+), Got\s+(-?\d+|[xXzZ]+))?')


def scenario_outcome(log: str, checks: List[Dict]) -> Dict[int, Optional[int]]:
    """test number -> the design's value (O): the compared value on PASS, Got on FAIL."""
    by_no = {c['test_no']: c for c in checks}
    out: Dict[int, Optional[int]] = {}
    for m in _TEST.finditer(log or ''):
        n = int(m.group(1))
        c = by_no.get(n)
        if c is None:
            continue
        if m.group(2) == 'PASS':
            out[n] = c['used']
        else:
            got = m.group(4)
            out[n] = int(got) % (1 << c['width']) if got and got.lstrip('-').isdigit() else None
    return out


def _vectors(w: int) -> List[Tuple[int, int, int]]:
    m = (1 << w) - 1
    top = 1 << (w - 1)
    edge = [0, 1, m, top, top - 1]
    rng = random.Random(1000 + w)
    out = [(k, a, b) for k in range(10) for a in edge for b in edge]
    out += [(k, rng.getrandbits(w), rng.getrandbits(w)) for k in range(10) for _ in range(40)]
    out += [(k, rng.getrandbits(w), rng.randrange(0, 2 * w)) for k in (5, 6, 7) for _ in range(20)]
    return out


def random_testbench(tb_text: str, w: int) -> Optional[str]:
    """The page's testbench with its scenario rows replaced by random inputs."""
    lines = tb_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r'\s*// Test 1 \[', ln)), None)
    if start is None:
        return None
    head = [ln for ln in lines[:start] if '$dump' not in ln]
    flags = 'wire zero' in tb_text and 'wire negative' in tb_text
    body = []
    for k, a, b in _vectors(w):
        show = '%h %b %b' if flags else '%h'
        sig = 'result, zero, negative' if flags else 'result'
        body.append(f"        a = {w}'h{a:x}; b = {w}'h{b:x}; opcode = 4'b{k:04b}; #10; "
                    f'$display("RND {k} {a:x} {b:x} {show}", {sig});')
    return '\n'.join(head + body + ['        $finish;', '    end', '', 'endmodule', ''])


def random_errors(tb_text: str, verilog: str, w: int) -> Tuple[Optional[Dict[str, int]], str]:
    """op -> examples whose result (or zero/negative) differs from the reference."""
    from src.testbench_generator import TestbenchGenerator
    rtb = random_testbench(tb_text, w)
    if rtb is None:
        return None, 'the testbench has an unexpected shape'
    log, err = simulate(rtb, verilog)
    if log is None:
        return None, err
    errors = {op: 0 for op in ALU_OPS}
    seen = 0
    for ln in log.splitlines():
        parts = ln.split()
        if len(parts) < 4 or parts[0] != 'RND':
            continue
        seen += 1
        k, a, b = int(parts[1]), int(parts[2], 16), int(parts[3], 16)
        op = ALU_OPS[k]
        want = TestbenchGenerator._reference_result(op, a, b, w)
        try:
            got = int(parts[4], 16)
        except (ValueError, IndexError):
            errors[op] += 1
            continue
        bad = got != want
        if len(parts) >= 7:
            bad = bad or parts[5] != str(int(want == 0)) or parts[6] != str((want >> (w - 1)) & 1)
        errors[op] += int(bad)
    return (errors if seen else None), ('' if seen else 'the random check printed nothing')


def evaluate(base: str, new: str, tb_text: str, checks: List[Dict], w: int,
             targets: List[str], base_state: Optional[Dict] = None) -> Dict:
    """Compare a changed design with the one it changes. base_state caches the base."""
    if base_state is None:
        base_state = baseline(base, tb_text, checks, w)
    ok, cerr = compiles(new)
    if not ok:
        return {'accepted': False, 'reason': f'the changed design does not compile: {cerr}'}
    log, err = simulate(tb_text, new)
    if log is None:
        return {'accepted': False, 'reason': f'the changed design does not run with the testbench: {err}'}
    after = scenario_outcome(log, checks)
    before = base_state['scenario']
    right = lambda o, c: o is not None and o == c['ref']
    newly_wrong = [c for c in checks if c['ref'] is not None
                   and right(before.get(c['test_no']), c) and not right(after.get(c['test_no']), c)]
    fixed = [c for c in checks if c['ref'] is not None
             and not right(before.get(c['test_no']), c) and right(after.get(c['test_no']), c)]
    still = [c for c in checks if c['ref'] is not None and c['op'] in targets
             and not right(after.get(c['test_no']), c)]
    rnd_after, rerr = random_errors(tb_text, new, w)
    rnd_before = base_state['random']
    worse = sorted(op for op in ALU_OPS if rnd_after and rnd_before and rnd_after[op] > rnd_before[op])
    res = {'scenario_fixed': len(fixed), 'scenario_newly_wrong': len(newly_wrong),
           'scenario_still_wrong': len(still),
           'scenario_right_before': sum(right(before.get(c['test_no']), c) for c in checks if c['ref'] is not None),
           'scenario_right_after': sum(right(after.get(c['test_no']), c) for c in checks if c['ref'] is not None),
           'scenario_total': sum(1 for c in checks if c['ref'] is not None),
           'random_total': len(_vectors(w)),
           'random_wrong_before': sum(rnd_before.values()) if rnd_before else None,
           'random_wrong_after': sum(rnd_after.values()) if rnd_after else None,
           'random_worse_ops': worse}
    if newly_wrong:
        ex = newly_wrong[0]
        res.update(accepted=False, reason=f"{len(newly_wrong)} example(s) that were right are now wrong, "
                                          f"e.g. {ex['op']} ({ex['tag'][4:]})")
    elif rnd_after is None:
        res.update(accepted=False, reason=f'the random check could not run: {rerr}')
    elif worse:
        res.update(accepted=False, reason='on random inputs the change makes ' + ', '.join(worse) + ' worse')
    elif still:
        # a fix that only repairs the examples it was shown (e.g. a special case for A = 0, B = 0)
        # is no fix; the operation must be right everywhere
        res.update(accepted=False, reason=f"{len(still)} failing example(s) of {', '.join(targets)} are still wrong")
    elif any(rnd_after[o] for o in targets):
        bad = {o: rnd_after[o] for o in targets if rnd_after[o]}
        res.update(accepted=False, reason='the examples pass, but on random inputs '
                   + ', '.join(f'{o} is still wrong {n} times' for o, n in bad.items())
                   + ' -- the change may only cover the examples it was shown')
    else:
        res.update(accepted=True, reason='')
    return res


def baseline(base: str, tb_text: str, checks: List[Dict], w: int) -> Dict:
    log, _ = simulate(tb_text, base)
    rnd, _ = random_errors(tb_text, base, w)
    return {'scenario': scenario_outcome(log or '', checks), 'random': rnd}


# ----------------------------------------------------------------- model

_SYSTEM = ('You fix one spot in a Verilog design. You may change only the numbered lines you are '
           'given; every other line stays exactly as it is. Answer with JSON only.')


def numbered(verilog: str) -> str:
    return '\n'.join(f'{i:4d} | {ln}' for i, ln in enumerate(verilog.splitlines(), 1))


def fix_prompt(verilog: str, lines: List[Dict], facts: str, note: str = '') -> str:
    allowed = ', '.join(str(ln['no']) for ln in lines)
    return (f"The design, with line numbers:\n\n{numbered(verilog)}\n\n"
            f"Lines you may change: {allowed}.\n\n{facts}\n"
            + (f"\nThe engineer adds: {note.strip()}\n" if note and note.strip() else '')
            + '\nReturn JSON of this form:\n'
              '{"changes": [{"line": <number>, "new": "<the new text of that line; it may span several lines, '
              'or be empty to delete it>"}], "why": "<one or two sentences>", '
              '"needs_other_lines": [<numbers of other lines that would also have to change, if any>]}\n'
              'Change only what the failing examples need. Do not write special cases for the example inputs: '
              'the design must follow the rules for every input.')


def parse_changes(text: str) -> Tuple[Optional[Dict], str]:
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        return None, 'the model did not return JSON'
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f'the model returned JSON that could not be read ({e.msg})'
    ch = d.get('changes')
    if not isinstance(ch, list):
        return None, 'the model returned no list of changes'
    out = []
    for c in ch:
        try:
            out.append({'line': int(c['line']), 'new': str(c.get('new', ''))})
        except (KeyError, TypeError, ValueError):
            return None, 'a change names no line number'
    return {'changes': out, 'why': str(d.get('why') or ''),
            'needs_other_lines': [int(x) for x in d.get('needs_other_lines') or [] if str(x).isdigit()]}, ''


def apply_changes(verilog: str, changes: List[Dict], allowed: List[int]) -> Tuple[Optional[str], str]:
    outside = sorted({c['line'] for c in changes if c['line'] not in allowed})
    if outside:
        return None, 'the model changed lines it was not given: ' + ', '.join(map(str, outside))
    if not changes:
        return None, 'the model proposed no change'
    lines = verilog.splitlines()
    for c in sorted(changes, key=lambda c: -c['line']):
        i = c['line'] - 1
        new = c['new'].split('\n') if c['new'] != '' else []
        lines[i:i + 1] = new
    out = '\n'.join(lines) + ('\n' if verilog.endswith('\n') else '')
    if out == verilog:
        return None, 'the proposed lines are the same as before'
    return out, ''


def diff(a: str, b: str, name: str = 'design.v') -> str:
    return ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                        f'{name} (before)', f'{name} (after)', n=2))


# ----------------------------------------------------------------- facts from the database

def _pv(v, w):
    v = int(v) % (1 << w)
    s = v - (1 << w) if v >> (w - 1) else v
    return f'{v:#x}' + (f' ({s} as signed)' if s < 0 else '')


def run_context(s, run_id: int) -> Dict:
    from src.xdb.ingest import artifact_text
    from src.xdb.explain import _overflow_mode
    run = s.get(M.SimRun, run_id)
    tb = s.get(M.Testbench, run.testbench_id)
    sess = s.get(M.Session, tb.session_id)
    design = s.get(M.Design, sess.design_id) if sess.design_id else None
    w = (design.bitwidth if design else None) or 32
    checks, failing = [], {}
    for ci, cr in s.execute(select(M.CheckItem, M.CheckResult)
                            .join(M.CheckResult, M.CheckResult.check_id == M.CheckItem.id)
                            .where(M.CheckResult.sim_run_id == run_id, M.CheckItem.seq == 1)):
        if not ci.tag.startswith('BDD:') or not ci.test_no or (ci.port or 'result') != 'result':
            continue
        checks.append({'test_no': ci.test_no, 'tag': ci.tag, 'op': ci.operation, 'width': ci.width or w,
                       'ref': int(ci.expected_ref) if ci.expected_ref is not None else None,
                       'used': int(ci.expected_used) if ci.expected_used is not None else None})
        if cr.verdict in _DESIGN_FAULT and ci.expected_ref is not None:
            st = ci.stimulus or {}
            got = _pv(cr.observed_value, w) if cr.observed_value is not None else 'unknown'
            failing.setdefault(ci.operation, []).append(
                f"- {ci.operation} with A = {_pv(st.get('a', 0), w)}, B = {_pv(st.get('b', 0), w)}: "
                f"the design gives {got}, correct is {_pv(ci.expected_ref, w)}"
                + (f"; the scenario expected {_pv(ci.expected_bdd, w)}" if ci.expected_bdd is not None
                   and int(ci.expected_bdd) != int(ci.expected_ref) else ''))
    ov_mode, ov_answer = _overflow_mode(s, sess.id)
    tb_art = s.get(M.Artifact, tb.artifact)
    dut_art = s.get(M.Artifact, run.dut_artifact) if run.dut_artifact else None
    return {'run': run, 'testbench': tb, 'session': sess, 'design': design, 'width': w,
            'checks': checks, 'failing': failing, 'overflow_mode': ov_mode, 'overflow_answer': ov_answer,
            'tb_text': artifact_text(tb_art) if tb_art else None,
            'verilog': artifact_text(dut_art) if dut_art else None,
            'duv_id': run.duv_id}


def facts_text(ctx: Dict, ops: List[str], port: Optional[str] = None) -> str:
    rules = (L.library_for('alu') or {}).get('rules') or {}
    parts = []
    for op in ops:
        ex = ctx['failing'].get(op) or []
        if ex:
            parts.append(f'Failing examples of {op} ({len(ex)}):\n' + '\n'.join(ex[:8]))
    rl = [f"- {op}: {rules[op]['text']}" for op in ops if op in rules and rules[op].get('text')]
    if port:
        fr = L.flag_rule('alu', port, ctx.get('overflow_mode'))
        if fr:
            rl.append(f"- {port} flag: {fr['text']}")
    if rl:
        parts.append('The rules these examples are judged by:\n' + '\n'.join(rl))
    return '\n\n'.join(parts)


# ----------------------------------------------------------------- propose, keep, discard

def propose(run_id: int, groups: List[Dict], call: Optional[Caller], note: str = '', llm: Optional[str] = None,
            proposal: Optional[Dict] = None, url=None) -> Tuple[Optional[Dict], Optional[str]]:
    """groups: [{'ops': [...], 'port': ...}], fixed one after another on the changing design.
    With proposal ({'changes': [...]}, one group) no model is called."""
    with xdb.session_scope(url) as s:
        ctx = run_context(s, run_id)
    if not ctx['verilog'] or not ctx['tb_text']:
        return None, 'the design or the testbench of this simulation is not stored'
    base = cur = ctx['verilog']
    w, checks = ctx['width'], ctx['checks']
    state = baseline(base, ctx['tb_text'], checks, w)
    results = []
    for g in groups:
        ops, port = [o for o in g.get('ops') or [] if o in ALU_OPS], g.get('port')
        if g.get('lines'):
            # lines the user selected in the file view (D52); the checks are the same
            src_lines = cur.splitlines()
            chosen = sorted({int(n) for n in g['lines'] if str(n).isdigit() and 1 <= int(n) <= len(src_lines)})
            lines = [{'no': n, 'text': src_lines[n - 1]} for n in chosen]
            if not ops:                       # no card: the operations that fail anywhere
                ops = [o for o in ctx['failing'] if o in ALU_OPS]
        else:
            lines = []
            for op in ops:
                lines += locate(cur, op, port)
            lines = sorted({ln['no']: ln for ln in lines}.values(), key=lambda ln: ln['no'])
        r = {'ops': ops, 'port': port, 'lines': lines}
        if not lines:
            results.append({**r, 'accepted': False,
                            'reason': 'no lines selected' if g.get('lines') is not None
                            else 'the code of this operation was not found'})
            continue
        if proposal is not None:
            parsed, perr = {'changes': proposal.get('changes') or [], 'why': proposal.get('why', ''),
                            'needs_other_lines': []}, ''
        else:
            facts = facts_text(ctx, ops, port)
            if g.get('finding'):     # the card's text: verdicts and, where proven, the cause
                facts = f"What the analysis of the simulation found:\n{g['finding']}\n\n{facts}"
            text = call(_SYSTEM, fix_prompt(cur, lines, facts, note))
            parsed, perr = parse_changes(text)
        if parsed is None:
            results.append({**r, 'accepted': False, 'reason': perr})
            continue
        r.update(why=parsed['why'], needs_other_lines=parsed['needs_other_lines'], changes=parsed['changes'])
        new, aerr = apply_changes(cur, parsed['changes'], [ln['no'] for ln in lines])
        if new is None:
            results.append({**r, 'accepted': False, 'reason': aerr})
            continue
        ev = evaluate(cur, new, ctx['tb_text'], checks, w, ops, state)
        r.update(ev, diff=diff(cur, new))
        if ev['accepted']:
            cur = new
            state = baseline(cur, ctx['tb_text'], checks, w)
        results.append(r)
    accepted = [r for r in results if r.get('accepted')]
    final = baseline(cur, ctx['tb_text'], checks, w) if accepted else state
    first = baseline(base, ctx['tb_text'], checks, w) if accepted else state
    right = lambda st: sum(1 for c in checks if c['ref'] is not None and st['scenario'].get(c['test_no']) == c['ref'])
    summary = {'scenario_total': sum(1 for c in checks if c['ref'] is not None),
               'scenario_right_before': right(first), 'scenario_right_after': right(final),
               'random_total': len(_vectors(w)),
               'random_wrong_before': sum(first['random'].values()) if first['random'] else None,
               'random_wrong_after': sum(final['random'].values()) if final['random'] else None}
    with xdb.session_scope(url) as s:
        fb = M.Feedback(subject_type='duv', subject_id=ctx['duv_id'] or 0, kind='correction',
                        body=note or None,
                        proposed={'action': 'local_fix', 'llm': llm, 'sim_run_id': run_id,
                                  'human_note': bool(note and note.strip()), 'from_chat': proposal is not None,
                                  'groups': [{k: v for k, v in r.items() if k != 'lines'} | {
                                      'lines': [ln['no'] for ln in r.get('lines') or []]} for r in results],
                                  'summary': summary, 'new_code': cur if accepted else None},
                        status='open' if accepted else 'rejected')
        s.add(fb)
        s.flush()
        fb_id = fb.id
    return {'feedback_id': fb_id, 'accepted': bool(accepted), 'groups': results, 'summary': summary,
            'diff': diff(base, cur) if accepted else ''}, None


def keep(project_root, feedback_id: int, design_path: str, url=None) -> Tuple[Optional[Dict], Optional[str]]:
    """Write the accepted design as a new file next to the old one; the caller stores and simulates it."""
    from src import run_report
    from src.xdb.repair import _next_fix_path
    old, err = run_report.safe_path(design_path, project_root)
    if err or not old:
        return None, err or 'no design file'
    with xdb.session_scope(url) as s:
        fb = s.get(M.Feedback, feedback_id)
        if fb is None or (fb.proposed or {}).get('action') != 'local_fix':
            return None, 'no such fix'
        if fb.status != 'open':
            return None, f'this fix is already {fb.status}'
        code = (fb.proposed or {}).get('new_code')
        if not code:
            return None, 'this fix has no accepted change'
        new = _next_fix_path(old)
        new.write_text(code, encoding='utf-8')
        fb.status = 'applied'
        rel = new.resolve().relative_to(Path(project_root).resolve()).as_posix()
        fb.result_ref = f'file:{rel}'
        sim_run = (fb.proposed or {}).get('sim_run_id')
    return {'design_path': rel, 'design_filename': new.name, 'feedback_id': feedback_id,
            'sim_run_id': sim_run}, None


def discard(feedback_id: int, url=None) -> None:
    with xdb.session_scope(url) as s:
        fb = s.get(M.Feedback, feedback_id)
        if fb is not None and fb.status == 'open':
            fb.status = 'rejected'


# ----------------------------------------------------------------- chat

CHAT_ALGORITHM = 'chat_v1'
_CHAT_SYSTEM = (
    'You help a hardware engineer with one problem found when their design was simulated. '
    'Use only the facts given below (the problem, its examples with the design value, the scenario '
    'value and the correct value, the design lines, and the rules); say so when they do not answer '
    'the question. Refer to example values and line numbers. Answer in English, briefly. '
    'If the engineer wants the code changed, add the change at the end as a JSON block: '
    '```json\n{"changes": [{"line": <number>, "new": "<new text>"}], "why": "<short reason>"}\n``` '
    'using only the listed lines. The change is tested before it is applied, so never say it has been applied.')


def chat_facts(ctx: Dict, card: Dict) -> Tuple[str, List[Dict]]:
    ops = [o for o in (card.get('ops') or []) if o in ALU_OPS]
    port = card.get('port')
    lines: Dict[int, Dict] = {}
    for op in ops:
        for ln in locate(ctx['verilog'] or '', op, port):
            lines[ln['no']] = ln
    sel = card.get('selection') or {}
    if sel.get('file') == 'design' and sel.get('start'):
        src_lines = (ctx['verilog'] or '').splitlines()
        for n in range(int(sel['start']), int(sel.get('end') or sel['start']) + 1):
            if 1 <= n <= len(src_lines):
                lines[n] = {'no': n, 'text': src_lines[n - 1]}
    if not ops:
        ops = [o for o in ctx['failing'] if o in ALU_OPS]
    code = '\n'.join(f"{ln['no']:4d} | {ln['text']}" for ln in sorted(lines.values(), key=lambda x: x['no']))
    txt = ((f"The problem: {card.get('title', '')}\n{card.get('detail', '')}\n\n" if card.get('title') else '')
           + (f"The engineer selected lines {sel.get('start')}-{sel.get('end')} of the "
              f"{'design' if sel.get('file') == 'design' else 'scenario file'}:\n{sel.get('text', '')[:3000]}\n\n"
              if sel.get('text') else '')
           + f"Design lines:\n{code or '(not found)'}\n\n{facts_text(ctx, ops, port)}")
    if ctx.get('overflow_answer'):
        txt += f"\n\nThe user chose in Step 0: overflow means {ctx['overflow_answer']}"
    return txt, sorted(lines.values(), key=lambda x: x['no'])


def _history(s, run_id: int, thread: str) -> List[Tuple[str, str]]:
    rows = s.execute(select(M.ExplanationTarget, M.Explanation)
                     .join(M.Explanation, M.Explanation.target_id == M.ExplanationTarget.id)
                     .where(M.ExplanationTarget.sim_run_id == run_id, M.ExplanationTarget.mode == 'chat')
                     .order_by(M.ExplanationTarget.id)).all()
    return [(t.question or '', e.text_polished or '') for t, e in rows
            if any(isinstance(q, dict) and q.get('thread') == thread for q in (t.q_t or []))]


def chat(run_id: int, card: Dict, message: str, call: Caller, thread: Optional[str] = None,
         llm: Optional[str] = None, llm_call_id: Callable[[], Optional[int]] = lambda: None,
         url=None) -> Tuple[Optional[Dict], Optional[str]]:
    thread = thread or uuid.uuid4().hex
    with xdb.session_scope(url) as s:
        ctx = run_context(s, run_id)
        history = _history(s, run_id, thread)
    if not ctx['verilog']:
        return None, 'the design of this simulation is not stored'
    facts, lines = chat_facts(ctx, card)
    convo = ''.join(f'\nEngineer: {q}\nYou: {a}\n' for q, a in history[-6:])
    prompt = f"Facts:\n{facts}\n\nConversation so far:{convo or ' (none)'}\n\nEngineer: {message.strip()}\nYou:"
    answer = (call(_CHAT_SYSTEM, prompt) or '').strip()
    proposal = None
    m = re.search(r'```json\s*(\{.*?\})\s*```', answer, re.S)
    if m:
        parsed, _ = parse_changes(m.group(1))
        if parsed and parsed['changes']:
            proposal = parsed
        answer = (answer[:m.start()] + answer[m.end():]).strip()
    call_id = llm_call_id()
    with xdb.session_scope(url) as s:
        t = M.ExplanationTarget(entry='nl', question=message.strip(), subject_type='sim_run', subject_id=run_id,
                                sim_run_id=run_id, p_t=[], scope='local', mode='chat',
                                q_t=[{'thread': thread, 'turn': len(history) + 1,
                                      'ops': card.get('ops') or [], 'title': card.get('title')}])
        s.add(t)
        s.flush()
        s.add(M.Explanation(target_id=t.id, algorithm=CHAT_ALGORITHM, selected=[],
                            structure={'thread': thread, 'turn': len(history) + 1, 'llm': llm,
                                       'proposal': proposal},
                            text_rule=facts, text_polished=answer, polish_llm_call=call_id,
                            evidence={'sim_run_id': run_id, 'lines': [ln['no'] for ln in lines]}))
    return {'thread': thread, 'answer': answer, 'proposal': proposal, 'lines': lines}, None
