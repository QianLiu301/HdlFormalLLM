"""
Requirements confirmation (web Step 0).

One LLM call splits the user's design description into atomic requirements
and lists the points it leaves open, each with concrete options and a
preselected default (an unchanged standard description needs no call, see
design_templates.py). The user edits the list and picks answers; the result
is saved as one JSON file under output/requirements/. Step 1 (DUV) and
Step 2 (BDD) both receive the path, read the same file, and append the same
text to their prompts (prompt_store.decisions_block), so the two generators
can no longer resolve an ambiguity in two different ways.

The benchmark runners never send a requirements file, so their prompts are
unchanged.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_REQUIREMENTS = 30
MAX_QUESTIONS = 8

# Questions the tool always asks for a module type: the generated design and
# the generated scenarios must agree on them, and descriptions rarely say.
# The wording is fixed because the explainability rule library reads the
# answer back (src/xdb/library.py, overflow_mode_from_answer).
OVERFLOW_QUESTION = 'What does the overflow flag mean?'
OVERFLOW_SIGNED = ("Signed overflow: the ADD/SUB result does not fit in W-bit two's complement "
                   "(e.g. 8-bit 127 + 1)")
OVERFLOW_CARRY = ('Unsigned carry/borrow: ADD carries out of the top bit, SUB borrows when A < B '
                  '(e.g. 8-bit 255 + 1)')
FIXED_QUESTIONS = {
    'alu': [{'topic': 'overflow', 'question': OVERFLOW_QUESTION,
             'options': [OVERFLOW_SIGNED, OVERFLOW_CARRY], 'default': 0,
             'why': ('"Overflow" is used for both. The design and the scenarios are generated '
                     'separately, so they must be told the same meaning. The preselected answer '
                     "is what the tool's design template uses."),
             'fixed': True}],
}
FIXED_QUESTIONS['alu_comb'] = FIXED_QUESTIONS['alu']

# Points the reference model has already fixed. They are not choices: the verdicts
# use these rules (TestbenchGenerator._reference_result, src/xdb/library.py), so an
# answer that differs could only be reported as wrong. Step 0 shows them, drops the
# model's questions about them, and both generators receive them. Keep in step with
# those rules. A question naming these operations is about them: in 21 logged ALU
# analyses every question on a shift or on SLT/SLTU was settled by the rules.
SETTLED_TOPICS = {'alu': re.compile(r'\b(SLL|SRL|SRA|SLTU?)\b|\bshift', re.I)}
SETTLED_TOPICS['alu_comb'] = SETTLED_TOPICS['alu']


def settled_facts(module_type: Optional[str], bitwidth=None) -> List[Dict]:
    """What the reference model fixes for this module type, [{id, fact}]."""
    if (module_type or '').strip() not in SETTLED_TOPICS:
        return []
    try:
        w = int(bitwidth or 32)
    except (TypeError, ValueError):
        w = 32
    power_of_two = w > 0 and w & (w - 1) == 0
    amount = f'the low {w.bit_length() - 1} bits of B' if power_of_two else f'B masked with {w - 1}'
    facts = [f'A shift moves by {amount}; the other bits of B are ignored.',
             'SLL and SRL fill with zeros; SRA copies the sign bit of A.',
             'SLT compares as signed numbers, SLTU as unsigned. Both are strict, so A == B gives 0, '
             'and the result is 0 or 1.',
             f'ADD and SUB wrap to {w} bits.',
             'zero is 1 when result is 0, and negative is the top bit of result, for every operation.']
    return [{'id': f'F{i}', 'fact': f} for i, f in enumerate(facts, 1)]


def settled_lines(module_type: Optional[str], bitwidth=None) -> str:
    """The same facts as text for the analysis prompt."""
    return '\n'.join(f"- {f['fact']}" for f in settled_facts(module_type, bitwidth)) \
        or '(nothing is fixed for this module type)'


def add_settled(analysis: Dict, module_type: Optional[str], bitwidth=None) -> Dict:
    """Drop the model's questions about settled points and list the facts instead."""
    facts = settled_facts(module_type, bitwidth)
    if not facts:
        return analysis
    topic = SETTLED_TOPICS[(module_type or '').strip()]
    qs = [q for q in analysis.get('questions') or [] if not topic.search(q.get('question') or '')]
    for i, q in enumerate(qs, 1):
        q['id'] = f'Q{i}'
    return {**analysis, 'questions': qs, 'settled': facts}


def add_fixed_questions(analysis: Dict, module_type: Optional[str]) -> Dict:
    """Append the tool's fixed questions for this module type. A question the
    model asked about the same topic is dropped: the fixed wording wins, so
    the answer can be read back. Ids are renumbered Q1..Qn."""
    fixed = FIXED_QUESTIONS.get((module_type or '').strip(), [])
    if not fixed:
        return analysis
    qs = list(analysis.get('questions') or [])
    for fq in fixed:
        qs = [q for q in qs if fq['topic'] not in (q.get('question') or '').lower()]
    qs += [{k: v for k, v in fq.items() if k != 'topic'} for fq in fixed]
    for i, q in enumerate(qs, 1):
        q['id'] = f'Q{i}'
    return {**analysis, 'questions': qs}


def parse_analysis(text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Pull the JSON object out of an LLM reply and normalise it.

    Returns (analysis, error). Models wrap JSON in fences or add a sentence
    despite being told not to, so take the outermost {...} rather than
    requiring the reply to be pure JSON.
    """
    if not text or not text.strip():
        return None, 'empty response from the model'
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.M)
    start, end = s.find('{'), s.rfind('}')
    if start < 0 or end <= start:
        return None, 'no JSON object found in the model response'
    repaired = False
    try:
        raw = json.loads(s[start:end + 1])
    except ValueError as e:
        # Models without a JSON mode sometimes drop a comma or leave a quote
        # unescaped. Repair the syntax; the user reviews every requirement and
        # answer before saving, and the page says the reply was repaired.
        try:
            import json_repair
            raw = json_repair.loads(s[start:end + 1])
        except Exception:
            raw = None
        if not isinstance(raw, dict) or not raw:
            return None, f'model returned invalid JSON: {e}'
        repaired = True
    if not isinstance(raw, dict):
        return None, 'model returned JSON that is not an object'

    reqs = []
    for i, r in enumerate(raw.get('requirements') or [], 1):
        t = r.get('text') if isinstance(r, dict) else r
        if isinstance(t, str) and t.strip():
            reqs.append({'id': f'REQ-{len(reqs) + 1}', 'text': t.strip(), 'source': 'llm'})

    qs = []
    for q in raw.get('questions') or []:
        if not isinstance(q, dict):
            continue
        text_q = (q.get('question') or '').strip()
        opts = [str(o).strip() for o in (q.get('options') or []) if str(o).strip()]
        if not text_q or len(opts) < 2:
            continue                      # 没有可选项的问题无法回答，丢弃
        try:
            d = int(q.get('default', 0))
        except (TypeError, ValueError):
            d = 0
        qs.append({'id': f'Q{len(qs) + 1}', 'question': text_q, 'options': opts[:4],
                   'default': d if 0 <= d < len(opts[:4]) else 0,
                   'why': (q.get('why') or '').strip()})

    if not reqs and not qs:
        return None, 'the model returned no requirements and no questions'
    return {'requirements': reqs[:MAX_REQUIREMENTS], 'questions': qs[:MAX_QUESTIONS],
            'json_repaired': repaired}, None


def build_record(payload: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """Validate what the browser sends back after the user confirmed.

    Decisions are rebuilt here from questions + chosen option index rather
    than trusted as free text, so every saved answer is one of the offered
    options (or an explicit custom answer marked as such).
    """
    reqs: List[Dict] = []
    for r in payload.get('requirements') or []:
        t = (r.get('text') or '').strip() if isinstance(r, dict) else ''
        if not t:
            continue
        src = r.get('source') if r.get('source') in ('llm', 'edited', 'user', 'template') else 'user'
        reqs.append({'id': f'REQ-{len(reqs) + 1}', 'text': t, 'source': src})

    questions, decisions = [], []
    for q in payload.get('questions') or []:
        if not isinstance(q, dict):
            continue
        opts = [str(o) for o in (q.get('options') or [])]
        qtext = (q.get('question') or '').strip()
        if not qtext or not opts:
            continue
        qid = f'Q{len(questions) + 1}'
        default = q.get('default', 0) if isinstance(q.get('default'), int) else 0
        questions.append({'id': qid, 'question': qtext, 'options': opts,
                          'default': default, 'why': q.get('why') or ''})
        custom = (q.get('custom') or '').strip()
        chosen = q.get('chosen')
        if custom:
            answer, source = custom, 'custom'
        elif isinstance(chosen, int) and 0 <= chosen < len(opts):
            answer = opts[chosen]
            source = 'default' if chosen == default else 'user'
        else:
            return None, f'question {qtext!r} has no answer'
        decisions.append({'id': f'D{len(decisions) + 1}', 'question_id': qid,
                          'question': qtext, 'answer': answer, 'source': source})

    if not reqs and not decisions:
        return None, 'nothing to save: no requirements and no decisions'
    if len(reqs) > MAX_REQUIREMENTS:
        return None, f'too many requirements (max {MAX_REQUIREMENTS})'

    return {
        'version': 1,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'module_type': payload.get('module_type'),
        'config': payload.get('config') or {},
        'description': payload.get('description') or '',
        'analysis_source': payload.get('source') or 'llm',     # template: no model was involved
        'analysis_llm': payload.get('llm'),
        'analysis_model': payload.get('model'),
        'accepted_all_defaults': bool(payload.get('accepted_all_defaults')),
        'requirements': reqs,
        'questions': questions,
        'decisions': decisions,
        # from the rules, not from the page: this is what the verdicts will use
        'settled': settled_facts(payload.get('module_type'), (payload.get('config') or {}).get('bitwidth')),
    }, None


def save(record: Dict, project_root) -> Path:
    out = Path(project_root) / 'output' / 'requirements'
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = out / f"req_{record.get('module_type') or 'design'}_{stamp}_{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


def load(filepath, project_root) -> Tuple[Optional[Dict], Optional[str]]:
    """Read a confirmed record. Only files under output/requirements/ are accepted."""
    root = (Path(project_root) / 'output' / 'requirements').resolve()
    p = Path(filepath)
    if not p.is_absolute():
        p = Path(project_root) / p
    try:
        p = p.resolve()
    except OSError as e:
        return None, f'bad requirements path: {e}'
    if root not in p.parents:
        return None, f'requirements file must be under output/requirements/: {filepath}'
    if not p.exists():
        return None, f'requirements file not found: {filepath}'
    try:
        return json.loads(p.read_text(encoding='utf-8')), None
    except (OSError, ValueError) as e:
        return None, f'failed to read requirements file {filepath}: {e}'
