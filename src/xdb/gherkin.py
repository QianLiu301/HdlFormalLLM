"""
Minimal Gherkin reader for what the explainability database needs:
steps (with their role in the BDX sense) and Examples rows.

BDX (arXiv 2607.24881, §IV-C) treats a scenario as an implication P -> Q:
Given and When steps form the antecedent P, Then steps the consequent Q.
And / But / * inherit the role of the step before them.

Row numbering follows the testbench generator's IDs (ALU-ADD-001/r3 is the
third data row of the first Examples table, counting every table of the
scenario in order).
"""

import re
from typing import Dict, List, Optional

_STEP_RE = re.compile(r'^\s*(Given|When|Then|And|But|\*)\s+(.*\S)\s*$')
_SCEN_RE = re.compile(r'^\s*(Scenario Outline|Scenario Template|Scenario|Example)\s*:\s*(.*)$')
_BLOCK_END_RE = re.compile(r'^\s*(Scenario Outline|Scenario Template|Scenario|Example|'
                           r'Background|Rule|Feature)\s*:|^\s*@')
_ROLE = {'Given': 'P', 'When': 'P', 'Then': 'Q'}


def _cells(line: str) -> List[str]:
    # Same rule as testbench_generator._table_cells (empty cells kept in place),
    # so a row the generator counts as malformed is malformed here too.
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in s.split('|')]


def parse_steps(lines: List[str]) -> List[Dict]:
    steps, last_role = [], 'P'
    for line in lines:
        m = _STEP_RE.match(line)
        if not m:
            continue
        kw, body = m.group(1), m.group(2)
        role = _ROLE.get(kw, last_role)
        last_role = role
        steps.append({'keyword': kw, 'role': role, 'text': body})
    return steps


def parse_scenario(block: str) -> Dict:
    """Parse one scenario block (tags, header, steps, Examples)."""
    lines = block.splitlines()
    title, is_outline, tags = None, False, []
    step_lines, rows, header, in_examples = [], [], None, False
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('@') and title is None:
            tags += [t[1:] for t in s.split() if t.startswith('@')]
            continue
        m = _SCEN_RE.match(line)
        if m and title is None:
            is_outline = m.group(1) in ('Scenario Outline', 'Scenario Template')
            title = m.group(2).strip()
            continue
        if re.match(r'^(Examples|Scenarios)\s*:', s):
            in_examples, header = True, None
            continue
        if in_examples and s.startswith('|'):
            cells = _cells(s)
            if header is None:
                header = cells
            else:
                rows.append({'row_no': len(rows) + 1,
                             'vals': dict(zip(header, cells)) if len(cells) == len(header)
                             else {'_raw': cells, '_header': header},
                             'malformed': len(cells) != len(header)})
            continue
        if not in_examples:
            step_lines.append(line)
    return {'title': title, 'is_outline': is_outline, 'tags': tags,
            'steps': parse_steps(step_lines), 'rows': rows}


def background_steps(feature_text: str) -> List[Dict]:
    """Steps of the Feature's Background block (shared context of every scenario)."""
    lines = feature_text.splitlines()
    out, inside = [], False
    for line in lines:
        if re.match(r'^\s*Background\s*:', line):
            inside = True
            continue
        if inside and _BLOCK_END_RE.match(line):
            break
        if inside:
            out.append(line)
    return parse_steps(out)


def feature_title(feature_text: str) -> Optional[str]:
    m = re.search(r'^\s*Feature\s*:\s*(.+)$', feature_text, flags=re.M)
    return m.group(1).strip() if m else None
