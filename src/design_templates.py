"""
The standard description of each design type, and what follows from it.

Step 0 fills in these descriptions (static/bdd_generator.html, BDD_TEMPLATES, must
stay the same text; a test compares them). When the user keeps one unchanged there
is nothing for a model to interpret, so the requirements and questions come from
here: instantly, with no model call, and the same every time. Only a description
the user wrote or edited goes to a model.

The texts agree with the design templates (prompts/duv_*) and, for the ALU, with
the reference model. The questions are the real open points found in the logged
analyses; what a design template or the reference already decides is not asked.
"""

from typing import Dict, List, Optional


def _size(config: Optional[Dict], key: str, default: int) -> int:
    try:
        return int((config or {}).get(key) or default)
    except (TypeError, ValueError):
        return default


def _alu_description(w: int, _depth: int, _stages: int) -> str:
    return f"""{w}-bit ALU with 4-bit opcode selecting:
- ADD  (opcode 0000): A + B
- SUB  (opcode 0001): A - B
- AND  (opcode 0010): A & B
- OR   (opcode 0011): A | B
- XOR  (opcode 0100): A ^ B
- SLL  (opcode 0101): A << B, shift amount is the low {w.bit_length() - 1} bits of B
- SRL  (opcode 0110): A >> B, zero-filled
- SRA  (opcode 0111): A >>> B, sign-extended
- SLT  (opcode 1000): 1 if A < B as signed values, else 0
- SLTU (opcode 1001): 1 if A < B as unsigned values, else 0
- Zero flag: high when result is 0, for every operation
- Overflow flag, meaningful for ADD and SUB
- Boundary value tests (0, 1, max, min, all-ones)
- Contrast SRL against SRA on a negative operand
- Contrast SLT against SLTU on the same operand pair"""


def _counter_description(w: int, _depth: int, _stages: int) -> str:
    return f"""{w}-bit Counter with:
- UP mode (increment)
- DOWN mode (decrement)
- UP-DOWN mode (ping-pong)
- Load preset value
- Enable control
- Overflow flag
- Zero flag"""


def _regfile_description(w: int, depth: int, _stages: int) -> str:
    return f"""{depth}x{w}-bit Register File with:
- Dual read ports
- Single write port
- x0 always returns 0
- Concurrent read/write test"""


def _cpu_description(_w: int, _depth: int, stages: int) -> str:
    return f"""{stages}-stage RISC-V CPU (RV32I subset) with:
- R-type: ADD, SUB, AND, OR, XOR, SLT
- I-type: ADDI, ANDI, ORI, XORI, SLTI, LW
- S-type: SW
- B-type: BEQ, BNE, BLT, BGE
- J-type: JAL, JALR
- Forwarding: EX/MEM and MEM/WB to EX
- Load-use hazard: stall one cycle after LW
- Control hazard: flush the pipeline on a taken branch or jump
- Signed comparison: contrast SLT against unsigned interpretation
- x0 stays zero on any write"""


def _alu_requirements(w: int, _depth: int, _stages: int) -> List[str]:
    amount = f'B[{w.bit_length() - 2}:0]'
    return [f'ADD (opcode 0000): result = A + B, wrapped to {w} bits.',
            f'SUB (opcode 0001): result = A - B, wrapped to {w} bits.',
            'AND (opcode 0010): result = A & B.',
            'OR (opcode 0011): result = A | B.',
            'XOR (opcode 0100): result = A ^ B.',
            f'SLL (opcode 0101): result = A shifted left by {amount}, filled with zeros.',
            f'SRL (opcode 0110): result = A shifted right by {amount}, filled with zeros.',
            f'SRA (opcode 0111): result = A shifted right by {amount}, filled with the sign bit of A.',
            'SLT (opcode 1000): result = 1 if A < B as signed numbers, else 0.',
            'SLTU (opcode 1001): result = 1 if A < B as unsigned numbers, else 0.',
            f'zero is 1 when result is 0, and negative is result[{w - 1}], for every operation.',
            'overflow follows the meaning chosen in Step 0 for ADD and SUB, and is 0 for every other operation.',
            'On reset (rst = 1), result, zero, overflow and negative are 0 after the next rising clock edge.']


def _counter_requirements(_w: int, _depth: int, _stages: int) -> List[str]:
    return ['When rst_n is low, count and overflow become 0.',
            'When load is high, count becomes load_value at the rising clock edge, whatever enable and mode are. '
            'rst_n has priority over load.',
            'When enable and load are low, count keeps its value.',
            'UP mode: count increases by 1 at each enabled rising clock edge and wraps from the maximum to 0.',
            'DOWN mode: count decreases by 1 at each enabled rising clock edge and wraps from 0 to the maximum.',
            'UP-DOWN mode: count moves between 0 and the maximum, reversing direction at each end.',
            'overflow is 1 when count wraps in UP or DOWN mode and at both ends in UP-DOWN mode, otherwise 0.',
            'zero is 1 whenever count is 0, in the same cycle.']


def _regfile_requirements(_w: int, _depth: int, _stages: int) -> List[str]:
    return ['Both read ports can read any register at the same time, with no clock delay.',
            'A write takes effect at the rising clock edge when wen is 1.',
            'A read returns the value last written to that register.',
            'Register 0 always reads as 0, and writes to it are ignored.',
            'When rst_n is low, every register becomes 0.',
            'A read of the register being written in the same cycle returns the old value.']


def _cpu_requirements(_w: int, _depth: int, _stages: int) -> List[str]:
    return ['R-type ADD, SUB, AND, OR, XOR and SLT write their result to rd.',
            'I-type ADDI, ANDI, ORI, XORI and SLTI compute with rs1 and the sign-extended immediate, '
            'and write the result to rd.',
            'LW loads the word at address rs1 + imm into rd.',
            'SW stores rs2 at address rs1 + imm.',
            'BEQ, BNE, BLT and BGE move the PC to PC + imm when the comparison holds, and fall through otherwise.',
            'JAL writes PC + 4 to rd and moves the PC to PC + imm.',
            'JALR writes PC + 4 to rd and moves the PC to rs1 + imm.',
            'Results in EX/MEM and MEM/WB are forwarded to EX, so a dependent instruction needs no stall, '
            'except after LW.',
            'An instruction right after LW that reads the loaded register stalls one cycle.',
            'A taken branch or jump flushes the younger instructions, so they never write anything.',
            'SLT, SLTI, BLT and BGE compare as signed numbers.',
            'x0 always reads as 0, and any write to it is ignored.']


_COUNTER_QUESTIONS = [
    {'question': 'In UP-DOWN mode, which way does count move after reset or load?',
     'options': ['Up on the first enabled edge after reset or load',
                 'Down on the first enabled edge after reset or load',
                 'Up after reset; a load keeps the direction it had before'],
     'default': 0, 'why': 'The first expected count after reset or load differs.'},
    {'question': 'How long does the overflow flag stay high?',
     'options': ['One clock cycle for each wrap or reversal', 'Until the next load or reset'],
     'default': 0, 'why': 'A test that checks overflow one cycle later expects a different value.'},
    {'question': 'In UP-DOWN mode, how long does count stay at an end value?',
     'options': ['It reverses on the next enabled edge, with no extra cycle',
                 'It stays for one extra enabled cycle, then reverses'],
     'default': 0, 'why': 'The count sequence around 0 and the maximum differs by one cycle.'},
]

_TEMPLATES = {
    'alu': (_alu_description, _alu_requirements, []),
    'counter': (_counter_description, _counter_requirements, _COUNTER_QUESTIONS),
    'regfile': (_regfile_description, _regfile_requirements, []),
    'cpu': (_cpu_description, _cpu_requirements, []),
}


def _sizes(config: Optional[Dict]) -> tuple:
    return _size(config, 'bitwidth', 32), _size(config, 'depth', 32), _size(config, 'pipeline_stages', 5)


def description(module_type: Optional[str], config: Optional[Dict] = None) -> Optional[str]:
    """The standard description of this design, or None for an unknown type."""
    t = _TEMPLATES.get((module_type or '').strip())
    return t[0](*_sizes(config)) if t else None


def standard_analysis(module_type: Optional[str], config: Optional[Dict], text: str) -> Optional[Dict]:
    """Requirements and questions of an unchanged standard description, else None."""
    standard = description(module_type, config)
    if standard is None or (text or '').strip() != standard.strip():
        return None
    _, requirements, questions = _TEMPLATES[(module_type or '').strip()]
    return {'requirements': [{'id': f'REQ-{i}', 'text': t, 'source': 'template'}
                             for i, t in enumerate(requirements(*_sizes(config)), 1)],
            'questions': [{**q, 'id': f'Q{i}'} for i, q in enumerate(questions, 1)],
            'json_repaired': False, 'source': 'template'}
