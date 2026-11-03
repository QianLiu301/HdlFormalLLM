"""
Who is wrong? Classify one evaluated assertion from three values:

    O  observed output of the design
    B  expected value written in the BDD scenario
    R  expected value computed by the tool's reference model

    O = B = R        pass
    O != B, O = R    scenario_wrong     the design is right, the scenario's value is not
    O != B, B = R    design_wrong       the scenario is right, the design is not
    all different    both_wrong
    O = B != R       consistent_wrong   test passes, but scenario and design deviate the
                                        same way (typical when the design was generated
                                        from the scenario: specification-first workflow)
    R or B missing   undecided

All values are compared as unsigned integers masked to the port width, which
is how the testbench compares them. A non-numeric observation (x / z) never
equals a number.
"""

from typing import Optional, Union

Value = Optional[Union[int, str]]


def _norm(v: Value, width: Optional[int]):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        try:
            v = int(s, 0)
        except ValueError:
            return s.lower()           # 'x', 'z', ... kept as text
    v = int(v)
    if width:
        v &= (1 << width) - 1
    return v


def classify(observed: Value, bdd: Value, ref: Value, width: Optional[int] = None) -> str:
    o, b, r = (_norm(x, width) for x in (observed, bdd, ref))
    if o is None or b is None or r is None:
        return 'undecided'
    if o == b:
        return 'pass' if b == r else 'consistent_wrong'
    if o == r:
        return 'scenario_wrong'
    if b == r:
        return 'design_wrong'
    return 'both_wrong'


def observed_value(status: str, got: Value, expected_used: Value, width: Optional[int] = None):
    """O: the log prints the observed value only on FAIL; on PASS it equals what
    the testbench compared against."""
    if status == 'fail':
        return _norm(got, width)
    if status == 'pass':
        return _norm(expected_used, width)
    return None
