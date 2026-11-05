"""What goes back to the model when a design does not compile (D30):
main._compile_report turns iverilog's messages into short, unique lines."""
import contextlib
import io
import shutil

import pytest

pytestmark = pytest.mark.skipif(shutil.which('iverilog') is None, reason='iverilog not installed')

# the error class of the first real trial: wires assigned in an always block,
# each use reported again with the same "declared here" line
WIRES_IN_ALWAYS = """module m(input wire clk, input wire [1:0] sel, output wire [7:0] y);
    wire [7:0] a;
    wire [7:0] b;
    always @(*) begin
        case (sel)
            2'd0: a = 8'd1;
            2'd1: a = 8'd2;
            default: a = 8'd0;
        endcase
        b = a;
    end
    assign y = b;
endmodule
"""


def _report(tmp_path, code, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    f = tmp_path / 'design.v'
    f.write_text(code, encoding='utf-8')
    return main._compile_report(f, **kw)


def test_lines_are_unique_and_name_the_design(tmp_path):
    lines, err = _report(tmp_path, WIRES_IN_ALWAYS)
    assert err is None and lines
    assert len(lines) == len(set(lines))
    assert all(ln.startswith('design line ') for ln in lines), lines
    assert not any(': :' in ln or str(tmp_path) in ln for ln in lines), lines
    # the wording depends on the iverilog version
    assert any(ln.startswith('design line 2: ') and 'declared here as' in ln for ln in lines), lines
    assert {ln.split('error: ')[1].split(' ')[0].strip("'") for ln in lines if 'error: ' in ln} == {'a', 'b'}


def test_long_reports_say_how_much_was_left_out(tmp_path):
    full, _ = _report(tmp_path, WIRES_IN_ALWAYS)
    short, _ = _report(tmp_path, WIRES_IN_ALWAYS, limit=2)
    assert short == full[:2] + [f'... and {len(full) - 2} more error lines']


def test_a_design_that_compiles_has_no_lines(tmp_path):
    assert _report(tmp_path, 'module ok(input wire a, output wire y); assign y = a; endmodule\n') == ([], None)
