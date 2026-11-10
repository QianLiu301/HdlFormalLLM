// Independent check of the ALU reference model (tests/test_reference_vs_verilog.py).
// Written from the Verilog operators themselves, not from the Python rules: the
// simulator decides what >>>, $signed and the carry out mean. Overflow is derived
// differently from src/xdb/library.py (a W+1-bit signed sum), so the two agree only
// if both are right.
module ref_alu #(parameter W = 8) (
    input      [3:0]   op,
    input      [W-1:0] a,
    input      [W-1:0] b,
    output reg [W-1:0] y,
    output             zero,
    output             negative,
    output reg         ovf_signed,   // ADD/SUB result outside the signed W-bit range
    output reg         ovf_carry     // ADD carries out of the top bit, SUB borrows
);
    localparam SW = $clog2(W);
    wire [SW-1:0]       sh   = b[SW-1:0];
    wire signed [W:0]   ssum = $signed(a) + $signed(b);
    wire signed [W:0]   sdif = $signed(a) - $signed(b);
    wire [W:0]          usum = {1'b0, a} + {1'b0, b};
    wire [W:0]          udif = {1'b0, a} - {1'b0, b};

    always @* begin
        ovf_signed = 1'b0;
        ovf_carry  = 1'b0;
        case (op)
            4'd0: begin y = a + b; ovf_signed = (ssum != {y[W-1], y}); ovf_carry = usum[W]; end
            4'd1: begin y = a - b; ovf_signed = (sdif != {y[W-1], y}); ovf_carry = udif[W]; end
            4'd2: y = a & b;
            4'd3: y = a | b;
            4'd4: y = a ^ b;
            4'd5: y = a << sh;
            4'd6: y = a >> sh;
            4'd7: y = $signed(a) >>> sh;
            4'd8: y = {{(W-1){1'b0}}, ($signed(a) < $signed(b))};
            4'd9: y = {{(W-1){1'b0}}, (a < b)};
            default: y = {W{1'b0}};
        endcase
    end
    assign zero     = (y == {W{1'b0}});
    assign negative = y[W-1];
endmodule
