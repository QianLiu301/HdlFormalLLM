// comparator_8bit — MUTANT (bug07_signed_lt.v) — lt uses signed comparison; wrong when exactly one operand has MSB set
module comparator_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       eq,
    output wire       gt,
    output wire       lt
);
    assign eq = (a == b);
    assign gt = (a > b);
    assign lt = ($signed(a) < $signed(b));  // BUG: signed compare
endmodule
