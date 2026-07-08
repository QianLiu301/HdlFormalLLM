// comparator_8bit — MUTANT (bug05_lt_off_by_one.v) — lt implemented as <=; lt and eq both high at equality
module comparator_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       eq,
    output wire       gt,
    output wire       lt
);
    assign eq = (a == b);
    assign gt = (a > b);
    assign lt = (a <= b);  // BUG: boundary included
endmodule
