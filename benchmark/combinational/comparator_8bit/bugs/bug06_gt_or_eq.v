// comparator_8bit — MUTANT (bug06_gt_or_eq.v) — gt implemented as >=; gt and eq both high at equality
module comparator_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       eq,
    output wire       gt,
    output wire       lt
);
    assign eq = (a == b);
    assign gt = (a >= b);  // BUG: boundary included
    assign lt = (a < b);
endmodule
