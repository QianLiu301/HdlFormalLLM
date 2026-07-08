// comparator_8bit — MUTANT (bug04_eq_ignores_bit0.v) — eq ignores bit0: values differing only in LSB compare equal
module comparator_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       eq,
    output wire       gt,
    output wire       lt
);
    assign eq = (((a ^ b) & 8'hFE) == 8'h00);  // BUG: bit0 masked
    assign gt = (a > b);
    assign lt = (a < b);
endmodule
