// comparator_8bit — golden reference design
module comparator_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       eq,
    output wire       gt,
    output wire       lt
);
    assign eq = (a == b);
    assign gt = (a < b); // BUG: gt/lt swapped
    assign lt = (a > b); // BUG: gt/lt swapped
endmodule
