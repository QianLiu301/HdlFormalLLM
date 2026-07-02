// comparator_8bit — golden reference design
module comparator_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       eq,
    output wire       gt,
    output wire       lt
);
    assign eq = (a == b);
    assign gt = ($signed(a) > $signed(b)); // BUG: signed comparison
    assign lt = (a < b);
endmodule
