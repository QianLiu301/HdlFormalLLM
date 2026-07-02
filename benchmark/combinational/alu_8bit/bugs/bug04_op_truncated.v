// alu_8bit — golden reference design
module alu_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire [1:0] op, // BUG: opcode truncated to 2 bits
    output reg  [7:0] result,
    output reg        carry_out,
    output wire       zero
);
    always @(*) begin
        carry_out = 1'b0;
        result    = 8'h00;
        case ({1'b0, op})
            3'b000: {carry_out, result} = {1'b0, a} + {1'b0, b};
            3'b001: {carry_out, result} = {1'b0, a} - {1'b0, b};
            3'b010: result = a & b;
            3'b011: result = a | b;
            3'b100: result = a ^ b;
            3'b101: result = ~a;
            3'b110: {carry_out, result} = {a, 1'b0};
            3'b111: begin result = a >> 1; carry_out = a[0]; end
        endcase
    end
    assign zero = (result == 8'h00);
endmodule
