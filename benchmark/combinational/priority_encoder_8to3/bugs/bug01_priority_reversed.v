// priority_encoder_8to3 — BUG: priority reversed (bit 0 wins)
module priority_encoder_8to3 (
    input  wire [7:0] in,
    output reg  [2:0] out,
    output wire       valid
);
    always @(*) begin
        casez (in)
            8'b???????1: out = 3'd0; // BUG: lowest bit has highest priority
            8'b??????10: out = 3'd1;
            8'b?????100: out = 3'd2;
            8'b????1000: out = 3'd3;
            8'b???10000: out = 3'd4;
            8'b??100000: out = 3'd5;
            8'b?1000000: out = 3'd6;
            8'b10000000: out = 3'd7;
            default:     out = 3'd0;
        endcase
    end
    assign valid = (in != 8'h00);
endmodule
