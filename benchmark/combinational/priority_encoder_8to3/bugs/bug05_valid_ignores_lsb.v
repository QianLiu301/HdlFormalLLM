// priority_encoder_8to3 — MUTANT (bug05_valid_ignores_lsb.v) — valid ignores in[0]; in==0x01 reported as invalid
module priority_encoder_8to3 (
    input  wire [7:0] in,
    output reg  [2:0] out,
    output wire       valid
);
    always @(*) begin
        casez (in)
            8'b1???????: out = 3'd7;
            8'b01??????: out = 3'd6;
            8'b001?????: out = 3'd5;
            8'b0001????: out = 3'd4;
            8'b00001???: out = 3'd3;
            8'b000001??: out = 3'd2;
            8'b0000001?: out = 3'd1;
            8'b00000001: out = 3'd0;
            default:     out = 3'd0;
        endcase
    end
    assign valid = (in[7:1] != 7'h00);  // BUG: LSB not counted
endmodule
