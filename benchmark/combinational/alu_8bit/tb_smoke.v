// Deterministic smoke stimulus for alu_8bit (used by validate.py for differential testing)
`timescale 1ns/1ps
module tb_smoke;
    reg  [7:0] a, b;
    reg  [2:0] op;
    wire [7:0] result;
    wire carry_out, zero;
    integer i, j, k;
    reg [7:0] corners [0:4];

    alu_8bit dut (.a(a), .b(b), .op(op), .result(result),
                  .carry_out(carry_out), .zero(zero));

    initial begin
        corners[0]=8'h00; corners[1]=8'h01; corners[2]=8'h7F;
        corners[3]=8'h80; corners[4]=8'hFF;
        for (i = 0; i < 5; i = i + 1)
            for (j = 0; j < 5; j = j + 1)
                for (k = 0; k < 8; k = k + 1) begin
                    a = corners[i]; b = corners[j]; op = k[2:0];
                    #1 $display("a=%h b=%h op=%b r=%h c=%b z=%b", a, b, op, result, carry_out, zero);
                end
        for (i = 0; i < 200; i = i + 1) begin
            a = $random; b = $random; op = $random;
            #1 $display("a=%h b=%h op=%b r=%h c=%b z=%b", a, b, op, result, carry_out, zero);
        end
        $finish;
    end
endmodule
