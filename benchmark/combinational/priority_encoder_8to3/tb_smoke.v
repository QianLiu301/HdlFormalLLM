`timescale 1ns/1ps
module tb_smoke;
    reg  [7:0] in;
    wire [2:0] out;
    wire valid;
    integer i;

    priority_encoder_8to3 dut (.in(in), .out(out), .valid(valid));

    initial begin
        // exhaustive: all 256 inputs
        for (i = 0; i < 256; i = i + 1) begin
            in = i[7:0];
            #1 $display("in=%h out=%d v=%b", in, out, valid);
        end
        $finish;
    end
endmodule
