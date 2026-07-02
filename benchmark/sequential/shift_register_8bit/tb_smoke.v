`timescale 1ns/1ps
module tb_smoke;
    reg clk, rst_n, load, shift_en, dir, serial_in;
    reg [7:0] d;
    wire [7:0] q;
    wire serial_out;
    integer i;

    shift_register_8bit dut (.clk(clk), .rst_n(rst_n), .load(load), .d(d),
                             .shift_en(shift_en), .dir(dir),
                             .serial_in(serial_in), .q(q), .serial_out(serial_out));

    always #5 clk = ~clk;

    task show; begin
        $display("t=%0t ld=%b se=%b dir=%b si=%b d=%h q=%h so=%b", $time, load, shift_en, dir, serial_in, d, q, serial_out);
    end endtask

    initial begin
        clk = 0; rst_n = 0; load = 0; shift_en = 0; dir = 0; serial_in = 0; d = 8'h00;
        @(negedge clk) show; rst_n = 1;
        // load pattern, shift left 8 times with serial_in=1
        load = 1; d = 8'hA5; @(negedge clk) show; load = 0;
        shift_en = 1; dir = 0; serial_in = 1;
        for (i = 0; i < 8; i = i + 1) begin @(negedge clk) show; end
        // shift right 8 times with serial_in=0
        load = 1; d = 8'hA5; @(negedge clk) show; load = 0;
        dir = 1; serial_in = 0;
        for (i = 0; i < 8; i = i + 1) begin @(negedge clk) show; end
        // load priority: both load & shift_en
        load = 1; shift_en = 1; d = 8'h3C; @(negedge clk) show; load = 0;
        @(negedge clk) show;
        // hold
        shift_en = 0; @(negedge clk) show; @(negedge clk) show;
        // mixed pseudo-random
        for (i = 0; i < 40; i = i + 1) begin
            {load, shift_en, dir, serial_in} = $random; d = $random;
            @(negedge clk) show;
        end
        $finish;
    end
endmodule
