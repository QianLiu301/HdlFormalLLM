`timescale 1ns/1ps
module tb_smoke;
    reg clk, rst_n, load, en, up_down;
    reg [7:0] d;
    wire [7:0] q;
    wire overflow;
    integer i;

    counter_8bit dut (.clk(clk), .rst_n(rst_n), .load(load), .d(d),
                      .en(en), .up_down(up_down), .q(q), .overflow(overflow));

    always #5 clk = ~clk;

    task show; begin
        $display("t=%0t ld=%b en=%b ud=%b d=%h q=%h ov=%b", $time, load, en, up_down, d, q, overflow);
    end endtask

    initial begin
        clk = 0; rst_n = 0; load = 0; en = 0; up_down = 1; d = 8'h00;
        @(negedge clk) show; rst_n = 1;
        // count up from 0 for 5 cycles
        en = 1; up_down = 1;
        for (i = 0; i < 5; i = i + 1) begin @(negedge clk) show; end
        // load 0xFE then count up across the wrap
        load = 1; d = 8'hFE; @(negedge clk) show; load = 0;
        for (i = 0; i < 4; i = i + 1) begin @(negedge clk) show; end
        // hold
        en = 0; @(negedge clk) show; @(negedge clk) show;
        // load 0x01 then count down across the wrap
        load = 1; d = 8'h01; @(negedge clk) show; load = 0;
        en = 1; up_down = 0;
        for (i = 0; i < 4; i = i + 1) begin @(negedge clk) show; end
        // load priority over en (both high)
        load = 1; d = 8'h55; @(negedge clk) show; load = 0;
        @(negedge clk) show;
        // mid-operation async reset (away from clock edge)
        #2 rst_n = 0; #1 show; rst_n = 1;
        @(negedge clk) show;
        // random-ish mixed run
        for (i = 0; i < 40; i = i + 1) begin
            {load, en, up_down} = $random; d = $random;
            @(negedge clk) show;
        end
        // overflow pulse must clear even while counting is disabled:
        // wrap at 0xFF -> 0x00 (ov=1), then en=0 and watch ov drop
        load = 1; en = 0; up_down = 1; d = 8'hFF; @(negedge clk) show; load = 0;
        en = 1; @(negedge clk) show;   // wrap: q=00, ov=1
        en = 0; @(negedge clk) show;   // golden: ov=0; sticky mutant: ov=1
        @(negedge clk) show;
        $finish;
    end
endmodule
