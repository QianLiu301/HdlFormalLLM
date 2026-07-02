`timescale 1ns/1ps
module tb_smoke;
    reg clk, rst_n;
    wire red, yellow, green;
    integer i;

    traffic_light dut (.clk(clk), .rst_n(rst_n), .red(red), .yellow(yellow), .green(green));

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst_n = 0;
        @(negedge clk); rst_n = 1;
        // observe three full cycles (3 * (8+3+6) = 51 cycles) plus slack
        for (i = 0; i < 60; i = i + 1) begin
            @(negedge clk) $display("i=%0d r=%b y=%b g=%b", i, red, yellow, green);
        end
        // mid-cycle reset back to RED
        rst_n = 0; @(negedge clk) $display("RESET r=%b y=%b g=%b", red, yellow, green); rst_n = 1;
        for (i = 0; i < 20; i = i + 1) begin
            @(negedge clk) $display("i=%0d r=%b y=%b g=%b", i, red, yellow, green);
        end
        $finish;
    end
endmodule
