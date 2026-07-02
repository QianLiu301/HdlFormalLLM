`timescale 1ns/1ps
module tb_smoke;
    reg clk, rst_n, din;
    wire detected;
    integer i;
    reg [31:0] stream;

    seq_detector_1011 dut (.clk(clk), .rst_n(rst_n), .din(din), .detected(detected));

    always #5 clk = ~clk;

    task feed(input b); begin
        din = b;
        @(negedge clk) $display("t=%0t din=%b det=%b", $time, din, detected);
    end endtask

    initial begin
        clk = 0; rst_n = 0; din = 0;
        @(negedge clk); rst_n = 1;
        // basic match: 1011
        feed(1); feed(0); feed(1); feed(1);
        // overlap: continue 011 -> second match (1011011 total)
        feed(0); feed(1); feed(1);
        // non-match fallback: 1010
        feed(1); feed(0); feed(1); feed(0);
        // run of ones then 011: 111011 -> one match
        feed(1); feed(1); feed(1); feed(0); feed(1); feed(1);
        // reset forgets partial match "101"
        feed(1); feed(0); feed(1);
        rst_n = 0; @(negedge clk) $display("t=%0t RESET det=%b", $time, detected); rst_n = 1;
        feed(1);  // would complete 1011 if history survived reset
        // pseudo-random stream
        stream = $random;
        for (i = 0; i < 64; i = i + 1) begin
            feed(stream[i % 32]);
            if (i % 32 == 31) stream = $random;
        end
        $finish;
    end
endmodule
