`timescale 1ns/1ps
// 定向探针：验证 golden 1011 检测器 (a) Mealy 同拍脉冲 (b) 1011011 重叠触发两次
module tb;
    reg clk = 0, rst_n = 0, din = 0;
    wire detected;
    seq_detector_1011 dut(.clk(clk), .rst_n(rst_n), .din(din), .detected(detected));
    always #5 clk = ~clk;
    integer errors = 0;
    integer i;
    reg [6:0] stream = 7'b1011011;   // 送入顺序：bit6 -> bit0（最老的在前）
    reg [6:0] expect_det = 7'b0001001; // 第 4 位和第 7 位（Mealy 同拍）

    task chk(input cond, input [255:0] name);
        if (!cond) begin errors = errors + 1; $display("PROBE FAIL: %0s", name); end
    endtask

    initial begin
        #12 rst_n = 1;
        @(negedge clk);
        for (i = 6; i >= 0; i = i - 1) begin
            din = stream[i];
            #3;  // 输入稳定后、时钟沿前采样（Mealy：detected 应在本拍有效）
            chk(detected === expect_det[i],
                expect_det[i] ? "detected should PULSE this cycle" : "detected must stay low");
            @(negedge clk);
        end
        din = 0;
        if (errors == 0) $display("PROBE PASSED");
        else $display("PROBE FAILED: %0d errors", errors);
        $finish;
    end
endmodule
