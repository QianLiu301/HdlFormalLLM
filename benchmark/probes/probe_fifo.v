`timescale 1ns/1ps
// 定向探针：验证 golden FIFO 的 (a) FIFO 顺序 (b) dout 寄存一拍语义 (c) 空读保持
module tb;
    reg clk = 0, rst_n = 0, wr_en = 0, rd_en = 0;
    reg [7:0] din = 0;
    wire [7:0] dout;
    wire full, empty;
    sync_fifo_8x8 dut(.clk(clk), .rst_n(rst_n), .wr_en(wr_en), .din(din),
                      .rd_en(rd_en), .dout(dout), .full(full), .empty(empty));
    always #5 clk = ~clk;
    integer errors = 0;

    task chk(input cond, input [255:0] name);
        if (!cond) begin errors = errors + 1; $display("PROBE FAIL: %0s", name); end
    endtask

    initial begin
        #12 rst_n = 1;                       // 复位释放
        @(negedge clk) chk(empty === 1'b1 && full === 1'b0, "reset: empty=1 full=0");
        chk(dout === 8'h00, "reset: dout=00");

        // 写入 0x55, 0x66
        wr_en = 1; din = 8'h55;
        @(negedge clk) din = 8'h66;
        @(negedge clk) wr_en = 0;

        // 第一次读：接受读的下一拍 dout 才出现 0x55（寄存语义）
        rd_en = 1;
        chk(dout === 8'h00, "before 1st read accepted: dout still 00");
        @(negedge clk) rd_en = 0;
        chk(dout === 8'h55, "one cycle after 1st accepted read: dout=55 (FIFO order + registered)");

        // 第二次读
        rd_en = 1;
        @(negedge clk) rd_en = 0;
        chk(dout === 8'h66, "one cycle after 2nd read: dout=66");
        chk(empty === 1'b1, "after two reads: empty=1");

        // 空读：忽略且 dout 保持
        rd_en = 1;
        @(negedge clk) rd_en = 0;
        chk(dout === 8'h66, "read-while-empty ignored: dout holds 66");
        chk(empty === 1'b1, "read-while-empty: still empty");

        if (errors == 0) $display("PROBE PASSED");
        else $display("PROBE FAILED: %0d errors", errors);
        $finish;
    end
endmodule
