`timescale 1ns/1ps
// 定向探针 v2：与复位相位无关——等第一个 GREEN 上升点，然后逐拍数每个状态的长度
module tb;
    reg clk = 0, rst_n = 0;
    wire red, yellow, green;
    traffic_light dut(.clk(clk), .rst_n(rst_n), .red(red), .yellow(yellow), .green(green));
    always #5 clk = ~clk;
    integer errors = 0;
    integer n, i;

    task chk(input cond, input [255:0] name);
        if (!cond) begin errors = errors + 1; $display("PROBE FAIL: %0s", name); end
    endtask

    // 数当前状态还持续多少个 negedge 采样点（进入时已在该状态的第 1 拍）
    task count_state(input integer which /*0=R 1=G 2=Y*/, output integer len);
        begin
            len = 0;
            while ((which == 0 && red) || (which == 1 && green) || (which == 2 && yellow)) begin
                len = len + 1;
                chk(red + yellow + green == 1, "one-hot every cycle");
                @(negedge clk);
            end
        end
    endtask

    initial begin
        #12 rst_n = 1;
        chk(red === 1'b1, "reset state is RED");
        // 等待第一次进入 GREEN（对齐到状态边界）
        @(negedge clk);
        while (!green) @(negedge clk);
        count_state(1, n); $display("GREEN  len=%0d", n); chk(n == 8, "GREEN lasts exactly 8 cycles");
        count_state(2, n); $display("YELLOW len=%0d", n); chk(n == 3, "YELLOW lasts exactly 3 cycles");
        count_state(0, n); $display("RED    len=%0d", n); chk(n == 6, "RED lasts exactly 6 cycles");
        count_state(1, n); $display("GREEN2 len=%0d", n); chk(n == 8, "GREEN again 8 cycles (periodic)");
        chk(yellow === 1'b1, "order GREEN->YELLOW (never GREEN->RED)");

        if (errors == 0) $display("PROBE PASSED");
        else $display("PROBE FAILED: %0d errors", errors);
        $finish;
    end
endmodule
