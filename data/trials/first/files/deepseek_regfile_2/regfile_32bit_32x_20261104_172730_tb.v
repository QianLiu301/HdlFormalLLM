// ==========================================================================
// Register File Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: deepseek
// Generated: 2026-11-04 17:27:30
// Configuration: 32 registers x 32 bits
// ==========================================================================

`timescale 1ns/1ps

module regfile_tb;

    parameter WIDTH = 32;
    parameter DEPTH = 32;
    parameter ADDR_WIDTH = 5;

    reg clk;
    reg rst_n;
    // Read port 1
    reg [ADDR_WIDTH-1:0] raddr1;
    wire [WIDTH-1:0] rdata1;
    // Read port 2
    reg [ADDR_WIDTH-1:0] raddr2;
    wire [WIDTH-1:0] rdata2;
    // Write port
    reg wen;
    reg [ADDR_WIDTH-1:0] waddr;
    reg [WIDTH-1:0] wdata;

    integer total, passed, failed;

    regfile dut (
        .clk(clk),
        .rst_n(rst_n),
        .raddr1(raddr1),
        .rdata1(rdata1),
        .raddr2(raddr2),
        .rdata2(rdata2),
        .wen(wen),
        .waddr(waddr),
        .wdata(wdata)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        // Initialize
        total = 0; passed = 0; failed = 0;
        rst_n = 0; wen = 0; raddr1 = 0; raddr2 = 0; waddr = 0; wdata = 0;
        #20 rst_n = 1;

        $display("\n========================================");
        $display("Register File Testbench - deepseek");
        $display("Configuration: 32 x 32-bit");
        $display("========================================\n");

        // row 1: write R1 = 100
        wen = 1; waddr = 5'd1; wdata = 32'd100;
        @(posedge clk); #1; wen = 0;

        // row 2: write R2 = 255
        wen = 1; waddr = 5'd2; wdata = 32'd255;
        @(posedge clk); #1; wen = 0;

        // row 3: write R3 = 4294967295
        wen = 1; waddr = 5'd3; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;

        // row 4: write R4 = 0
        wen = 1; waddr = 5'd4; wdata = 32'd0;
        @(posedge clk); #1; wen = 0;

        // row 5: write R5 = 305419896
        wen = 1; waddr = 5'd5; wdata = 32'd305419896;
        @(posedge clk); #1; wen = 0;

        // row 6: write R1 = 999
        wen = 1; waddr = 5'd1; wdata = 32'd999;
        @(posedge clk); #1; wen = 0;

        // row 7: write R2 = 111
        wen = 1; waddr = 5'd2; wdata = 32'd111;
        @(posedge clk); #1; wen = 0;

        // row 8: write R3 = 222
        wen = 1; waddr = 5'd3; wdata = 32'd222;
        @(posedge clk); #1; wen = 0;

        // row 9: write R4 = 333
        wen = 1; waddr = 5'd4; wdata = 32'd333;
        @(posedge clk); #1; wen = 0;

        // row 10: write R5 = 444
        wen = 1; waddr = 5'd5; wdata = 32'd444;
        @(posedge clk); #1; wen = 0;

        // row 11: write R10 = 2863311530
        wen = 1; waddr = 5'd10; wdata = 32'd2863311530;
        @(posedge clk); #1; wen = 0;

        // row 12: write R11 = 1111
        wen = 1; waddr = 5'd11; wdata = 32'd1111;
        @(posedge clk); #1; wen = 0;

        // row 13: write R12 = 2222
        wen = 1; waddr = 5'd12; wdata = 32'd2222;
        @(posedge clk); #1; wen = 0;

        // row 14: write R13 = 3333
        wen = 1; waddr = 5'd13; wdata = 32'd3333;
        @(posedge clk); #1; wen = 0;

        // row 15: write R14 = 4444
        wen = 1; waddr = 5'd14; wdata = 32'd4444;
        @(posedge clk); #1; wen = 0;

        // Test: Register 0 should always be 0 [GEN:RF-R0]
        wen = 1; waddr = 5'd0; wdata = 32'd255;
        #10;
        wen = 0; raddr1 = 5'd0;
        #10;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test R0: PASS - R0 is always 0 [GEN:RF-R0]");
        end else begin
            failed = failed + 1;
            $display("✗ Test R0: FAIL - R0 should be 0, got %0d [GEN:RF-R0]", rdata1);
        end

        // Summary
        $display("\n========================================");
        $display("Test Summary");
        $display("========================================");
        $display("Total:  %0d", total);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        if (failed == 0)
            $display("\n🎉 ALL TESTS PASSED!");
        else
            $display("\n⚠️  SOME TESTS FAILED");
        $display("========================================\n");
        $finish;
    end

endmodule