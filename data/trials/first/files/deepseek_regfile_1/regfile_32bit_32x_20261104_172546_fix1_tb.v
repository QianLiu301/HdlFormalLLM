// ==========================================================================
// Register File Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: deepseek
// Generated: 2026-11-04 17:25:54
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

        // row 3: write R5 = 4294967295
        wen = 1; waddr = 5'd5; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;

        // row 4: write R7 = 42
        wen = 1; waddr = 5'd7; wdata = 32'd42;
        @(posedge clk); #1; wen = 0;

        // row 5: write R9 = 123456789
        wen = 1; waddr = 5'd9; wdata = 32'd123456789;
        @(posedge clk); #1; wen = 0;

        // row 6: write R3 = 777
        wen = 1; waddr = 5'd3; wdata = 32'd777;
        @(posedge clk); #1; wen = 0;

        // row 7: write R4 = 888
        wen = 1; waddr = 5'd4; wdata = 32'd888;
        @(posedge clk); #1; wen = 0;

        // row 8: write R6 = 999
        wen = 1; waddr = 5'd6; wdata = 32'd999;
        @(posedge clk); #1; wen = 0;

        // row 9: write R17 = 2020
        wen = 1; waddr = 5'd17; wdata = 32'd2020;
        @(posedge clk); #1; wen = 0;

        // row 10: read R1, expect 100 [BDD:RF-READ-001/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd100) begin
            passed = passed + 1;
            $display("✓ row 10: PASS - R%0d = %0d [BDD:RF-READ-001/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 10: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r1]", 5'd1, 32'd100, rdata1);
        end

        // row 11: read R2, expect 255 [BDD:RF-READ-001/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd255) begin
            passed = passed + 1;
            $display("✓ row 11: PASS - R%0d = %0d [BDD:RF-READ-001/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 11: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r2]", 5'd2, 32'd255, rdata1);
        end

        // row 12: read R5, expect 4294967295 [BDD:RF-READ-001/r3]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 12: PASS - R%0d = %0d [BDD:RF-READ-001/r3]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 12: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r3]", 5'd5, 32'd4294967295, rdata1);
        end

        // row 13: read R7, expect 42 [BDD:RF-READ-001/r4]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd42) begin
            passed = passed + 1;
            $display("✓ row 13: PASS - R%0d = %0d [BDD:RF-READ-001/r4]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 13: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r4]", 5'd7, 32'd42, rdata1);
        end

        // row 14: read R9, expect 123456789 [BDD:RF-READ-001/r5]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd123456789) begin
            passed = passed + 1;
            $display("✓ row 14: PASS - R%0d = %0d [BDD:RF-READ-001/r5]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 14: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r5]", 5'd9, 32'd123456789, rdata1);
        end

        // row 15: read R3, expect 777 [BDD:RF-READ-001/r6]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd777) begin
            passed = passed + 1;
            $display("✓ row 15: PASS - R%0d = %0d [BDD:RF-READ-001/r6]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 15: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r6]", 5'd3, 32'd777, rdata1);
        end

        // row 16: read R4, expect 888 [BDD:RF-READ-001/r7]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd888) begin
            passed = passed + 1;
            $display("✓ row 16: PASS - R%0d = %0d [BDD:RF-READ-001/r7]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 16: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r7]", 5'd4, 32'd888, rdata1);
        end

        // row 17: read R6, expect 999 [BDD:RF-READ-001/r8]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd999) begin
            passed = passed + 1;
            $display("✓ row 17: PASS - R%0d = %0d [BDD:RF-READ-001/r8]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 17: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r8]", 5'd6, 32'd999, rdata1);
        end

        // row 18: read R1, expect 100 [BDD:RF-READ-002/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd100) begin
            passed = passed + 1;
            $display("✓ row 18: PASS - R%0d = %0d [BDD:RF-READ-002/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 18: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r1]", 5'd1, 32'd100, rdata1);
        end

        // row 19: read R2, expect 255 [BDD:RF-READ-002/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd255) begin
            passed = passed + 1;
            $display("✓ row 19: PASS - R%0d = %0d [BDD:RF-READ-002/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 19: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r2]", 5'd2, 32'd255, rdata1);
        end

        // row 20: read R5, expect 4294967295 [BDD:RF-READ-002/r3]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 20: PASS - R%0d = %0d [BDD:RF-READ-002/r3]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 20: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r3]", 5'd5, 32'd4294967295, rdata1);
        end

        // row 21: read R7, expect 42 [BDD:RF-READ-002/r4]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd42) begin
            passed = passed + 1;
            $display("✓ row 21: PASS - R%0d = %0d [BDD:RF-READ-002/r4]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 21: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r4]", 5'd7, 32'd42, rdata1);
        end

        // row 22: read R9, expect 123456789 [BDD:RF-READ-002/r5]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd123456789) begin
            passed = passed + 1;
            $display("✓ row 22: PASS - R%0d = %0d [BDD:RF-READ-002/r5]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 22: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r5]", 5'd9, 32'd123456789, rdata1);
        end

        // row 23: read R3, expect 777 [BDD:RF-READ-002/r6]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd777) begin
            passed = passed + 1;
            $display("✓ row 23: PASS - R%0d = %0d [BDD:RF-READ-002/r6]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 23: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r6]", 5'd3, 32'd777, rdata1);
        end

        // row 24: read R4, expect 888 [BDD:RF-READ-002/r7]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd888) begin
            passed = passed + 1;
            $display("✓ row 24: PASS - R%0d = %0d [BDD:RF-READ-002/r7]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 24: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r7]", 5'd4, 32'd888, rdata1);
        end

        // row 25: read R6, expect 999 [BDD:RF-READ-002/r8]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd999) begin
            passed = passed + 1;
            $display("✓ row 25: PASS - R%0d = %0d [BDD:RF-READ-002/r8]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 25: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r8]", 5'd6, 32'd999, rdata1);
        end

        // row 26: write R8 = 11
        wen = 1; waddr = 5'd8; wdata = 32'd11;
        @(posedge clk); #1; wen = 0;
        // row 26: read R8, expect 11 [BDD:RF-WRITE_READ-001/r1]
        raddr1 = 5'd8; #1;
        total = total + 1;
        if (rdata1 == 32'd11) begin
            passed = passed + 1;
            $display("✓ row 26: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r1]", 5'd8, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 26: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r1]", 5'd8, 32'd11, rdata1);
        end

        // row 27: write R10 = 22
        wen = 1; waddr = 5'd10; wdata = 32'd22;
        @(posedge clk); #1; wen = 0;
        // row 27: read R10, expect 22 [BDD:RF-WRITE_READ-001/r2]
        raddr1 = 5'd10; #1;
        total = total + 1;
        if (rdata1 == 32'd22) begin
            passed = passed + 1;
            $display("✓ row 27: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r2]", 5'd10, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 27: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r2]", 5'd10, 32'd22, rdata1);
        end

        // row 28: write R11 = 33
        wen = 1; waddr = 5'd11; wdata = 32'd33;
        @(posedge clk); #1; wen = 0;
        // row 28: read R11, expect 33 [BDD:RF-WRITE_READ-001/r3]
        raddr1 = 5'd11; #1;
        total = total + 1;
        if (rdata1 == 32'd33) begin
            passed = passed + 1;
            $display("✓ row 28: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r3]", 5'd11, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 28: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r3]", 5'd11, 32'd33, rdata1);
        end

        // row 29: write R12 = 44
        wen = 1; waddr = 5'd12; wdata = 32'd44;
        @(posedge clk); #1; wen = 0;
        // row 29: read R12, expect 44 [BDD:RF-WRITE_READ-001/r4]
        raddr1 = 5'd12; #1;
        total = total + 1;
        if (rdata1 == 32'd44) begin
            passed = passed + 1;
            $display("✓ row 29: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r4]", 5'd12, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 29: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r4]", 5'd12, 32'd44, rdata1);
        end

        // row 30: write R13 = 55
        wen = 1; waddr = 5'd13; wdata = 32'd55;
        @(posedge clk); #1; wen = 0;
        // row 30: read R13, expect 55 [BDD:RF-WRITE_READ-001/r5]
        raddr1 = 5'd13; #1;
        total = total + 1;
        if (rdata1 == 32'd55) begin
            passed = passed + 1;
            $display("✓ row 30: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r5]", 5'd13, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 30: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r5]", 5'd13, 32'd55, rdata1);
        end

        // row 31: write R14 = 111
        wen = 1; waddr = 5'd14; wdata = 32'd111;
        @(posedge clk); #1; wen = 0;
        // row 31: read R14, expect 111 [BDD:RF-WRITE_READ-002/r1]
        raddr1 = 5'd14; #1;
        total = total + 1;
        if (rdata1 == 32'd111) begin
            passed = passed + 1;
            $display("✓ row 31: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r1]", 5'd14, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 31: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r1]", 5'd14, 32'd111, rdata1);
        end

        // row 32: write R14 = 222
        wen = 1; waddr = 5'd14; wdata = 32'd222;
        @(posedge clk); #1; wen = 0;
        // row 32: read R14, expect 222 [BDD:RF-WRITE_READ-002/r2]
        raddr1 = 5'd14; #1;
        total = total + 1;
        if (rdata1 == 32'd222) begin
            passed = passed + 1;
            $display("✓ row 32: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r2]", 5'd14, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 32: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r2]", 5'd14, 32'd222, rdata1);
        end

        // row 33: write R15 = 333
        wen = 1; waddr = 5'd15; wdata = 32'd333;
        @(posedge clk); #1; wen = 0;
        // row 33: read R15, expect 333 [BDD:RF-WRITE_READ-002/r3]
        raddr1 = 5'd15; #1;
        total = total + 1;
        if (rdata1 == 32'd333) begin
            passed = passed + 1;
            $display("✓ row 33: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r3]", 5'd15, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 33: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r3]", 5'd15, 32'd333, rdata1);
        end

        // row 34: write R15 = 444
        wen = 1; waddr = 5'd15; wdata = 32'd444;
        @(posedge clk); #1; wen = 0;
        // row 34: read R15, expect 444 [BDD:RF-WRITE_READ-002/r4]
        raddr1 = 5'd15; #1;
        total = total + 1;
        if (rdata1 == 32'd444) begin
            passed = passed + 1;
            $display("✓ row 34: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r4]", 5'd15, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 34: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r4]", 5'd15, 32'd444, rdata1);
        end

        // row 35: write R16 = 555
        wen = 1; waddr = 5'd16; wdata = 32'd555;
        @(posedge clk); #1; wen = 0;
        // row 35: read R16, expect 555 [BDD:RF-WRITE_READ-002/r5]
        raddr1 = 5'd16; #1;
        total = total + 1;
        if (rdata1 == 32'd555) begin
            passed = passed + 1;
            $display("✓ row 35: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r5]", 5'd16, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 35: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r5]", 5'd16, 32'd555, rdata1);
        end

        // row 36: write R0 = 123
        wen = 1; waddr = 5'd0; wdata = 32'd123;
        @(posedge clk); #1; wen = 0;

        // row 37: write R0 = 456
        wen = 1; waddr = 5'd0; wdata = 32'd456;
        @(posedge clk); #1; wen = 0;

        // row 38: write R0 = 789
        wen = 1; waddr = 5'd0; wdata = 32'd789;
        @(posedge clk); #1; wen = 0;

        // row 39: write R0 = 1010
        wen = 1; waddr = 5'd0; wdata = 32'd1010;
        @(posedge clk); #1; wen = 0;

        // row 40: write R0 = 4294967295
        wen = 1; waddr = 5'd0; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;

        // row 41: read R0, expect 0 [BDD:RF-READ-003/r1]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 41: PASS - R%0d = %0d [BDD:RF-READ-003/r1]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 41: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r1]", 5'd0, 32'd0, rdata1);
        end

        // row 42: read R0, expect 0 [BDD:RF-READ-003/r2]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 42: PASS - R%0d = %0d [BDD:RF-READ-003/r2]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 42: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r2]", 5'd0, 32'd0, rdata1);
        end

        // row 43: read R0, expect 0 [BDD:RF-READ-003/r3]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 43: PASS - R%0d = %0d [BDD:RF-READ-003/r3]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 43: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r3]", 5'd0, 32'd0, rdata1);
        end

        // row 44: read R0, expect 0 [BDD:RF-READ-003/r4]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 44: PASS - R%0d = %0d [BDD:RF-READ-003/r4]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 44: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r4]", 5'd0, 32'd0, rdata1);
        end

        // row 45: read R0, expect 0 [BDD:RF-READ-003/r5]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 45: PASS - R%0d = %0d [BDD:RF-READ-003/r5]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 45: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r5]", 5'd0, 32'd0, rdata1);
        end

        // row 46: read R0, expect 0 [BDD:RF-READ-004/r1]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 46: PASS - R%0d = %0d [BDD:RF-READ-004/r1]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 46: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r1]", 5'd0, 32'd0, rdata1);
        end

        // row 47: read R0, expect 0 [BDD:RF-READ-004/r2]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 47: PASS - R%0d = %0d [BDD:RF-READ-004/r2]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 47: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r2]", 5'd0, 32'd0, rdata1);
        end

        // row 48: read R0, expect 0 [BDD:RF-READ-004/r3]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 48: PASS - R%0d = %0d [BDD:RF-READ-004/r3]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 48: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r3]", 5'd0, 32'd0, rdata1);
        end

        // row 49: read R0, expect 0 [BDD:RF-READ-004/r4]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 49: PASS - R%0d = %0d [BDD:RF-READ-004/r4]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 49: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r4]", 5'd0, 32'd0, rdata1);
        end

        // row 50: read R0, expect 0 [BDD:RF-READ-004/r5]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 50: PASS - R%0d = %0d [BDD:RF-READ-004/r5]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 50: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r5]", 5'd0, 32'd0, rdata1);
        end

        // row 51: write R1 = 200
        wen = 1; waddr = 5'd1; wdata = 32'd200;
        @(posedge clk); #1; wen = 0;

        // row 52: write R2 = 300
        wen = 1; waddr = 5'd2; wdata = 32'd300;
        @(posedge clk); #1; wen = 0;

        // row 53: write R5 = 400
        wen = 1; waddr = 5'd5; wdata = 32'd400;
        @(posedge clk); #1; wen = 0;

        // row 54: write R7 = 500
        wen = 1; waddr = 5'd7; wdata = 32'd500;
        @(posedge clk); #1; wen = 0;

        // row 55: write R9 = 600
        wen = 1; waddr = 5'd9; wdata = 32'd600;
        @(posedge clk); #1; wen = 0;

        // row 56: write R1 = 200
        wen = 1; waddr = 5'd1; wdata = 32'd200;
        @(posedge clk); #1; wen = 0;
        // row 56: read R1, expect 200 [BDD:RF-WRITE_READ-003/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd200) begin
            passed = passed + 1;
            $display("✓ row 56: PASS - R%0d = %0d [BDD:RF-WRITE_READ-003/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 56: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-003/r1]", 5'd1, 32'd200, rdata1);
        end

        // row 57: write R2 = 300
        wen = 1; waddr = 5'd2; wdata = 32'd300;
        @(posedge clk); #1; wen = 0;
        // row 57: read R2, expect 300 [BDD:RF-WRITE_READ-003/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd300) begin
            passed = passed + 1;
            $display("✓ row 57: PASS - R%0d = %0d [BDD:RF-WRITE_READ-003/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 57: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-003/r2]", 5'd2, 32'd300, rdata1);
        end

        // row 58: write R5 = 400
        wen = 1; waddr = 5'd5; wdata = 32'd400;
        @(posedge clk); #1; wen = 0;
        // row 58: read R5, expect 400 [BDD:RF-WRITE_READ-003/r3]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd400) begin
            passed = passed + 1;
            $display("✓ row 58: PASS - R%0d = %0d [BDD:RF-WRITE_READ-003/r3]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 58: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-003/r3]", 5'd5, 32'd400, rdata1);
        end

        // row 59: write R7 = 500
        wen = 1; waddr = 5'd7; wdata = 32'd500;
        @(posedge clk); #1; wen = 0;
        // row 59: read R7, expect 500 [BDD:RF-WRITE_READ-003/r4]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd500) begin
            passed = passed + 1;
            $display("✓ row 59: PASS - R%0d = %0d [BDD:RF-WRITE_READ-003/r4]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 59: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-003/r4]", 5'd7, 32'd500, rdata1);
        end

        // row 60: write R9 = 600
        wen = 1; waddr = 5'd9; wdata = 32'd600;
        @(posedge clk); #1; wen = 0;
        // row 60: read R9, expect 600 [BDD:RF-WRITE_READ-003/r5]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd600) begin
            passed = passed + 1;
            $display("✓ row 60: PASS - R%0d = %0d [BDD:RF-WRITE_READ-003/r5]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 60: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-003/r5]", 5'd9, 32'd600, rdata1);
        end

        // row 61: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 62: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 63: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 64: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 65: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 66: read R1, expect 0 [BDD:RF-READ-005/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 66: PASS - R%0d = %0d [BDD:RF-READ-005/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 66: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r1]", 5'd1, 32'd0, rdata1);
        end

        // row 67: read R2, expect 0 [BDD:RF-READ-005/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 67: PASS - R%0d = %0d [BDD:RF-READ-005/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 67: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r2]", 5'd2, 32'd0, rdata1);
        end

        // row 68: read R5, expect 0 [BDD:RF-READ-005/r3]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 68: PASS - R%0d = %0d [BDD:RF-READ-005/r3]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 68: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r3]", 5'd5, 32'd0, rdata1);
        end

        // row 69: read R7, expect 0 [BDD:RF-READ-005/r4]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 69: PASS - R%0d = %0d [BDD:RF-READ-005/r4]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 69: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r4]", 5'd7, 32'd0, rdata1);
        end

        // row 70: read R9, expect 0 [BDD:RF-READ-005/r5]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 70: PASS - R%0d = %0d [BDD:RF-READ-005/r5]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 70: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r5]", 5'd9, 32'd0, rdata1);
        end

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