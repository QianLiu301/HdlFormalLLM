// ==========================================================================
// Register File Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: gemini
// Generated: 2026-11-04 17:40:11
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
        $display("Register File Testbench - gemini");
        $display("Configuration: 32 x 32-bit");
        $display("========================================\n");

        // row 1: write R1 = 100
        wen = 1; waddr = 5'd1; wdata = 32'd100;
        @(posedge clk); #1; wen = 0;

        // row 2: write R2 = 255
        wen = 1; waddr = 5'd2; wdata = 32'd255;
        @(posedge clk); #1; wen = 0;

        // row 3: write R3 = 12345
        wen = 1; waddr = 5'd3; wdata = 32'd12345;
        @(posedge clk); #1; wen = 0;

        // row 4: write R4 = 4294967295
        wen = 1; waddr = 5'd4; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;

        // row 5: write R5 = 0
        wen = 1; waddr = 5'd5; wdata = 32'd0;
        @(posedge clk); #1; wen = 0;

        // row 6: write R6 = 987654321
        wen = 1; waddr = 5'd6; wdata = 32'd987654321;
        @(posedge clk); #1; wen = 0;

        // row 7: write R1 = 111
        wen = 1; waddr = 5'd1; wdata = 32'd111;
        @(posedge clk); #1; wen = 0;

        // row 8: write R7 = 16777215
        wen = 1; waddr = 5'd7; wdata = 32'd16777215;
        @(posedge clk); #1; wen = 0;

        // row 9: read R1, expect 111 [BDD:RF-READ-001/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd111) begin
            passed = passed + 1;
            $display("✓ row 9: PASS - R%0d = %0d [BDD:RF-READ-001/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 9: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r1]", 5'd1, 32'd111, rdata1);
        end

        // row 10: read R2, expect 255 [BDD:RF-READ-001/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd255) begin
            passed = passed + 1;
            $display("✓ row 10: PASS - R%0d = %0d [BDD:RF-READ-001/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 10: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r2]", 5'd2, 32'd255, rdata1);
        end

        // row 11: read R3, expect 12345 [BDD:RF-READ-001/r3]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd12345) begin
            passed = passed + 1;
            $display("✓ row 11: PASS - R%0d = %0d [BDD:RF-READ-001/r3]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 11: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r3]", 5'd3, 32'd12345, rdata1);
        end

        // row 12: read R4, expect 4294967295 [BDD:RF-READ-001/r4]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 12: PASS - R%0d = %0d [BDD:RF-READ-001/r4]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 12: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r4]", 5'd4, 32'd4294967295, rdata1);
        end

        // row 13: read R5, expect 0 [BDD:RF-READ-001/r5]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 13: PASS - R%0d = %0d [BDD:RF-READ-001/r5]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 13: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r5]", 5'd5, 32'd0, rdata1);
        end

        // row 14: read R6, expect 987654321 [BDD:RF-READ-001/r6]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd987654321) begin
            passed = passed + 1;
            $display("✓ row 14: PASS - R%0d = %0d [BDD:RF-READ-001/r6]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 14: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r6]", 5'd6, 32'd987654321, rdata1);
        end

        // row 15: read R7, expect 16777215 [BDD:RF-READ-001/r7]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd16777215) begin
            passed = passed + 1;
            $display("✓ row 15: PASS - R%0d = %0d [BDD:RF-READ-001/r7]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 15: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r7]", 5'd7, 32'd16777215, rdata1);
        end

        // row 16: read R1, expect 111 [BDD:RF-READ-001/r8]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd111) begin
            passed = passed + 1;
            $display("✓ row 16: PASS - R%0d = %0d [BDD:RF-READ-001/r8]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 16: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r8]", 5'd1, 32'd111, rdata1);
        end

        // row 17: write R8 = 42
        wen = 1; waddr = 5'd8; wdata = 32'd42;
        @(posedge clk); #1; wen = 0;
        // row 17: read R8, expect 42 [BDD:RF-WRITE_READ-001/r1]
        raddr1 = 5'd8; #1;
        total = total + 1;
        if (rdata1 == 32'd42) begin
            passed = passed + 1;
            $display("✓ row 17: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r1]", 5'd8, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 17: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r1]", 5'd8, 32'd42, rdata1);
        end

        // row 18: write R9 = 1
        wen = 1; waddr = 5'd9; wdata = 32'd1;
        @(posedge clk); #1; wen = 0;
        // row 18: read R9, expect 1 [BDD:RF-WRITE_READ-001/r2]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd1) begin
            passed = passed + 1;
            $display("✓ row 18: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r2]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 18: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r2]", 5'd9, 32'd1, rdata1);
        end

        // row 19: write R10 = 2147483647
        wen = 1; waddr = 5'd10; wdata = 32'd2147483647;
        @(posedge clk); #1; wen = 0;
        // row 19: read R10, expect 2147483647 [BDD:RF-WRITE_READ-001/r3]
        raddr1 = 5'd10; #1;
        total = total + 1;
        if (rdata1 == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ row 19: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r3]", 5'd10, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 19: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r3]", 5'd10, 32'd2147483647, rdata1);
        end

        // row 20: write R11 = 0
        wen = 1; waddr = 5'd11; wdata = 32'd0;
        @(posedge clk); #1; wen = 0;
        // row 20: read R11, expect 0 [BDD:RF-WRITE_READ-001/r4]
        raddr1 = 5'd11; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 20: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r4]", 5'd11, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 20: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r4]", 5'd11, 32'd0, rdata1);
        end

        // row 21: write R12 = 3333333333
        wen = 1; waddr = 5'd12; wdata = 32'd3333333333;
        @(posedge clk); #1; wen = 0;
        // row 21: read R12, expect 3333333333 [BDD:RF-WRITE_READ-001/r5]
        raddr1 = 5'd12; #1;
        total = total + 1;
        if (rdata1 == 32'd3333333333) begin
            passed = passed + 1;
            $display("✓ row 21: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r5]", 5'd12, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 21: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r5]", 5'd12, 32'd3333333333, rdata1);
        end

        // row 22: write R0 = 123
        wen = 1; waddr = 5'd0; wdata = 32'd123;
        @(posedge clk); #1; wen = 0;

        // row 23: write R0 = 456
        wen = 1; waddr = 5'd0; wdata = 32'd456;
        @(posedge clk); #1; wen = 0;

        // row 24: write R0 = 789
        wen = 1; waddr = 5'd0; wdata = 32'd789;
        @(posedge clk); #1; wen = 0;

        // row 25: write R0 = 1011
        wen = 1; waddr = 5'd0; wdata = 32'd1011;
        @(posedge clk); #1; wen = 0;

        // row 26: write R0 = 1213
        wen = 1; waddr = 5'd0; wdata = 32'd1213;
        @(posedge clk); #1; wen = 0;

        // row 27: read R0, expect 0 [BDD:RF-READ-002/r1]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 27: PASS - R%0d = %0d [BDD:RF-READ-002/r1]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 27: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r1]", 5'd0, 32'd0, rdata1);
        end

        // row 28: read R0, expect 0 [BDD:RF-READ-002/r2]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 28: PASS - R%0d = %0d [BDD:RF-READ-002/r2]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 28: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r2]", 5'd0, 32'd0, rdata1);
        end

        // row 29: read R0, expect 0 [BDD:RF-READ-002/r3]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 29: PASS - R%0d = %0d [BDD:RF-READ-002/r3]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 29: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r3]", 5'd0, 32'd0, rdata1);
        end

        // row 30: read R0, expect 0 [BDD:RF-READ-002/r4]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 30: PASS - R%0d = %0d [BDD:RF-READ-002/r4]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 30: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r4]", 5'd0, 32'd0, rdata1);
        end

        // row 31: read R0, expect 0 [BDD:RF-READ-002/r5]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 31: PASS - R%0d = %0d [BDD:RF-READ-002/r5]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 31: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-002/r5]", 5'd0, 32'd0, rdata1);
        end

        // row 32: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 33: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 34: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 35: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 36: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 37: read R1, expect 0 [BDD:RF-READ-003/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 37: PASS - R%0d = %0d [BDD:RF-READ-003/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 37: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r1]", 5'd1, 32'd0, rdata1);
        end

        // row 38: read R2, expect 0 [BDD:RF-READ-003/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 38: PASS - R%0d = %0d [BDD:RF-READ-003/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 38: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r2]", 5'd2, 32'd0, rdata1);
        end

        // row 39: read R3, expect 0 [BDD:RF-READ-003/r3]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 39: PASS - R%0d = %0d [BDD:RF-READ-003/r3]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 39: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r3]", 5'd3, 32'd0, rdata1);
        end

        // row 40: read R4, expect 0 [BDD:RF-READ-003/r4]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 40: PASS - R%0d = %0d [BDD:RF-READ-003/r4]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 40: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r4]", 5'd4, 32'd0, rdata1);
        end

        // row 41: read R5, expect 0 [BDD:RF-READ-003/r5]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 41: PASS - R%0d = %0d [BDD:RF-READ-003/r5]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 41: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r5]", 5'd5, 32'd0, rdata1);
        end

        // row 42: read R6, expect 0 [BDD:RF-READ-003/r6]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 42: PASS - R%0d = %0d [BDD:RF-READ-003/r6]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 42: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r6]", 5'd6, 32'd0, rdata1);
        end

        // row 43: read R7, expect 0 [BDD:RF-READ-003/r7]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 43: PASS - R%0d = %0d [BDD:RF-READ-003/r7]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 43: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r7]", 5'd7, 32'd0, rdata1);
        end

        // row 44: read R8, expect 0 [BDD:RF-READ-003/r8]
        raddr1 = 5'd8; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 44: PASS - R%0d = %0d [BDD:RF-READ-003/r8]", 5'd8, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 44: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r8]", 5'd8, 32'd0, rdata1);
        end

        // row 45: write R13 = 1000
        wen = 1; waddr = 5'd13; wdata = 32'd1000;
        @(posedge clk); #1; wen = 0;
        // row 45: read R13, expect 1000 [BDD:RF-WRITE_READ-002/r1]
        raddr1 = 5'd13; #1;
        total = total + 1;
        if (rdata1 == 32'd1000) begin
            passed = passed + 1;
            $display("✓ row 45: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r1]", 5'd13, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 45: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r1]", 5'd13, 32'd1000, rdata1);
        end

        // row 46: write R14 = 2000
        wen = 1; waddr = 5'd14; wdata = 32'd2000;
        @(posedge clk); #1; wen = 0;
        // row 46: read R14, expect 2000 [BDD:RF-WRITE_READ-002/r2]
        raddr1 = 5'd14; #1;
        total = total + 1;
        if (rdata1 == 32'd2000) begin
            passed = passed + 1;
            $display("✓ row 46: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r2]", 5'd14, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 46: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r2]", 5'd14, 32'd2000, rdata1);
        end

        // row 47: write R15 = 4294967295
        wen = 1; waddr = 5'd15; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;
        // row 47: read R15, expect 4294967295 [BDD:RF-WRITE_READ-002/r3]
        raddr1 = 5'd15; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 47: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r3]", 5'd15, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 47: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r3]", 5'd15, 32'd4294967295, rdata1);
        end

        // row 48: write R16 = 555
        wen = 1; waddr = 5'd16; wdata = 32'd555;
        @(posedge clk); #1; wen = 0;
        // row 48: read R16, expect 555 [BDD:RF-WRITE_READ-002/r4]
        raddr1 = 5'd16; #1;
        total = total + 1;
        if (rdata1 == 32'd555) begin
            passed = passed + 1;
            $display("✓ row 48: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r4]", 5'd16, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 48: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r4]", 5'd16, 32'd555, rdata1);
        end

        // row 49: write R17 = 0
        wen = 1; waddr = 5'd17; wdata = 32'd0;
        @(posedge clk); #1; wen = 0;
        // row 49: read R17, expect 0 [BDD:RF-WRITE_READ-002/r5]
        raddr1 = 5'd17; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 49: PASS - R%0d = %0d [BDD:RF-WRITE_READ-002/r5]", 5'd17, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 49: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-002/r5]", 5'd17, 32'd0, rdata1);
        end

        // row 50: read R18, expect 10 [BDD:RF-READ-004/r1]
        raddr1 = 5'd18; #1;
        total = total + 1;
        if (rdata1 == 32'd10) begin
            passed = passed + 1;
            $display("✓ row 50: PASS - R%0d = %0d [BDD:RF-READ-004/r1]", 5'd18, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 50: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r1]", 5'd18, 32'd10, rdata1);
        end

        // row 51: read R19, expect 100 [BDD:RF-READ-004/r2]
        raddr1 = 5'd19; #1;
        total = total + 1;
        if (rdata1 == 32'd100) begin
            passed = passed + 1;
            $display("✓ row 51: PASS - R%0d = %0d [BDD:RF-READ-004/r2]", 5'd19, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 51: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r2]", 5'd19, 32'd100, rdata1);
        end

        // row 52: read R20, expect 4294967295 [BDD:RF-READ-004/r3]
        raddr1 = 5'd20; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 52: PASS - R%0d = %0d [BDD:RF-READ-004/r3]", 5'd20, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 52: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r3]", 5'd20, 32'd4294967295, rdata1);
        end

        // row 53: read R21, expect 0 [BDD:RF-READ-004/r4]
        raddr1 = 5'd21; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 53: PASS - R%0d = %0d [BDD:RF-READ-004/r4]", 5'd21, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 53: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r4]", 5'd21, 32'd0, rdata1);
        end

        // row 54: read R22, expect 12345 [BDD:RF-READ-004/r5]
        raddr1 = 5'd22; #1;
        total = total + 1;
        if (rdata1 == 32'd12345) begin
            passed = passed + 1;
            $display("✓ row 54: PASS - R%0d = %0d [BDD:RF-READ-004/r5]", 5'd22, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 54: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r5]", 5'd22, 32'd12345, rdata1);
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