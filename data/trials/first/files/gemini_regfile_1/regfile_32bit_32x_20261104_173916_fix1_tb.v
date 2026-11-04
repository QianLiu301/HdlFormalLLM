// ==========================================================================
// Register File Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: gemini
// Generated: 2026-11-04 17:39:25
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

        // row 4: write R4 = 987654321
        wen = 1; waddr = 5'd4; wdata = 32'd987654321;
        @(posedge clk); #1; wen = 0;

        // row 5: write R5 = 4294967295
        wen = 1; waddr = 5'd5; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;

        // row 6: write R6 = 1
        wen = 1; waddr = 5'd6; wdata = 32'd1;
        @(posedge clk); #1; wen = 0;

        // row 7: write R7 = 16777215
        wen = 1; waddr = 5'd7; wdata = 32'd16777215;
        @(posedge clk); #1; wen = 0;

        // row 8: read R1, expect 100 [BDD:RF-READ-001/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd100) begin
            passed = passed + 1;
            $display("✓ row 8: PASS - R%0d = %0d [BDD:RF-READ-001/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 8: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r1]", 5'd1, 32'd100, rdata1);
        end

        // row 9: read R2, expect 255 [BDD:RF-READ-001/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd255) begin
            passed = passed + 1;
            $display("✓ row 9: PASS - R%0d = %0d [BDD:RF-READ-001/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 9: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r2]", 5'd2, 32'd255, rdata1);
        end

        // row 10: read R3, expect 12345 [BDD:RF-READ-001/r3]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd12345) begin
            passed = passed + 1;
            $display("✓ row 10: PASS - R%0d = %0d [BDD:RF-READ-001/r3]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 10: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r3]", 5'd3, 32'd12345, rdata1);
        end

        // row 11: read R4, expect 987654321 [BDD:RF-READ-001/r4]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd987654321) begin
            passed = passed + 1;
            $display("✓ row 11: PASS - R%0d = %0d [BDD:RF-READ-001/r4]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 11: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r4]", 5'd4, 32'd987654321, rdata1);
        end

        // row 12: read R5, expect 4294967295 [BDD:RF-READ-001/r5]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 12: PASS - R%0d = %0d [BDD:RF-READ-001/r5]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 12: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r5]", 5'd5, 32'd4294967295, rdata1);
        end

        // row 13: read R6, expect 1 [BDD:RF-READ-001/r6]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd1) begin
            passed = passed + 1;
            $display("✓ row 13: PASS - R%0d = %0d [BDD:RF-READ-001/r6]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 13: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r6]", 5'd6, 32'd1, rdata1);
        end

        // row 14: read R7, expect 16777215 [BDD:RF-READ-001/r7]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd16777215) begin
            passed = passed + 1;
            $display("✓ row 14: PASS - R%0d = %0d [BDD:RF-READ-001/r7]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 14: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-001/r7]", 5'd7, 32'd16777215, rdata1);
        end

        // row 15: write R8 = 42
        wen = 1; waddr = 5'd8; wdata = 32'd42;
        @(posedge clk); #1; wen = 0;
        // row 15: read R8, expect 42 [BDD:RF-WRITE_READ-001/r1]
        raddr1 = 5'd8; #1;
        total = total + 1;
        if (rdata1 == 32'd42) begin
            passed = passed + 1;
            $display("✓ row 15: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r1]", 5'd8, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 15: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r1]", 5'd8, 32'd42, rdata1);
        end

        // row 16: write R9 = 1024
        wen = 1; waddr = 5'd9; wdata = 32'd1024;
        @(posedge clk); #1; wen = 0;
        // row 16: read R9, expect 1024 [BDD:RF-WRITE_READ-001/r2]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd1024) begin
            passed = passed + 1;
            $display("✓ row 16: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r2]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 16: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r2]", 5'd9, 32'd1024, rdata1);
        end

        // row 17: write R10 = 65535
        wen = 1; waddr = 5'd10; wdata = 32'd65535;
        @(posedge clk); #1; wen = 0;
        // row 17: read R10, expect 65535 [BDD:RF-WRITE_READ-001/r3]
        raddr1 = 5'd10; #1;
        total = total + 1;
        if (rdata1 == 32'd65535) begin
            passed = passed + 1;
            $display("✓ row 17: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r3]", 5'd10, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 17: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r3]", 5'd10, 32'd65535, rdata1);
        end

        // row 18: write R11 = 2147483647
        wen = 1; waddr = 5'd11; wdata = 32'd2147483647;
        @(posedge clk); #1; wen = 0;
        // row 18: read R11, expect 2147483647 [BDD:RF-WRITE_READ-001/r4]
        raddr1 = 5'd11; #1;
        total = total + 1;
        if (rdata1 == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ row 18: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r4]", 5'd11, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 18: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r4]", 5'd11, 32'd2147483647, rdata1);
        end

        // row 19: write R12 = 0
        wen = 1; waddr = 5'd12; wdata = 32'd0;
        @(posedge clk); #1; wen = 0;
        // row 19: read R12, expect 0 [BDD:RF-WRITE_READ-001/r5]
        raddr1 = 5'd12; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 19: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r5]", 5'd12, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 19: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r5]", 5'd12, 32'd0, rdata1);
        end

        // row 20: write R13 = 13579
        wen = 1; waddr = 5'd13; wdata = 32'd13579;
        @(posedge clk); #1; wen = 0;
        // row 20: read R13, expect 13579 [BDD:RF-WRITE_READ-001/r6]
        raddr1 = 5'd13; #1;
        total = total + 1;
        if (rdata1 == 32'd13579) begin
            passed = passed + 1;
            $display("✓ row 20: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r6]", 5'd13, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 20: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r6]", 5'd13, 32'd13579, rdata1);
        end

        // row 21: write R14 = 24680
        wen = 1; waddr = 5'd14; wdata = 32'd24680;
        @(posedge clk); #1; wen = 0;
        // row 21: read R14, expect 24680 [BDD:RF-WRITE_READ-001/r7]
        raddr1 = 5'd14; #1;
        total = total + 1;
        if (rdata1 == 32'd24680) begin
            passed = passed + 1;
            $display("✓ row 21: PASS - R%0d = %0d [BDD:RF-WRITE_READ-001/r7]", 5'd14, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 21: FAIL - R%0d expected %0d, got %0d [BDD:RF-WRITE_READ-001/r7]", 5'd14, 32'd24680, rdata1);
        end

        // row 22: write R0 = 123
        wen = 1; waddr = 5'd0; wdata = 32'd123;
        @(posedge clk); #1; wen = 0;

        // row 23: write R0 = 45678
        wen = 1; waddr = 5'd0; wdata = 32'd45678;
        @(posedge clk); #1; wen = 0;

        // row 24: write R0 = 987654321
        wen = 1; waddr = 5'd0; wdata = 32'd987654321;
        @(posedge clk); #1; wen = 0;

        // row 25: write R0 = 4294967295
        wen = 1; waddr = 5'd0; wdata = 32'd4294967295;
        @(posedge clk); #1; wen = 0;

        // row 26: write R0 = 1
        wen = 1; waddr = 5'd0; wdata = 32'd1;
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

        // row 32: write R15 = 111
        wen = 1; waddr = 5'd15; wdata = 32'd111;
        @(posedge clk); #1; wen = 0;

        // row 33: write R16 = 222
        wen = 1; waddr = 5'd16; wdata = 32'd222;
        @(posedge clk); #1; wen = 0;

        // row 34: write R17 = 333
        wen = 1; waddr = 5'd17; wdata = 32'd333;
        @(posedge clk); #1; wen = 0;

        // row 35: write R1 = 500
        wen = 1; waddr = 5'd1; wdata = 32'd500;
        @(posedge clk); #1; wen = 0;

        // row 36: write R18 = 444
        wen = 1; waddr = 5'd18; wdata = 32'd444;
        @(posedge clk); #1; wen = 0;

        // row 37: write R19 = 555
        wen = 1; waddr = 5'd19; wdata = 32'd555;
        @(posedge clk); #1; wen = 0;

        // row 38: write R20 = 666
        wen = 1; waddr = 5'd20; wdata = 32'd666;
        @(posedge clk); #1; wen = 0;

        // row 39: read R1, expect 500 [BDD:RF-READ-003/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd500) begin
            passed = passed + 1;
            $display("✓ row 39: PASS - R%0d = %0d [BDD:RF-READ-003/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 39: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r1]", 5'd1, 32'd500, rdata1);
        end

        // row 40: read R2, expect 255 [BDD:RF-READ-003/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd255) begin
            passed = passed + 1;
            $display("✓ row 40: PASS - R%0d = %0d [BDD:RF-READ-003/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 40: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r2]", 5'd2, 32'd255, rdata1);
        end

        // row 41: read R3, expect 12345 [BDD:RF-READ-003/r3]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd12345) begin
            passed = passed + 1;
            $display("✓ row 41: PASS - R%0d = %0d [BDD:RF-READ-003/r3]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 41: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r3]", 5'd3, 32'd12345, rdata1);
        end

        // row 42: read R4, expect 987654321 [BDD:RF-READ-003/r4]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd987654321) begin
            passed = passed + 1;
            $display("✓ row 42: PASS - R%0d = %0d [BDD:RF-READ-003/r4]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 42: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r4]", 5'd4, 32'd987654321, rdata1);
        end

        // row 43: read R5, expect 4294967295 [BDD:RF-READ-003/r5]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ row 43: PASS - R%0d = %0d [BDD:RF-READ-003/r5]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 43: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r5]", 5'd5, 32'd4294967295, rdata1);
        end

        // row 44: read R6, expect 1 [BDD:RF-READ-003/r6]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd1) begin
            passed = passed + 1;
            $display("✓ row 44: PASS - R%0d = %0d [BDD:RF-READ-003/r6]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 44: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r6]", 5'd6, 32'd1, rdata1);
        end

        // row 45: read R7, expect 16777215 [BDD:RF-READ-003/r7]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd16777215) begin
            passed = passed + 1;
            $display("✓ row 45: PASS - R%0d = %0d [BDD:RF-READ-003/r7]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 45: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r7]", 5'd7, 32'd16777215, rdata1);
        end

        // row 46: read R8, expect 42 [BDD:RF-READ-003/r8]
        raddr1 = 5'd8; #1;
        total = total + 1;
        if (rdata1 == 32'd42) begin
            passed = passed + 1;
            $display("✓ row 46: PASS - R%0d = %0d [BDD:RF-READ-003/r8]", 5'd8, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 46: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r8]", 5'd8, 32'd42, rdata1);
        end

        // row 47: read R9, expect 1024 [BDD:RF-READ-003/r9]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd1024) begin
            passed = passed + 1;
            $display("✓ row 47: PASS - R%0d = %0d [BDD:RF-READ-003/r9]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 47: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r9]", 5'd9, 32'd1024, rdata1);
        end

        // row 48: read R10, expect 65535 [BDD:RF-READ-003/r10]
        raddr1 = 5'd10; #1;
        total = total + 1;
        if (rdata1 == 32'd65535) begin
            passed = passed + 1;
            $display("✓ row 48: PASS - R%0d = %0d [BDD:RF-READ-003/r10]", 5'd10, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 48: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r10]", 5'd10, 32'd65535, rdata1);
        end

        // row 49: read R11, expect 2147483647 [BDD:RF-READ-003/r11]
        raddr1 = 5'd11; #1;
        total = total + 1;
        if (rdata1 == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ row 49: PASS - R%0d = %0d [BDD:RF-READ-003/r11]", 5'd11, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 49: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r11]", 5'd11, 32'd2147483647, rdata1);
        end

        // row 50: read R12, expect 0 [BDD:RF-READ-003/r12]
        raddr1 = 5'd12; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 50: PASS - R%0d = %0d [BDD:RF-READ-003/r12]", 5'd12, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 50: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r12]", 5'd12, 32'd0, rdata1);
        end

        // row 51: read R13, expect 13579 [BDD:RF-READ-003/r13]
        raddr1 = 5'd13; #1;
        total = total + 1;
        if (rdata1 == 32'd13579) begin
            passed = passed + 1;
            $display("✓ row 51: PASS - R%0d = %0d [BDD:RF-READ-003/r13]", 5'd13, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 51: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r13]", 5'd13, 32'd13579, rdata1);
        end

        // row 52: read R14, expect 24680 [BDD:RF-READ-003/r14]
        raddr1 = 5'd14; #1;
        total = total + 1;
        if (rdata1 == 32'd24680) begin
            passed = passed + 1;
            $display("✓ row 52: PASS - R%0d = %0d [BDD:RF-READ-003/r14]", 5'd14, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 52: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r14]", 5'd14, 32'd24680, rdata1);
        end

        // row 53: read R15, expect 111 [BDD:RF-READ-003/r15]
        raddr1 = 5'd15; #1;
        total = total + 1;
        if (rdata1 == 32'd111) begin
            passed = passed + 1;
            $display("✓ row 53: PASS - R%0d = %0d [BDD:RF-READ-003/r15]", 5'd15, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 53: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r15]", 5'd15, 32'd111, rdata1);
        end

        // row 54: read R16, expect 222 [BDD:RF-READ-003/r16]
        raddr1 = 5'd16; #1;
        total = total + 1;
        if (rdata1 == 32'd222) begin
            passed = passed + 1;
            $display("✓ row 54: PASS - R%0d = %0d [BDD:RF-READ-003/r16]", 5'd16, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 54: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r16]", 5'd16, 32'd222, rdata1);
        end

        // row 55: read R17, expect 333 [BDD:RF-READ-003/r17]
        raddr1 = 5'd17; #1;
        total = total + 1;
        if (rdata1 == 32'd333) begin
            passed = passed + 1;
            $display("✓ row 55: PASS - R%0d = %0d [BDD:RF-READ-003/r17]", 5'd17, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 55: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r17]", 5'd17, 32'd333, rdata1);
        end

        // row 56: read R18, expect 444 [BDD:RF-READ-003/r18]
        raddr1 = 5'd18; #1;
        total = total + 1;
        if (rdata1 == 32'd444) begin
            passed = passed + 1;
            $display("✓ row 56: PASS - R%0d = %0d [BDD:RF-READ-003/r18]", 5'd18, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 56: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r18]", 5'd18, 32'd444, rdata1);
        end

        // row 57: read R19, expect 555 [BDD:RF-READ-003/r19]
        raddr1 = 5'd19; #1;
        total = total + 1;
        if (rdata1 == 32'd555) begin
            passed = passed + 1;
            $display("✓ row 57: PASS - R%0d = %0d [BDD:RF-READ-003/r19]", 5'd19, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 57: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r19]", 5'd19, 32'd555, rdata1);
        end

        // row 58: read R20, expect 666 [BDD:RF-READ-003/r20]
        raddr1 = 5'd20; #1;
        total = total + 1;
        if (rdata1 == 32'd666) begin
            passed = passed + 1;
            $display("✓ row 58: PASS - R%0d = %0d [BDD:RF-READ-003/r20]", 5'd20, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 58: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-003/r20]", 5'd20, 32'd666, rdata1);
        end

        // row 59: write R1 = 777
        wen = 1; waddr = 5'd1; wdata = 32'd777;
        @(posedge clk); #1; wen = 0;

        // row 60: write R21 = 888
        wen = 1; waddr = 5'd21; wdata = 32'd888;
        @(posedge clk); #1; wen = 0;

        // row 61: write R22 = 999
        wen = 1; waddr = 5'd22; wdata = 32'd999;
        @(posedge clk); #1; wen = 0;

        // row 62: write R23 = 1010
        wen = 1; waddr = 5'd23; wdata = 32'd1010;
        @(posedge clk); #1; wen = 0;

        // row 63: write R24 = 1111
        wen = 1; waddr = 5'd24; wdata = 32'd1111;
        @(posedge clk); #1; wen = 0;

        // row 64: read R1, expect 777 [BDD:RF-READ-004/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd777) begin
            passed = passed + 1;
            $display("✓ row 64: PASS - R%0d = %0d [BDD:RF-READ-004/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 64: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r1]", 5'd1, 32'd777, rdata1);
        end

        // row 65: read R21, expect 888 [BDD:RF-READ-004/r2]
        raddr1 = 5'd21; #1;
        total = total + 1;
        if (rdata1 == 32'd888) begin
            passed = passed + 1;
            $display("✓ row 65: PASS - R%0d = %0d [BDD:RF-READ-004/r2]", 5'd21, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 65: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r2]", 5'd21, 32'd888, rdata1);
        end

        // row 66: read R22, expect 999 [BDD:RF-READ-004/r3]
        raddr1 = 5'd22; #1;
        total = total + 1;
        if (rdata1 == 32'd999) begin
            passed = passed + 1;
            $display("✓ row 66: PASS - R%0d = %0d [BDD:RF-READ-004/r3]", 5'd22, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 66: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r3]", 5'd22, 32'd999, rdata1);
        end

        // row 67: read R23, expect 1010 [BDD:RF-READ-004/r4]
        raddr1 = 5'd23; #1;
        total = total + 1;
        if (rdata1 == 32'd1010) begin
            passed = passed + 1;
            $display("✓ row 67: PASS - R%0d = %0d [BDD:RF-READ-004/r4]", 5'd23, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 67: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r4]", 5'd23, 32'd1010, rdata1);
        end

        // row 68: read R24, expect 1111 [BDD:RF-READ-004/r5]
        raddr1 = 5'd24; #1;
        total = total + 1;
        if (rdata1 == 32'd1111) begin
            passed = passed + 1;
            $display("✓ row 68: PASS - R%0d = %0d [BDD:RF-READ-004/r5]", 5'd24, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 68: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-004/r5]", 5'd24, 32'd1111, rdata1);
        end

        // row 69: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 70: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 71: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 72: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 73: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 74: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 75: reset
        rst_n = 0; @(posedge clk); #1; rst_n = 1; #1;

        // row 76: read R1, expect 0 [BDD:RF-READ-005/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 76: PASS - R%0d = %0d [BDD:RF-READ-005/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 76: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r1]", 5'd1, 32'd0, rdata1);
        end

        // row 77: read R2, expect 0 [BDD:RF-READ-005/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 77: PASS - R%0d = %0d [BDD:RF-READ-005/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 77: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r2]", 5'd2, 32'd0, rdata1);
        end

        // row 78: read R3, expect 0 [BDD:RF-READ-005/r3]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 78: PASS - R%0d = %0d [BDD:RF-READ-005/r3]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 78: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r3]", 5'd3, 32'd0, rdata1);
        end

        // row 79: read R4, expect 0 [BDD:RF-READ-005/r4]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 79: PASS - R%0d = %0d [BDD:RF-READ-005/r4]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 79: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r4]", 5'd4, 32'd0, rdata1);
        end

        // row 80: read R5, expect 0 [BDD:RF-READ-005/r5]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 80: PASS - R%0d = %0d [BDD:RF-READ-005/r5]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 80: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r5]", 5'd5, 32'd0, rdata1);
        end

        // row 81: read R6, expect 0 [BDD:RF-READ-005/r6]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 81: PASS - R%0d = %0d [BDD:RF-READ-005/r6]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 81: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r6]", 5'd6, 32'd0, rdata1);
        end

        // row 82: read R7, expect 0 [BDD:RF-READ-005/r7]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 82: PASS - R%0d = %0d [BDD:RF-READ-005/r7]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 82: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-005/r7]", 5'd7, 32'd0, rdata1);
        end

        // row 83: write R1 = 1000
        wen = 1; waddr = 5'd1; wdata = 32'd1000;
        @(posedge clk); #1; wen = 0;

        // row 84: write R2 = 2000
        wen = 1; waddr = 5'd2; wdata = 32'd2000;
        @(posedge clk); #1; wen = 0;

        // row 85: write R3 = 3000
        wen = 1; waddr = 5'd3; wdata = 32'd3000;
        @(posedge clk); #1; wen = 0;

        // row 86: write R4 = 4000
        wen = 1; waddr = 5'd4; wdata = 32'd4000;
        @(posedge clk); #1; wen = 0;

        // row 87: write R0 = 5000
        wen = 1; waddr = 5'd0; wdata = 32'd5000;
        @(posedge clk); #1; wen = 0;

        // row 88: write R5 = 6000
        wen = 1; waddr = 5'd5; wdata = 32'd6000;
        @(posedge clk); #1; wen = 0;

        // row 89: write R6 = 7000
        wen = 1; waddr = 5'd6; wdata = 32'd7000;
        @(posedge clk); #1; wen = 0;

        // row 90: read R1, expect 1000 [BDD:RF-READ-006/r1]
        raddr1 = 5'd1; #1;
        total = total + 1;
        if (rdata1 == 32'd1000) begin
            passed = passed + 1;
            $display("✓ row 90: PASS - R%0d = %0d [BDD:RF-READ-006/r1]", 5'd1, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 90: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r1]", 5'd1, 32'd1000, rdata1);
        end

        // row 91: read R2, expect 2000 [BDD:RF-READ-006/r2]
        raddr1 = 5'd2; #1;
        total = total + 1;
        if (rdata1 == 32'd2000) begin
            passed = passed + 1;
            $display("✓ row 91: PASS - R%0d = %0d [BDD:RF-READ-006/r2]", 5'd2, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 91: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r2]", 5'd2, 32'd2000, rdata1);
        end

        // row 92: read R3, expect 3000 [BDD:RF-READ-006/r3]
        raddr1 = 5'd3; #1;
        total = total + 1;
        if (rdata1 == 32'd3000) begin
            passed = passed + 1;
            $display("✓ row 92: PASS - R%0d = %0d [BDD:RF-READ-006/r3]", 5'd3, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 92: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r3]", 5'd3, 32'd3000, rdata1);
        end

        // row 93: read R4, expect 4000 [BDD:RF-READ-006/r4]
        raddr1 = 5'd4; #1;
        total = total + 1;
        if (rdata1 == 32'd4000) begin
            passed = passed + 1;
            $display("✓ row 93: PASS - R%0d = %0d [BDD:RF-READ-006/r4]", 5'd4, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 93: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r4]", 5'd4, 32'd4000, rdata1);
        end

        // row 94: read R0, expect 0 [BDD:RF-READ-006/r5]
        raddr1 = 5'd0; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 94: PASS - R%0d = %0d [BDD:RF-READ-006/r5]", 5'd0, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 94: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r5]", 5'd0, 32'd0, rdata1);
        end

        // row 95: read R5, expect 6000 [BDD:RF-READ-006/r6]
        raddr1 = 5'd5; #1;
        total = total + 1;
        if (rdata1 == 32'd6000) begin
            passed = passed + 1;
            $display("✓ row 95: PASS - R%0d = %0d [BDD:RF-READ-006/r6]", 5'd5, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 95: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r6]", 5'd5, 32'd6000, rdata1);
        end

        // row 96: read R6, expect 7000 [BDD:RF-READ-006/r7]
        raddr1 = 5'd6; #1;
        total = total + 1;
        if (rdata1 == 32'd7000) begin
            passed = passed + 1;
            $display("✓ row 96: PASS - R%0d = %0d [BDD:RF-READ-006/r7]", 5'd6, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 96: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r7]", 5'd6, 32'd7000, rdata1);
        end

        // row 97: read R7, expect 0 [BDD:RF-READ-006/r8]
        raddr1 = 5'd7; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 97: PASS - R%0d = %0d [BDD:RF-READ-006/r8]", 5'd7, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 97: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r8]", 5'd7, 32'd0, rdata1);
        end

        // row 98: read R8, expect 0 [BDD:RF-READ-006/r9]
        raddr1 = 5'd8; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 98: PASS - R%0d = %0d [BDD:RF-READ-006/r9]", 5'd8, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 98: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r9]", 5'd8, 32'd0, rdata1);
        end

        // row 99: read R9, expect 0 [BDD:RF-READ-006/r10]
        raddr1 = 5'd9; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 99: PASS - R%0d = %0d [BDD:RF-READ-006/r10]", 5'd9, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 99: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-006/r10]", 5'd9, 32'd0, rdata1);
        end

        // row 107: read R25, expect 0 [BDD:RF-READ-007/r1]
        raddr1 = 5'd25; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 107: PASS - R%0d = %0d [BDD:RF-READ-007/r1]", 5'd25, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 107: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r1]", 5'd25, 32'd0, rdata1);
        end

        // row 108: read R26, expect 0 [BDD:RF-READ-007/r2]
        raddr1 = 5'd26; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 108: PASS - R%0d = %0d [BDD:RF-READ-007/r2]", 5'd26, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 108: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r2]", 5'd26, 32'd0, rdata1);
        end

        // row 109: read R27, expect 0 [BDD:RF-READ-007/r3]
        raddr1 = 5'd27; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 109: PASS - R%0d = %0d [BDD:RF-READ-007/r3]", 5'd27, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 109: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r3]", 5'd27, 32'd0, rdata1);
        end

        // row 110: read R28, expect 0 [BDD:RF-READ-007/r4]
        raddr1 = 5'd28; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 110: PASS - R%0d = %0d [BDD:RF-READ-007/r4]", 5'd28, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 110: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r4]", 5'd28, 32'd0, rdata1);
        end

        // row 111: read R29, expect 0 [BDD:RF-READ-007/r5]
        raddr1 = 5'd29; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 111: PASS - R%0d = %0d [BDD:RF-READ-007/r5]", 5'd29, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 111: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r5]", 5'd29, 32'd0, rdata1);
        end

        // row 112: read R30, expect 0 [BDD:RF-READ-007/r6]
        raddr1 = 5'd30; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 112: PASS - R%0d = %0d [BDD:RF-READ-007/r6]", 5'd30, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 112: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r6]", 5'd30, 32'd0, rdata1);
        end

        // row 113: read R31, expect 0 [BDD:RF-READ-007/r7]
        raddr1 = 5'd31; #1;
        total = total + 1;
        if (rdata1 == 32'd0) begin
            passed = passed + 1;
            $display("✓ row 113: PASS - R%0d = %0d [BDD:RF-READ-007/r7]", 5'd31, rdata1);
        end else begin
            failed = failed + 1;
            $display("✗ row 113: FAIL - R%0d expected %0d, got %0d [BDD:RF-READ-007/r7]", 5'd31, 32'd0, rdata1);
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