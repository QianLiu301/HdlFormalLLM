// ==========================================================================
// ALU Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: qwen
// Generated: 2026-11-10 12:56:10
// Bitwidth: 32
// Test Cases: 50
// Quality Score: 90.6/100
// ==========================================================================

`timescale 1ns/1ps

module alu_32bit_tb;

    // Parameters
    parameter WIDTH = 32;

    // Signals
    reg clk;
    reg rst;
    reg [WIDTH-1:0] a;
    reg [WIDTH-1:0] b;
    reg [3:0] opcode;
    wire [WIDTH-1:0] result;
    wire zero;
    wire overflow;
    wire negative;

    // Test counters
    integer total, passed, failed;

    // DUT instantiation
    alu_32bit dut (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .opcode(opcode),
        .result(result),
        .zero(zero),
        .overflow(overflow),
        .negative(negative)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;


    // Waveform dump for GTKWave
    initial begin
        $dumpfile("alu_32bit_tb.vcd");
        $dumpvars(0, alu_32bit_tb);
    end
    // Test stimulus
    initial begin
        // Initialize
        total = 0;
        passed = 0;
        failed = 0;
        rst = 1;
        a = 0;
        b = 0;
        opcode = 0;
        #20 rst = 0;

        $display("\n======================================================================");
        $display("ALU Testbench - qwen");
        $display("======================================================================\n");

        // Test 1 [BDD:ALU-ADD-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 1: PASS [BDD:ALU-ADD-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 1: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r1]", 32'd0, result);
        end

        // Test 2 [BDD:ALU-ADD-001/r2]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 2: PASS [BDD:ALU-ADD-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 2: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r2]", 32'd0, result);
        end

        // Test 3 [BDD:ALU-ADD-001/r3]
        a = 32'd2147483647;
        b = 32'd1;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 3: PASS [BDD:ALU-ADD-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 3: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r3]", 32'd2147483648, result);
        end

        // Test 4 [BDD:ALU-ADD-001/r4]
        a = 32'd2147483647;
        b = 32'd2147483647;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd4294967294) begin
            passed = passed + 1;
            $display("✓ Test 4: PASS [BDD:ALU-ADD-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 4: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r4]", 32'd4294967294, result);
        end

        // Test 5 [BDD:ALU-ADD-001/r5]
        a = 32'd1000000;
        b = 32'd2000000;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd3000000) begin
            passed = passed + 1;
            $display("✓ Test 5: PASS [BDD:ALU-ADD-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 5: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r5]", 32'd3000000, result);
        end

        // Test 6 [BDD:ALU-SUB-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 6: PASS [BDD:ALU-SUB-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 6: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r1]", 32'd0, result);
        end

        // Test 7 [BDD:ALU-SUB-001/r2]
        a = 32'd0;
        b = 32'd1;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 7: PASS [BDD:ALU-SUB-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 7: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r2]", 32'd4294967295, result);
        end

        // Test 8 [BDD:ALU-SUB-001/r3]
        a = -32'd2147483648;
        b = 32'd2147483647;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 8: PASS [BDD:ALU-SUB-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 8: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r3]", 32'd1, result);
        end

        // Test 9 [BDD:ALU-SUB-001/r4]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 9: PASS [BDD:ALU-SUB-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 9: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r4]", 32'd0, result);
        end

        // Test 10 [BDD:ALU-SUB-001/r5]
        a = 32'd3000000;
        b = 32'd1000000;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd2000000) begin
            passed = passed + 1;
            $display("✓ Test 10: PASS [BDD:ALU-SUB-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 10: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r5]", 32'd2000000, result);
        end

        // Test 11 [BDD:ALU-AND-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 11: PASS [BDD:ALU-AND-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 11: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r1]", 32'd0, result);
        end

        // Test 12 [BDD:ALU-AND-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 12: PASS [BDD:ALU-AND-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 12: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r2]", 32'd4294967295, result);
        end

        // Test 13 [BDD:ALU-AND-001/r3]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 13: PASS [BDD:ALU-AND-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 13: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r3]", 32'd0, result);
        end

        // Test 14 [BDD:ALU-AND-001/r4]
        a = 32'd2863311530;
        b = 32'd3681400539;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd2684354560) begin
            passed = passed + 1;
            $display("✓ Test 14: PASS [BDD:ALU-AND-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 14: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r4]", 32'd2684354560, result);
        end

        // Test 15 [BDD:ALU-AND-001/r5]
        a = 32'd1;
        b = 32'd2;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 15: PASS [BDD:ALU-AND-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 15: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r5]", 32'd0, result);
        end

        // Test 16 [BDD:ALU-OR-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 16: PASS [BDD:ALU-OR-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 16: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r1]", 32'd0, result);
        end

        // Test 17 [BDD:ALU-OR-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 17: PASS [BDD:ALU-OR-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 17: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r2]", 32'd4294967295, result);
        end

        // Test 18 [BDD:ALU-OR-001/r3]
        a = 32'd0;
        b = 32'd4294967295;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 18: PASS [BDD:ALU-OR-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 18: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r3]", 32'd4294967295, result);
        end

        // Test 19 [BDD:ALU-OR-001/r4]
        a = 32'd2863311530;
        b = 32'd3681400539;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 19: PASS [BDD:ALU-OR-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 19: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r4]", 32'd4294967295, result);
        end

        // Test 20 [BDD:ALU-OR-001/r5]
        a = 32'd1;
        b = 32'd2;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd3) begin
            passed = passed + 1;
            $display("✓ Test 20: PASS [BDD:ALU-OR-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 20: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r5]", 32'd3, result);
        end

        // Test 21 [BDD:ALU-XOR-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 21: PASS [BDD:ALU-XOR-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 21: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r1]", 32'd0, result);
        end

        // Test 22 [BDD:ALU-XOR-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 22: PASS [BDD:ALU-XOR-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 22: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r2]", 32'd0, result);
        end

        // Test 23 [BDD:ALU-XOR-001/r3]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 23: PASS [BDD:ALU-XOR-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 23: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r3]", 32'd4294967295, result);
        end

        // Test 24 [BDD:ALU-XOR-001/r4]
        a = 32'd2863311530;
        b = 32'd3681400539;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd1610612735) begin
            passed = passed + 1;
            $display("✓ Test 24: PASS [BDD:ALU-XOR-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 24: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r4]", 32'd1610612735, result);
        end

        // Test 25 [BDD:ALU-XOR-001/r5]
        a = 32'd1;
        b = 32'd2;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd3) begin
            passed = passed + 1;
            $display("✓ Test 25: PASS [BDD:ALU-XOR-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 25: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r5]", 32'd3, result);
        end

        // Test 26 [BDD:ALU-SLL-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 26: PASS [BDD:ALU-SLL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 26: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r1]", 32'd0, result);
        end

        // Test 27 [BDD:ALU-SLL-001/r2]
        a = 32'd1;
        b = 32'd31;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 27: PASS [BDD:ALU-SLL-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 27: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r2]", 32'd2147483648, result);
        end

        // Test 28 [BDD:ALU-SLL-001/r3]
        a = 32'd1;
        b = 32'd32;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 28: PASS [BDD:ALU-SLL-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 28: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r3]", 32'd1, result);
        end

        // Test 29 [BDD:ALU-SLL-001/r4]
        a = 32'd1;
        b = 32'd33;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd2) begin
            passed = passed + 1;
            $display("✓ Test 29: PASS [BDD:ALU-SLL-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 29: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r4]", 32'd2, result);
        end

        // Test 30 [BDD:ALU-SLL-001/r5]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd4294967294) begin
            passed = passed + 1;
            $display("✓ Test 30: PASS [BDD:ALU-SLL-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 30: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r5]", 32'd4294967294, result);
        end

        // Test 31 [BDD:ALU-SRL-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 31: PASS [BDD:ALU-SRL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 31: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r1]", 32'd0, result);
        end

        // Test 32 [BDD:ALU-SRL-001/r2]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd1073741824) begin
            passed = passed + 1;
            $display("✓ Test 32: PASS [BDD:ALU-SRL-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 32: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r2]", 32'd1073741824, result);
        end

        // Test 33 [BDD:ALU-SRL-001/r3]
        a = 32'd2147483648;
        b = 32'd31;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 33: PASS [BDD:ALU-SRL-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 33: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r3]", 32'd1, result);
        end

        // Test 34 [BDD:ALU-SRL-001/r4]
        a = 32'd2147483648;
        b = 32'd32;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 34: PASS [BDD:ALU-SRL-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 34: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r4]", 32'd2147483648, result);
        end

        // Test 35 [BDD:ALU-SRL-001/r5]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ Test 35: PASS [BDD:ALU-SRL-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 35: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r5]", 32'd2147483647, result);
        end

        // Test 36 [BDD:ALU-SRA-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 36: PASS [BDD:ALU-SRA-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 36: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r1]", 32'd0, result);
        end

        // Test 37 [BDD:ALU-SRA-001/r2]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd3221225472) begin
            passed = passed + 1;
            $display("✓ Test 37: PASS [BDD:ALU-SRA-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 37: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r2]", 32'd3221225472, result);
        end

        // Test 38 [BDD:ALU-SRA-001/r3]
        a = 32'd2147483648;
        b = 32'd31;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 38: PASS [BDD:ALU-SRA-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 38: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r3]", 32'd2147483648, result);
        end

        // Test 39 [BDD:ALU-SRA-001/r4]
        a = 32'd2147483648;
        b = 32'd32;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 39: PASS [BDD:ALU-SRA-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 39: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r4]", 32'd2147483648, result);
        end

        // Test 40 [BDD:ALU-SRA-001/r5]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 40: PASS [BDD:ALU-SRA-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 40: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r5]", 32'd4294967295, result);
        end

        // Test 41 [BDD:ALU-SLT-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 41: PASS [BDD:ALU-SLT-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 41: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r1]", 32'd0, result);
        end

        // Test 42 [BDD:ALU-SLT-001/r2]
        a = -32'd1;
        b = 32'd0;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 42: PASS [BDD:ALU-SLT-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 42: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r2]", 32'd1, result);
        end

        // Test 43 [BDD:ALU-SLT-001/r3]
        a = 32'd0;
        b = -32'd1;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 43: PASS [BDD:ALU-SLT-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 43: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r3]", 32'd0, result);
        end

        // Test 44 [BDD:ALU-SLT-001/r4]
        a = 32'd2147483647;
        b = -32'd2147483648;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 44: PASS [BDD:ALU-SLT-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 44: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r4]", 32'd0, result);
        end

        // Test 45 [BDD:ALU-SLT-001/r5]
        a = -32'd2147483648;
        b = 32'd2147483647;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 45: PASS [BDD:ALU-SLT-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 45: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r5]", 32'd1, result);
        end

        // Test 46 [BDD:ALU-SLTU-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 46: PASS [BDD:ALU-SLTU-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 46: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r1]", 32'd0, result);
        end

        // Test 47 [BDD:ALU-SLTU-001/r2]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 47: PASS [BDD:ALU-SLTU-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 47: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r2]", 32'd0, result);
        end

        // Test 48 [BDD:ALU-SLTU-001/r3]
        a = 32'd1;
        b = 32'd4294967295;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 48: PASS [BDD:ALU-SLTU-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 48: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r3]", 32'd1, result);
        end

        // Test 49 [BDD:ALU-SLTU-001/r4]
        a = 32'd2147483647;
        b = -32'd2147483648;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 49: PASS [BDD:ALU-SLTU-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 49: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r4]", 32'd1, result);
        end

        // Test 50 [BDD:ALU-SLTU-001/r5]
        a = -32'd2147483648;
        b = 32'd2147483647;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 50: PASS [BDD:ALU-SLTU-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 50: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r5]", 32'd0, result);
        end

        // Summary
        $display("\n======================================================================");
        $display("Test Summary");
        $display("======================================================================");
        $display("Total:  %0d", total);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);

        if (failed == 0)
            $display("\n🎉 ALL TESTS PASSED!");
        else
            $display("\n⚠️  SOME TESTS FAILED");

        $display("======================================================================\n");
        $finish;
    end

endmodule