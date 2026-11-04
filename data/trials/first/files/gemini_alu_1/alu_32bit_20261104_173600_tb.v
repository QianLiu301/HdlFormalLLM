// ==========================================================================
// ALU Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: gemini
// Generated: 2026-11-04 17:36:00
// Bitwidth: 32
// Test Cases: 65
// Quality Score: 92.2/100
// ==========================================================================

`timescale 1ns/1ps

module alu_tb;

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
    alu dut (
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
        $dumpfile("alu_tb.vcd");
        $dumpvars(0, alu_tb);
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
        $display("ALU Testbench - gemini");
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
        b = 32'd4294967295;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd4294967294) begin
            passed = passed + 1;
            $display("✓ Test 2: PASS [BDD:ALU-ADD-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 2: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r2]", 32'd4294967294, result);
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
        a = 32'd2147483648;
        b = 32'd4294967295;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ Test 4: PASS [BDD:ALU-ADD-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 4: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r4]", 32'd2147483647, result);
        end

        // Test 5 [BDD:ALU-ADD-001/r5]
        a = 32'd100;
        b = 32'd50;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd150) begin
            passed = passed + 1;
            $display("✓ Test 5: PASS [BDD:ALU-ADD-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 5: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r5]", 32'd150, result);
        end

        // Test 6 [BDD:ALU-ADD-001/r6]
        a = 32'd4294967196;
        b = 32'd4294967246;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd4294967146) begin
            passed = passed + 1;
            $display("✓ Test 6: PASS [BDD:ALU-ADD-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 6: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r6]", 32'd4294967146, result);
        end

        // Test 7 [BDD:ALU-SUB-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 7: PASS [BDD:ALU-SUB-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 7: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r1]", 32'd0, result);
        end

        // Test 8 [BDD:ALU-SUB-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 8: PASS [BDD:ALU-SUB-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 8: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r2]", 32'd0, result);
        end

        // Test 9 [BDD:ALU-SUB-001/r3]
        a = 32'd2147483647;
        b = 32'd4294967295;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 9: PASS [BDD:ALU-SUB-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 9: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r3]", 32'd2147483648, result);
        end

        // Test 10 [BDD:ALU-SUB-001/r4]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ Test 10: PASS [BDD:ALU-SUB-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 10: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r4]", 32'd2147483647, result);
        end

        // Test 11 [BDD:ALU-SUB-001/r5]
        a = 32'd100;
        b = 32'd50;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd50) begin
            passed = passed + 1;
            $display("✓ Test 11: PASS [BDD:ALU-SUB-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 11: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r5]", 32'd50, result);
        end

        // Test 12 [BDD:ALU-SUB-001/r6]
        a = 32'd4294967196;
        b = 32'd4294967246;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd4294967290) begin
            passed = passed + 1;
            $display("✓ Test 12: PASS [BDD:ALU-SUB-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 12: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r6]", 32'd4294967290, result);
        end

        // Test 13 [BDD:ALU-AND-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 13: PASS [BDD:ALU-AND-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 13: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r1]", 32'd0, result);
        end

        // Test 14 [BDD:ALU-AND-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 14: PASS [BDD:ALU-AND-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 14: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r2]", 32'd4294967295, result);
        end

        // Test 15 [BDD:ALU-AND-001/r3]
        a = 32'd0;
        b = 32'd4294967295;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 15: PASS [BDD:ALU-AND-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 15: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r3]", 32'd0, result);
        end

        // Test 16 [BDD:ALU-AND-001/r4]
        a = 32'd170;
        b = 32'd85;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 16: PASS [BDD:ALU-AND-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 16: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r4]", 32'd0, result);
        end

        // Test 17 [BDD:ALU-AND-001/r5]
        a = 32'd2147483648;
        b = 32'd4294967295;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 17: PASS [BDD:ALU-AND-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 17: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r5]", 32'd2147483648, result);
        end

        // Test 18 [BDD:ALU-AND-001/r6]
        a = 32'd123456789;
        b = 32'd987654321;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd67108865) begin
            passed = passed + 1;
            $display("✓ Test 18: PASS [BDD:ALU-AND-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 18: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r6]", 32'd67108865, result);
        end

        // Test 19 [BDD:ALU-OR-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 19: PASS [BDD:ALU-OR-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 19: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r1]", 32'd0, result);
        end

        // Test 20 [BDD:ALU-OR-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 20: PASS [BDD:ALU-OR-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 20: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r2]", 32'd4294967295, result);
        end

        // Test 21 [BDD:ALU-OR-001/r3]
        a = 32'd0;
        b = 32'd4294967295;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 21: PASS [BDD:ALU-OR-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 21: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r3]", 32'd4294967295, result);
        end

        // Test 22 [BDD:ALU-OR-001/r4]
        a = 32'd170;
        b = 32'd85;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd255) begin
            passed = passed + 1;
            $display("✓ Test 22: PASS [BDD:ALU-OR-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 22: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r4]", 32'd255, result);
        end

        // Test 23 [BDD:ALU-OR-001/r5]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd2147483649) begin
            passed = passed + 1;
            $display("✓ Test 23: PASS [BDD:ALU-OR-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 23: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r5]", 32'd2147483649, result);
        end

        // Test 24 [BDD:ALU-OR-001/r6]
        a = 32'd123456789;
        b = 32'd987654321;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd1044012757) begin
            passed = passed + 1;
            $display("✓ Test 24: PASS [BDD:ALU-OR-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 24: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r6]", 32'd1044012757, result);
        end

        // Test 25 [BDD:ALU-XOR-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 25: PASS [BDD:ALU-XOR-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 25: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r1]", 32'd0, result);
        end

        // Test 26 [BDD:ALU-XOR-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 26: PASS [BDD:ALU-XOR-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 26: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r2]", 32'd0, result);
        end

        // Test 27 [BDD:ALU-XOR-001/r3]
        a = 32'd0;
        b = 32'd4294967295;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 27: PASS [BDD:ALU-XOR-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 27: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r3]", 32'd4294967295, result);
        end

        // Test 28 [BDD:ALU-XOR-001/r4]
        a = 32'd170;
        b = 32'd85;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd255) begin
            passed = passed + 1;
            $display("✓ Test 28: PASS [BDD:ALU-XOR-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 28: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r4]", 32'd255, result);
        end

        // Test 29 [BDD:ALU-XOR-001/r5]
        a = 32'd2147483648;
        b = 32'd4294967295;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ Test 29: PASS [BDD:ALU-XOR-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 29: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r5]", 32'd2147483647, result);
        end

        // Test 30 [BDD:ALU-XOR-001/r6]
        a = 32'd123456789;
        b = 32'd987654321;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd976903892) begin
            passed = passed + 1;
            $display("✓ Test 30: PASS [BDD:ALU-XOR-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 30: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r6]", 32'd976903892, result);
        end

        // Test 31 [BDD:ALU-SLL-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 31: PASS [BDD:ALU-SLL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 31: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r1]", 32'd0, result);
        end

        // Test 32 [BDD:ALU-SLL-001/r2]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 32: PASS [BDD:ALU-SLL-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 32: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r2]", 32'd4294967295, result);
        end

        // Test 33 [BDD:ALU-SLL-001/r3]
        a = 32'd1;
        b = 32'd1;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd2) begin
            passed = passed + 1;
            $display("✓ Test 33: PASS [BDD:ALU-SLL-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 33: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r3]", 32'd2, result);
        end

        // Test 34 [BDD:ALU-SLL-001/r4]
        a = 32'd1;
        b = 32'd31;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd2147483648) begin
            passed = passed + 1;
            $display("✓ Test 34: PASS [BDD:ALU-SLL-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 34: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r4]", 32'd2147483648, result);
        end

        // Test 35 [BDD:ALU-SLL-001/r5]
        a = 32'd1;
        b = 32'd32;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 35: PASS [BDD:ALU-SLL-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 35: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r5]", 32'd1, result);
        end

        // Test 36 [BDD:ALU-SLL-001/r6]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 36: PASS [BDD:ALU-SLL-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 36: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r6]", 32'd0, result);
        end

        // Test 37 [BDD:ALU-SLL-001/r7]
        a = 32'd12345;
        b = 32'd5;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd395040) begin
            passed = passed + 1;
            $display("✓ Test 37: PASS [BDD:ALU-SLL-001/r7]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 37: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r7]", 32'd395040, result);
        end

        // Test 38 [BDD:ALU-SRL-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 38: PASS [BDD:ALU-SRL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 38: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r1]", 32'd0, result);
        end

        // Test 39 [BDD:ALU-SRL-001/r2]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 39: PASS [BDD:ALU-SRL-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 39: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r2]", 32'd4294967295, result);
        end

        // Test 40 [BDD:ALU-SRL-001/r3]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd2147483647) begin
            passed = passed + 1;
            $display("✓ Test 40: PASS [BDD:ALU-SRL-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 40: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r3]", 32'd2147483647, result);
        end

        // Test 41 [BDD:ALU-SRL-001/r4]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd1073741824) begin
            passed = passed + 1;
            $display("✓ Test 41: PASS [BDD:ALU-SRL-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 41: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r4]", 32'd1073741824, result);
        end

        // Test 42 [BDD:ALU-SRL-001/r5]
        a = 32'd1;
        b = 32'd31;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 42: PASS [BDD:ALU-SRL-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 42: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r5]", 32'd0, result);
        end

        // Test 43 [BDD:ALU-SRL-001/r6]
        a = 32'd1;
        b = 32'd32;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 43: PASS [BDD:ALU-SRL-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 43: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r6]", 32'd1, result);
        end

        // Test 44 [BDD:ALU-SRL-001/r7]
        a = 32'd123456789;
        b = 32'd5;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd3858024) begin
            passed = passed + 1;
            $display("✓ Test 44: PASS [BDD:ALU-SRL-001/r7]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 44: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r7]", 32'd3858024, result);
        end

        // Test 45 [BDD:ALU-SRA-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 45: PASS [BDD:ALU-SRA-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 45: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r1]", 32'd0, result);
        end

        // Test 46 [BDD:ALU-SRA-001/r2]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 46: PASS [BDD:ALU-SRA-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 46: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r2]", 32'd4294967295, result);
        end

        // Test 47 [BDD:ALU-SRA-001/r3]
        a = 32'd4294967295;
        b = 32'd1;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd4294967295) begin
            passed = passed + 1;
            $display("✓ Test 47: PASS [BDD:ALU-SRA-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 47: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r3]", 32'd4294967295, result);
        end

        // Test 48 [BDD:ALU-SRA-001/r4]
        a = 32'd2147483648;
        b = 32'd1;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd3221225472) begin
            passed = passed + 1;
            $display("✓ Test 48: PASS [BDD:ALU-SRA-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 48: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r4]", 32'd3221225472, result);
        end

        // Test 49 [BDD:ALU-SRA-001/r5]
        a = 32'd1;
        b = 32'd31;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 49: PASS [BDD:ALU-SRA-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 49: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r5]", 32'd0, result);
        end

        // Test 50 [BDD:ALU-SRA-001/r6]
        a = 32'd1;
        b = 32'd32;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 50: PASS [BDD:ALU-SRA-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 50: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r6]", 32'd1, result);
        end

        // Test 51 [BDD:ALU-SRA-001/r7]
        a = 32'd123456789;
        b = 32'd5;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd3858024) begin
            passed = passed + 1;
            $display("✓ Test 51: PASS [BDD:ALU-SRA-001/r7]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 51: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r7]", 32'd3858024, result);
        end

        // Test 52 [BDD:ALU-SLT-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 52: PASS [BDD:ALU-SLT-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 52: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r1]", 32'd0, result);
        end

        // Test 53 [BDD:ALU-SLT-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 53: PASS [BDD:ALU-SLT-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 53: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r2]", 32'd0, result);
        end

        // Test 54 [BDD:ALU-SLT-001/r3]
        a = 32'd1;
        b = 32'd0;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 54: PASS [BDD:ALU-SLT-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 54: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r3]", 32'd0, result);
        end

        // Test 55 [BDD:ALU-SLT-001/r4]
        a = 32'd0;
        b = 32'd1;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 55: PASS [BDD:ALU-SLT-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 55: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r4]", 32'd1, result);
        end

        // Test 56 [BDD:ALU-SLT-001/r5]
        a = 32'd2147483647;
        b = 32'd2147483648;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 56: PASS [BDD:ALU-SLT-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 56: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r5]", 32'd0, result);
        end

        // Test 57 [BDD:ALU-SLT-001/r6]
        a = 32'd2147483648;
        b = 32'd2147483647;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 57: PASS [BDD:ALU-SLT-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 57: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r6]", 32'd1, result);
        end

        // Test 58 [BDD:ALU-SLT-001/r7]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 58: PASS [BDD:ALU-SLT-001/r7]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 58: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r7]", 32'd1, result);
        end

        // Test 59 [BDD:ALU-SLTU-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 59: PASS [BDD:ALU-SLTU-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 59: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r1]", 32'd0, result);
        end

        // Test 60 [BDD:ALU-SLTU-001/r2]
        a = 32'd4294967295;
        b = 32'd4294967295;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 60: PASS [BDD:ALU-SLTU-001/r2]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 60: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r2]", 32'd0, result);
        end

        // Test 61 [BDD:ALU-SLTU-001/r3]
        a = 32'd1;
        b = 32'd0;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 61: PASS [BDD:ALU-SLTU-001/r3]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 61: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r3]", 32'd0, result);
        end

        // Test 62 [BDD:ALU-SLTU-001/r4]
        a = 32'd0;
        b = 32'd1;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 62: PASS [BDD:ALU-SLTU-001/r4]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 62: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r4]", 32'd1, result);
        end

        // Test 63 [BDD:ALU-SLTU-001/r5]
        a = 32'd2147483647;
        b = 32'd2147483648;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd1) begin
            passed = passed + 1;
            $display("✓ Test 63: PASS [BDD:ALU-SLTU-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 63: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r5]", 32'd1, result);
        end

        // Test 64 [BDD:ALU-SLTU-001/r6]
        a = 32'd2147483648;
        b = 32'd2147483647;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 64: PASS [BDD:ALU-SLTU-001/r6]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 64: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r6]", 32'd0, result);
        end

        // Test 65 [BDD:ALU-SLTU-001/r7]
        a = 32'd4294967295;
        b = 32'd0;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 65: PASS [BDD:ALU-SLTU-001/r7]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 65: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r7]", 32'd0, result);
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