// ==========================================================================
// ALU Testbench - Generated from BDD
// ==========================================================================
// LLM Provider: gemini
// Generated: 2026-11-04 17:38:19
// Bitwidth: 32
// Test Cases: 12
// Quality Score: 68.8/100
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

        // Test 2 [BDD:ALU-ADD-001/r5]
        a = 32'd100;
        b = 32'd200;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        if (result == 32'd300) begin
            passed = passed + 1;
            $display("✓ Test 2: PASS [BDD:ALU-ADD-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 2: FAIL - Expected %d, Got %d [BDD:ALU-ADD-001/r5]", 32'd300, result);
        end

        // Test 3 [BDD:ALU-SUB-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 3: PASS [BDD:ALU-SUB-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 3: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r1]", 32'd0, result);
        end

        // Test 4 [BDD:ALU-SUB-001/r5]
        a = 32'd300;
        b = 32'd100;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        if (result == 32'd200) begin
            passed = passed + 1;
            $display("✓ Test 4: PASS [BDD:ALU-SUB-001/r5]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 4: FAIL - Expected %d, Got %d [BDD:ALU-SUB-001/r5]", 32'd200, result);
        end

        // Test 5 [BDD:ALU-AND-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 5: PASS [BDD:ALU-AND-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 5: FAIL - Expected %d, Got %d [BDD:ALU-AND-001/r1]", 32'd0, result);
        end

        // Test 6 [BDD:ALU-OR-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 6: PASS [BDD:ALU-OR-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 6: FAIL - Expected %d, Got %d [BDD:ALU-OR-001/r1]", 32'd0, result);
        end

        // Test 7 [BDD:ALU-XOR-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0100;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 7: PASS [BDD:ALU-XOR-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 7: FAIL - Expected %d, Got %d [BDD:ALU-XOR-001/r1]", 32'd0, result);
        end

        // Test 8 [BDD:ALU-SLL-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0101;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 8: PASS [BDD:ALU-SLL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 8: FAIL - Expected %d, Got %d [BDD:ALU-SLL-001/r1]", 32'd0, result);
        end

        // Test 9 [BDD:ALU-SRL-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0110;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 9: PASS [BDD:ALU-SRL-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 9: FAIL - Expected %d, Got %d [BDD:ALU-SRL-001/r1]", 32'd0, result);
        end

        // Test 10 [BDD:ALU-SRA-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b0111;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 10: PASS [BDD:ALU-SRA-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 10: FAIL - Expected %d, Got %d [BDD:ALU-SRA-001/r1]", 32'd0, result);
        end

        // Test 11 [BDD:ALU-SLT-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b1000;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 11: PASS [BDD:ALU-SLT-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 11: FAIL - Expected %d, Got %d [BDD:ALU-SLT-001/r1]", 32'd0, result);
        end

        // Test 12 [BDD:ALU-SLTU-001/r1]
        a = 32'd0;
        b = 32'd0;
        opcode = 4'b1001;
        #10;
        total = total + 1;
        if (result == 32'd0) begin
            passed = passed + 1;
            $display("✓ Test 12: PASS [BDD:ALU-SLTU-001/r1]");
        end else begin
            failed = failed + 1;
            $display("✗ Test 12: FAIL - Expected %d, Got %d [BDD:ALU-SLTU-001/r1]", 32'd0, result);
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