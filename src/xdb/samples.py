"""
Canned designs and scenarios, shared by the tests (tests/test_xdb.py), the
browser-check staging (tests/ui/stage_run.py) and the mock model of the trial
runner (src/xdb/trial.py --llms mock).

Each design type has a sample with known faults on both sides, so every
verdict class shows up:

ALU        the Gemini feature of artifact/ablation (variant A) with two wrong
           expected values (SUB r7, SRA r5) and one planted wrong zero flag
           (ADD r7); ALU compares SLTU signed.
regfile    RF_FEATURE expects R1 = 100 after a later row wrote 300; RF_BUGGY
           does not protect register 0.
cpu        CPU_FEATURE: SUB 10 - 30 expects 20, BLT -5 < 3 expects "no", a row
           gives x1 two values, a row writes x0, a row's instruction writes
           another register than its rd column; CPU_BUGGY compares SLT/SLTI
           unsigned.

Fixed versions (FIXED_ALU, RF_FIXED, CPU_OK) are what a successful design
repair returns. Not used by the website itself.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def alu_feature() -> str:
    """The ALU sample feature (needs the repository's artifact/ folder)."""
    feat = (ROOT / 'artifact/ablation/variant_A_original.feature').read_text(encoding='utf-8')
    return (feat.replace('@add @arithmetic', '@add @arithmetic @req_1')
                .replace('@sltu', '@sltu @req_2', 1)
                # plant one flag contradiction (ADD row 7: 100 + 200 = 300 is not zero)
                .replace('| 300             | False     |', '| 300             | True      |', 1))


ALU_RECORD = {'module_type': 'alu', 'created_at': '2026-11-03T10:00:00', 'questions': [], 'decisions': [],
              'requirements': [
                  {'id': 'REQ-1', 'text': 'ADD returns a + b modulo 2^32', 'source': 'llm'},
                  {'id': 'REQ-2', 'text': 'SLTU compares unsigned', 'source': 'llm'},
                  {'id': 'REQ-3', 'text': 'Reset clears outputs', 'source': 'user'}]}


ALU = """
module alu_32bit #(parameter WIDTH = 32) (
    input  wire clk, input wire rst,
    input  wire [WIDTH-1:0] a, b,
    input  wire [3:0] opcode,
    output reg  [WIDTH-1:0] result,
    output wire zero, output reg overflow, output wire negative
);
    always @(*) begin
        overflow = 1'b0;
        case (opcode)
            4'b0000: begin result = a + b;
                     overflow = (a[WIDTH-1] == b[WIDTH-1]) && (result[WIDTH-1] != a[WIDTH-1]); end
            4'b0001: begin result = a - b;
                     overflow = (a[WIDTH-1] != b[WIDTH-1]) && (result[WIDTH-1] != a[WIDTH-1]); end
            4'b0010: result = a & b;
            4'b0011: result = a | b;
            4'b0100: result = a ^ b;
            4'b0101: result = a << b[4:0];
            4'b0110: result = a >> b[4:0];
            4'b0111: result = $signed(a) >>> b[4:0];
            4'b1000: result = ($signed(a) < $signed(b)) ? 1 : 0;
            4'b1001: result = ($signed(a) < $signed(b)) ? 1 : 0;   // BUG: SLTU compares signed
            default: result = 0;
        endcase
    end
    assign zero = (result == 0);
    assign negative = result[WIDTH-1];
endmodule
"""


FIXED_ALU = ALU.replace("4'b1001: result = ($signed(a) < $signed(b)) ? 1 : 0;   // BUG: SLTU compares signed",
                        "4'b1001: result = (a < b) ? 1 : 0;")


RF_FEATURE = """Feature: 32x32-bit Register File Verification
  As a hardware verification engineer
  I want to verify the Register File implementation
  So that I can ensure correct read/write behavior

  Background:
    Given the Register File has 32 registers of 32-bit width

  @write @req_1
  Scenario Outline: Write values into registers
    Given the register file is ready
    When I write <data> to register <addr>
    Then the write is accepted

    Examples:
      | addr | data       |
      | 1    | 100        |
      | 2    | 255        |
      | 5    | 4294967295 |
      | 1    | 300        |

  @read @req_1
  Scenario Outline: Read back values written earlier
    Given the register file holds earlier writes
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected   |
      | 1    | 100        |
      | 2    | 255        |
      | 5    | 4294967295 |
      | 9    | 0          |

  @write_read @req_1
  Scenario Outline: Write and immediately read back
    Given the register file is ready
    When I write <data> to register <addr>
    And I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | data | expected |
      | 7    | 42   | 42       |
      | 12   | 7    | 7        |

  @write @req_2
  Scenario Outline: Writes to register 0 are ignored
    Given the register file is ready
    When I write <data> to register <addr>
    Then the write is discarded

    Examples:
      | addr | data |
      | 0    | 123  |

  @read @req_2
  Scenario Outline: Register 0 always reads as zero
    Given register 0 is hardwired
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected |
      | 0    | 0        |

  @reset @req_3
  Scenario Outline: Reset clears every register
    Given registers hold values
    When reset is asserted
    Then register <addr> is cleared

    Examples:
      | addr |
      | 1    |

  @read @req_3
  Scenario Outline: Everything reads zero after reset
    Given reset has completed
    When I read register <addr>
    Then the value should be <expected>

    Examples:
      | addr | expected |
      | 1    | 0        |
      | 7    | 0        |
"""


RF_BUGGY = """module regfile #(parameter WIDTH = 32, parameter DEPTH = 32, parameter ADDR_WIDTH = 5) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [ADDR_WIDTH-1:0] raddr1,
    output wire [WIDTH-1:0]      rdata1,
    input  wire [ADDR_WIDTH-1:0] raddr2,
    output wire [WIDTH-1:0]      rdata2,
    input  wire                  wen,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [WIDTH-1:0]      wdata
);
    reg [WIDTH-1:0] registers [0:DEPTH-1];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) registers[i] <= 0;
        end else if (wen) begin
            registers[waddr] <= wdata;          // BUG: register 0 is not protected
        end
    end
    assign rdata1 = registers[raddr1];          // BUG: register 0 is not hardwired
    assign rdata2 = (raddr2 == 0) ? 0 : registers[raddr2];
endmodule
"""


RF_FIXED = (RF_BUGGY.replace("registers[waddr] <= wdata;          // BUG: register 0 is not protected",
                             "if (waddr != 0) registers[waddr] <= wdata;")
            .replace("assign rdata1 = registers[raddr1];          // BUG: register 0 is not hardwired",
                     "assign rdata1 = (raddr1 == 0) ? 0 : registers[raddr1];"))


RF_RECORD = {'module_type': 'regfile', 'created_at': '2026-11-03T12:00:00', 'questions': [], 'decisions': [],
             'requirements': [
                 {'id': 'REQ-1', 'text': 'A read returns the value last written to that register.', 'source': 'llm'},
                 {'id': 'REQ-2', 'text': 'Register 0 always reads as zero; writes to it are ignored.', 'source': 'llm'},
                 {'id': 'REQ-3', 'text': 'Reset clears every register.', 'source': 'llm'},
                 {'id': 'REQ-4', 'text': 'Both read ports can read any register at the same time.', 'source': 'llm'}]}


RF_DUT = {'module_type': 'regfile', 'module_name': 'regfile', 'bitwidth': 32, 'depth': 32}


CPU_FEATURE = """Feature: RISC-V CPU Verification (5-stage pipeline)
  As a hardware verification engineer
  I want to verify the RISC-V CPU implementation
  So that I can ensure correct instruction execution

  Background:
    Given the CPU is a 5-stage pipeline RV32I processor
    And all registers are initialized to 0

  @arithmetic @req_1
  Scenario Outline: Instructions that write a register
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction      | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | add x3, x1, x2   | 1   | 10        | 2   | 20        | 3  | 30       |
      | sub x3, x1, x2   | 1   | 10        | 2   | 30        | 3  | 20       |
      | addi x3, x1, 100 | 1   | 50        | 0   | 0         | 3  | 150      |
      | add x3, x1, x1   | 1   | 5         | 1   | 7         | 3  | 12       |
      | addi x0, x1, 5   | 1   | 10        | 0   | 0         | 0  | 15       |
      | add x5, x1, x2   | 1   | 10        | 2   | 20        | 3  | 30       |

  @logical @req_1
  Scenario Outline: Bitwise instructions
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction     | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | and x3, x1, x2  | 1   | 0xFF      | 2   | 0x0F      | 3  | 0x0F     |
      | or x3, x1, x2   | 1   | 0xF0      | 2   | 0x0F      | 3  | 0xFF     |
      | xori x3, x1, 15 | 1   | 0xFF      | 0   | 0         | 3  | 0xF0     |

  @compare @req_2
  Scenario Outline: Signed comparison
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction     | rs1 | rs1_value | rs2 | rs2_value | rd | expected |
      | slt x3, x1, x2  | 1   | -1        | 2   | 1         | 3  | 1        |
      | slt x3, x1, x2  | 1   | 5         | 2   | 3         | 3  | 0        |
      | slti x3, x1, 0  | 1   | -1        | 0   | 0         | 3  | 1        |

  @memory @req_3
  Scenario Outline: Load and store, one instruction per row
    Given register x<rs1> contains address <rs1_value>
    And memory at that address contains <mem_value>
    When I execute <instruction>
    Then register x<rd> should contain <expected>

    Examples:
      | instruction    | rs1 | rs1_value | rs2 | rs2_value | rd | mem_value | expected |
      | lw x3, 0(x1)   | 1   | 0x100     | 0   | 0         | 3  | 0x1234    | 0x1234   |
      | lw x4, 4(x1)   | 1   | 0x180     | 0   | 0         | 4  | 0x55      | 0x55     |
      | sw x2, 0(x1)   | 1   | 0x140     | 2   | 0x7BC     | 0  | 0         | 0x7BC    |

  @branch @req_4
  Scenario Outline: Branch taken or not taken
    Given register x<rs1> contains <rs1_value>
    And register x<rs2> contains <rs2_value>
    When I execute <instruction>
    Then the branch is taken should be <taken>

    Examples:
      | instruction    | rs1 | rs1_value | rs2 | rs2_value | taken |
      | beq x1, x2, 8  | 1   | 10        | 2   | 10        | yes   |
      | beq x1, x2, 8  | 1   | 10        | 2   | 20        | no    |
      | bne x1, x2, 8  | 1   | 10        | 2   | 20        | yes   |
      | blt x1, x2, 8  | 1   | -5        | 2   | 3         | no    |
      | bge x1, x2, 8  | 1   | -5        | 2   | 3         | no    |
"""


CPU_OK = """// Minimal RV32I subset CPU for the explainability tests: one instruction per
// cycle, a load takes two (the testbench's data memory reads synchronously).
module riscv_cpu (
    input  wire        clk,
    input  wire        rst_n,
    output wire [31:0] imem_addr,
    input  wire [31:0] imem_data,
    output wire [31:0] dmem_addr,
    output wire [31:0] dmem_wdata,
    output wire        dmem_wen,
    output wire [3:0]  dmem_be,
    input  wire [31:0] dmem_rdata,
    output wire [31:0] debug_pc,
    output wire [31:0] debug_inst
);
    reg [31:0] pc;
    reg [31:0] x [0:31];
    reg        load_wait;
    integer k;

    wire [31:0] inst = imem_data;
    wire [6:0]  opc  = inst[6:0];
    wire [4:0]  rd   = inst[11:7];
    wire [2:0]  f3   = inst[14:12];
    wire [4:0]  rs1  = inst[19:15];
    wire [4:0]  rs2  = inst[24:20];
    wire [6:0]  f7   = inst[31:25];
    wire [31:0] a    = (rs1 == 0) ? 32'd0 : x[rs1];
    wire [31:0] b    = (rs2 == 0) ? 32'd0 : x[rs2];
    wire [31:0] imm_i = {{20{inst[31]}}, inst[31:20]};
    wire [31:0] imm_s = {{20{inst[31]}}, inst[31:25], inst[11:7]};
    wire [31:0] imm_b = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0};

    wire is_r = opc == 7'b0110011, is_i = opc == 7'b0010011, is_lw = opc == 7'b0000011,
         is_sw = opc == 7'b0100011, is_br = opc == 7'b1100011;
    wire [31:0] op2 = is_r ? b : imm_i;

    reg [31:0] alu;
    always @(*) begin
        case (f3)
            3'b000: alu = (is_r && f7[5]) ? a - op2 : a + op2;
            3'b010: alu = ($signed(a) < $signed(op2)) ? 32'd1 : 32'd0;     // SLT / SLTI
            3'b100: alu = a ^ op2;
            3'b110: alu = a | op2;
            3'b111: alu = a & op2;
            default: alu = 32'd0;
        endcase
    end

    reg taken;
    always @(*) begin
        case (f3)
            3'b000: taken = a == b;
            3'b001: taken = a != b;
            3'b100: taken = $signed(a) < $signed(b);
            3'b101: taken = $signed(a) >= $signed(b);
            default: taken = 1'b0;
        endcase
    end

    assign imem_addr  = pc;
    assign dmem_addr  = is_sw ? a + imm_s : a + imm_i;
    assign dmem_wdata = b;
    assign dmem_wen   = is_sw & rst_n;
    assign dmem_be    = 4'b1111;
    assign debug_pc   = pc;
    assign debug_inst = inst;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 0; load_wait <= 0;
            for (k = 0; k < 32; k = k + 1) x[k] <= 0;
        end else if (is_lw && !load_wait) begin
            load_wait <= 1;                                   // address out, data next cycle
        end else begin
            load_wait <= 0;
            if ((is_r || is_i) && rd != 0) x[rd] <= alu;
            if (is_lw && rd != 0) x[rd] <= dmem_rdata;
            pc <= (is_br && taken) ? pc + imm_b : pc + 4;
        end
    end
endmodule
"""


CPU_BUGGY = CPU_OK.replace("3'b010: alu = ($signed(a) < $signed(op2)) ? 32'd1 : 32'd0;     // SLT / SLTI",
                           "3'b010: alu = (a < op2) ? 32'd1 : 32'd0;     // BUG: SLT compares unsigned")


CPU_RECORD = {'module_type': 'cpu', 'created_at': '2026-11-03T14:00:00', 'questions': [], 'decisions': [],
              'requirements': [
                  {'id': 'REQ-1', 'text': 'Arithmetic and logical instructions write their result to rd.',
                   'source': 'llm'},
                  {'id': 'REQ-2', 'text': 'SLT and SLTI compare signed values.', 'source': 'llm'},
                  {'id': 'REQ-3', 'text': 'LW loads a word from memory; SW stores a word to memory.',
                   'source': 'llm'},
                  {'id': 'REQ-4', 'text': 'Branches are taken exactly when their condition holds.',
                   'source': 'llm'}]}


CPU_DUT = {'module_type': 'cpu', 'module_name': 'riscv_cpu', 'bitwidth': 32, 'pipeline_stages': 5}

