# register_file_16x8 — 16 x 8-bit Register File (RISC-V style R0)

## Overview
Register file with two asynchronous (combinational) read ports and one
synchronous write port. Register 0 is HARDWIRED TO ZERO: writes to
address 0 are ignored and reads of address 0 always return 0x00.

## Ports
| Name    | Dir    | Width | Description                          |
|---------|--------|-------|--------------------------------------|
| clk     | input  | 1     | Clock (rising edge write)            |
| rst_n   | input  | 1     | Async reset, active low (all regs 0) |
| we      | input  | 1     | Write enable                         |
| waddr   | input  | 4     | Write address                        |
| wdata   | input  | 8     | Write data                           |
| raddr_a | input  | 4     | Read address, port A                 |
| raddr_b | input  | 4     | Read address, port B                 |
| rdata_a | output | 8     | Read data, port A (combinational)    |
| rdata_b | output | 8     | Read data, port B (combinational)    |

## Behavior
- Reads are combinational (no clock needed) and independent per port.
- A read in the same cycle as a write to the same address returns the OLD value
  (write becomes visible after the clock edge).
- Writes to address 0 are silently ignored; reading address 0 gives 0x00 always.

## Functional Points
- FP-1: written value is readable from both ports after the clock edge.
- FP-2: the two read ports are independent (different addresses simultaneously).
- FP-3: R0 reads as 0x00 even after an attempted write to address 0.
- FP-4: write only occurs when we=1.
- FP-5: read-during-write to the same address returns the old value.
- FP-6: all 16 addresses are distinct storage locations (no aliasing).
- FP-7: reset clears all registers.
