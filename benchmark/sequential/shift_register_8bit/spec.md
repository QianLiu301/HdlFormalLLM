# shift_register_8bit — 8-bit Universal Shift Register

## Overview
Synchronous 8-bit shift register with parallel load, bidirectional shift,
and serial input/output. Asynchronous active-low reset.
Priority: reset > load > shift.

## Ports
| Name       | Dir    | Width | Description                                  |
|------------|--------|-------|----------------------------------------------|
| clk        | input  | 1     | Clock (rising edge)                          |
| rst_n      | input  | 1     | Asynchronous reset, active low (q <= 0)      |
| load       | input  | 1     | Parallel load (q <= d), priority over shift  |
| d          | input  | 8     | Parallel load value                          |
| shift_en   | input  | 1     | Shift enable                                 |
| dir        | input  | 1     | 0 = shift left (toward MSB), 1 = shift right |
| serial_in  | input  | 1     | Bit shifted into the vacated position        |
| q          | output | 8     | Register value                               |
| serial_out | output | 1     | Bit about to be shifted out (combinational): q[7] when dir=0, q[0] when dir=1 |

## Behavior
- Shift left  (dir=0): q <= {q[6:0], serial_in}; the old q[7] leaves the register.
- Shift right (dir=1): q <= {serial_in, q[7:1]}; the old q[0] leaves the register.
- serial_out continuously shows the bit that WOULD leave next: q[7] if dir=0, q[0] if dir=1.
- load=1 wins over shift_en=1 on the same edge.

## Functional Points
- FP-1: asynchronous active-low reset clears q.
- FP-2: parallel load sets q to d on the clock edge.
- FP-3: left shift moves every bit up; serial_in enters at bit 0.
- FP-4: right shift moves every bit down; serial_in enters at bit 7.
- FP-5: serial_out equals q[7] when dir=0 and q[0] when dir=1.
- FP-6: load has priority when load and shift_en are both high.
- FP-7: q holds when load=0 and shift_en=0.
