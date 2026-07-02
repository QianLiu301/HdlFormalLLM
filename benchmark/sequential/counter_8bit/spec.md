# counter_8bit — 8-bit Up/Down Counter with Load

## Overview
Synchronous 8-bit counter with asynchronous active-low reset,
parallel load, count enable, and direction control.
Priority: reset > load > count.

## Ports
| Name     | Dir    | Width | Description                              |
|----------|--------|-------|------------------------------------------|
| clk      | input  | 1     | Clock (rising edge)                      |
| rst_n    | input  | 1     | Asynchronous reset, ACTIVE LOW           |
| load     | input  | 1     | Synchronous parallel load (q <= d)       |
| d        | input  | 8     | Load value                               |
| en       | input  | 1     | Count enable                             |
| up_down  | input  | 1     | 1 = count up, 0 = count down             |
| q        | output | 8     | Counter value                            |
| overflow | output | 1     | 1-cycle pulse on wrap (FF->00 up, 00->FF down) |

## Behavior
- reset (rst_n = 0): q <= 0x00, overflow <= 0, immediately (async).
- load = 1 (and not in reset): on clock edge q <= d, regardless of en.
- en = 1, load = 0: count up (q+1) if up_down = 1, else down (q-1). Wraps around.
- en = 0, load = 0: hold value.
- overflow: registered; high for exactly one cycle after the wrap transition
  (0xFF->0x00 counting up, 0x00->0xFF counting down). Load never sets overflow.

## Functional Points
- FP-1: asynchronous active-low reset clears q to 0x00 at any time.
- FP-2: counting up increments by 1 each enabled clock edge.
- FP-3: counting down decrements by 1 each enabled clock edge.
- FP-4: wrap-around up: 0xFF -> 0x00 with a 1-cycle overflow pulse.
- FP-5: wrap-around down: 0x00 -> 0xFF with a 1-cycle overflow pulse.
- FP-6: parallel load takes effect on clock edge and has priority over counting.
- FP-7: en = 0 holds the current value.
- FP-8: overflow is never asserted by load or normal (non-wrap) counting.
