# traffic_light — Traffic Light Controller FSM

## Overview
A Moore FSM cycling GREEN -> YELLOW -> RED -> GREEN with fixed durations
counted in clock cycles. Outputs are one-hot. Asynchronous active-low reset
enters RED (safe state) with its full duration.

## Parameters (defaults)
| Name    | Default | Meaning                 |
|---------|---------|-------------------------|
| G_TICKS | 8       | GREEN duration (cycles) |
| Y_TICKS | 3       | YELLOW duration         |
| R_TICKS | 6       | RED duration            |

## Ports
| Name   | Dir    | Width | Description                    |
|--------|--------|-------|--------------------------------|
| clk    | input  | 1     | Clock (rising edge)            |
| rst_n  | input  | 1     | Async reset, active low -> RED |
| red    | output | 1     | RED lamp                       |
| yellow | output | 1     | YELLOW lamp                    |
| green  | output | 1     | GREEN lamp                     |

## Behavior
- Each state lasts exactly its configured number of cycles, then advances.
- Sequence: RED (R_TICKS) -> GREEN (G_TICKS) -> YELLOW (Y_TICKS) -> RED ...
- Outputs are one-hot at all times (exactly one lamp on).

## Functional Points
- FP-1: reset enters RED and stays exactly R_TICKS cycles.
- FP-2: GREEN lasts exactly G_TICKS cycles.
- FP-3: YELLOW lasts exactly Y_TICKS cycles.
- FP-4: state order is RED -> GREEN -> YELLOW -> RED (YELLOW never followed by GREEN).
- FP-5: outputs are one-hot in every cycle.
- FP-6: the cycle repeats periodically with period G_TICKS+Y_TICKS+R_TICKS.
