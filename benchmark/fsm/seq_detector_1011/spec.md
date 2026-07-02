# seq_detector_1011 — Overlapping "1011" Sequence Detector (Mealy FSM)

## Overview
A Mealy finite-state machine that monitors a serial bit stream `din`
(one bit per clock cycle) and pulses `detected` for exactly one cycle
whenever the most recent four bits equal "1011" (oldest bit first).
Detection is OVERLAPPING: the trailing "1" of a match may begin the
next match (input 1011011 fires twice, at bit 4 and bit 7).

## Ports
| Name     | Dir    | Width | Description                                  |
|----------|--------|-------|----------------------------------------------|
| clk      | input  | 1     | Clock (rising edge samples din)              |
| rst_n    | input  | 1     | Asynchronous reset, active low               |
| din      | input  | 1     | Serial input bit                             |
| detected | output | 1     | Mealy output: high during the cycle in which the 4th matching bit ("1") is present |

## State encoding (informative)
- S0: no relevant history
- S1: seen "1"
- S2: seen "10"
- S3: seen "101"
- In S3 with din=1: detected=1 (combinational), next state S1 (overlap: last "1" reused).

## Functional Points
- FP-1: the exact sequence 1-0-1-1 raises detected in the 4th cycle.
- FP-2: detected is a Mealy output: asserted combinationally in the same cycle the final '1' arrives, for exactly one cycle.
- FP-3: overlapping detection: 1011011 produces two pulses.
- FP-4: 1-0-1-0 does not fire; the FSM falls back to the "10" history state.
- FP-5: a run of 1s (111...) keeps the FSM waiting in the "seen 1" state; 11011 fires once.
- FP-6: reset returns to S0; a partial match is forgotten after reset.
