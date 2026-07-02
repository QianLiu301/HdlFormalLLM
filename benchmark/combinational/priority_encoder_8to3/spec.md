# priority_encoder_8to3 — 8-to-3 Priority Encoder

## Overview
Combinational priority encoder. Bit 7 has the HIGHEST priority.
`out` is the index of the highest set bit of `in`; `valid` indicates
that at least one input bit is set.

## Ports
| Name  | Dir    | Width | Description                          |
|-------|--------|-------|--------------------------------------|
| in    | input  | 8     | Request lines (bit 7 = highest prio) |
| out   | output | 3     | Index of highest set bit             |
| valid | output | 1     | High when in != 0                    |

## Functional Points
- FP-1: out equals the index of the highest set bit (bit 7 wins over all).
- FP-2: lower bits are ignored when a higher bit is set (e.g., in=0b1000_0001 -> out=7).
- FP-3: valid is high if and only if in != 0.
- FP-4: when in == 0, valid is low and out is 0.
- FP-5: each single-bit input (one-hot) maps to its own index (8 cases).
