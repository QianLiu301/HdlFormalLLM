# comparator_8bit — 8-bit Unsigned Magnitude Comparator

## Overview
A purely combinational comparator for two 8-bit UNSIGNED values.
Exactly one of the three outputs is high at any time.

## Ports
| Name | Dir    | Width | Description        |
|------|--------|-------|--------------------|
| a    | input  | 8     | Operand A          |
| b    | input  | 8     | Operand B          |
| eq   | output | 1     | High when a == b   |
| gt   | output | 1     | High when a > b (unsigned) |
| lt   | output | 1     | High when a < b (unsigned) |

## Functional Points
- FP-1: eq is high if and only if a == b.
- FP-2: gt is high if and only if a > b, using UNSIGNED comparison
        (e.g., 0x80 > 0x7F must set gt, not lt).
- FP-3: lt is high if and only if a < b (unsigned).
- FP-4: outputs are one-hot: exactly one of eq/gt/lt is high for every input pair.
- FP-5: boundary behavior: adjacent values (b = a+1, b = a-1) resolve correctly.
