# alu_8bit — 8-bit Arithmetic Logic Unit

## Overview
A purely combinational 8-bit ALU supporting 8 operations selected by a 3-bit opcode.

## Ports
| Name      | Dir    | Width | Description                        |
|-----------|--------|-------|------------------------------------|
| a         | input  | 8     | Operand A                          |
| b         | input  | 8     | Operand B                          |
| op        | input  | 3     | Operation select                   |
| result    | output | 8     | Operation result                   |
| carry_out | output | 1     | Carry / borrow / shifted-out bit   |
| zero      | output | 1     | High when result == 0              |

## Operations
| op  | Name | result                | carry_out            |
|-----|------|-----------------------|----------------------|
| 000 | ADD  | a + b (low 8 bits)    | carry of a + b       |
| 001 | SUB  | a - b (low 8 bits)    | borrow (1 if a < b)  |
| 010 | AND  | a & b                 | 0                    |
| 011 | OR   | a \| b                | 0                    |
| 100 | XOR  | a ^ b                 | 0                    |
| 101 | NOT  | ~a (b ignored)        | 0                    |
| 110 | SHL  | a << 1                | a[7] (shifted out)   |
| 111 | SHR  | a >> 1                | a[0] (shifted out)   |

## Functional Points (ground truth for scenario completeness)
- FP-1: ADD produces correct 8-bit sum for all operand values.
- FP-2: ADD sets carry_out on unsigned overflow (e.g., 0xFF + 0x01).
- FP-3: SUB produces correct 8-bit difference (two's complement wrap).
- FP-4: SUB sets carry_out (borrow) exactly when a < b.
- FP-5: AND/OR/XOR produce correct bitwise results with carry_out = 0.
- FP-6: NOT inverts operand a and ignores b.
- FP-7: SHL shifts left by one; carry_out receives the old MSB a[7].
- FP-8: SHR shifts right by one; carry_out receives the old LSB a[0].
- FP-9: zero flag is high if and only if result == 0x00, for every operation.
