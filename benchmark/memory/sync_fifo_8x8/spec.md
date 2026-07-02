# sync_fifo_8x8 — Synchronous FIFO, 8 entries x 8 bits

## Overview
Single-clock FIFO with registered read data and full/empty flags.
Asynchronous active-low reset.

## Ports
| Name  | Dir    | Width | Description                                |
|-------|--------|-------|--------------------------------------------|
| clk   | input  | 1     | Clock (rising edge)                        |
| rst_n | input  | 1     | Async reset, active low (FIFO emptied)     |
| wr_en | input  | 1     | Write request                              |
| din   | input  | 8     | Write data                                 |
| rd_en | input  | 1     | Read request                               |
| dout  | output | 8     | Read data, REGISTERED (valid the cycle after an accepted read) |
| full  | output | 1     | High when 8 entries are stored             |
| empty | output | 1     | High when 0 entries are stored             |

## Behavior
- A write is ACCEPTED only when full is low; writes while full are ignored
  (even if a simultaneous read frees a slot that same edge).
- A read is ACCEPTED only when empty is low; reads while empty are ignored.
- Simultaneous accepted read + write keeps the occupancy count unchanged.
- dout updates only on an accepted read; otherwise it holds its value.
- Data ordering is strictly first-in first-out.

## Functional Points
- FP-1: reset empties the FIFO (empty=1, full=0).
- FP-2: data emerges in FIFO order.
- FP-3: full asserts exactly at 8 stored entries; a 9th write is ignored.
- FP-4: empty asserts exactly at 0 entries; a read while empty is ignored and dout holds.
- FP-5: write-while-full is dropped even with a simultaneous read.
- FP-6: simultaneous read+write at intermediate occupancy leaves count unchanged.
- FP-7: dout is registered: it shows accepted-read data one cycle later and holds between reads.
