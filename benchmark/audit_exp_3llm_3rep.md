# Failure-Attribution Audit — pilot run `exp_3llm_3rep`

**Date:** 2026-07-08 · **Data:** deepseek (61 exp) + qwen (59 exp), 9 modules × ~3 reps
**Question:** are the persistent golden-pass failures caused by (a) benchmark defects
(golden bugs / spec ambiguity) or (b) genuine LLM weaknesses?

## Verdict

**All 6 contested golden designs are correct, and no spec ambiguity was found.
Every audited failure is a genuine LLM error.** The specs already state explicitly
the semantics the LLMs got wrong (registered FIFO dout, Mealy same-cycle output,
read-old-value, exact FSM durations, dir encoding, priority order).

### Golden correctness evidence

Directed probe testbenches (in `benchmark/probes/`, run with
`iverilog -g2005 -o p.vvp <golden.v> <probe.v> && vvp p.vvp`):

| Probe | Verifies | Result |
|---|---|---|
| `probe_fifo.v` | FIFO order; dout registered one-cycle-after-read; read-while-empty ignored & dout holds | PASS |
| `probe_seq.v` | Mealy pulse in the same cycle as the 4th bit; `1011011` fires exactly twice (overlap) | PASS |
| `probe_traffic.v` | GREEN=8 / YELLOW=3 / RED=6 cycles exactly; one-hot; periodic (phase-independent counting) | PASS |

counter_8bit / shift_register_8bit / register_file_16x8 goldens were verified by
line-by-line review against their specs (all contested points are unambiguous).

## LLM testbench error taxonomy (from 36 failure logs + 5 deep dives)

| ID | Error class | Evidence (module × llm/rep) | Notes |
|----|---|---|---|
| T1 | **SystemVerilog despite explicit "Verilog-2005 only" prompt** | seq/qwen r2 r3, traffic/deepseek r2 r3, regfile/deepseek r2, shift/deepseek r2 r3, fifo/qwen r2, … | task body without begin/end, var decl in unnamed block, size cast, SV for-loop. Instruction-following failure; qwen-dominant (qwen compile rate 39% vs deepseek 75%) |
| T2 | **Clock-edge race**: control signals changed with blocking assigns at `posedge+0` | counter/deepseek r1 (q=0x81 vs 0x06), fifo/deepseek r3 read task, shift/deepseek r1 (probable) | DUT may sample old or new value → nondeterministic accept; textbook TB anti-pattern |
| T3 | **Mealy output sampled one cycle late** (`@(posedge clk); #1; check`) | seq/deepseek r1 r2 r3 | spec states "high during the cycle in which the 4th bit is present" |
| T4 | **Cumulative off-by-one drift** in duration counting (`wait_cycles(N); @(posedge clk);` = N+1 per scenario) | traffic/deepseek r1 (4 of 23 checks fail after drift crosses a state boundary) | early scenarios pass, later ones fail — looks "flaky", is deterministic drift |
| T5 | **BDD-hallucination propagation**: TB follows its own wrong BDD scenario over the spec | fifo/deepseek r3 — TB comments literally say "This contradicts spec but we follow scenario for test" | key finding for the BDD-as-IR thesis: errors injected at BDD stage survive into the TB |
| T6 | **Driving DUT outputs** (`q = 8'h05;` on a DUT output) | counter/qwen r1 r2 r3, shift/qwen r1 r2 | interface misunderstanding; iverilog: "q is not a valid l-value" |
| T7 | **Wrong expectation on explicit semantics** (e.g. expects non-registered dout, read-new-value) | fifo/deepseek r2, regfile/qwen r1 r2 | reading-comprehension failure, spec explicit |

## Decisions

1. **No spec changes.** The hypothesized ambiguities (FIFO read latency, Mealy vs
   Moore, read-during-write, duration counting) are all explicitly specified.
   Softening specs or coaching prompts (e.g. "drive at negedge") would lower
   benchmark difficulty and is NOT applied.
2. **Keep `-g2005` + "no SystemVerilog" prompt rule.** T1 failures are legitimate
   instruction-following findings, not harness unfairness, because the prompt
   states the constraint explicitly.
3. `benchmark/probes/` added as regression evidence of golden correctness.

## Paper-relevant takeaways

- Golden-pass failures decompose into syntax discipline (T1, T6) vs semantic
  timing (T2–T4, T7) vs pipeline-specific (T5). T5 is unique ammunition for
  Phase 2: feedback loops must be able to *repair the BDD*, not just the TB.
- The category gradient (combinational ≫ FSM/memory pass rates) is driven by
  timing-sensitive checks, not module size.
