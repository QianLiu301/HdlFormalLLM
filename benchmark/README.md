# BDD-Hardware Benchmark Suite (v0.1)

A benchmark for evaluating **LLM-generated BDD scenarios and testbenches**
for hardware verification. First batch: **9 modules, 5 categories,
28 fault-injected mutants, 60 numbered functional points**.

## Structure

```
benchmark/
├── index.json                  # machine-readable module index
├── validate.py                 # quality gate (see below)
└── <category>/<module>/
    ├── spec.md                 # natural-language spec + numbered Functional Points (FP-n)
    ├── golden.v                # reference design (verified with iverilog)
    ├── tb_smoke.v              # deterministic smoke stimulus (differential-test driver)
    ├── manifest.json           # metadata: difficulty, bug list with taxonomy labels
    └── bugs/bugNN_*.v          # single-fault mutants of golden.v
```

## Modules

| Category      | Module                 | Difficulty | FPs | Bugs |
|---------------|------------------------|-----------|-----|------|
| combinational | alu_8bit               | easy      | 9   | 4    |
| combinational | comparator_8bit        | easy      | 5   | 3    |
| combinational | priority_encoder_8to3  | easy      | 5   | 3    |
| sequential    | counter_8bit           | medium    | 8   | 3    |
| sequential    | shift_register_8bit    | medium    | 7   | 3    |
| fsm           | seq_detector_1011      | medium    | 6   | 3    |
| fsm           | traffic_light          | medium    | 6   | 3    |
| memory        | sync_fifo_8x8          | hard      | 7   | 3    |
| memory        | register_file_16x8     | hard      | 7   | 3    |

## How the benchmark is used

1. **Scenario Completeness** — give an LLM the `spec.md` (without the FP list)
   and ask for BDD scenarios; score = fraction of numbered Functional Points
   covered by at least one generated scenario.
2. **Conversion Rate** — fraction of generated BDD scenarios that convert to
   compiling, running Verilog testbenches.
3. **Simulation Pass Rate** — generated testbench must PASS on `golden.v`
   (a testbench that fails the golden design is itself wrong).
4. **Mutation Score** — fraction of `bugs/` mutants the generated testbench
   detects (reports FAIL on the mutant after passing golden).

## Bug taxonomy

Mutants are labeled with the project-wide taxonomy
(`src/bug_taxonomy.py`): `FUNCTIONAL`, `INTERFACE`, `TIMING`, `STRUCTURAL`,
plus a `detection_difficulty` tier (easy / medium / hard).
Every mutant is a **single behavioral fault** — it compiles cleanly and
differs from golden only in behavior.

## Validation (quality gate)

```
python benchmark/validate.py              # all modules
python benchmark/validate.py alu_8bit     # one module
```

Checks, per module: golden compiles & simulates; every mutant compiles;
every mutant is **killed** (produces different output than golden under
`tb_smoke.v`'s deterministic stimulus). A surviving mutant would be an
invalid bug and fails validation.

Current status: **9/9 golden OK, 28/28 mutants killed.**

Requires `iverilog`/`vvp` (already in the project Dockerfile).
