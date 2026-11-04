# Research outline: what the website work can be used for

Status: living document, started 2026-11-04. We are **building features
first** and decide on papers later. This file records, while it is fresh,
what each feature contributes, what evidence it already produces, and what
would have to be measured to turn it into a paper. Nothing here is a
commitment to write any of these papers.

Branch: `bdx-explainability` (working main branch). Design decisions D1–D28:
`docs/explainability_db_design.md`. Setup: `docs/explainability_db_setup.md`.

---

## 1. What exists (2026-11-04)

| Asset | What it does | Where |
|---|---|---|
| Web flow Steps 0–4 | requirements Q&A → design (LLM) → BDD scenarios (LLM) → template testbench (no LLM) → simulation | `main.py`, `static/bdd_generator.html` |
| Step-0 fixed questions | asks what ALU "overflow" means; both generators and the rule library follow the answer | `requirements_store.FIXED_QUESTIONS`, D22 |
| Scenario index v2 | every assertion of the testbench tagged with its scenario row, stimulus, expected values (B, R, used); Verilog unchanged | `testbench_generator.py`, `*_tb.scenarios.json` |
| Verdict model | per assertion, from Observed / BDD-expected / Reference: pass, scenario_wrong, design_wrong, both_wrong, consistent_wrong, undecided | `src/xdb/verdict.py`, D16–D18 |
| Rule libraries (hand-written, LLM-free) | ALU (10 ops, 3 flags, 2 overflow meanings), register file (sequence replay with the cause of every read), RV32I subset (17 instructions, rows judged as executed) | `src/xdb/library.py` |
| Canonical steps | surface step → canonical step → port predicate, all steps of the prompt formats map | `library.py`, `mapping.py` |
| Explanations | local (complete + contrastive), global (when does the spec say X), requirement status by subject port; plain-language rendering first, technical behind a toggle | `src/xdb/explain.py`, D20, D24, D25 |
| Flag probe | second simulation that observes outputs the testbench never compares (ALU flags) | `src/xdb/probe.py`, D23 |
| Repair | deterministic scenario correction (new feature version); design regeneration with verdict-filtered feedback (ALU, register file, CPU) | `src/xdb/repair.py`, D26 |
| Provenance database | 23 tables: every generated file version, LLM call, check, verdict, explanation, feedback; `feedback:<id>` version chains | `src/xdb/models.py`, PostgreSQL |
| Tests | 116 pytest tests (DB end-to-end, rule libraries vs. generator reference, template regressions), 2 browser checks over 3 design types | `tests/` |

Field observations so far (anecdotal, one run each; useful as motivating
examples, not as results):

- Gemini ALU feature: two expected values wrong (SUB r7, SRA r5), design
  bug in SLTU (injected) — the verdicts separate the two sides correctly.
- DeepSeek used "overflow" to mean unsigned carry/borrow → led to D22.
- Mistral produced a design that does not compile → compile errors kept.
- DeepSeek returned invalid JSON in Step 0 → repair-and-retry.
- The explanation layer exposed four bugs in the deterministic testbench
  generator (operations guessed from English words in titles, missing
  expected values checked against 0, CPU stores checked against x0,
  load/store offsets ignored). See commits 3ab7993, ab9960a, 41d90a9.

---

## 2. Candidate papers

Each entry: question, claim we would like to make, what is already in place,
what is missing, how to measure it. Ordered by how much of the work exists.

### P1 — Who is wrong? Attributing failures between generated specifications and generated designs

- **Question.** When both the design and its BDD scenarios come from LLMs, a
  failing check says only that they disagree. Can a deterministic reference
  decide which side is wrong, and how often is neither side trustworthy?
- **Claim.** A three-value comparison (O, B, R) per assertion, with
  hand-written rule libraries as R, attributes failures with high precision
  and makes "consistent_wrong" (passing tests that are wrong) visible.
- **In place.** Verdict model, rule libraries for three design types,
  per-row causes for stateful designs, plain explanations, DB.
- **Missing.** Ground truth for attribution: inject faults on both sides
  (design mutants from `src/bug_taxonomy.py`; scenario mutants = perturbed
  Examples cells) and check that the verdicts name the injected side.
  Real-run statistics over several LLMs.
- **Metrics.** Attribution precision/recall per verdict class; undecided
  rate; share of passing checks that are consistent_wrong; canonical-step
  mapping coverage of real scenarios.
- **Related.** AutoBench, LLM4Cov, CVDP (they score testbenches/designs, not
  who is wrong); our own arXiv 2512.17814.

### P2 — Closing the loop: verdict-filtered feedback for LLM hardware repair

- **Question.** Does it help to tell the design model only the failures that
  are the design's fault, with the rule each one violates, instead of the
  raw test log? And how far does deterministic scenario correction get?
- **Claim.** Verdict-filtered feedback converges in fewer iterations and
  with fewer regressions than raw-log feedback, because scenario errors are
  never turned into design changes.
- **In place.** Both repair actions, version chains in the DB, feedback
  rows, the prompt block (`prompt_store.repair_feedback_block`).
- **Missing.** A batch harness that runs the loop without the browser
  (roadmap R1); a raw-log baseline prompt; repeated runs per LLM.
- **Metrics.** Success rate and iterations to all-pass; regressions
  (checks that passed before and fail after); tokens and cost per repaired
  design; scenario cells corrected per feature.
- **Connection.** This is the professor's phase 2 (simulation-feedback
  loop) in its first form.

### P3 — Explanations that stay true (workshop; CAUSE / self-explaining systems)

- **Question.** How much of a hardware verification result can be explained
  without an LLM, and are those explanations verifiably correct?
- **Claim.** Explanations built from rule libraries and the simulation
  record are checkable (every number is reproduced by an independent model)
  and cover most failures; an LLM is only needed for wording.
- **In place.** Explanation engine, automatic self-checks
  (`rule_agrees_with_reference`), two independent reference
  implementations per design type, BDX-style structure (P → Q, local /
  global / contrastive).
- **Missing.** An explanation-coverage measure (roadmap R2); a comparison
  with LLM-written explanations of the same failures (correctness judged
  against the verdicts).
- **Metrics.** Explanation coverage (checks with a decided verdict and a
  named cause / all checks); factual errors in LLM explanations vs. ours.

### P4 — Tool demo

The website with the database as a demo (DATE / ASP-DAC university booth,
DAC demo night, or a workshop demo track). Needs: a stable deployment or a
local install script, a 3-minute scenario (ALU with a wrong scenario and a
design bug → Problems card → Details → both repairs → all green), and the
history view (roadmap R4).

### P5 — Pitfalls of LLM-written BDD for stateful designs (short paper / section)

Observations to collect: rows that depend on other scenarios' state
(register file), self-contradictory rows (one register given two values,
results expected in x0), operations named only in English titles, missing
expected columns. The explanation layer already classifies them; the paper
would count them over real runs.

---

## 3. Data the database already collects

Per session: the Step-0 record and decisions, every design and feature
version (with parent and reason), every LLM call (prompt, response,
model served, sampling, latency), testbench and log artifacts, every check
with B / R / O and verdict, probe observations, explanations with their
structure, feedback rows with what was applied.

Missing for the papers above: wall-clock time per step; whether the user
accepted or ignored a suggested repair (only applied repairs are stored);
the injected-fault ground truth (to be generated by a harness, not by users).

---

## 4. BDX / CAUSE wishlist vs. status

| Wishlist item | Status |
|---|---|
| Explanation coverage (measure, compute) | partial: per-run checks and requirement status; a coverage metric is roadmap R2 |
| Real examples (RISC-V) and "unreal" examples | ALU, register file, RV32I subset; the wind turbine was dropped (user decision) |
| Nice interface with a database of generated content | done for the current run; browsing history is roadmap R4 |
| NL → formal → NL | canonical steps carry port predicates (P → Q); generating SVA from them is roadmap R5 |
| Spec vs. implementation validation | done: verdicts, specification check, flag probe |
| Workflow / back-annotation RTL/SVA → scenarios | first step: scenario correction writes reference values back; counterexample rows are roadmap R3 |
| Deterministic, LLM-free approach | done for verdicts, explanations, scenario correction |

---

## 5. Feature roadmap (recommended order) and which paper each feeds

| # | Feature | Feeds | Effort |
|---|---|---|---|
| R1 | **Trial runner** (done: `python -m src.xdb.trial`): run Steps 0–4 + explanations + repair loops headless for several LLMs × designs × repetitions; runs.csv / runs.jsonl / summary.md per trial | all; P1, P2 directly | runs on the user's Mac with API keys |
| R2 | **Coverage view + "Add examples for the gaps"**: per design type, the input classes the rules distinguish (ALU: operation × zero / max / sign boundary / overflow on-off; register file: overwrite before read, read after reset, x0; CPU: instruction × sign combinations, taken / not taken); show which ones no scenario covers and add rows with reference values as a new feature version | P1, P3 (coverage), phase 2 | 2–3 days |
| R3 | **Deep check with counterexamples**: random + corner-case differential simulation of the design against the rule library (iverilog only); each mismatch becomes a concrete example that can be added to the scenarios | P1, P3, back-annotation | 1–2 days; formal proof (yosys-smtbmc) optional later |
| R4 | **History page**: past sessions, version chains of features and designs, verdict changes before / after each repair | P2, P4 | 1–2 days |
| R5 | **SVA view**: show the scenarios and rules as SystemVerilog assertions (NL → formal), optionally run them | P3, NL → formal → NL | 2 days |
| R6 | Counter (reference model, multi-cycle stimulus; the template's up-down check never fails today) | coverage of design types | 2–3 days |
| R7 | Free-text questions mapped to explanations (LLM only for parsing) | P4 | 2 days |

Decided 2026-11-04: in every template, a row that states no expected value
is not checked (reason "scenario states no expected result"); only the
oracle "spec" fills in the reference value.
