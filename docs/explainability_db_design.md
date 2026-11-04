# Explainability Database — Design (v0.3)

Status: **M1–M5 implemented** (schema v2 with verdicts, step-by-step
ingestion, rule libraries for ALU, register file and RV32I CPU, explanation
engine, Step-4 explanations, repair; see `explainability_db_setup.md`).
Counter and M6 pending. The ORM models in
`src/xdb/models.py` and the migrations are the authoritative schema; §5 shows
the v0.2 baseline and §5a the v0.3 changes. Branch: `bdx-explainability`
(the working main branch; fixes from other branches are merged into it).

Goal: add explainability to the HdlFormalLLM website. Every generated item
(requirement, BDD scenario, step, check, simulation verdict, explanation,
user correction) is stored with its provenance and its links to the others,
so the site can answer *why* a check passed or failed, *which requirement* it
serves, *who* produced each piece, and *what changed* after a correction.

Reference: Dominik & Drechsler, *Behavior-Driven Explainability* (arXiv
2607.24881). We reuse its concepts (Section 3) but are not bound to them.

---

## 0. Decision log

| # | Decision | Date |
|---|----------|------|
| D1 | Users: hardware developers, researchers, reviewers | 2026-11-03 |
| D2 | v1 explanation types: **global** and **local**, complete only. Counterfactual / action-oriented → v2 (needs a solver) | 2026-11-03 |
| D3 | Entry points: **click** on a scenario / example row / check / requirement (primary, deterministic) **and** free-text question (secondary; an LLM maps it to the same subjects, the explanation is then computed exactly as for a click) | 2026-11-03 |
| D4 | Web workflow is ingested first; benchmark workflow later | 2026-11-03 |
| D5 | Designs in scope: existing ones (ALU, ALU-comb, counter, register file, CPU). Wind turbine dropped | 2026-11-03 |
| D6 | Database: **PostgreSQL + pgvector**, hosted on **Neon, region AWS Frankfurt (`aws-eu-central-1`)**. Local development also on PostgreSQL (no SQLite/Postgres dialect split) | 2026-11-03 |
| D7 | No user accounts. All content is **publicly readable**. Operations that call an LLM (generate, regenerate, NL question) are behind an **access password** + rate limit to protect API credit | 2026-11-03 |
| D8 | Canonical steps: **three layers** (surface text → canonical step → signal predicate), Section 4. v1 mapping is rule-based | 2026-11-03 |
| D9 | Nothing is overwritten: every edit creates a new revision (needed for correction requests) | 2026-11-03 |
| D10 | Explanations are computed by deterministic rules and **stored** with their evidence. An LLM may only rephrase them; rephrasings are checked for added facts | 2026-11-03 |
| D11 | Every LLM call is kept, including failed ones | 2026-11-03 |
| D12 | Semantic search (embeddings) is v2. Embedding provider decided then (OpenAI API vs. local open model) | 2026-11-03 |
| D13 | Scope of this branch: the website's Steps 0–4 plus two new steps, **5 explain** and **6 repair**. Benchmark, golden designs, mutants, Yosys analysis, uploads and batch tools are set aside (kept, not changed) | 2026-11-03 |
| D14 | Every evaluated assertion gets a **verdict** from three values — O observed, B expected by the BDD scenario, R expected by the tool's reference model (`TestbenchGenerator._reference_result`, `_rv32i_reference`): pass, scenario_wrong, design_wrong, both_wrong, consistent_wrong (O = B ≠ R), undecided (no reference). "Who is wrong" is the first line of every explanation | 2026-11-03 |
| D15 | The testbench Verilog is not changed. Its side index (`*_tb.scenarios.json`, v2) records per assertion the applied stimulus, compared port, B and R. Columns the testbench does not compare (e.g. ALU flags) stay unchecked; explanations label such Then steps "stated, not verified" | 2026-11-03 |
| D16 | Versions are kept **per file**: a regenerated or corrected feature / Verilog is a new row with `parent_id` and `reason`. Scenarios and rows are views of one file version, so every displayed state can be simulated | 2026-11-03 |
| D17 | Every generated file is stored when it is saved (Step 1, Step 2), including versions later discarded; Step-0 requirements and decisions are stored when a step uses them. Render's disk is wiped on deploy, so waiting for Step 4 would lose them | 2026-11-03 |
| D18 | Files are stored uncompressed for now; `artifact.encoding` allows switching to gzip later without a schema change. `python -m src.xdb stats` shows usage against the 512 MB plan limit | 2026-11-03 |
| D19 | Rules (operation semantics) come from a rule library written by us (provenance `human`, checked against the reference model); canonical-step predicates are expressions (SVA-compatible). Explanations are stored as structured data and rendered at display time, **in English only** (product output contains no Chinese; decided 2026-11-03) | 2026-11-03 |
| D20 | v1 explanation modes: complete and **contrastive** (another observed row of the same scenario that behaved differently, as X₁,C in the BDX paper); solver-based counterfactuals later | 2026-11-03 |
| D22 | Step 0 always asks, for ALU designs, what "overflow" means (signed overflow — the design template's definition, recommended — or unsigned carry/borrow). The answer goes into both prompts like every decision, and the rule library uses it as the overflow rule; without an answer the template's definition applies. Reason: the first field observation (DeepSeek, §6a) | 2026-11-03 |
| D23 | **Flag probe.** Outputs the testbench does not compare (ALU zero / negative / overflow) are observed by a second simulation: a copy of the generated testbench with one `$display` per test after its settling delay — same DUT instance, clock and stimulus. Stored as its own testbench + sim_run (`probe_of` = the main run), judged per flag with O (probe), B (scenario, when stated) and R (flag rule); never part of the pass rate. The original testbench file is not touched. Rule values and verdicts are recomputed with the session's current overflow definition | 2026-11-03 |
| D24 | **Requirement status by subject port.** A requirement is judged on the output it is about — the first output port its text names (rule, reproducible). Statuses: verified (testbench checks of that port all pass), verified_by_probe, fail, stated_only (nothing simulated the port), not_tested, not_covered. Replaces "all checks of the linked scenarios passed", which marked requirements about uncompared flags as verified | 2026-11-03 |
| D25 | **Plain-language first.** Step 4 opens with a Problems card: "No problems found" or "N problems found", one line per problem (grouped by operation, concrete inputs and values, who is at fault: design / scenario / not tested / coverage), a Details button, and one line "What was checked". No internal ids (scenario keys, check tags, REQ numbers) in it. The detailed sections fold into "Technical details". Explanations carry a plain rendering (`render_plain`) shown first; the technical text is behind a toggle. A requirement whose scenarios produced no tests is "not tested" (whether an output is compared is decided by the whole testbench) | 2026-11-03 |
| D26 | **Repair acts on the verdicts, never on the raw test outcome.** (a) *Scenario correction*: every Examples cell whose check is scenario_wrong / both_wrong / consistent_wrong (result: reference model; flags: flag rule via the probe) is set to the reference value — deterministic, no model call. The corrected text is written to a NEW file `<stem>_fixN.feature` (the old one is kept), stored as the next feature version (parent_id, reason `feedback:<id>`); the feature keeps the provenance of the model that wrote it and only the changed Examples rows get `prov_kind='rule'`, `prov_ref='feedback:<id>'`. Refused if the file on disk is not the one the simulation was generated from (catalog spans would be stale) or if a cell no longer holds the simulated value. (b) *Design regeneration* (ALU): Step 1 is called again with the failing checks of verdict design_wrong / both_wrong / consistent_wrong (inputs, correct value, the design's value; one example per operation first, at most 12), the rules they follow, and the simulated Verilog, appended after the Step-0 decisions. Scenario-side failures never reach the design prompt. The new design continues the same run (session), is stored synchronously as the next design version with reason `feedback:<id>`, and the call row carries `repair_feedback_id`. Both are recorded as a `feedback` row (subject = the simulation, kind correction, `proposed.action` = apply_reference_values / regenerate_design, status open → applied / failed, `result_ref` = feature:<id> / duv:<id>). Both rerun Step 3 and Step 4 | 2026-11-03 |
| D27 | **Register file: explanations over a sequence.** The web testbench runs every Examples row in document order and carries the register contents from row to row and from scenario to scenario (what prompts/bdd_regfile asks for). The scenario index records every executed row — writes and resets as setup rows with operation and stimulus, reads with B, R and the value compared — and the Verilog is unchanged. R of a read is the generator's model; the rule library has an independent replay (`regfile_replay`) that must give the same value for every read (test) and also yields the *cause* of the value: the last write since the last reset, the reset itself, the initial reset, or register 0. Explanations name the cause by example number and scenario title; for a wrong scenario value they name the earlier write it belongs to and what made it obsolete (a later write, a reset, the register-0 rule); for a wrong design value they recognise stale data and stored register-0 writes. A requirement is judged on rdata1 (rdata2 only when named; the testbench never reads port 2, which "What was checked" says). A read without an expected value is not checked (since 2026-11-04, the same rule as ALU and counter: "scenario states no expected result"); with oracle "spec" it is compared with the reference model, so its verdict is pass or design_wrong. Checks the testbench adds by itself (GEN, e.g. register 0 after a write) count as design failures. No flag probe and no flag specification check for this type | 2026-11-03 |
| D28 | **RV32I CPU: a row is judged as the CPU executes it.** The testbench runs each Examples row as its own program (ADDI loads the registers of the rs1/rs2 columns, the instruction runs, SW stores the register of the rd column; a branch skips a marker store). The library decodes the instruction itself (`cpu_operands`) and computes R from the values the CPU actually sees (`cpu_reference`): a register given two values takes the second, x0 reads 0 and ignores writes. That R replaces the testbench's nominal value at ingestion (kept as `stimulus.tb_reference`). Rows whose meaning differs from what they say are classified: *notes* (same register twice, a value for x0, rd = x0) keep a verdict and are named in the explanation; *issues* make the check meaningless (verdict undecided) and are attributed — to the scenario (the instruction reads a register the row does not set, writes another register than the rd column) or to the testbench (load/store offset ignored, SW checked as a register, x29/x30 used). A branch is judged as taken / not taken (1/0); a slot that holds neither 0 nor the marker is a design error. Explanations show the working (10 − 30 = −20; −5 < 3 is true) and recognise unsigned comparisons and add/subtract swaps. Requirements on branches are judged on the decision, all others on every check of their scenarios; examples that cannot be checked are not evidence. Scenario correction skips rows that contradict themselves | 2026-11-03 |
| D21 | Order of designs: ALU (incl. alu_comb) → regfile → CPU → counter (counter needs a reference model and multi-cycle stimulus first) | 2026-11-03 |

---

## 1. Users and their questions

Imagined against the current 4-step website (Step 0 requirements → Step 1
DUV → Step 2 BDD → Step 3 testbench → Step 4 simulation).

| User | Example question | Type |
|------|------------------|------|
| Developer | Why does this check fail on the golden design? | local |
| Developer | Row 3 of ADD expects −2147483648 — why not 2147483648? | local |
| Developer | When is the overflow flag set? | global |
| Developer | In UP-DOWN mode, which way does the counter count after a load? | global (answer comes from a Step-0 decision) |
| Developer | Is reset behaviour tested at all? | coverage |
| Researcher | Which requirements do GPT and Claude scenarios each cover? | comparison |
| Researcher | Which requirements did no LLM cover? | coverage |
| Researcher | Which model and prompt version produced this scenario? | provenance |
| Researcher | Why was this mutant not detected? | local (v2, benchmark) |
| Reviewer | Where does this number in the paper come from? | provenance |
| Reviewer | What evidence supports this explanation? | evidence |
| Any | Scenario 5 has a wrong expected value — please fix it. | **correction** |
| Any | This scenario misses the reset case — please add it. | **correction** |
| Any | This explanation is wrong. | **feedback** |

Corrections are not explanations, but they drive two requirements: keep every
old revision (D9), and record who asked for what and why — itself research
data (where do LLMs err most?).

---

## 2. Concepts (mapping to BDX)

| BDX | Here |
|-----|------|
| Model **M** (spec, tests, implementation) | requirements, feature file, scenarios, testbench, DUV — all stored as revisions + artifacts |
| Belief **B** (observed behaviour) | simulation runs, check results, extracted trace events |
| Alphabet **Σ** (all steps) | canonical steps of a design (Section 4) |
| Scenario **s = P_s → Q_s** | `P_s` = Given/When steps, `Q_s` = Then steps (`scenario_step.role`) |
| Target **T = (P_T, Q_T)** | `explanation_target` (from a click or an NL question) |
| Explanation **X** | `explanation`: selected scenarios `S_T` + rendered Given/When steps |
| global / local | `P_T = ∅` / `P_T ≠ ∅` |

BDX computation (paper §IV-C): (1) translate T into Σ; (2) `S_T` = scenarios
with `Q_T ⊆ Q_s`, maximising `|P_T ∩ P_s|`; (3) render the Given/When steps of
`S_T` as text. Gaps we must close: step (1) is unspecified; canonical steps
are assumed but LLM-generated text is not canonical; ties in step (2) need a
fixed rule (we use: larger overlap, then fewer extra steps, then
`scenario_key` order).

---

## 3. Competency questions

Every question must be answerable from the database. **v1** = needed for the
first website release. New questions should only need a new query, not a new
table — that is the test of the schema.

### Traceability
- **Q1** (v1) Which scenarios / example rows cover requirement R, and did their checks pass on the golden design?
- **Q2** (v1) Requirement R is uncovered: no scenario, scenario without a check (`not_tested`), check only `weak`, or check failed?
- **Q3** (v1) Which checks belong to scenario S, and which checks were added by the generator itself (`GEN:` tags)?
- **Q4** (v2) A mutant was detected: by which check, scenario, requirement?
- **Q5** (v2) Which requirements are violated by no mutant (benchmark blind spots)?

### Explanation (BDX)
- **Q6** (v1, global) Under which conditions does outcome `Q_T` occur? (e.g. *When is overflow = 1?*)
- **Q7** (v1, local) Why did check C on example row r produce the observed value?
- **Q8** (v1) Which canonical steps does a free-text question refer to?
- **Q9** (v1) Which explanations exist for subject X, by which algorithm/version, and on what evidence?
- **Q10** (v2, counterfactual) What minimal change to the inputs avoids `Q_T`?

### Provenance and reproducibility
- **Q11** (v1) Who produced item X (human / LLM model + prompt version / rule id), and when?
- **Q12** (v1) Show the raw prompt and response of the LLM call that produced X.
- **Q13** (v1) Re-run: given a simulation run, retrieve the exact DUV, testbench and simulator version used.

### Corrections and versions
- **Q14** (v1) What is the revision history of scenario S, and which feedback caused each change?
- **Q15** (v1) Which corrections are still open?
- **Q16** (v2) Which kinds of errors do users report most often, per LLM?

### Comparison and evaluation (paper)
- **Q17** (v1) For design D, which requirements does LLM A cover that LLM B does not?
- **Q18** (v1) Which surface steps could not be mapped to a canonical step (per LLM)?
- **Q19** (v2) Explanation coverage of the deterministic vs. the LLM method on the same runs.
- **Q20** (v2) Explanation quality metrics (Section 7) per method.

### Search
- **Q21** (v2) Which scenarios / requirements / explanations are semantically closest to a question? (pgvector)

---

## 4. Canonical steps (Σ)

**Problem.** The same step is written differently by different LLMs:

```
Given I have operand A = <A>          Given input a is <A>
When I perform the ADD operation      When the ALU executes ADD (op=0000)
  with opcode 0000
Then the result should be <R>         Then the output equals <R>
```

BDX compares steps as sets (`Q_T ⊆ Q_s`, overlap of P). Without a single
canonical form these comparisons — and every coverage or cross-LLM
statistic — are wrong.

**Design: three layers.**

```
layer 1  surface text     "Given I have operand A = 5"        (as generated, kept verbatim)
            │ mapping (method + confidence)
layer 2  canonical step   "operand A = <value>"   params {value: 5}
            │ definition
layer 3  signal predicate a == 5                  (over the DUV's ports)
```

- A canonical step belongs to one design type and has a `role`
  (`context` = Given, `action` = When, `outcome` = Then), a text template
  with typed parameters, and a predicate template over **ports**
  (`port`, operator, value, optional cycle offset).
- Canonical steps are derived from the port list plus the step patterns
  the BDD prompts already prescribe (`prompts/bdd_*/v1.yaml`); the
  testbench generator already parses these patterns, so v1 mapping is
  **rule-based (regex), deterministic**.
- Steps that match no rule stay unmapped and are flagged; the unmapped rate
  is a quality metric (Q18).
- Later mapping methods (embedding, LLM, human) are stored side by side with
  `method` and `confidence` and can be compared.
- Layer 3 is the bridge to formal properties (SVA atoms), wishlist item
  "NL → formal → NL".

---

## 5. Schema (PostgreSQL)

Conventions:

- **Revisions.** Editable entities (`requirement`, `scenario`, `example_row`,
  `explanation`) carry `uid` (stable identity across revisions), `rev`,
  `supersedes` (previous row id), `is_current`, `created_at`. Old rows are
  never updated except to clear `is_current`.
- **Provenance.** Generated rows carry `prov_kind ∈ {human, llm, rule, import}`
  and `prov_ref` (`llm_call.id`, rule id, or feedback id).
- **Artifacts** (feature files, DUV, testbench, logs) are content-addressed
  by SHA-256 and stored in the DB (all are KB-sized). VCD waveforms are **not**
  stored: they are re-created by re-simulating the stored DUV + testbench
  (deterministic), which keeps the DB within the free storage limit.

```sql
-- ===== core objects =====
CREATE TABLE design (
  id          bigserial PRIMARY KEY,
  module_type text NOT NULL,          -- alu | alu_comb | counter | regfile | cpu
  bitwidth    int,
  source      text NOT NULL,          -- web | benchmark
  name        text NOT NULL,          -- e.g. alu_16bit
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE port (
  design_id  bigint REFERENCES design(id),
  name       text NOT NULL,
  direction  text NOT NULL,           -- input | output | inout
  width      int  NOT NULL,
  PRIMARY KEY (design_id, name)
);

CREATE TABLE artifact (
  sha256     char(64) PRIMARY KEY,
  kind       text NOT NULL,           -- requirements | feature | duv | testbench | sim_log | golden | mutant
  content    bytea NOT NULL,
  orig_name  text,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- one website workflow run (Steps 0-4), no personal data stored
CREATE TABLE session (
  id         bigserial PRIMARY KEY,
  design_id  bigint REFERENCES design(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  note       text
);

-- every LLM call, including failures (D11); successor of SQLite llm_calls
CREATE TABLE llm_call (
  id              bigserial PRIMARY KEY,
  session_id      bigint REFERENCES session(id),
  task_type       text,               -- requirements | bdd | duv | nl_question | polish | judge
  provider        text, model_requested text, model_served text,
  prompt_name     text, prompt_version text,
  system_prompt   text, prompt text, response text,
  params          jsonb,              -- sampling actually sent / ignored
  usage           jsonb,              -- tokens, cost if known
  latency_ms      int, success boolean NOT NULL, error text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- ===== model M: requirements, scenarios, steps =====
CREATE TABLE requirement (
  id bigserial PRIMARY KEY, uid uuid NOT NULL, rev int NOT NULL,
  supersedes bigint REFERENCES requirement(id), is_current boolean NOT NULL,
  design_id  bigint REFERENCES design(id),
  session_id bigint REFERENCES session(id),      -- NULL for benchmark FPs
  req_key    text NOT NULL,                      -- FP-4 | REQ-3
  text       text NOT NULL,
  origin     text NOT NULL,                      -- spec_fp | req_step | user
  prov_kind  text NOT NULL, prov_ref text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE feature (
  id bigserial PRIMARY KEY,
  session_id bigint REFERENCES session(id),
  artifact   char(64) REFERENCES artifact(sha256),
  title      text,
  prov_kind  text NOT NULL, prov_ref text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE scenario (
  id bigserial PRIMARY KEY, uid uuid NOT NULL, rev int NOT NULL,
  supersedes bigint REFERENCES scenario(id), is_current boolean NOT NULL,
  feature_id   bigint REFERENCES feature(id),
  scenario_key text NOT NULL,                    -- ALU-ADD-001 (existing deterministic tag)
  title        text, tags text[], is_outline boolean NOT NULL,
  prov_kind    text NOT NULL, prov_ref text,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE example_row (
  id bigserial PRIMARY KEY, uid uuid NOT NULL, rev int NOT NULL,
  supersedes bigint REFERENCES example_row(id), is_current boolean NOT NULL,
  scenario_id bigint REFERENCES scenario(id),
  row_no      int NOT NULL,                      -- ALU-ADD-001/r3
  vals        jsonb NOT NULL,                    -- {"A": "5", "B": "10", "Expected_Result": "15"}
  prov_kind   text NOT NULL, prov_ref text,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE canonical_step (                    -- alphabet Σ
  id          bigserial PRIMARY KEY,
  module_type text NOT NULL,
  role        text NOT NULL,                     -- context | action | outcome
  template    text NOT NULL,                     -- 'operand A = <value>'
  params      jsonb NOT NULL,                    -- {"value": "int"}
  predicate   jsonb,                             -- {"port":"a","op":"==","value":"<value>","cycle":0}
  version     int NOT NULL DEFAULT 1,
  UNIQUE (module_type, template, version)
);

CREATE TABLE scenario_step (
  scenario_id   bigint REFERENCES scenario(id),
  position      int NOT NULL,
  keyword       text NOT NULL,                   -- Given | When | Then | And | But
  role          text NOT NULL,                   -- P (Given/When) | Q (Then)
  surface_text  text NOT NULL,                   -- layer 1, verbatim
  PRIMARY KEY (scenario_id, position)
);

CREATE TABLE step_mapping (                      -- layer 1 -> layer 2, several methods side by side
  scenario_id bigint, position int,
  canonical_step_id bigint REFERENCES canonical_step(id),   -- NULL = unmapped
  params      jsonb,
  method      text NOT NULL,                     -- rule | embedding | llm | human
  method_ref  text,                              -- rule id / model / llm_call id
  confidence  real,
  PRIMARY KEY (scenario_id, position, method),
  FOREIGN KEY (scenario_id, position) REFERENCES scenario_step
);

CREATE TABLE scenario_requirement (
  scenario_id    bigint REFERENCES scenario(id),
  requirement_id bigint REFERENCES requirement(id),
  method         text NOT NULL,                  -- rule | judge_llm | human
  confidence     real, prov_ref text,
  PRIMARY KEY (scenario_id, requirement_id, method)
);

-- ===== tests and belief B =====
CREATE TABLE testbench (
  id bigserial PRIMARY KEY,
  session_id bigint REFERENCES session(id),
  artifact   char(64) REFERENCES artifact(sha256),
  generator  text NOT NULL,                      -- template | llm
  prov_kind  text NOT NULL, prov_ref text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE check_item (
  id bigserial PRIMARY KEY,
  testbench_id   bigint REFERENCES testbench(id),
  tag            text NOT NULL,                  -- BDD:ALU-ADD-001/r3 | GEN:RF-R0
  scenario_id    bigint REFERENCES scenario(id),
  example_row_id bigint REFERENCES example_row(id),
  strength       text NOT NULL DEFAULT 'normal'  -- normal | weak
);

CREATE TABLE sim_run (
  id bigserial PRIMARY KEY,
  testbench_id bigint REFERENCES testbench(id),
  dut_artifact char(64) REFERENCES artifact(sha256),
  target       text NOT NULL,                    -- golden | generated_duv | mutant:<file>
  simulator    text NOT NULL,                    -- 'iverilog 12.0'
  status       text NOT NULL,                    -- ok | compile_error | timeout
  log_artifact char(64) REFERENCES artifact(sha256),
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE check_result (
  sim_run_id bigint REFERENCES sim_run(id),
  check_id   bigint REFERENCES check_item(id),
  status     text NOT NULL,                      -- pass | fail | not_tested
  expected   jsonb, observed jsonb,              -- {"result": "15"} / {"result": "14"}
  sim_time   bigint, message text,
  PRIMARY KEY (sim_run_id, check_id)
);

CREATE TABLE trace_event (                       -- key events only, never full waveforms
  sim_run_id bigint REFERENCES sim_run(id),
  sim_time   bigint NOT NULL,
  port       text NOT NULL,
  value      text NOT NULL,
  canonical_step_id bigint REFERENCES canonical_step(id)
);

-- ===== explanations =====
CREATE TABLE explanation_target (                -- T = (P_T, Q_T)
  id bigserial PRIMARY KEY,
  entry        text NOT NULL,                    -- click | nl
  question     text,                             -- NL text if entry = nl
  subject_type text, subject_id bigint,          -- what was clicked
  p_t          jsonb NOT NULL,                   -- [{canonical_step_id, params}]  ([] = global)
  q_t          jsonb NOT NULL,
  scope        text NOT NULL,                    -- global | local
  mode         text NOT NULL DEFAULT 'complete', -- complete | counterfactual (v2)
  mapping_llm_call bigint REFERENCES llm_call(id),  -- NL -> Σ, if any
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE explanation (
  id bigserial PRIMARY KEY, uid uuid NOT NULL, rev int NOT NULL,
  supersedes bigint REFERENCES explanation(id), is_current boolean NOT NULL,
  target_id      bigint REFERENCES explanation_target(id),
  algorithm      text NOT NULL,                  -- bdx_v1
  selected       jsonb NOT NULL,                 -- S_T: [{scenario_id, example_row_id, overlap}]
  text_rule      text NOT NULL,                  -- deterministic rendering
  text_polished  text,                           -- optional LLM rephrasing
  polish_llm_call bigint REFERENCES llm_call(id),
  evidence       jsonb NOT NULL,                 -- check_result / trace_event ids
  auto_checks    jsonb,                          -- {"faithful": true, "minimal": true, "no_added_facts": true}
  created_at     timestamptz NOT NULL DEFAULT now()
);

-- ===== corrections and feedback =====
CREATE TABLE feedback (
  id bigserial PRIMARY KEY,
  subject_type text NOT NULL, subject_id bigint NOT NULL,
  kind         text NOT NULL,                    -- correction | error_report | rating | comment
  body         text,
  proposed     jsonb,                            -- proposed change, if any
  status       text NOT NULL DEFAULT 'open',     -- open | applied | rejected
  result_ref   text,                             -- new revision id once applied
  created_at   timestamptz NOT NULL DEFAULT now()
);

-- ===== v2 =====
-- fault(design_id, file, taxonomy, difficulty, description, artifact)
-- fault_violates(fault_id, requirement_id, prov_kind, prov_ref)
-- embedding(subject_type, subject_id, model, dim, vec vector(N))  -- separate table:
--   changing the embedding model never touches the core tables
```

Indexes (sketch): partial unique `(uid) WHERE is_current` on each revisioned
table; `(scenario_id)` on `check_item`, `step_mapping`; `(sim_run_id)` on
`check_result`, `trace_event`; GIN on `example_row.vals`.

---

## 5a. Schema changes in v0.3 (migration 0002)

| Change | Why |
|--------|-----|
| new `duv` (origin llm/upload/golden/mutant, parent_id + reason, workflow_mode, prompt_has_bdd, based_on_feature_id, prompt_has_decisions, compiles, provenance) | a design has many concrete Verilog implementations; pairing with the feature it came from (D16, D17) |
| new `decision` (question, options, default, answer, answer source, why) | Step-0 answers explain choices both generators made |
| `feature` + parent_id, reason, workflow_mode, prompt_has_decisions | file-level versions (D16) |
| `check_item` + seq, test_no, port, width, operation, stimulus, expected_bdd (B), expected_ref (R), expected_used, oracle; unique (testbench, tag, seq) | one row per assertion, with what it actually does (D15) |
| `check_result` + observed_value (O), verdict | who is wrong (D14) |
| `sim_run` + duv_id | exact feature ↔ Verilog pair that was simulated |
| `explanation_target` + feature_id, sim_run_id; `explanation` + structure | context of an explanation; structured storage (D19) |
| `artifact` + encoding | compression later (D18) |
| `sim_run` + probe_of (migration 0004) | links a flag-probe run to the main simulation it observes (D23) |
| `duv` + compile_error (migration 0003) | first iverilog error of the Step-1 check; the only explanation when a design cannot be simulated |

Values B, R, O are stored as exact unsigned integers masked to the port width
(`NUMERIC(40,0)`), the way the testbench compares them.

Validated on a real Gemini feature file: of three failing ALU rows, SUB row 7
(B = 4294967245, R = O = 4294967246) and SRA row 5 are `scenario_wrong`, SLTU
row 4 (B = R = 0, O = 1, injected bug) is `design_wrong`
(`tests/test_xdb.py::test_who_is_wrong_on_the_real_feature`).

---

## 6. Explanation algorithm v1 (sketch)

Global target `T = (∅, Q_T)` — e.g. click "why?" on outcome step
"overflow flag = True":

```sql
-- scenarios whose Then-steps contain every canonical step in Q_T (relational division)
SELECT m.scenario_id
FROM step_mapping m JOIN scenario_step s USING (scenario_id, position)
JOIN scenario sc ON sc.id = m.scenario_id AND sc.is_current
WHERE s.role = 'Q' AND m.method = 'rule'
  AND m.canonical_step_id = ANY(:q_t_ids)
GROUP BY m.scenario_id
HAVING count(DISTINCT m.canonical_step_id) = cardinality(:q_t_ids);
```

Local target (click on a failing check of row r): `P_T` = canonical P-steps of
row r instantiated with `example_row.vals`, `Q_T` = the outcome step of the
check with the **observed** value. Candidates as above, ranked by
`|P_T ∩ P_s|` (same canonical step **and** same parameter values), ties broken
by fewer extra steps, then `scenario_key`. `X` = Given/When steps of `S_T`
rendered with the row's values, plus expected vs. observed from
`check_result`.

---

## 6a. What M2 implements

- `src/xdb/library.py` — ALU library v1, hand-written: 9 canonical steps
  (width, operand A, operand B incl. "shift amount B", operation/opcode,
  result, zero / overflow / negative / carry flags) with regex patterns and
  port predicates; 10 operation rules (expression + English sentence + an
  independent implementation) and 3 flag rules. Rules are tested against
  `TestbenchGenerator._reference_result` on random inputs (8/16/32 bit).
- `src/xdb/mapping.py` — every stored step is mapped at ingestion
  (`step_mapping`, method `rule`); unmatched steps are stored with
  `canonical_step_id = NULL`. On the Gemini ALU feature: 63/63 steps mapped.
- `src/xdb/explain.py` — `explain_check` (local: verdict, scenario steps
  instantiated for the row, applied stimulus, rule with the concrete
  calculation, design output vs. rule, contrasting row of the same scenario,
  provenance, evidence, and Then steps the testbench does not compare — each
  of these is checked against the flag rules, so a scenario that states a
  wrong flag is reported even though nothing simulated it);
  `explain_global` (BDX global, P_T = ∅: every row whose instantiated Q
  contains port = value, grouped by operation, plus the general rule; each
  row is checked against the rule, rows the rule predicts but the
  specification omits are listed, and when all contradicting rows fit a
  known alternative meaning — e.g. "overflow" used as unsigned
  carry/borrow — that is said explicitly. Alternatives
  (`library.FLAG_ALTERNATIVES`) only describe a specification, they are
  never used as rules);
  `explain_requirement`. Results are stored in `explanation_target` /
  `explanation` (structure + English rendering) and reused.
- CLI: `python -m src.xdb runs | remap | explain ...`.
- First field observation (DeepSeek v4 Flash, 16-bit ALU, 2026-11-03): 61/61
  checks passed, yet all 5 rows stating overflow = 1 contradict signed
  overflow (the design template's definition) and match carry/borrow. The
  testbench never compares the flag, so only the specification check sees
  it (`test_global_explanation_detects_carry_meaning_of_overflow`).

---

## 6b. What M3 implements

- Endpoints (read-only, no model call, so no access password; `{"enabled": false}`
  without `DATABASE_URL`): `/api/explain/summary` (finds the simulation by the
  SHA-256 of its testbench and log, returns verdict counts, per-check verdicts,
  requirement ids and the specification check; `ready: false` until the
  background write has finished), `/api/explain/check`, `/api/explain/global`,
  `/api/explain/requirement`.
- `explain.check_specification(feature)`: for each flag the scenarios state,
  every row against the rule, with the alternative meaning when all
  contradictions share one.
- Step 4 page: section "Who Is Wrong? — Explanations" (headline naming the
  compared ports, verdict chips, the flag probe's result, specification check
  with Why?), a verdict chip, flag-probe findings and Why? on every scenario
  row, explanation in a modal. The requirements table takes its status from
  D24 (with the subject port and the evidence: simulation or flag probe),
  has its own Why? column, and shows "not compared" only for the requirement
  it concerns. English only.
  Verified in headless Chromium (`tests/ui/explain_panel.js`), including a
  390 px wide viewport.

---

## 6c. Plain-language view (D25)

`explain.build_problems` turns one simulation into a list of problems,
most severe first: design wrong (per operation, with one example and the
count), scenario wrong (wrong expected result, wrong flag), the design
repeating a scenario's mistake, flags wrong in the design (rows whose result
is already wrong are folded into that problem as "a consequence"), scenarios
the generator skipped (its reason translated, `plain_reason`) together with
the requirement they leave untested, and requirements no scenario covers.
`summarize_run` returns it with an overview; `/api/explain/*` return
`plain` next to `text`.

Found while testing and fixed since (commit 3ab7993, merged): the testbench
generator's operation detection fell back to plain words in the scenario
title, so a title such as "Compare signed and unsigned …" was tested as AND.
It now reads only the title and the When step, in an operation-like context,
and leaves ambiguous scenarios unmapped (reported as "not tested"). Rows it
drops because their Operation cell names no known operation are listed as
skipped in the results (`rows_unmapped` in the scenario index), so such a
scenario reads "partly tested".

## 6d. Repair (M4, D26)

Code: `src/xdb/repair.py`; endpoints in `main.py`; buttons on the Problems
card.

| Action | Trigger | Model call | Output | Then |
|---|---|---|---|---|
| Correct the scenarios | `repairs.scenario_values > 0` | none | `<stem>_fixN.feature`, feature version `feedback:<id>` | Step 3 + 4 again (`/api/repair/scenarios` does both) |
| Regenerate the design | `repairs.design_failures > 0`, ALU, design generated by a model on this page | the design's model (Step-1 settings) | new design file, design version `feedback:<id>` | `/api/repair/rerun` (Step 3 + 4) |

`summarize_run` returns `repairs = {scenario_values, design_failures,
design_regeneration, feature_reason, duv_reason}`; the last two tell the page
whether the shown run already uses a repaired version.

Cell edits (`apply_corrections`): the scenario's span comes from the
testbench index catalog; data rows are counted across all Examples tables of
the scenario (the generator's row numbering, so `row_no` addresses the row of
the check tag); the column is the one the canonical outcome step reads
(`step_mapping.params[*].col` of ALU.Q.RESULT / ALU.Q.ZERO / …); the old cell
value must equal the stored row value, and the new value keeps the cell's
width and number style (hex, negative decimal, True/False case). Spans are
edited last-first so earlier offsets stay valid.

What the design prompt receives (`prompt_store.repair_feedback_block`):

```
IMPORTANT - Repair of a previous version:
A previous version of this design was simulated and made the errors listed
below. Write a corrected version: fix these errors, keep what was correct, and
keep the module name, the ports, their order and their widths exactly as they are.

The previous version of this design failed these checks (3 in total):
- SLTU with a = 4294967295 (0xFFFFFFFF), b = 0 (opcode 1001): result must be 0, the design gave 1
- ...

The rules these checks follow:
- SLTU: SLTU is 1 when A < B as unsigned numbers, else 0

The previous version:
```verilog
...
```
Return the complete corrected module.
```

Measured in the tests (`tests/test_xdb.py`, the Gemini feature with an
injected SLTU bug): scenario correction changes exactly SUB r7, SRA r5 and
the planted ADD r7 zero flag; the rerun has verdicts {pass 64, design_wrong 3}
and no scenario problem. Design regeneration (fake model returning the fixed
SLTU) gives {pass 65, scenario_wrong 2}. Browser check:
`tests/ui/repair_flow.js` runs both buttons in sequence and ends with
"No problems found".

Limitations: design regeneration is wired for ALU in the non-streaming Step-1
endpoint only; a repaired design that does not compile is saved and reported,
the shown result stays the previous one; there is no undo button (the old
files are kept, and Step 1/2 can be rerun); repairs on the public site are not
moderated (open issue below) — the scenario correction only ever writes values
computed by the rule library, the design regeneration costs API credit.

For the paper, the feedback rows give, per run: how many cells the rules
corrected, how many design repairs a model needed until its checks passed
(chain of duv versions with reason `feedback:`), and the verdict counts before
and after each step.

---

## 6e. Register file (M5a, D27)

Data flow for one read, `RF-READ-001/r1` of the test feature (R1 is written
100 in example 1 and 300 in example 4 of "Write values into registers"; the
read scenario still expects 100):

```
index detail   test 5, READ, addr 1, B = 100, R = 300, compared with 100
replay         cause = write at test 4 (example 4), value 300; earlier: 100 (test 1)
verdict        O = 300 (design), B = 100, R = 300  ->  scenario_wrong
plain text     The scenario is wrong, not the design. It expects 100, but R1 holds 300
               at this point — and that is what the design returns.
               100 was written by example 1 of “Write values into registers”, but
               example 4 of the same scenario overwrote it with 300 before this read.
repair         Examples cell expected 100 -> 300 (outcome column of RF.Q.RDATA)
```

Canonical steps: `RF.P.CONFIG`, `RF.P.STATE` (Givens that only describe the
carried-over state), `RF.A.WRITE`, `RF.A.READ`, `RF.A.RESET`, `RF.Q.RDATA`,
`RF.Q.CLEARED` (stated, not compared), `RF.Q.WRITE`. All steps of the prompt's
example format map.

Design repair for a register file sends each failing read with its position
in the sequence and the writes/resets it depends on, e.g. "read of R0 at
step 12 of the sequence: rdata1 must be 0, the design gave 123. Before it:
step 11 wrote 123 to R0 (must be ignored); register 0 must always read 0",
plus failing checks the testbench adds by itself and the rules for WRITE,
RESET and the failing reads.

Limitations: the sequence semantics differ from Gherkin's scenario isolation;
a scenario written as if it started from an empty register file shows up as
scenario_wrong, with the write from another scenario named as the cause —
the explanation makes the dependency visible rather than hiding it. Read
port 2 and same-cycle read/write are not exercised by the testbench.

---

## 6f. RV32I CPU (M5b, D28)

Test sample (`tests/test_xdb.py`, CPU_FEATURE on a CPU whose SLT/SLTI compare
unsigned), 19 checks:

| Row | What the explanation says | Verdict |
|---|---|---|
| sub x3, x1, x2, x1 = 10, x2 = 30, expects 20 | 10 − 30 = −20 | scenario wrong |
| add x3, x1, x1, x1 = 5 then 7, expects 12 | x1 has two values, the second is in effect: 14 | scenario wrong (not auto-corrected) |
| addi x0, x1, 5, expects 15 | x0 always stays 0 | scenario wrong |
| slt x3, x1, x2, x1 = −1, x2 = 1 | design gives 0; seems to compare unsigned | design wrong |
| blt x1, x2, 8, −5 vs 3, expects not taken | −5 < 3 is true: taken | scenario wrong |
| lw x4, 4(x1) | checked at x1 + 4 | pass |
| sw x2, 0(x1) with rd = 0 (the prompt's own format) | checked in memory | pass |
| add x5, x1, x2 with rd column 3 | skipped by the testbench: other registers than the columns | Partly tested |

The LW and SW rows were bugs of the testbench generator: offsets were
ignored and a store row with an rd column was checked against x0. They are
fixed in commit 41d90a9 (together with: rows whose instruction names other
registers than the columns, x29/x30, unaligned or result-area addresses and
contended memory words are skipped with a reason), merged here. An ALU/counter
row without an expected value is no longer checked against 0 but skipped
(ab9960a, reason "scenario states no expected result", shown as "Partly
tested"). Testbenches of features without such rows
are byte-identical. For testbenches generated before the fix, `cpu_reference`
still reports the ignored offset and the store checked as a register as
testbench issues (`stimulus.offset_applied` / `path` tell the two apart).

Repair: scenario correction writes −20, 0 and "yes" (cell style kept, negative
values signed); design regeneration lists each failing instruction with its
operands. Browser check: `stage_run.py cpu` + `repair_flow.js`.

Limitations: the PC and pipeline timing are not observed; JAL/JALR/LUI and
byte/half-word accesses are outside the supported subset; the explanation
assumes each row starts from the registers its own ADDIs set.

---

## 7. Evaluating explanations

| Criterion | Meaning | Measurement |
|-----------|---------|-------------|
| Faithfulness | every statement agrees with spec and observed run | automatic: re-check stated conditions against `check_result` / `trace_event` |
| Relevance (precision) | selected scenarios are relevant | gold `S_T` for a labelled question set |
| Completeness (recall) | no relevant scenario missing (global) | same gold set |
| Minimality | removing any step breaks the explanation | automatic, per step |
| No added facts | an LLM rephrasing adds nothing absent from the evidence | automatic diff of facts vs. `text_rule` + LLM judge, reported separately |
| Usefulness | helps a person find the problem | task study: participants localise injected faults (existing mutants) with vs. without explanations; time and accuracy |

---

## 8. Deployment

- **Database:** Neon, region AWS Frankfurt (`aws-eu-central-1`, fixed at
  project creation). Free plan per Neon's docs (verify before relying on it):
  0.5 GB storage per project, scale-to-zero after 5 min idle (first request
  afterwards is slower), pgvector included. Size estimate for v1: KB-sized
  artifacts and text, no waveforms → well below 0.5 GB.
- **Connection:** `DATABASE_URL` environment variable (Neon pooled
  connection string, SSL required); never committed.
- **App:** stays on Render (`render.yaml`), Frankfurt region to sit next to
  the DB.
- **Access:** reading is public; LLM-calling operations require the access
  password and are rate-limited (D7).
- **Privacy (GDPR):** no accounts, no IP addresses or other personal data
  stored; a short privacy note on the site says that submitted content is
  public.
- **Local development:** PostgreSQL on the developer machine (Postgres.app or
  Homebrew), same schema. Schema changes via Alembic migrations; access via
  SQLAlchemy.
- **Existing SQLite** (`output/experiments/experiments.db`): untouched for
  now; a one-off import script moves `llm_calls` and run artifacts later.

---

## 9. Phases (superseded by the milestones below)

Milestones (2026-11-03): **M1** verdicts + schema v2 + step-by-step storage (done) ·
**M2** ALU canonical-step and rule library, explanation engine (local, global,
contrastive) (done, see §6a) · **M3** Step-4 UI: verdict summary and "why?" panel (English) (done, §6b) ·
**M4** repair: apply reference value (new feature version) / regenerate design (done, §6d) ·
**M5** regfile (done, §6e), CPU (done, §6f); counter reference model · **M6** free-text questions.

Original phase list:

1. Review this document.
2. Schema + migrations; ingestion of the web workflow (Steps 0–4) on a local
   Postgres; tests.
3. Canonical steps + rule-based mapping for each design type; unmapped report.
4. Explanation algorithm v1 (global + local), stored with evidence and
   automatic checks.
5. Website: "why?" buttons on scenarios / rows / checks / requirements;
   correction form; revision history view.
6. Neon deployment.
7. v2: NL questions via LLM mapping, pgvector search, benchmark workflow
   (faults), counterfactual explanations.

---

## 10. Open issues

- Canonical-step coverage of the CPU design (instruction-level steps are
  richer than port assignments).
- How much of the counter's multi-cycle behaviour fits into single-cycle
  predicates (needs `cycle` offsets).
- Who reviews and applies corrections on a public site without accounts
  (moderation).
- Neon free-plan limits must be re-checked on the official pricing page at
  setup time.
