# Explainability Database — Design (v0.2)

Status: **phase 2 implemented** (schema, migrations, web-workflow ingestion,
v1 queries; see `explainability_db_setup.md`). Phases 3+ pending. Branch: `bdx-explainability`
(never merged into `master`).

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

## 9. Phases

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
