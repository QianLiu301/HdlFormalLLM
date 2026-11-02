# Explainability Database — Design Draft (v0.1)

Status: **draft for discussion** (no code yet). Branch: `bdx-explainability`.

Goal: add explainability to the HdlFormalLLM web platform. Every generated
artifact (requirement, BDD scenario, assertion, simulation verdict,
explanation) is stored with its provenance and its links to the others, so
that the platform can answer *why* a check passed or failed, *which
requirement* it serves, and *who* (human, LLM, or deterministic rule)
produced each piece.

We are not bound to the definitions of the BDX paper (arXiv 2607.24881);
it is related work, not a specification.

---

## 1. Competency questions

The schema is driven by the questions it must answer. Every question below
must be answerable with one SQL query (Section 6 gives examples).

| ID | Question |
|----|----------|
| CQ1 | Which scenarios cover requirement R, and did they pass on the golden design? |
| CQ2 | A fault (mutant) was detected: by which check, from which scenario, serving which requirement? |
| CQ3 | Requirement R is uncovered: was no scenario generated, or was one generated but its check never compiled/ran (`not_tested`) or was too weak? |
| CQ4 | For design D, which requirements does LLM A cover that LLM B does not? |
| CQ5 | Explanation E: who produced it (rule id / LLM model + prompt version), and on which evidence (check, log, trace)? |
| CQ6 | On the same runs, what is the explanation coverage of the deterministic generator vs. the LLM generator? |
| CQ7 | Scenario S: which LLM call produced it? Show the raw prompt and response. |
| CQ8 | Which requirements are violated by no fault at all (benchmark blind spots)? |
| CQ9 | For a failing check on the website, give a natural-language "why" built from Given/When/Then + observed value. |
| CQ10 | Find scenarios semantically similar to a free-text user question (website search). *(ChromaDB, Section 7)* |

---

## 2. Design principles

1. **SQLite is the single source of truth.** One file
   (`output/experiments/experiments.db`, extended — not a new DB), ships in
   the replication package, no server. Rejected alternatives: Neo4j (server,
   harder to package), hosted LLM-observability tools (only cover the LLM-call
   layer; data leaves the machine).
2. **Graph-shaped relational schema.** Node tables + link tables, so the data
   can be exported to a graph DB / visualizer later without redesign.
3. **Provenance on every generated row.** Columns
   `prov_kind ∈ {human, llm, rule, import}` and `prov_ref`
   (`llm_calls.id`, rule id, or file path). This is what makes the
   LLM-vs-deterministic comparison (CQ6) possible.
4. **Raw LLM output is kept** (already true: `llm_calls.prompt/response`).
   Structured rows point back to the call that produced them (CQ7).
   Benefits: reproducibility, re-scoring without new API cost, error analysis.
5. **Artifacts are content-addressed.** Feature files, testbenches, logs are
   stored by SHA-256; identical content is stored once and can never be
   silently overwritten.
6. **Stable IDs.** Scenario keys reuse the existing deterministic tags
   (`ALU-ADD-001/r3`, from commit ee0b481). Requirement keys reuse `FP-n`.
7. **Anti-leak preserved.** Ground-truth links (`fault_violates`,
   human `scenario_covers`) are never read by any prompt-building code.
8. **ChromaDB is a derived index, never a store.** It can be deleted and
   rebuilt from SQLite at any time (Section 7).

---

## 3. Two existing workflows that must be unified

| | Web workflow (`main.py`, Step 0–4) | Benchmark workflow (`benchmark/`) |
|---|---|---|
| Requirements | Step-0 requirement file (`output/requirements/`) | `spec.md` FP-n (ground truth, withheld) |
| Scenarios | Feature file, tagged `ALU-ADD-001/r3` | Feature text in `llm_calls.response`, **no IDs** |
| Checks | Template TB, every line tagged `[BDD:id]` / `[GEN:id]` | LLM-written TB, **untagged** |
| Faults | none | 64 mutants in `bugs/*.v` + `manifest.json` |
| Results | `*_tb.scenarios.json` + sim log | `benchmark_results` (aggregate only) |

Consequence: the full chain requirement → scenario → check → fault exists in
**neither** workflow today. The schema covers both; closing the gap needs
(a) scenario IDs + check tags in benchmark TBs, and (b) fault→requirement
labels (Section 8).

---

## 4. Entities and tables

Existing tables stay unchanged: `llm_calls`, `run_artifacts`,
`benchmark_results`, `feedback_results`, `baseline_runs`.
New tables (prefix `x_` to keep them apart):

```sql
-- ---------- static design data (imported from benchmark/ or web) ----------
CREATE TABLE x_design (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,          -- counter_8bit, alu_16bit_web ...
  category TEXT,                      -- combinational / sequential / fsm / memory / cpu
  source TEXT NOT NULL,               -- benchmark | web
  spec_artifact TEXT,                 -- -> x_artifact.sha256
  golden_artifact TEXT
);

CREATE TABLE x_requirement (
  id INTEGER PRIMARY KEY,
  design_id INTEGER NOT NULL REFERENCES x_design(id),
  req_key TEXT NOT NULL,              -- FP-4 / REQ-3
  text TEXT NOT NULL,
  origin TEXT NOT NULL,               -- spec_fp | req_step
  prov_kind TEXT NOT NULL, prov_ref TEXT,
  UNIQUE (design_id, req_key, origin)
);

CREATE TABLE x_fault (
  id INTEGER PRIMARY KEY,
  design_id INTEGER NOT NULL REFERENCES x_design(id),
  file TEXT NOT NULL,                 -- bug03_down_overflow_missing.v
  taxonomy TEXT, difficulty TEXT, description TEXT,
  artifact TEXT,                      -- mutant source sha256
  UNIQUE (design_id, file)
);

-- ground truth, hand-labelled once (NEW work, see Section 8)
CREATE TABLE x_fault_violates (
  fault_id INTEGER REFERENCES x_fault(id),
  requirement_id INTEGER REFERENCES x_requirement(id),
  prov_kind TEXT NOT NULL, prov_ref TEXT,
  PRIMARY KEY (fault_id, requirement_id)
);

-- ---------- per-run generated data ----------
CREATE TABLE x_run (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL,               -- same key as llm_calls.run_id
  design_id INTEGER REFERENCES x_design(id),
  workflow TEXT NOT NULL,             -- benchmark | web | feedback
  llm TEXT, model_requested TEXT, model_served TEXT,
  prompt_version TEXT, rep INTEGER, arm TEXT, iteration INTEGER,
  created_at TEXT NOT NULL, config_json TEXT
);

CREATE TABLE x_artifact (
  sha256 TEXT PRIMARY KEY,
  kind TEXT NOT NULL,                 -- spec|golden|mutant|feature|testbench|duv|log|trace|report
  path TEXT,                          -- original location (informational)
  content BLOB,                       -- NULL if large and kept on disk
  created_at TEXT NOT NULL
);

CREATE TABLE x_scenario (
  id INTEGER PRIMARY KEY,
  run_pk INTEGER NOT NULL REFERENCES x_run(id),
  scenario_key TEXT NOT NULL,         -- ALU-ADD-001/r3
  title TEXT, given_text TEXT, when_text TEXT, then_text TEXT,
  feature_artifact TEXT REFERENCES x_artifact(sha256),
  prov_kind TEXT NOT NULL, prov_ref TEXT,   -- llm + llm_calls.id
  UNIQUE (run_pk, scenario_key)
);

-- which requirement a scenario serves; may come from a human, a judge LLM,
-- or a rule -> several rows per pair are allowed, distinguished by method
CREATE TABLE x_scenario_covers (
  scenario_id INTEGER REFERENCES x_scenario(id),
  requirement_id INTEGER REFERENCES x_requirement(id),
  method TEXT NOT NULL,               -- human | judge_llm | rule
  confidence REAL,
  prov_ref TEXT,
  PRIMARY KEY (scenario_id, requirement_id, method)
);

CREATE TABLE x_check (
  id INTEGER PRIMARY KEY,
  run_pk INTEGER NOT NULL REFERENCES x_run(id),
  tag TEXT NOT NULL,                  -- BDD:ALU-ADD-001/r3 | GEN:RF-R0
  scenario_id INTEGER REFERENCES x_scenario(id),   -- NULL for GEN checks
  kind TEXT NOT NULL,                 -- sim_assert | sva (future)
  strength TEXT DEFAULT 'normal',     -- normal | weak (e.g. count <= max)
  tb_artifact TEXT REFERENCES x_artifact(sha256)
);

CREATE TABLE x_sim_result (
  id INTEGER PRIMARY KEY,
  run_pk INTEGER NOT NULL REFERENCES x_run(id),
  target TEXT NOT NULL,               -- 'golden' or 'fault:<x_fault.id>'
  check_id INTEGER REFERENCES x_check(id),        -- NULL = whole-TB verdict
  status TEXT NOT NULL,               -- pass | fail | not_tested | compile_error | timeout
  message TEXT,                       -- e.g. "expected 0x00 got 0xFF"
  log_artifact TEXT REFERENCES x_artifact(sha256)
);

-- ---------- explanations ----------
CREATE TABLE x_explanation (
  id INTEGER PRIMARY KEY,
  subject_type TEXT NOT NULL,         -- sim_result | requirement | fault | scenario
  subject_id INTEGER NOT NULL,
  question TEXT NOT NULL,             -- why_fail | why_pass | coverage | why_not | what_if
  text TEXT NOT NULL,
  evidence_json TEXT NOT NULL,        -- ids of checks / results / log lines used
  generator TEXT NOT NULL,            -- rule:<rule_id> | llm:<model>
  prompt_version TEXT, llm_call_id INTEGER,       -- -> llm_calls.id
  validation TEXT,                    -- NULL | consistent | inconsistent (auto-check vs evidence)
  created_at TEXT NOT NULL
);
```

Size estimate: 9 LLMs × 20 designs × 5 reps × ~20 scenarios ≈ 18k
scenarios, ~10⁵ sim results — trivial for SQLite.

---

## 5. Ingesting what already exists

| Source | Target |
|--------|--------|
| `benchmark/*/spec.md` FP list | `x_requirement` (origin `spec_fp`, prov `import`) |
| `benchmark/*/manifest.json` bugs | `x_fault` |
| `output/requirements/*` (Step 0) | `x_requirement` (origin `req_step`) |
| `*_tb.scenarios.json` + sim log (web) | `x_scenario`, `x_check`, `x_sim_result` |
| `llm_calls` (bdd_generation) | `x_scenario.prov_ref` (benchmark scenarios need a parser) |
| `benchmark_results` | `x_run` (aggregate scores stay where they are) |

---

## 6. Example queries

CQ1 — scenarios covering a requirement, with golden verdict:
```sql
SELECT s.scenario_key, c.method, r2.status
FROM x_requirement q
JOIN x_scenario_covers c ON c.requirement_id = q.id
JOIN x_scenario s        ON s.id = c.scenario_id
LEFT JOIN x_check k      ON k.scenario_id = s.id
LEFT JOIN x_sim_result r2 ON r2.check_id = k.id AND r2.target = 'golden'
WHERE q.req_key = 'FP-4' AND q.design_id = :design;
```

CQ2 — detected fault → check → scenario → requirement:
```sql
SELECT f.file, k.tag, s.scenario_key, q.req_key
FROM x_sim_result r
JOIN x_fault f    ON r.target = 'fault:' || f.id
JOIN x_check k    ON k.id = r.check_id
LEFT JOIN x_scenario s        ON s.id = k.scenario_id
LEFT JOIN x_scenario_covers c ON c.scenario_id = s.id
LEFT JOIN x_requirement q     ON q.id = c.requirement_id
WHERE r.status = 'fail' AND r.run_pk = :run;
```

CQ8 — requirements no fault violates:
```sql
SELECT q.req_key FROM x_requirement q
WHERE q.design_id = :design AND q.origin = 'spec_fp'
  AND NOT EXISTS (SELECT 1 FROM x_fault_violates v WHERE v.requirement_id = q.id);
```

Deterministic explanation coverage (CQ6), definition:
requirement R is *explained* in run X iff some scenario S with
`(S, R) ∈ x_scenario_covers` has a check that **passes on golden** and
**fails on a fault F with `(F, R) ∈ x_fault_violates`**.
`EC(X) = |explained R| / |R violated by at least one fault|`.

---

## 7. ChromaDB: role and limits

ChromaDB is an embedded vector database (local, Python, no server). It
answers "what is *semantically similar* to this text", not exact relations.

**Use it for (derived index, rebuilt from SQLite):**
- Website search (CQ10): user types "why does overflow go high after load?"
  → retrieve the nearest scenarios / requirements / explanations, then show
  their exact linked data from SQLite.
- Duplicate detection among LLM-generated scenarios (quality metric).
- Retrieval of similar past scenarios as few-shot examples for the Phase-2
  feedback loop.

**Do not use it for:**
- Coverage or explanation metrics — similarity is approximate and depends on
  the embedding model; metrics must stay exact and deterministic.
- Storing anything that exists only there.

**Collections:** `scenarios`, `requirements`, `explanations`. Each Chroma id
equals the SQLite primary key (`scenario:123`); metadata holds `design`,
`run_id`, `llm`. The embedding model name and version are recorded so the
index is reproducible. Rebuild command: one script that reads SQLite and
re-embeds.

**Trade-off:** adds a dependency (`chromadb` + an embedding model). It is
optional — the platform works without it; only CQ10 needs it. Recommended to
add after the SQLite core is stable.

---

## 8. Feasibility and open issues

| Item | Effort | Risk / note |
|------|--------|-------------|
| Schema + import of static data (designs, FP, faults) | low | — |
| Web-workflow ingestion (tags already exist) | low | — |
| `x_fault_violates` labels for 64 mutants | low–medium (manual, ~1–2 days) | Some FPs may have no mutant → add mutants (CQ8 finds them) |
| Scenario IDs + `[BDD:]` tags in **benchmark** TBs | medium | TBs are LLM-written; tagging must be requested in the prompt and verified by parser; untagged checks fall back to whole-TB verdict |
| `x_scenario_covers` (scenario → requirement) | medium | Who decides? Options: human (gold, small subset), judge LLM (scalable, stored with method + confidence), rule (keyword/signal match). Keep all, compare. |
| Deterministic "why-fail" explanations from Given/When/Then + log | low | Template-based; validation = recompute from evidence |
| Counterfactual (why-not / what-if) | high | Needs a solver (SymbiYosys); later phase |
| Explanation quality evaluation | high (research question) | Proposed formal criteria: consistency with trace, sufficiency, minimality |
| ChromaDB search | low–medium | Optional, after core |

---

## 9. Proposed phases

1. **This document** reviewed (competency questions + schema).
2. `src/xdb.py`: schema, import of benchmark static data, web ingestion; tests.
3. Fault → requirement labels; EC metric; deterministic why-fail explanations.
4. Website: explanation panel per failing check, requirement coverage view.
5. Benchmark TB tagging; ChromaDB search; formal layer.
