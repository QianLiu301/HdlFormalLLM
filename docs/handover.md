# Handover: context for a new session (2026-11-04)

Development of the explainability track ran in one long cloud session until
2026-11-04. From then on the work happens in a Claude Code session on the
user's Mac, in the local project folder. This file is the entry point: it
says what the user wants, what the sources were, how we work, where things
stand and what comes next. Details live in the documents it points to.

Read first: `CLAUDE.md`, then `docs/explainability_db_design.md` (decision
log D1–D32, §1 users, §2 BDX mapping, §6d–§6h milestones and trial findings),
`docs/explainability_db_setup.md` (how to run everything),
`docs/research_outline.md` (candidate papers and feature roadmap R1–R7).

## 1. What the user wants

- PhD project of Qian Liu: LLM-based BDD for hardware design and
  verification. The website (Steps 0–4) is the platform.
- **Now: build website features first; papers are decided later** (user,
  2026-11-04). `docs/research_outline.md` records what each feature could
  contribute, without commitments.
- Explainability: the website explains test results, preferably without an
  LLM and verifiably (deterministic rule libraries, every number reproduced
  by an independent model). An LLM is acceptable for wording and for parsing
  free-text questions, never for deciding who is wrong.

## 2. Sources the user shared and what we took from them

| Source | What we took |
|---|---|
| Drechsler & Liu, "LLM-based BDD for HW Design", arXiv 2512.17814 (the user's own paper) | the 4-step web flow this project is built on |
| Dominik & Drechsler, BDX, arXiv 2607.24881 (PDF given to the cloud session; ask the user for it if needed) | scenario as P → Q, canonical steps Σ, local / global / contrastive explanations, explanation coverage. Mapping and the gaps we closed: design doc §2. Not binding: the collaborator left; we only want explainability on our website (user) |
| CAUSE (DFG RTG 2972, self-explaining systems) wishlist from the group | verifiable, deterministic, LLM-free explanations; NL → formal → NL; spec vs. implementation; back-annotation. Status table: research outline §4. The "unreal" wind-turbine example was dropped (user) |
| ChromaDB (seen by the user) | considered; chose PostgreSQL + pgvector on Neon for provenance, revisions and relational queries (design doc §5, D-log) |
| Related work: AutoBench, LLM4Cov, CVDP, FVEval | they score testbenches/designs; none decides *who* is wrong (P1 in the outline) |

The user's answers that shaped the design: users are hardware developers,
researchers, reviewers (D1); v1 = local + global explanations (D2); click
first, free-text questions later (D3, roadmap R7); web flow before the
benchmark (D4); existing designs only (D5); no user accounts.

## 3. How we work (user preferences)

- Chat with the user in Chinese; code, docs and all product output (UI,
  explanations, stored rule text) in English.
- The user is on macOS (zsh); give Mac commands, one per code block.
- Local only: commit locally; push to `bdx-explainability` only when the
  user says so; never push to master.
- Never show a key or password in chat or in files. The pre-commit hook
  (`tools/check_secrets.py`) must stay enabled (D32).
- Claims are checked against data before they are reported (re-simulate,
  compare models); a verdict is only evidence if the testbench did what the
  scenario says and the model was given the interface contract (D29, D31).
- The user tests in the browser; report results plainly, including failures.

## 4. Where things stand

Done: explainability DB and verdicts (M1), rule libraries and explanations
(M2), Step-4 UI (M3), repairs (M4), register file (M5a), RV32I CPU (M5b),
batch trial runner (R1), D29 (steps the testbench cannot run), D30
(compile-error repair), D31 (CPU memory timing, prompt duv_cpu v2), D32
(credentials). 153 tests (`tests/`), browser checks in `tests/ui/`.

Trials (results in `data/trials/`, findings in design doc §6g, §6h):
- `first`: DeepSeek + Gemini × ALU / register file / CPU × 2 → D29 and the
  parser fixes.
- `cpu_fix`: CPU only. Gemini hit its quota (429); DeepSeek timed out or
  returned empty answers in 2 of 3 runs; the one finished DeepSeek CPU was
  blamed for wrong loads, which was the testbench's memory timing (D31).
  The compile-error repair has not yet run on real model output.

## 5. Next steps

1. Check the local setup with the mock model:
   `python -m src.xdb.trial run --llms mock --tag dryrun_local`.
2. Re-run the CPU with DeepSeek (prompt v2, fixed memory timing, compile
   fixes on): `python -m src.xdb.trial run --llms deepseek --designs cpu --reps 3 --tag cpu_v2`,
   then read `output/trials/cpu_v2/` (summary.md, runs.jsonl, logs/).
   Gemini again when its quota allows.
3. Check whether other design types have unstated interface contracts like
   D31 (timing of reads, reset polarity, flag meanings).
4. Then the roadmap in `docs/research_outline.md` §5: R2 coverage view,
   R3 deep check with counterexamples, R4 history page, R5 SVA view,
   R6 counter reference model, R7 free-text questions.

Running the tests locally needs iverilog and a local, wipeable PostgreSQL
for `XDB_TEST_URL` (never the Neon database); see the setup document.
