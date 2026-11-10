# Handover: context for a new session (updated 2026-11-05)

Development of the explainability track ran in one long cloud session until
2026-11-04. From then on the work happens in a Claude Code session on the
user's Mac, in the local project folder. This file is the entry point: it
says what the user wants, what the sources were, how we work, where things
stand and what comes next. Details live in the documents it points to.

Read first: `CLAUDE.md`, then `docs/explainability_db_design.md` (decision
log D1–D46, §1 users, §2 BDX mapping, §6d–§6j milestones and trial findings, §7a the
explanation standard, §7b where the reference rules come from),
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
(credentials), D33 (a CPU check counts only if the program got there), D34
(empty answers), D35 (Step 4 says when it is waiting), D36 (the model list), D37 (a
card is one thing to fix), D38 (what a card says must hold for the whole run), D39 (Step 0 offers only what can be honoured), D40 (an unchanged standard description needs no model), D41 (the Step 1 and Step 2 boxes reach the model), D42 (Step 0 shows the fixed rules only for an edited description), D43 (what the testbench cannot show is never called verified), D44 (Step 4 uses the testbench path Step 3 reports), D45 (the ALU reference checked against Verilog; rule sources in §7b), D46 (per-card code lines, checked local fixes and chat). 270 tests (`tests/`), browser checks in `tests/ui/`.

Live site: master is 23dcb9a = the release before this work + the model list of D36
(24fd642) + the Docker image installing `requirements-core.txt` instead of the
Windows export in `requirements.txt`. Everything else stays on
`bdx-explainability` until the user merges it; the Dockerfile change arrives with
that merge.

Trials (`first`, `cpu_fix` in `data/trials/`; `cpu_v2`, `cpu_v3` in
`output/trials/`, local; findings in design doc §6g–§6i):
- `first`: DeepSeek + Gemini × ALU / register file / CPU × 2 → D29 and the
  parser fixes.
- `cpu_fix`: CPU only. Gemini hit its quota (429); DeepSeek timed out or
  returned empty answers in 2 of 3 runs; the one finished DeepSeek CPU was
  blamed for wrong loads, which was the testbench's memory timing (D31).
- `cpu_v2`, `cpu_v3`: gpt-5-mini, CPU × 3, before and after D33/D34. All 178
  scenario values agree with an independent instruction-set simulator; `cpu_v3`
  ends with no failing check in all 3 runs, 94 checks, one real LW bug repaired.
  The compile-error repair worked on real model output. JAL is wrong in all 3
  designs and no scenario tests it: "no failing check" is not "a correct CPU".

## 5. Next steps

0. After the master update, check on Render that no `GLM_MODEL` or `QWEN_MODEL` still
   names a retired model and that `QWEN_API_KEY` is set: `load_config` reads only
   the environment there, and a `*_MODEL` overrides the default in the code.
1. Optional: the same CPU trial with DeepSeek (balance 7.09 CNY on 2026-11-04,
   no balance error so far) or Gemini when its quota allows, to compare models:
   `python -m src.xdb.trial run --llms deepseek --designs cpu --reps 3 --tag cpu_v3_deepseek`.
2. Check whether other design types have unstated interface contracts like
   D31 (timing of reads, reset polarity, flag meanings).
3. R3 first: the independent simulator of §6i (`output/trials/cpu_v3/check/`) finds
   what the padded one-instruction programs cannot (JAL, forwarding, stalls).
4. Then the roadmap in `docs/research_outline.md` §5: R2 coverage view,
   R4 history page (could also store what only `output/trials/` holds today, such
   as a failed Step 0 per run), R5 SVA view, R6 counter reference model,
   R7 free-text questions.

Running the tests locally needs iverilog, pytest and a local, wipeable PostgreSQL
with pgvector for `XDB_TEST_URL` (never the Neon database: the tests drop the
schema); step by step for the Mac in the setup document, §8. Runs of the website
and of the trial runner do go to Neon, from any machine with `DATABASE_URL`.
