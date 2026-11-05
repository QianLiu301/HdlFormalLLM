# HdlFormalLLM — Project Memory

## What this project is

PhD research platform: **LLM-based BDD (Behavior-Driven Development) for hardware
verification**. A Flask web app that uses 9 LLM providers to generate Verilog
designs, BDD test scenarios, testbenches, and run simulations (iverilog/Yosys).

**Research goal (agreed with professor, 2026-07):** two phases
1. **Benchmark + multi-LLM comparison** — build a BDD-hardware benchmark
   (specs + golden designs + fault-injected mutants), run all LLMs under
   identical settings, score them objectively. ← infrastructure DONE, real
   experiments pending
2. **Simulation-feedback loop** — feed coverage/failure results back to the
   LLM to iteratively fill test gaps; argue BDD as intermediate representation
   beats direct testbench regeneration. ← NOT started

Own prior work: "LLM-based BDD for HW Design" (Drechsler & Liu, arXiv
2512.17814) — the user's own paper, not a competitor. Related work to
cite/compare: AutoBench (TUM), LLM4Cov
(VTS 2026), CVDP (NVIDIA), FVEval (Berkeley). Target venues: ASP-DAC / DATE /
ISVLSI; workshop first (MLCAD/ICLAD).

## Architecture

- `main.py` — Flask app (~2100 lines), all routes. Port 5000 local.
- `src/llm_providers.py` — 9 providers (Groq/Gemini/OpenAI/Claude/DeepSeek/
  Grok/Qwen/Mistral/Together) behind `LLMFactory.create_provider(name)`.
  Generic entry: `provider._call_api(prompt, max_tokens=, system_prompt=)`.
- `src/experiment_logger.py` — SQLite logger (`output/experiments/experiments.db`).
  Auto-instruments every `_call_api*` via `LLMProvider.__init_subclass__` —
  zero changes needed in providers. Tag calls with
  `with call_context(task_type=, run_id=, module_name=):`.
  Tables: `llm_calls` (every API call) + `benchmark_results` (scores).
- `benchmark/` — the benchmark suite (v0.1): 9 modules × 5 categories,
  64 single-fault mutants, 60 numbered functional points.
  - Per module: `spec.md` (NL spec + FP-n ground truth), `golden.v`
    (iverilog-verified), `tb_smoke.v` (deterministic stimulus), `bugs/*.v`
    (mutants labeled with taxonomy from `src/bug_taxonomy.py`), `manifest.json`.
  - `validate.py` — quality gate: golden must sim, every mutant must compile
    AND behave differently from golden (differential testing). Run after any
    benchmark change. Current: 9/9 golden OK, 64/64 mutants killed.
  - `run_experiments.py` — batch runner: (module × LLM × rep) →
    BDD generation (FP list withheld!) → testbench → golden gate →
    mutation testing → optional LLM judge for FP coverage.
    CLI: `--llms mock|groq,... --modules ... --reps N --run-id X --judge Y`;
    `summary --run-id X`. `mock` LLM = pipeline test, no API keys.
- `static/bdd_generator.html` — original 4-step UI (monolithic, 4700+ lines).
- `static/experiment_dashboard.html` — experiment dashboard at `/dashboard`:
  run experiments from browser, per-LLM score meters, CSV/JSONL/ZIP export.

## Scoring metrics (the paper's evaluation)

1. **Compile Rate** — generated TB compiles with iverilog
2. **Golden Pass** — TB must print "TEST PASSED" on the golden design
3. **Mutation Score** (headline metric) — fraction of fault mutants the TB
   detects (any non-PASS on mutant = detected)
4. **Completeness** (optional) — judge LLM counts FP coverage of the BDD

Anti-leak design: FP list withheld from the tested LLM; only port declaration
(never golden implementation) given for TB generation.

## Critical gotchas (hard-won fixes — do not regress)

- **Windows encoding**: ALWAYS pass `encoding='utf-8'` to read_text/write_text/
  open, and `encoding='utf-8', errors='replace'` to subprocess.run(text=True).
  Windows defaults to GBK and crashes on the em dashes in spec.md.
- **requests proxy semantics**: `proxies=None` means "use env vars", NOT
  "no proxy". Domestic providers (Qwen, DeepSeek) must return
  `{'http': None, 'https': None}` from `_get_proxies()` and pass it explicitly
  — otherwise they route through the user's proxy and fail.
- **Never save non-Verilog as .v**: LLM API failures fall back to template
  text; the hardware-stream endpoint guards with a `'module' not in code`
  check. Keep that guard.
- **User's local proxy**: v2rayN-style, SOCKS5 on 10808 / HTTP on 10809.
  Config in `config/llm_config.json` (gitignored; example file committed).
- **Credentials (D32)**: never put a key in a URL (Gemini uses the `x-goog-api-key` header); pass error texts that leave the machine through `src.redact.redact_secrets`; trial outputs are redacted in `write_outputs`. Pre-commit hook `tools/check_secrets.py` (`git config core.hooksPath .githooks`). 2026-11-04: a Gemini key and the Neon password reached the public repo (runs.jsonl, .env.save) — both to be rotated by the user; never print key values in chat.
- **Tests and Neon**: on a machine with `.env` the app reaches Neon by default, and `tests/test_xdb.py` runs `DROP SCHEMA public CASCADE` on `XDB_TEST_URL`. Keep `tests/conftest.py` (disables `.env` for tests); never point `XDB_TEST_URL` at Neon. Setup on the Mac: `docs/explainability_db_setup.md` §8.
- **CPU memory timing (D31)**: the CPU testbench reads both memories combinationally; `prompts/duv_cpu/v2` (web default) states it. A registered read made textbook pipelines fail every load.
- iverilog required for all simulation paths (Windows: bleyer.org/icarus,
  check "Add to PATH"). Included in Dockerfile.

## How to run

```bash
python main.py                     # web app → localhost:5000, dashboard at /dashboard
python benchmark/validate.py       # benchmark quality gate
python benchmark/run_experiments.py --llms mock --reps 1 --run-id smoke   # pipeline smoke test
python -m src.experiment_logger stats|recent|export   # raw LLM call log
```

Deploy: Render.com (render.yaml) or Docker. Render disk is ephemeral —
download CSV/ZIP from the dashboard after important runs.

## Conventions

- Branches (user decision 2026-11-04): **`bdx-explainability` is the working
  main branch.** Fixes made on other branches (`claude/*`) are merged into it;
  master is updated from it later, when the user decides. Commit and push
  there; do not push to master.
- User (钱柳/QianLiu301) now works on **macOS (zsh, Terminal)**; project at
  `~/PycharmProjects/HdlFormalLLM`. Give Mac commands only (bash/zsh syntax,
  Homebrew, `open`, Cmd shortcuts) — no PowerShell. Earlier work was on Windows,
  so keep the UTF-8 gotchas below. Tests in browser.
- **New session? Read `docs/handover.md` first** (goals, sources the user shared, how we work, status, next steps).
- Workflow (user request 2026-11-04): work in a Claude Code session on the user's Mac, in the local project folder; trial data and logs stay local (read them in `output/trials/`); commit locally and push to `bdx-explainability` only when the user says so. Cloud sessions cannot use the folder (Remote Control folder not offered to cloud sessions).
- Explain things in Chinese to the user; code/comments/docs in the repo mostly
  English (some Chinese comments in experiment tooling are fine).

## Explainability track (branch `bdx-explainability`, 2026-11)

Group wishlist ("BDX"): explanation coverage, real (RISC-V) + "unreal"
(wind turbine = CAUSE demonstrator) examples, website UI backed by a DB of
generated content, NL->formal->NL, spec-vs-impl validation, back-annotation
RTL/SVA -> scenarios, deterministic LLM-free baseline (CAUSE = DFG RTG 2972,
self-explaining systems; prefers verifiable, non-LLM explanations).
Not bound to the BDX paper (Dominik & Drechsler, arXiv 2607.24881).
Design doc: `docs/explainability_db_design.md` (v0.2: PostgreSQL + pgvector on
Neon Frankfurt; canonical steps in three layers; revisions + provenance on
every row; v1 = global/local explanations, click-first). Code in `src/xdb/`
(SQLAlchemy models, Alembic migrations, ingestion hooked into
`/api/run-report`, inert without `DATABASE_URL`, read from env or a gitignored
`.env`; Step 1/2 saves are stored via `_log_artifact`). Each assertion gets a
verdict from O/B/R (observed / BDD expected / reference model in
TestbenchGenerator): pass, scenario_wrong, design_wrong, both_wrong,
consistent_wrong, undecided — testbench Verilog is unchanged, the data comes
from the `*_tb.scenarios.json` index v2 (ALU only so far). Scope on this branch:
web Steps 0–4 (+explain, repair); benchmark/golden/mutants set aside.
Milestones M1, M2 done (M2: `src/xdb/library.py` ALU canonical steps + rules,
`mapping.py`, `explain.py`; CLI `python -m src.xdb runs|remap|explain`), M3 done
(`/api/explain/*` + Step-4 "Who Is Wrong?" panel with Why? buttons; browser
check `tests/ui/explain_panel.js`; the app sits behind the access-password
screen, `#main-app` must be shown in automated tests). Step 0 always asks
what ALU "overflow" means (`requirements_store.FIXED_QUESTIONS`); the rule
library follows the answer. Flag probe (`src/xdb/probe.py`, D23): a copy of
the testbench with one $display per test observes zero/negative/overflow, stored
as a separate sim_run with `probe_of`; main-run queries must exclude it
(`probe_of IS NULL` / testbench.generator != 'probe'). Requirement status by
subject port (D24, `explain.requirement_subject`). Step 4 shows a plain-language
Problems card first (D25, `explain.build_problems`, `render_plain`; no internal
ids), the old sections folded into "Technical details". M4 done (D26, `src/xdb/repair.py`: "Correct the scenarios" sets
scenario_wrong cells to the reference value in a new `<stem>_fixN.feature`,
no model; "Regenerate the design" re-calls Step 1 for ALU with the failing
checks + rules + previous Verilog (`repair_of` in /api/generate-hardware,
`prompt_store.repair_feedback_block`); both store a `feedback` row and the
new file as version `feedback:<id>`, then rerun Step 3+4; browser check
`tests/ui/repair_flow.js` with `tests/ui/stage_run.py`). M5a done (D27,
register file: rows run in order with state carried across scenarios; index
records writes/resets as setup rows; `library.regfile_replay` gives each
read's value and cause; explanations name the write/reset behind a value;
subject port rdata1; repair works for regfile; `stage_run.py regfile`).
M5b done (D28, CPU: `library.cpu_operands`/`cpu_reference` judge each row as
executed — R replaces the testbench's nominal value at ingestion; notes
(same register twice, x0) vs. issues (scenario: unset/mismatched registers;
testbench: offsets, SW-as-register, x29/x30) → verdict undecided; branches
as taken 1/0; `stage_run.py cpu`). Template testbench fixes merged here:
3ab7993 (no operation guessed from English words in titles), ab9960a (a
row without expected value is not asserted — ALU/counter, and since
2026-11-04 register-file reads and CPU rows too; reason
`TestbenchGenerator.NO_EXPECTED_RESULT`, `tests/test_no_expected_result.py`),
41d90a9 (CPU: stores with an rd column checked in memory, load/store offsets,
register/column mismatches, x29/x30, bad addresses, memory conflicts skipped
with a reason; `tests/test_template_testbenches.py`). Papers are decided
later (user, 2026-11-04: build website features first); candidate papers,
evidence and the feature roadmap R1–R7 are in `docs/research_outline.md`.
R1 done: batch trial runner `python -m src.xdb.trial run --llms … --designs …
--reps N --tag T` (drives the web endpoints in-process, stores runs in the DB
as `trial:<tag>`, writes output/trials/<tag>/summary.md|runs.csv|runs.jsonl;
`--llms mock` = canned samples from `src/xdb/samples.py`, no API). Its dry
run found that the testbench parser dropped the last Examples row of files
without a final newline (all web-generated features); fixed 2026-11-04.
First real trial (data/trials/first, design doc §6g) → D29: checks of a
scenario with a Given/When step the testbench does not do (library steps
`executable: False`, `library.not_executable`, `mapping.blocked_steps`; or an
unrecognised When step) are undecided, reported as testbench limits, never
repaired; parser keeps blank table cells (`testbench_generator._table_cells`,
empty operand → skipped `EMPTY_OPERAND`); trial `export` copies run files. D30 compile-error repair: `compile_fix_of {duv_filepath, testbench_path?}` in /api/generate-hardware (`main._compile_fix_feedback`: kind 'compile' = design alone fails iverilog -g2012, 'interface' = only fails with the testbench, ports from its `dut` instance; `prompt_store._REPAIR_INTRO[kind]`; any design type, DB optional; with DB a feedback row `fix_compile_error` via `repair.open_compile_feedback`); UI buttons in Step 1 (`renderDuvCompile`) and Step 4 (`offerSimFix`, then /api/repair/rerun); trial `--compile-fix N`, mock `--mock-fault compile|interface`; browser check `tests/ui/compile_fix.js` with `tests/ui/mock_server.py` (app + mock model). Ingestion retries deadlock victims (`ingest._retry`); generator file names are unique within one second. D33: a CPU check counts only if the program got there: the testbench records `fetched[]` / `written[]` from the memory ports; "- not reached" = status not_tested, no verdict; a branch whose target store never happened = "- did not continue after the branch", design_wrong (`library.CPU_NOT_REACHED`, `CPU_ASTRAY`, `explain._cpu_not_reached`). JAL/JALR rows are skipped, so "no failing check" does not cover jumps, forwarding or stalls (§6i). D34: an empty answer is logged as failed, and with finish_reason length retried once with twice the budget (`experiment_logger._retry_budget`). D35: Step 4 shows a waiting line, up to 4 minutes, with Try again; a query to Neon takes about 75 ms from the Mac, so avoid per-row queries. D36: models offered: Qwen 3 Coder Next, Gemma 4 31B, GLM 5.3 Flash (`reasoning_effort` low) on the GWDG key, plus OpenAI, Gemini, DeepSeek, Mistral on own accounts; check `chat-ai.academiccloud.de/v1/models` and the GWDG model list before adding one, and try it through the web flow (§6j). Master got this list alone (24fd642). D37: one card per root cause, signed values, a cause only when `library.alu_mixup` reproduces every example; the rules are in design doc §7a, the fixed example in `tests/data/unsigned_alu`. Product output (explanations, UI text,
stored rule text) is **English only** — no Chinese; chat with the user stays Chinese. Setup: `docs/explainability_db_setup.md`; tests: `tests/test_xdb.py`
(need a wipeable local Postgres via `XDB_TEST_URL` + iverilog — never Neon;
`service postgresql start` if the container restarted).

## Next steps (agreed plan)

1. User installs iverilog locally, runs mock end-to-end in the dashboard,
   then a small real run (deepseek/groq, 2-3 modules).
2. Scale benchmark 9 → 20+ modules (same recipe; run validate.py each time).
3. Full experiment: 9 LLMs × all modules × 5 reps → first dataset for paper.
4. Phase 2: simulation-feedback loop (coverage gaps → LLM → new BDD scenarios
   → re-simulate), compare against direct-TB-regeneration baseline.
