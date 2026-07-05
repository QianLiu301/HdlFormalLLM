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

Related work to cite/compare: Bremen/DFKI "LLM-based BDD for HW Design"
(arXiv 2512.17814, the only direct competitor), AutoBench (TUM), LLM4Cov
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
  28 single-fault mutants, 60 numbered functional points.
  - Per module: `spec.md` (NL spec + FP-n ground truth), `golden.v`
    (iverilog-verified), `tb_smoke.v` (deterministic stimulus), `bugs/*.v`
    (mutants labeled with taxonomy from `src/bug_taxonomy.py`), `manifest.json`.
  - `validate.py` — quality gate: golden must sim, every mutant must compile
    AND behave differently from golden (differential testing). Run after any
    benchmark change. Current: 9/9 golden OK, 28/28 mutants killed.
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

- Development branch pattern: `claude/*` feature branches, merged to `master`.
- User (钱柳/QianLiu301) works locally on Windows/PowerShell, tests in browser.
- Explain things in Chinese to the user; code/comments/docs in the repo mostly
  English (some Chinese comments in experiment tooling are fine).

## Next steps (agreed plan)

1. User installs iverilog locally, runs mock end-to-end in the dashboard,
   then a small real run (deepseek/groq, 2-3 modules).
2. Scale benchmark 9 → 20+ modules (same recipe; run validate.py each time).
3. Full experiment: 9 LLMs × all modules × 5 reps → first dataset for paper.
4. Phase 2: simulation-feedback loop (coverage gaps → LLM → new BDD scenarios
   → re-simulate), compare against direct-TB-regeneration baseline.
