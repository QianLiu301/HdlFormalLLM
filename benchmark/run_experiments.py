#!/usr/bin/env python3
"""
    Batch Experiment Runner

    Iterates through the benchmark suite × LLMs × repetition count, executing the full pipeline for each combination:

    spec.md (with functional requirement answers removed)
    → [LLM] Generate BDD scenarios (Gherkin)
    → [LLM] Generate self-checking Verilog testbench
    → iverilog compilation + golden simulation (must result in "TEST PASSED")
    → Mutation testing on each bug variant (detected = "TEST PASSED" not printed)
    → (Optional) LLM judge evaluates functional requirement coverage

    All LLM calls are automatically logged via `src/experiment_logger` (tagged with `run_id` / `module_name`),
    structured results are written to the `benchmark_results` table in the same SQLite database,
    and generated artifacts are saved under `output/benchmark_runs/<run_id>/`.

    Usage:
    # Smoke test (no API calls; uses built-in mock to verify the entire pipeline)
    python benchmark/run_experiments.py --llms mock --reps 1 --run-id smoke

    # Real experiment: 2 LLMs × all 9 modules × 3 repetitions
    python benchmark/run_experiments.py --llms groq,gemini --reps 3 --run-id exp001

    # Run specific modules only + use groq as the coverage judge
    python benchmark/run_experiments.py --llms groq --modules alu_8bit,sync_fifo_8x8 \\
    --judge groq --run-id exp002

    # View result summary
    python benchmark/run_experiments.py summary --run-id exp001
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = PROJECT_ROOT / "benchmark"
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_logger import call_context, DB_PATH  # noqa: E402

SIM_TIMEOUT = 30  # seconds, to prevent the LLM-generated testbench from entering an infinite loop.
LLM_MAX_TOKENS = 8000

BDD_SYSTEM = "You are a hardware verification engineer expert in BDD (Behavior-Driven Development)."
BDD_PROMPT = """Below is the specification of a hardware module.

{spec}

Write BDD test scenarios in Gherkin format (Feature / Scenario / Given / When / Then)
that thoroughly verify this module. Cover normal operation, boundary values,
corner cases, and any priority or flag behavior described in the spec.
Output ONLY the Gherkin feature file content, no explanations.
"""

TB_SYSTEM = "You are a hardware verification engineer expert in Verilog testbench design."
TB_PROMPT = """Below is a hardware module specification, its port declaration, and BDD test
scenarios describing what must be verified.

=== SPECIFICATION ===
{spec}

=== MODULE PORT DECLARATION (instantiate exactly this interface) ===
{ports}

=== BDD SCENARIOS ===
{bdd}

Write a SELF-CHECKING Verilog-2005 testbench that implements these BDD scenarios.
Requirements:
- Testbench module must be named `tb`.
- Instantiate the DUT exactly as declared above.
- Check expected values with if-statements; on any mismatch print
  "TEST FAILED" (plus details) using $display.
- If ALL checks pass, print exactly "TEST PASSED" at the end.
- Always call $finish at the end. Do not use SystemVerilog-only features.
Output ONLY the Verilog code in a ```verilog code block.
"""

JUDGE_SYSTEM = "You are a strict hardware verification auditor."
JUDGE_PROMPT = """A hardware module has the following numbered functional points (FPs):

{fps}

Below are BDD scenarios generated for this module:

{bdd}

For each FP, decide whether at least one scenario genuinely tests it.
Respond with ONLY a JSON array of the covered FP numbers, e.g. [1,2,5].
"""

MOCK_BDD = """Feature: Mock feature
  Scenario: trivial
    Given the DUT
    When nothing happens
    Then nothing is checked
"""
MOCK_TB = """```verilog
module tb;
    initial begin
        $display("TEST PASSED");
        $finish;
    end
endmodule
```"""


# ---------------------------------------------------------------------------
# Result table
# ---------------------------------------------------------------------------

def _results_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("""CREATE TABLE IF NOT EXISTS benchmark_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, run_id TEXT, module TEXT, category TEXT,
        llm TEXT, model TEXT, rep INTEGER,
        scenarios_count INTEGER,
        fp_total INTEGER, fp_covered INTEGER, completeness REAL,
        tb_compiled INTEGER, golden_passed INTEGER,
        mutants_total INTEGER, mutants_detected INTEGER, mutation_score REAL,
        bdd_ms INTEGER, tb_ms INTEGER,
        error TEXT, artifacts_dir TEXT)""")
    return conn


def save_result(row: dict):
    conn = _results_conn()
    cols = ",".join(row.keys())
    ph = ",".join("?" * len(row))
    conn.execute(f"INSERT INTO benchmark_results ({cols}) VALUES ({ph})",
                 list(row.values()))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_spec(mod_dir: Path):
    """Return the spec and FP list (with function points removed).
    FPs serve as the scoring ground truth and must not be disclosed to the LLM under test."""
    text = (mod_dir / "spec.md").read_text(encoding="utf-8")
    m = re.split(r"^## Functional Points.*$", text, flags=re.M)
    spec_body = m[0].strip()
    fps = re.findall(r"^- (FP-\d+): (.+)$", text, flags=re.M)
    return spec_body, fps


def extract_ports(golden: Path) -> str:
    """Extract the module declaration (up to the `');'` at the end of the port list)
    without exposing the implementation."""
    text = golden.read_text(encoding="utf-8")
    m = re.search(r"module\s+\w+.*?\);", text, flags=re.S)
    return m.group(0) if m else text.split("\n")[0]


def extract_verilog(response: str) -> str:
    m = re.search(r"```(?:verilog|systemverilog|v)?\s*\n(.*?)```", response, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(module\s.*?endmodule)", response, flags=re.S)
    return m.group(1).strip() if m else response.strip()


def compile_and_run(dut: Path, tb: Path, workdir: Path):
    """Returns (compiled, passed, output). "Passed" is defined as
    output containing "TEST PASSED" and not containing "TEST FAILED"."""
    exe = workdir / "sim.vvp"
    # Explicitly specify encoding/errors: Windows defaults to GBK,
    # which will throw a `UnicodeDecodeError` when processing UTF-8 output.
    try:
        r = subprocess.run(["iverilog", "-g2005", "-o", str(exe), str(dut), str(tb)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=SIM_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, False, "compile timeout"
    if r.returncode != 0:
        return False, False, r.stderr
    try:
        r = subprocess.run(["vvp", str(exe)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=SIM_TIMEOUT)
    except subprocess.TimeoutExpired:
        return True, False, "simulation timeout"
    out = r.stdout + r.stderr
    passed = ("TEST PASSED" in out) and ("TEST FAILED" not in out)
    return True, passed, out


def _config_api_key(name: str):
    """Read the api_key of a certain provider from config/llm_config.json (do not take the model - the one in the configuration
The model name may expire, and the default value provided by the provider is more reliable)"""
    try:
        cfg = json.loads((PROJECT_ROOT / "config" / "llm_config.json").read_text(encoding="utf-8"))
        return cfg.get("providers", {}).get(name, {}).get("api_key") or None
    except Exception:
        return None


def make_provider(name: str):
    if name == "mock":
        return None
    from src.llm_providers import LLMFactory
    # First, create using the default method (the key from environment variables takes precedence and has been verified to work);
    # If the key is missing from environment variables (e.g., for Claude/Grok), use the key from the config file as a fallback.
    provider = LLMFactory.create_provider(name)
    if not hasattr(provider, "_call_api"):
        key = _config_api_key(name)
        if key:
            provider = LLMFactory.create_provider(name, api_key=key)
    # The factory silently falls back to LocalLLMProvider upon creation failure (template text, lacking _call_api).
    # —benchmark must fail hard; otherwise, it results in data contamination.
    if not hasattr(provider, "_call_api"):
        raise RuntimeError(f"provider '{name}' fell back to local templates — "
                           f"no usable api_key in env vars or config/llm_config.json")
    return provider


# Providers do not throw exceptions upon API failure; instead, they silently return the template text (which is friendly for interactive applications
# but pollutes benchmark data)—here, we detect this and retry; if retries are exhausted, we raise the error as-is and log it in the 'error' field.
FALLBACK_MARKER = "Test ALU operation with various input values"
LLM_RETRIES = 3
RETRY_BACKOFF = 10  # seconds; wait n * RETRY_BACKOFF after the n-th failure


def _is_fallback(text: str) -> bool:
    if FALLBACK_MARKER in text:
        return True
    # Another fallback branch _fallback_intent_json: a small snippet of {"scenario":..., "operation":...} JSON
    if '"operation"' in text and '"scenario"' in text and len(text) < 800:
        return True
    # OpenAI _fallback_text fallback sentence
    if text.startswith("Given ALU operation, When executed"):
        return True
    return False


def llm_call(provider, name, prompt, system):
    if name == "mock":
        return MOCK_BDD if "Gherkin" in prompt else MOCK_TB
    # OpenAI's _call_api enforces JSON mode (required by the main application); benchmarks require plain text,
    # so prioritize _call_api_text, while other providers return plain text by default.
    call = getattr(provider, "_call_api_text", None) or provider._call_api
    for attempt in range(1, LLM_RETRIES + 1):
        resp = call(prompt, max_tokens=LLM_MAX_TOKENS, system_prompt=system)
        text = (resp or "").strip()
        if text and not _is_fallback(text):
            return resp
        if attempt < LLM_RETRIES:
            print(f"  ⚠️ {name} API failure (fallback detected), "
                  f"retry {attempt}/{LLM_RETRIES - 1} in {attempt * RETRY_BACKOFF}s ...", flush=True)
            time.sleep(attempt * RETRY_BACKOFF)
    raise RuntimeError(f"{name} API failed {LLM_RETRIES} times (provider returned fallback text)")


# ---------------------------------------------------------------------------
# Single experimental unit: (module, llm, rep)
# ---------------------------------------------------------------------------

def run_one(mod_dir: Path, manifest: dict, llm: str, provider, rep: int,
            run_id: str, judge=None, judge_name=None) -> dict:
    module = manifest["name"]
    spec_body, fps = load_spec(mod_dir)
    golden = mod_dir / "golden.v"
    art_dir = (PROJECT_ROOT / "output" / "benchmark_runs" / run_id /
               module / llm / f"rep{rep}")
    art_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id, "module": module, "category": manifest["category"],
        "llm": llm, "model": getattr(provider, "model", "mock"), "rep": rep,
        "scenarios_count": 0, "fp_total": len(fps), "fp_covered": None,
        "completeness": None, "tb_compiled": 0, "golden_passed": 0,
        "mutants_total": len(manifest["bugs"]), "mutants_detected": 0,
        "mutation_score": None, "bdd_ms": None, "tb_ms": None,
        "error": None, "artifacts_dir": str(art_dir),
    }

    try:
        # 1) BDD Scenario Generation
        with call_context(task_type="bench_bdd", run_id=run_id, module_name=module):
            t0 = time.time()
            bdd = llm_call(provider, llm, BDD_PROMPT.format(spec=spec_body), BDD_SYSTEM)
            row["bdd_ms"] = int((time.time() - t0) * 1000)
        (art_dir / "scenarios.feature").write_text(bdd, encoding="utf-8")
        row["scenarios_count"] = len(re.findall(r"^\s*Scenario", bdd, flags=re.M))

        # 2) Testbench Generation
        with call_context(task_type="bench_tb", run_id=run_id, module_name=module):
            t0 = time.time()
            tb_resp = llm_call(provider, llm, TB_PROMPT.format(
                spec=spec_body, ports=extract_ports(golden), bdd=bdd), TB_SYSTEM)
            row["tb_ms"] = int((time.time() - t0) * 1000)
        tb_file = art_dir / "tb.v"
        tb_file.write_text(extract_verilog(tb_resp), encoding="utf-8")

        # 3) Golden simulation
        compiled, passed, out = compile_and_run(golden, tb_file, art_dir)
        (art_dir / "golden_sim.log").write_text(out, encoding="utf-8")
        row["tb_compiled"] = int(compiled)
        row["golden_passed"] = int(passed)

        # 4) Mutation testing (only meaningful if the golden test passes)
        if passed:
            detected = 0
            mut_log = []
            for bug in manifest["bugs"]:
                _, mut_passed, mout = compile_and_run(
                    mod_dir / "bugs" / bug["file"], tb_file, art_dir)
                caught = not mut_passed
                detected += int(caught)
                mut_log.append(f"{bug['file']}: {'DETECTED' if caught else 'ESCAPED'}")
            row["mutants_detected"] = detected
            row["mutation_score"] = round(detected / len(manifest["bugs"]), 3)
            (art_dir / "mutation.log").write_text("\n".join(mut_log), encoding="utf-8")

        # 5) Coverage Arbiter (Optional)
        if judge is not None and row["scenarios_count"] > 0:
            fp_text = "\n".join(f"{k}: {v}" for k, v in fps)
            with call_context(task_type="bench_judge", run_id=run_id, module_name=module):
                verdict = llm_call(judge, judge_name,
                                   JUDGE_PROMPT.format(fps=fp_text, bdd=bdd), JUDGE_SYSTEM)
            m = re.search(r"\[[\d,\s]*\]", verdict)
            if m:
                covered = set(json.loads(m.group(0)))
                row["fp_covered"] = len(covered)
                row["completeness"] = round(len(covered) / len(fps), 3)

    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"

    save_result(row)
    return row


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def print_summary(run_id=None):
    conn = _results_conn()
    where, params = ("WHERE run_id = ?", [run_id]) if run_id else ("", [])
    rows = conn.execute(f"""
        SELECT llm, COUNT(*),
               AVG(tb_compiled), AVG(golden_passed),
               AVG(mutation_score), AVG(completeness), AVG(scenarios_count)
        FROM benchmark_results {where}
        GROUP BY llm ORDER BY AVG(golden_passed) DESC""", params).fetchall()
    conn.close()
    if not rows:
        print("No results found.")
        return
    hdr = f"{'LLM':<12}{'Runs':>5}{'Compile%':>10}{'GoldenPass%':>13}{'MutScore':>10}{'Complete':>10}{'AvgScen':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        fmt = lambda v, pct=False: ("-" if v is None else
                                    f"{v * 100:.0f}%" if pct else f"{v:.2f}")
        print(f"{r[0]:<12}{r[1]:>5}{fmt(r[2], True):>10}{fmt(r[3], True):>13}"
              f"{fmt(r[4]):>10}{fmt(r[5]):>10}{fmt(r[6]):>9}")


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--run-id", default=None)
        print_summary(ap.parse_args().run_id)
        return

    ap = argparse.ArgumentParser(description="Run BDD-Hardware benchmark experiments")
    ap.add_argument("--llms", required=True,
                    help="comma-separated: groq,gemini,openai,claude,deepseek,grok,qwen,mistral,together or 'mock'")
    ap.add_argument("--modules", default=None, help="comma-separated module names (default: all)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per (module, llm)")
    ap.add_argument("--run-id", required=True, help="experiment batch id, e.g. exp001")
    ap.add_argument("--judge", default=None, help="LLM used to judge FP coverage (optional)")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel experiments (LLM calls are IO-bound; use 1-2 for rate-limited providers)")
    ap.add_argument("--append", action="store_true",
                    help="allow adding results to an existing run_id (careful: same module×llm×rep overwrites artifacts)")
    args = ap.parse_args()

    # Reusing a run_id causes the artifacts directories (run_id/module/llm/repN) to overwrite each other; this is rejected by default.
    conn = _results_conn()
    existing = conn.execute("SELECT COUNT(*) FROM benchmark_results WHERE run_id = ?",
                            (args.run_id,)).fetchone()[0]
    conn.close()
    if existing and not args.append:
        sys.exit(f"run_id '{args.run_id}' already has {existing} results in the DB.\n"
                 f"Pick a new --run-id, or pass --append to knowingly add to it.")

    manifests = sorted(BENCH_ROOT.glob("*/*/manifest.json"))
    if args.modules:
        wanted = set(args.modules.split(","))
        manifests = [m for m in manifests if m.parent.name in wanted]
    if not manifests:
        sys.exit("No matching modules found.")

    llms = args.llms.split(",")
    providers = {name: make_provider(name) for name in llms}
    judge = make_provider(args.judge) if args.judge and args.judge != "mock" else None

    tasks = []
    for mf in manifests:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        for llm in llms:
            for rep in range(1, args.reps + 1):
                tasks.append((mf, manifest, llm, rep))

    total = len(tasks)
    workers = max(1, min(args.workers, 8))
    print(f"🚀 run_id={args.run_id}: {len(manifests)} modules × {len(llms)} LLMs × "
          f"{args.reps} reps = {total} experiments, {workers} workers\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()
    done = [0]

    def run_task(mf, manifest, llm, rep):
        row = run_one(mf.parent, manifest, llm, providers[llm], rep,
                      args.run_id, judge=judge, judge_name=args.judge)
        with lock:
            done[0] += 1
            tag = f"[{done[0]}/{total}] {manifest['name']} × {llm} rep{rep}"
            if row["error"]:
                print(f"{tag}  ❌ {row['error']}", flush=True)
            else:
                ms = row["mutation_score"]
                print(f"{tag}  scenarios={row['scenarios_count']} "
                      f"compiled={bool(row['tb_compiled'])} "
                      f"golden={bool(row['golden_passed'])} "
                      f"mutation={ms if ms is not None else '-'} "
                      f"completeness={row['completeness'] if row['completeness'] is not None else '-'}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, *t) for t in tasks]
        for f in as_completed(futures):
            f.result()

    print(f"\n{'=' * 60}\nSummary for run_id={args.run_id}:\n")
    print_summary(args.run_id)
    print(f"\nArtifacts: output/benchmark_runs/{args.run_id}/")
    print(f"Raw LLM calls: python -m src.experiment_logger recent")


if __name__ == "__main__":
    main()
