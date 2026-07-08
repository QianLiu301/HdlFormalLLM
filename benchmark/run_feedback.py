#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 — simulation-feedback loop experiment (BDD-as-intermediate-representation study).

Two arms share the same one-shot baseline (iteration 0), then iterate:
  bdd : feedback -> revise BDD scenarios -> regenerate testbench from revised BDD
  tb  : feedback -> patch the testbench directly (BDD untouched)

Feedback given to the LLM = compiler errors, or golden-simulation failure lines.
Mutation results are NEVER fed back (they are the answer key); the mutation score
is measured blindly after every iteration. The FP list is also withheld (anti-leak),
so judge-based coverage gaps are NOT used as feedback in this version.

Results go to a dedicated `feedback_results` table (same SQLite DB), one row per
(module x llm x rep x arm x iteration). Converged tasks (golden passed, nothing to
fix) carry their scores forward so per-iteration averages stay comparable.

CLI:
    python benchmark/run_feedback.py --llms deepseek --modules traffic_light \\
        --iters 3 --reps 1 --run-id fb001 [--arms bdd,tb] [--workers 3]
    python benchmark/run_feedback.py summary --run-id fb001
    python benchmark/run_feedback.py --llms mock --iters 2 --run-id fb_smoke   # pipeline test
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = PROJECT_ROOT / "benchmark"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCH_ROOT))

import run_experiments as rx                          # noqa: E402
from src.experiment_logger import call_context, DB_PATH  # noqa: E402

FEEDBACK_MAX_CHARS = 3000

REVISE_BDD_PROMPT = """Below is a hardware module specification, the BDD scenarios previously written
for it, and feedback from running the generated testbench in a simulator.

=== SPECIFICATION ===
{spec}

=== CURRENT BDD SCENARIOS ===
{bdd}

=== SIMULATION FEEDBACK ===
{feedback}

Revise the BDD scenarios in Gherkin format to fix the problems the feedback reveals:
correct wrong expectations, fix mis-read timing semantics, and add scenarios for
behaviours that are missing. Keep scenarios that are already correct.
Output ONLY the complete revised Gherkin feature file content, no explanations.
"""

PATCH_TB_PROMPT = """Below is a hardware module specification, its port declaration, the Verilog
testbench previously written for it, and feedback from running it in a simulator.

=== SPECIFICATION ===
{spec}

=== MODULE PORT DECLARATION (instantiate exactly this interface) ===
{ports}

=== CURRENT TESTBENCH ===
{tb}

=== SIMULATION FEEDBACK ===
{feedback}

Fix the testbench according to the feedback. Keep it a SELF-CHECKING Verilog-2005
testbench: module named `tb`, check expected values with if-statements, print
"TEST FAILED" (plus details) on any mismatch, print exactly "TEST PASSED" at the
end if all checks pass, always call $finish, no SystemVerilog-only features.
Output ONLY the Verilog code in a ```verilog code block.
"""


# ---------------------------------------------------------------------------
# 结果表
# ---------------------------------------------------------------------------

def _fb_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, run_id TEXT, module TEXT, category TEXT,
        llm TEXT, model TEXT, rep INTEGER,
        arm TEXT, iteration INTEGER,
        feedback_kind TEXT, converged INTEGER,
        scenarios_count INTEGER,
        tb_compiled INTEGER, golden_passed INTEGER,
        mutants_total INTEGER, mutants_detected INTEGER, mutation_score REAL,
        error TEXT, artifacts_dir TEXT)""")
    return conn


def save_fb(row: dict):
    conn = _fb_conn()
    cols = ",".join(row.keys())
    ph = ",".join("?" * len(row))
    conn.execute(f"INSERT INTO feedback_results ({cols}) VALUES ({ph})", list(row.values()))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# 反馈构造与打分
# ---------------------------------------------------------------------------

def build_feedback(compiled: bool, passed: bool, sim_out: str):
    """从上一轮结果构造反馈文本；golden 通过则返回 (None, None) 表示收敛。
    注意：mutant 检出情况绝不进入反馈（那是评分答案）。"""
    if not compiled:
        lines = sim_out.strip().splitlines()[:15]
        text = ("The testbench FAILED TO COMPILE with `iverilog -g2005`. "
                "Compiler messages:\n" + "\n".join(lines))
        return "compile_error", text[:FEEDBACK_MAX_CHARS]
    if not passed:
        fails = [l for l in sim_out.splitlines()
                 if "FAIL" in l or "Expected" in l or "expected" in l]
        if not fails:
            fails = sim_out.strip().splitlines()[-15:]
        text = ("The testbench compiled, but when run against a KNOWN-CORRECT reference "
                "implementation of this specification it reported failures. The reference "
                "design is correct, so these failures mean the testbench's own expectations "
                "or timing are wrong:\n" + "\n".join(fails[:15]))
        return "golden_fail", text[:FEEDBACK_MAX_CHARS]
    return None, None


def score_tb(tb_file: Path, mod_dir: Path, manifest: dict, work_dir: Path):
    """golden 门控 + 盲测 mutation。返回 (compiled, passed, sim_out, detected, total)。"""
    compiled, passed, out = rx.compile_and_run(mod_dir / "golden.v", tb_file, work_dir)
    (work_dir / "golden_sim.log").write_text(out, encoding="utf-8")
    total = len(manifest["bugs"])
    detected = None
    if passed:
        detected = 0
        log = []
        for bug in manifest["bugs"]:
            _, mut_passed, _ = rx.compile_and_run(mod_dir / "bugs" / bug["file"], tb_file, work_dir)
            caught = not mut_passed
            detected += int(caught)
            log.append(f"{bug['file']}: {'DETECTED' if caught else 'ESCAPED'}")
        (work_dir / "mutation.log").write_text("\n".join(log), encoding="utf-8")
    return compiled, passed, out, detected, total


# ---------------------------------------------------------------------------
# 单个任务：module x llm x rep，跑 iter0 + 两臂各 iters 轮
# ---------------------------------------------------------------------------

def run_task(mod_dir: Path, manifest: dict, llm: str, provider, rep: int,
             run_id: str, iters: int, arms):
    module = manifest["name"]
    spec_body, _fps = rx.load_spec(mod_dir)
    golden = mod_dir / "golden.v"
    ports = rx.extract_ports(golden)
    base = PROJECT_ROOT / "output" / "feedback_runs" / run_id / module / llm / f"rep{rep}"
    model = getattr(provider, "model", "mock")

    def row_common(arm, iteration, art_dir):
        return {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": run_id, "module": module, "category": manifest["category"],
                "llm": llm, "model": model, "rep": rep, "arm": arm, "iteration": iteration,
                "feedback_kind": None, "converged": 0, "scenarios_count": 0,
                "tb_compiled": 0, "golden_passed": 0,
                "mutants_total": len(manifest["bugs"]), "mutants_detected": 0,
                "mutation_score": None, "error": None, "artifacts_dir": str(art_dir)}

    def count_scen(bdd):
        import re
        return len(re.findall(r"^\s*Scenario", bdd, flags=re.M))

    # ---- iteration 0：共享 one-shot 基线 -----------------------------------
    it0 = base / "iter0"
    it0.mkdir(parents=True, exist_ok=True)
    row = row_common("base", 0, it0)
    try:
        with call_context(task_type="fb_bdd0", run_id=run_id, module_name=module):
            bdd = rx.llm_call(provider, llm, rx.BDD_PROMPT.format(spec=spec_body), rx.BDD_SYSTEM)
        (it0 / "scenarios.feature").write_text(bdd, encoding="utf-8")
        with call_context(task_type="fb_tb0", run_id=run_id, module_name=module):
            tb_resp = rx.llm_call(provider, llm, rx.TB_PROMPT.format(
                spec=spec_body, ports=ports, bdd=bdd), rx.TB_SYSTEM)
        tb_text = rx.extract_verilog(tb_resp)
        (it0 / "tb.v").write_text(tb_text, encoding="utf-8")
        compiled, passed, out, det, tot = score_tb(it0 / "tb.v", mod_dir, manifest, it0)
        row.update({"scenarios_count": count_scen(bdd), "tb_compiled": int(compiled),
                    "golden_passed": int(passed),
                    "mutants_detected": det or 0,
                    "mutation_score": round(det / tot, 3) if det is not None else None})
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        save_fb(row)
        return f"{module} × {llm} rep{rep}: baseline ERROR {row['error']}"
    save_fb(row)

    # 两臂共享的起点状态
    state = {arm: {"bdd": bdd, "tb": tb_text, "compiled": compiled, "passed": passed,
                   "out": out, "det": det, "scen": row["scenarios_count"]}
             for arm in arms}

    # ---- 迭代 ---------------------------------------------------------------
    for k in range(1, iters + 1):
        for arm in arms:
            st = state[arm]
            it_dir = base / arm / f"iter{k}"
            it_dir.mkdir(parents=True, exist_ok=True)
            row = row_common(arm, k, it_dir)
            kind, fb = build_feedback(st["compiled"], st["passed"], st["out"])
            if fb is None:
                # 已收敛：分数带着走，保证每轮均值可比
                row.update({"feedback_kind": "converged", "converged": 1,
                            "scenarios_count": st["scen"],
                            "tb_compiled": int(st["compiled"]),
                            "golden_passed": int(st["passed"]),
                            "mutants_detected": st["det"] or 0,
                            "mutation_score": round(st["det"] / len(manifest["bugs"]), 3)
                                              if st["det"] is not None else None})
                save_fb(row)
                continue
            (it_dir / "feedback.txt").write_text(fb, encoding="utf-8")
            row["feedback_kind"] = kind
            try:
                if arm == "bdd":
                    with call_context(task_type="fb_bdd_revise", run_id=run_id, module_name=module):
                        st["bdd"] = rx.llm_call(provider, llm, REVISE_BDD_PROMPT.format(
                            spec=spec_body, bdd=st["bdd"], feedback=fb), rx.BDD_SYSTEM)
                    (it_dir / "scenarios.feature").write_text(st["bdd"], encoding="utf-8")
                    st["scen"] = count_scen(st["bdd"])
                    with call_context(task_type="fb_tb_regen", run_id=run_id, module_name=module):
                        tb_resp = rx.llm_call(provider, llm, rx.TB_PROMPT.format(
                            spec=spec_body, ports=ports, bdd=st["bdd"]), rx.TB_SYSTEM)
                else:  # arm == "tb"
                    with call_context(task_type="fb_tb_patch", run_id=run_id, module_name=module):
                        tb_resp = rx.llm_call(provider, llm, PATCH_TB_PROMPT.format(
                            spec=spec_body, ports=ports, tb=st["tb"], feedback=fb), rx.TB_SYSTEM)
                st["tb"] = rx.extract_verilog(tb_resp)
                (it_dir / "tb.v").write_text(st["tb"], encoding="utf-8")
                compiled, passed, out, det, tot = score_tb(it_dir / "tb.v", mod_dir, manifest, it_dir)
                st.update({"compiled": compiled, "passed": passed, "out": out, "det": det})
                row.update({"scenarios_count": st["scen"], "tb_compiled": int(compiled),
                            "golden_passed": int(passed),
                            "mutants_detected": det or 0,
                            "mutation_score": round(det / tot, 3) if det is not None else None})
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            save_fb(row)

    return (f"{module} × {llm} rep{rep}: base golden={int(passed)} -> " +
            ", ".join(f"{arm}@iter{iters} golden={int(state[arm]['passed'])}" for arm in arms))


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def print_summary(run_id=None):
    conn = _fb_conn()
    where, params = ("WHERE run_id = ?", [run_id]) if run_id else ("", [])
    rows = conn.execute(f"""
        SELECT arm, iteration, COUNT(*),
               AVG(tb_compiled), AVG(golden_passed),
               AVG(CASE WHEN golden_passed=1 THEN mutation_score END),
               SUM(error IS NOT NULL)
        FROM feedback_results {where}
        GROUP BY arm, iteration ORDER BY arm, iteration""", params).fetchall()
    conn.close()
    if not rows:
        print("No feedback results found.")
        return
    hdr = f"{'arm':<6}{'iter':>5}{'n':>4}{'Compile%':>10}{'GoldenPass%':>13}{'MutScore':>10}{'Errors':>8}"
    print(hdr)
    print("-" * len(hdr))
    for arm, it, n, c, g, m, e in rows:
        fmt = lambda v, pct=False: ("-" if v is None else f"{v*100:.0f}%" if pct else f"{v:.3f}")
        print(f"{arm:<6}{it:>5}{n:>4}{fmt(c, True):>10}{fmt(g, True):>13}{fmt(m):>10}{e:>8}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--run-id", default=None)
        print_summary(ap.parse_args().run_id)
        return

    ap = argparse.ArgumentParser(description="Run simulation-feedback loop experiments")
    ap.add_argument("--llms", required=True, help="comma-separated provider names or 'mock'")
    ap.add_argument("--modules", default=None, help="comma-separated module names (default: all)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3, help="feedback iterations per arm")
    ap.add_argument("--arms", default="bdd,tb", help="which arms to run (bdd,tb)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    conn = _fb_conn()
    existing = conn.execute("SELECT COUNT(*) FROM feedback_results WHERE run_id = ?",
                            (args.run_id,)).fetchone()[0]
    conn.close()
    if existing and not args.append:
        sys.exit(f"run_id '{args.run_id}' already has {existing} feedback results. "
                 f"Pick a new --run-id or pass --append.")

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ("bdd", "tb")]
    if not arms:
        sys.exit("no valid arms (use --arms bdd,tb)")

    manifests = sorted(BENCH_ROOT.glob("*/*/manifest.json"))
    if args.modules:
        wanted = set(args.modules.split(","))
        manifests = [m for m in manifests if m.parent.name in wanted]
    if not manifests:
        sys.exit("No matching modules found.")

    llms = args.llms.split(",")
    providers = {name: rx.make_provider(name) for name in llms}

    tasks = []
    for mf in manifests:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        for llm in llms:
            for rep in range(1, args.reps + 1):
                tasks.append((mf.parent, manifest, llm, rep))

    total = len(tasks)
    workers = max(1, min(args.workers, 8))
    print(f"🔁 run_id={args.run_id}: {len(manifests)} modules × {len(llms)} LLMs × "
          f"{args.reps} reps, arms={arms}, iters={args.iters} "
          f"({total} tasks, {workers} workers)\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()
    done = [0]

    def one(mod_dir, manifest, llm, rep):
        msg = run_task(mod_dir, manifest, llm, providers[llm], rep,
                       args.run_id, args.iters, arms)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{total}] {msg}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, *t) for t in tasks]
        for f in as_completed(futures):
            f.result()

    print(f"\n{'=' * 60}\nSummary for run_id={args.run_id}:\n")
    print_summary(args.run_id)
    print(f"\nArtifacts: output/feedback_runs/{args.run_id}/")


if __name__ == "__main__":
    main()
