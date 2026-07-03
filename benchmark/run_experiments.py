#!/usr/bin/env python3
"""
Batch Experiment Runner — 批量实验执行器

遍历 benchmark 题库 × LLM × 重复次数，对每个组合执行完整流水线：

  spec.md (去掉功能点答案)
    → [LLM] 生成 BDD 场景 (Gherkin)
    → [LLM] 生成自检 Verilog testbench
    → iverilog 编译 + golden 仿真 (必须 TEST PASSED)
    → 对每个 bug 变体做突变测试 (检出 = 未打印 TEST PASSED)
    → (可选) LLM 裁判评估功能点覆盖率

所有 LLM 调用经 src/experiment_logger 自动入库（带 run_id / module_name 标签），
结构化结果写入同一 SQLite 的 benchmark_results 表，
生成的 artifacts 保存在 output/benchmark_runs/<run_id>/ 下。

用法:
    # 冒烟测试（不调用任何 API，用内置 mock 验证整条流水线）
    python benchmark/run_experiments.py --llms mock --reps 1 --run-id smoke

    # 真实实验：2 个 LLM × 全部 9 模块 × 3 次重复
    python benchmark/run_experiments.py --llms groq,gemini --reps 3 --run-id exp001

    # 只跑部分模块 + 用 groq 当覆盖率裁判
    python benchmark/run_experiments.py --llms groq --modules alu_8bit,sync_fifo_8x8 \\
        --judge groq --run-id exp002

    # 查看结果汇总
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

SIM_TIMEOUT = 30       # 秒，防止 LLM 生成的 testbench 死循环
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
# 结果表
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
# 工具函数
# ---------------------------------------------------------------------------

def load_spec(mod_dir: Path):
    """返回 (去掉功能点的 spec, FP 列表)。FP 是评分标准答案，不能泄漏给被测 LLM。"""
    text = (mod_dir / "spec.md").read_text()
    m = re.split(r"^## Functional Points.*$", text, flags=re.M)
    spec_body = m[0].strip()
    fps = re.findall(r"^- (FP-\d+): (.+)$", text, flags=re.M)
    return spec_body, fps


def extract_ports(golden: Path) -> str:
    """提取 module 声明（到端口列表结束的 ');'），不泄漏实现。"""
    text = golden.read_text()
    m = re.search(r"module\s+\w+.*?\);", text, flags=re.S)
    return m.group(0) if m else text.split("\n")[0]


def extract_verilog(response: str) -> str:
    m = re.search(r"```(?:verilog|systemverilog|v)?\s*\n(.*?)```", response, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(module\s.*?endmodule)", response, flags=re.S)
    return m.group(1).strip() if m else response.strip()


def compile_and_run(dut: Path, tb: Path, workdir: Path):
    """返回 (compiled, passed, output)。passed 定义为输出含 TEST PASSED 且不含 TEST FAILED。"""
    exe = workdir / "sim.vvp"
    try:
        r = subprocess.run(["iverilog", "-g2005", "-o", str(exe), str(dut), str(tb)],
                           capture_output=True, text=True, timeout=SIM_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, False, "compile timeout"
    if r.returncode != 0:
        return False, False, r.stderr
    try:
        r = subprocess.run(["vvp", str(exe)], capture_output=True, text=True,
                           timeout=SIM_TIMEOUT)
    except subprocess.TimeoutExpired:
        return True, False, "simulation timeout"
    out = r.stdout + r.stderr
    passed = ("TEST PASSED" in out) and ("TEST FAILED" not in out)
    return True, passed, out


def make_provider(name: str):
    if name == "mock":
        return None
    from src.llm_providers import LLMFactory
    return LLMFactory.create_provider(name)


def llm_call(provider, name, prompt, system):
    if name == "mock":
        return MOCK_BDD if "Gherkin" in prompt else MOCK_TB
    return provider._call_api(prompt, max_tokens=LLM_MAX_TOKENS, system_prompt=system)


# ---------------------------------------------------------------------------
# 单个实验单元: (module, llm, rep)
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
        # 1) BDD 场景生成
        with call_context(task_type="bench_bdd", run_id=run_id, module_name=module):
            t0 = time.time()
            bdd = llm_call(provider, llm, BDD_PROMPT.format(spec=spec_body), BDD_SYSTEM)
            row["bdd_ms"] = int((time.time() - t0) * 1000)
        (art_dir / "scenarios.feature").write_text(bdd)
        row["scenarios_count"] = len(re.findall(r"^\s*Scenario", bdd, flags=re.M))

        # 2) Testbench 生成
        with call_context(task_type="bench_tb", run_id=run_id, module_name=module):
            t0 = time.time()
            tb_resp = llm_call(provider, llm, TB_PROMPT.format(
                spec=spec_body, ports=extract_ports(golden), bdd=bdd), TB_SYSTEM)
            row["tb_ms"] = int((time.time() - t0) * 1000)
        tb_file = art_dir / "tb.v"
        tb_file.write_text(extract_verilog(tb_resp))

        # 3) golden 仿真
        compiled, passed, out = compile_and_run(golden, tb_file, art_dir)
        (art_dir / "golden_sim.log").write_text(out)
        row["tb_compiled"] = int(compiled)
        row["golden_passed"] = int(passed)

        # 4) 突变测试（仅当 golden 通过才有意义）
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
            (art_dir / "mutation.log").write_text("\n".join(mut_log))

        # 5) 覆盖率裁判（可选）
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
# 汇总
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
                                    f"{v*100:.0f}%" if pct else f"{v:.2f}")
        print(f"{r[0]:<12}{r[1]:>5}{fmt(r[2], True):>10}{fmt(r[3], True):>13}"
              f"{fmt(r[4]):>10}{fmt(r[5]):>10}{fmt(r[6]):>9}")


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

    ap = argparse.ArgumentParser(description="Run BDD-Hardware benchmark experiments")
    ap.add_argument("--llms", required=True,
                    help="comma-separated: groq,gemini,openai,claude,deepseek,grok,qwen,mistral,together or 'mock'")
    ap.add_argument("--modules", default=None, help="comma-separated module names (default: all)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per (module, llm)")
    ap.add_argument("--run-id", required=True, help="experiment batch id, e.g. exp001")
    ap.add_argument("--judge", default=None, help="LLM used to judge FP coverage (optional)")
    args = ap.parse_args()

    manifests = sorted(BENCH_ROOT.glob("*/*/manifest.json"))
    if args.modules:
        wanted = set(args.modules.split(","))
        manifests = [m for m in manifests if m.parent.name in wanted]
    if not manifests:
        sys.exit("No matching modules found.")

    llms = args.llms.split(",")
    providers = {name: make_provider(name) for name in llms}
    judge = make_provider(args.judge) if args.judge and args.judge != "mock" else None

    total = len(manifests) * len(llms) * args.reps
    done = 0
    print(f"🚀 run_id={args.run_id}: {len(manifests)} modules × {len(llms)} LLMs × "
          f"{args.reps} reps = {total} experiments\n")

    for mf in manifests:
        manifest = json.loads(mf.read_text())
        for llm in llms:
            for rep in range(1, args.reps + 1):
                done += 1
                tag = f"[{done}/{total}] {manifest['name']} × {llm} rep{rep}"
                print(f"{tag} ...", flush=True)
                row = run_one(mf.parent, manifest, llm, providers[llm], rep,
                              args.run_id, judge=judge, judge_name=args.judge)
                if row["error"]:
                    print(f"  ❌ {row['error']}")
                else:
                    ms = row["mutation_score"]
                    print(f"  scenarios={row['scenarios_count']} "
                          f"compiled={bool(row['tb_compiled'])} "
                          f"golden={bool(row['golden_passed'])} "
                          f"mutation={ms if ms is not None else '-'} "
                          f"completeness={row['completeness'] if row['completeness'] is not None else '-'}")

    print(f"\n{'='*60}\nSummary for run_id={args.run_id}:\n")
    print_summary(args.run_id)
    print(f"\nArtifacts: output/benchmark_runs/{args.run_id}/")
    print(f"Raw LLM calls: python -m src.experiment_logger recent")


if __name__ == "__main__":
    main()
