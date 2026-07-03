#!/usr/bin/env python3
"""
Benchmark validation — 题库质量守门员

对 benchmark/ 下的每个模块执行三重检查：
1. golden.v + tb_smoke.v 能用 iverilog 编译并仿真成功
2. 每个 bug 变体能编译成功（bug 必须是功能性突变，不是语法错误）
3. 每个 sim_detectable 的 bug 变体，在相同激励下输出必须与 golden 不同
   （差分测试：输出相同的突变体是无效 bug，会被报告为 SURVIVED）

用法:
    python benchmark/validate.py            # 验证全部模块
    python benchmark/validate.py alu_8bit   # 只验证指定模块
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_ROOT = Path(__file__).parent


def run(cmd, cwd=None, timeout=60):
    # encoding/errors 显式指定：Windows 默认 GBK，会在 UTF-8 输出上抛 UnicodeDecodeError
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd, timeout=timeout)


def compile_and_sim(design: Path, tb: Path, workdir: Path, tag: str):
    """编译 design+tb 并运行仿真，返回 (ok, output_or_error)"""
    exe = workdir / f"{tag}.vvp"
    r = run(["iverilog", "-g2005", "-o", str(exe), str(design), str(tb)])
    if r.returncode != 0:
        return False, f"COMPILE FAILED:\n{r.stderr}"
    r = run(["vvp", str(exe)])
    if r.returncode != 0:
        return False, f"SIM FAILED:\n{r.stderr}"
    return True, r.stdout


def validate_module(mod_dir: Path) -> dict:
    manifest = json.loads((mod_dir / "manifest.json").read_text(encoding="utf-8"))
    golden = mod_dir / "golden.v"
    tb = mod_dir / "tb_smoke.v"
    result = {"name": manifest["name"], "golden_ok": False, "bugs": []}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ok, golden_out = compile_and_sim(golden, tb, tmp, "golden")
        result["golden_ok"] = ok
        if not ok:
            result["golden_error"] = golden_out
            return result

        for bug in manifest.get("bugs", []):
            bug_file = mod_dir / "bugs" / bug["file"]
            entry = {"file": bug["file"], "compiled": False, "killed": False}
            ok, bug_out = compile_and_sim(bug_file, tb, tmp, bug["file"])
            entry["compiled"] = ok
            if ok:
                entry["killed"] = (bug_out != golden_out)
            else:
                entry["error"] = bug_out
            entry["expected_detectable"] = bug.get("sim_detectable", True)
            entry["ok"] = entry["compiled"] and (entry["killed"] == entry["expected_detectable"])
            result["bugs"].append(entry)

    return result


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    modules = sorted(BENCH_ROOT.glob("*/*/manifest.json"))
    if only:
        modules = [m for m in modules if m.parent.name == only]
    if not modules:
        print("No modules found.")
        sys.exit(1)

    all_ok = True
    total_bugs = killed = 0
    for mf in modules:
        res = validate_module(mf.parent)
        status = "✅" if res["golden_ok"] else "❌"
        print(f"\n{status} {res['name']} (golden {'OK' if res['golden_ok'] else 'FAILED'})")
        if not res["golden_ok"]:
            print(res.get("golden_error", ""))
            all_ok = False
            continue
        for b in res["bugs"]:
            total_bugs += 1
            if b["killed"]:
                killed += 1
            mark = "✅" if b["ok"] else "❌"
            state = ("killed" if b["killed"] else "SURVIVED") if b["compiled"] else "COMPILE-FAIL"
            print(f"   {mark} {b['file']}: {state}")
            if not b["ok"]:
                all_ok = False
                if "error" in b:
                    print(f"      {b['error'][:500]}")

    print(f"\n{'='*50}")
    print(f"Modules: {len(modules)}  Mutants: {total_bugs}  Killed by smoke stimulus: {killed}")
    print("RESULT:", "✅ ALL VALID" if all_ok else "❌ ISSUES FOUND")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
