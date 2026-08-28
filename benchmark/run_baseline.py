#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline data collection — 阶段一完成后的干净基线

历史记录（llm_calls id <= 623）横跨多个已修复的缺陷期：Groq 流式不可达、
转发调用重复记账、OpenAI Step 2 完全未被记录、OpenAI 收到额外的 system
prompt。那批数据不能用于论文，因此需要用当前代码重新采集一份。

矩阵：provider × module_type × seed，每格跑完整 pipeline
    Step 1 DUV 生成 -> Step 2 BDD 生成 -> Step 3 Testbench -> Step 4 仿真

Step 3 是确定性模板编译器、Step 4 是 iverilog 仿真，都不调用 LLM，
所以采样参数只作用于 Step 1/2。

调用方式走 Flask test client 而不是直接调 generator：这样跑的是网页用户
实际经过的同一条代码路径，run_id / sampling / prompt 记录也一并生效。

用法:
    python benchmark/run_baseline.py --batch base01
    python benchmark/run_baseline.py --batch base01 --providers groq,deepseek
    python benchmark/run_baseline.py --batch base01 --resume      # 断点续跑
    python benchmark/run_baseline.py --batch base01 --export-only # 只导 CSV
"""

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import main as webapp                                    # noqa: E402
from src.experiment_logger import DB_PATH                # noqa: E402

# ---------------------------------------------------------------------------
# 矩阵定义
# ---------------------------------------------------------------------------
ALL_PROVIDERS = [
    'gemini', 'mistral', 'deepseek', 'openai', 'qwen',
    'gptoss', 'glm', 'together',
]
MODULE_TYPES = ['alu', 'counter']
SEEDS = [1, 2, 3, 4, 5]
BITWIDTH = 32
STEP1_TEMP = 0.1
STEP2_TEMP = 0.7

# Step 2 的需求文本，与网页 BDD_TEMPLATES 保持一致，避免基线与实际使用脱节
BDD_INPUT = {
    'alu': lambda bw: f"""{bw}-bit ALU with 4-bit opcode selecting:
- ADD  (opcode 0000): A + B
- SUB  (opcode 0001): A - B
- AND  (opcode 0010): A & B
- OR   (opcode 0011): A | B
- XOR  (opcode 0100): A ^ B
- SLL  (opcode 0101): A << B, shift amount is the low {bw.bit_length() - 1} bits of B
- SRL  (opcode 0110): A >> B, zero-filled
- SRA  (opcode 0111): A >>> B, sign-extended
- SLT  (opcode 1000): 1 if A < B as signed values, else 0
- SLTU (opcode 1001): 1 if A < B as unsigned values, else 0
- Zero flag: high when result is 0, for every operation
- Overflow flag: signed overflow, meaningful for ADD and SUB
- Boundary value tests (0, 1, max, min, all-ones)
- Contrast SRL against SRA on a negative operand
- Contrast SLT against SLTU on the same operand pair""",
    'counter': lambda bw: f"""{bw}-bit Counter with:
- UP mode (increment)
- DOWN mode (decrement)
- UP-DOWN mode (ping-pong)
- Load preset value
- Enable control
- Overflow flag
- Zero flag""",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 15      # 秒；第 n 次失败后等 n * RETRY_BACKOFF

# provider 在 API 失败时不抛异常，而是返回这些固定的兜底文本，端点因此会报
# success=True。基线必须识别它们，否则会把失败记成成功——这是整份数据集
# 可信度的前提。benchmark/run_experiments.py 里有同样的判定。
FALLBACK_MARKERS = (
    "Test ALU operation with various input values",
    "Given ALU operation, When executed",
)


def is_fallback(text) -> bool:
    if not text:
        return True
    t = str(text)
    if any(m in t for m in FALLBACK_MARKERS):
        return True
    # _fallback_intent_json：一小段 {"scenario":..., "operation":...}
    return '"operation"' in t and '"scenario"' in t and len(t) < 800


# ---------------------------------------------------------------------------
# 结果表
# ---------------------------------------------------------------------------
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.execute("""CREATE TABLE IF NOT EXISTS baseline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, batch TEXT, cell_key TEXT UNIQUE, run_id TEXT,
        provider TEXT, model_effective TEXT, module_type TEXT, seed INTEGER,
        step1_temp REAL, step2_temp REAL,
        duv_success INTEGER, duv_compile INTEGER, duv_attempts INTEGER,
        bdd_success INTEGER, bdd_parse_ok INTEGER, bdd_attempts INTEGER,
        tb_success INTEGER, tb_compile INTEGER,
        sim_success INTEGER,
        -- oracle 分解：同一份 BDD 生成两份 testbench，激励相同、期望值来源不同。
        -- bdd  臂期望值取自 BDD，失败可能是 DUV 错、也可能是 BDD 期望值错
        -- spec 臂期望值由生成器按规格重算，失败只可能是 DUV 错
        -- 于是 spec 通过而 bdd 失败 = BDD 的 oracle 有误，两者第一次可区分
        sim_success_spec INTEGER, sim_pass_rate REAL, sim_pass_rate_spec REAL,
        oracle_error INTEGER,
        total_tokens_in INTEGER, total_tokens_out INTEGER, total_latency_ms INTEGER,
        failure_stage TEXT, failure_type TEXT,
        duv_call_ids TEXT, bdd_call_ids TEXT,
        duv_path TEXT, bdd_path TEXT, tb_path TEXT,
        notes TEXT)""")

    # 迁移：表可能是旧版建的，CREATE TABLE IF NOT EXISTS 不会补列。
    # 旧批次的这些列留空即可——它们本来就没有跑过第二臂。
    have = {r[1] for r in conn.execute("PRAGMA table_info(baseline_runs)")}
    for col, typ in (('sim_success_spec', 'INTEGER'), ('sim_pass_rate', 'REAL'),
                     ('sim_pass_rate_spec', 'REAL'), ('oracle_error', 'INTEGER')):
        if col not in have:
            conn.execute(f"ALTER TABLE baseline_runs ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def save_run(row):
    conn = _conn()
    cols = ",".join(row)
    ph = ",".join("?" * len(row))
    conn.execute(f"INSERT OR REPLACE INTO baseline_runs ({cols}) VALUES ({ph})",
                 list(row.values()))
    conn.commit()
    conn.close()


def done_cells(batch):
    """已完成的矩阵格（按 cell_key，而非 run_id——run_id 由 Step 1 生成）。"""
    conn = _conn()
    rows = conn.execute("SELECT cell_key FROM baseline_runs WHERE batch = ?",
                        (batch,)).fetchall()
    conn.close()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def available_providers(wanted, probe=True):
    """挑出真正可用的 provider。

    只看能否构造是不够的：claude 的 key 无效、qwen 账户欠费时对象都能建起来，
    失败发生在 API 层。上一批因此在这两家上白跑了 60 次调用（10 runs × 3 重试
    × 2 家）。这里补一次极小的探针调用，把它们提前剔除。
    """
    sys.path.insert(0, str(PROJECT_ROOT / "benchmark"))
    import run_experiments as rx
    usable, skipped = [], {}
    for name in wanted:
        try:
            p = rx.make_provider(name)
        except Exception as e:
            skipped[name] = f"构造失败 {type(e).__name__}: {str(e)[:60]}"
            continue
        if not probe:
            usable.append(name)
            continue
        try:
            call = getattr(p, '_call_api_text', None) or p._call_api
            resp = call("Reply with exactly: OK", max_tokens=200) or ""
            if is_fallback(resp):
                skipped[name] = "探针调用失败（API 返回兜底文本；key 无效或账户异常）"
            else:
                usable.append(name)
        except Exception as e:
            skipped[name] = f"探针调用异常 {type(e).__name__}: {str(e)[:60]}"
    return usable, skipped


def classify_failure(stage, resp, err_text=""):
    """把失败归到 api_error / parse_error / compile_error / sim_fail / timeout。"""
    text = (err_text or "") + " " + json.dumps(resp or {}, ensure_ascii=False)[:600]
    low = text.lower()
    if 'timeout' in low or 'timed out' in low:
        return 'timeout'
    if stage == 'sim':
        return 'sim_fail'
    if 'compile' in low or 'iverilog' in low or 'syntax error' in low:
        return 'compile_error'
    if ('did not return' in low or 'no verilog' in low or 'parse' in low
            or 'empty' in low or 'not found' in low):
        return 'parse_error'
    return 'api_error'


def calls_for(run_id, task_type):
    """取该 run 某阶段的 llm_calls 记录。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, prompt_chars, response_chars, latency_ms, success, extra
           FROM llm_calls WHERE run_id = ? AND task_type = ? ORDER BY id""",
        (run_id, task_type)).fetchall()
    conn.close()
    return rows


def iverilog_ok(path: Path):
    """能否单独编译通过（不含 testbench）。iverilog 缺失时返回 None。

    标准必须与 simulation_runner 一致（-g2012）。此前这里用 -g2005，比实际仿真
    严格：例如在无名 begin/end 里声明 integer 会被 -g2005 拒绝却能正常仿真，
    于是 duv_compile 把一批实际可用的设计记成了编译失败。
    """
    import subprocess, tempfile
    if not path or not Path(path).is_file():
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                ["iverilog", "-g2012", "-o", str(Path(tmp) / "a.out"), str(path)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60)
        return 1 if r.returncode == 0 else 0
    except FileNotFoundError:
        return None
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 单次 run
# ---------------------------------------------------------------------------
def run_one(client, batch, provider, module_type, seed, session_id):
    # Step 1 是依赖链起点，后端会无条件新建 run_id（DUV 变则整条链失效），
    # 所以这里不指定 run_id，而是采纳它返回的那个——与网页前端同样的做法。
    # 矩阵位置由 cell_key 标识，用于断点续跑。
    cell_key = f"{batch}_{provider}_{module_type}_s{seed}"
    run_id = None
    row = {
        'created_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'batch': batch, 'cell_key': cell_key, 'run_id': None, 'provider': provider,
        'model_effective': None, 'module_type': module_type, 'seed': seed,
        'step1_temp': STEP1_TEMP, 'step2_temp': STEP2_TEMP,
        'duv_success': 0, 'duv_compile': None, 'duv_attempts': 0,
        'bdd_success': 0, 'bdd_parse_ok': 0, 'bdd_attempts': 0,
        'tb_success': 0, 'tb_compile': None, 'sim_success': 0,
        'sim_success_spec': None, 'sim_pass_rate': None,
        'sim_pass_rate_spec': None, 'oracle_error': None,
        'total_tokens_in': 0, 'total_tokens_out': 0, 'total_latency_ms': 0,
        'failure_stage': None, 'failure_type': None,
        'duv_call_ids': None, 'bdd_call_ids': None,
        'duv_path': None, 'bdd_path': None, 'tb_path': None, 'notes': None,
    }
    # batch 会被后端写进 llm_calls.extra，便于事后精确筛出本批数据
    common = {'llm': provider, 'session_id': session_id, 'batch': batch}

    # ---- Step 1: DUV ----
    duv = None
    for attempt in range(1, MAX_RETRIES + 1):
        row['duv_attempts'] = attempt
        try:
            resp = client.post('/api/generate-hardware', json=dict(
                common, module_type=module_type, bitwidth=BITWIDTH,
                sampling={'temperature': STEP1_TEMP, 'seed': seed})).get_json()
        except Exception as e:
            resp = {'success': False, 'error': f"{type(e).__name__}: {e}"}
        # 端点报 success 也可能是兜底文本，必须看内容
        if resp and resp.get('success') and not is_fallback(resp.get('full_content')):
            duv = resp
            break
        if resp and resp.get('success'):
            resp = dict(resp, success=False,
                        error='provider returned fallback text (API call failed silently)')
        # 失败时后端仍新建了 run_id，取回来以免这次调用的记录追踪不到
        if resp and resp.get('run_id'):
            run_id = row['run_id'] = resp['run_id']
        if attempt < MAX_RETRIES:
            time.sleep(attempt * RETRY_BACKOFF)

    if not duv:
        row['failure_stage'] = 'duv'
        row['failure_type'] = classify_failure('duv', resp)
        row['notes'] = str((resp or {}).get('error'))[:300]
        _attach_metrics(row, run_id)
        return row

    run_id = row['run_id'] = duv.get('run_id')
    row['duv_success'] = 1
    row['duv_path'] = duv.get('filepath')
    row['model_effective'] = ((duv.get('call_meta') or {}).get('model_effective'))
    row['duv_compile'] = iverilog_ok(duv.get('filepath'))

    # ---- Step 2: BDD ----
    bdd = None
    for attempt in range(1, MAX_RETRIES + 1):
        row['bdd_attempts'] = attempt
        try:
            resp = client.post('/api/generate', json=dict(
                common, module_type=module_type, run_id=run_id,
                input=BDD_INPUT[module_type](BITWIDTH),
                sampling={'temperature': STEP2_TEMP, 'seed': seed})).get_json()
        except Exception as e:
            resp = {'success': False, 'error': f"{type(e).__name__}: {e}"}
        if resp and resp.get('success') and not is_fallback(resp.get('full_content')):
            bdd = resp
            break
        if resp and resp.get('success'):
            resp = dict(resp, success=False,
                        error='provider returned fallback text (API call failed silently)')
        if attempt < MAX_RETRIES:
            time.sleep(attempt * RETRY_BACKOFF)

    if not bdd:
        row['failure_stage'] = 'bdd'
        row['failure_type'] = classify_failure('bdd', resp)
        row['notes'] = str((resp or {}).get('error'))[:300]
        _attach_metrics(row, run_id)
        return row

    row['bdd_success'] = 1
    row['bdd_path'] = bdd.get('filepath')
    # 解析是否成功：生成的 feature 文件里至少要有一个 Scenario
    content = bdd.get('full_content') or ''
    row['bdd_parse_ok'] = 1 if 'Scenario' in content else 0

    # ---- Step 3: Testbench（确定性模板，不调 LLM）----
    # 不传 module_name：后端从 dut_filepath 读出真实模块名（单一事实来源）
    tb_req = {
        # 带上 run_id：Step 3/4 没有 LLM 调用，产物只能靠 run_artifacts 挂回本条链
        'run_id': run_id,
        'bdd_filepath': bdd.get('filepath'),
        'dut_filepath': duv.get('filepath'),
        'dut_info': {'module_type': module_type,
                     'bitwidth': BITWIDTH, 'depth': 32, 'pipeline_stages': 5},
    }
    tb = client.post('/api/generate-testbench',
                     json={**tb_req, 'oracle_source': 'bdd'}).get_json()
    if not (tb and tb.get('success')):
        row['failure_stage'] = 'tb'
        row['failure_type'] = classify_failure('tb', tb)
        row['notes'] = str((tb or {}).get('error'))[:300]
        _attach_metrics(row, run_id)
        return row
    row['tb_success'] = 1
    row['tb_path'] = tb.get('filepath')

    # ---- Step 4: 仿真 ----
    def rel(p):
        try:
            return str(Path(p).resolve().relative_to(PROJECT_ROOT))
        except Exception:
            return p

    def simulate(tb_path):
        return client.post('/api/run-simulation', json={
            'run_id': run_id,
            'testbench_path': rel(tb_path),
            'dut_path': rel(duv.get('filepath')),
        }).get_json() or {}

    sim = simulate(tb.get('filepath'))
    row['sim_pass_rate'] = sim.get('pass_rate')
    if sim.get('success'):
        row['sim_success'] = 1
        row['tb_compile'] = 1
    else:
        row['tb_compile'] = 0 if 'compil' in json.dumps(sim).lower() else 1
        row['failure_stage'] = 'sim'
        row['failure_type'] = classify_failure('sim', sim)
        row['notes'] = str(sim.get('error') or sim.get('output'))[:300]

    # 第二臂：同一份 BDD、同样的激励，但期望值由生成器按规格重算。
    # 这条臂的 oracle 由构造保证正确，所以它的失败只可能来自 DUV。
    # 对照之下，「spec 通过而 bdd 失败」把 BDD 期望值写错这一类错误单独分离
    # 出来——此前这两种失败在 sim_success 这一个 0/1 里无法区分。
    tb_spec = client.post('/api/generate-testbench',
                          json={**tb_req, 'oracle_source': 'spec'}).get_json() or {}
    if tb_spec.get('success'):
        sim_spec = simulate(tb_spec.get('filepath'))
        row['sim_success_spec'] = 1 if sim_spec.get('success') else 0
        row['sim_pass_rate_spec'] = sim_spec.get('pass_rate')
        # 判定必须用通过率而不是 success：simulation_runner 的 success 只表示
        # vvp 退出码为 0，即「仿真跑起来了」。生成的 testbench 在断言失败时只
        # $display 不 $fatal，所以哪怕一半测试点失败 success 仍然是 True。
        #
        # spec 臂的 oracle 由构造保证正确，它的失败只可能来自 DUV；两臂激励
        # 完全相同，所以 spec 通过得更多的那部分，就是 BDD 期望值写错的测试点。
        a, b = row.get('sim_pass_rate'), row.get('sim_pass_rate_spec')
        if a is not None and b is not None:
            row['oracle_error'] = int(b > a)

    _attach_metrics(row, run_id)
    return row


def _attach_metrics(row, run_id):
    """从 llm_calls 汇总本次 run 的用量与调用 id。"""
    if not run_id:
        return
    duv_calls = calls_for(run_id, 'web_duv_generation')
    bdd_calls = calls_for(run_id, 'web_bdd_generation')
    row['duv_call_ids'] = ",".join(str(r['id']) for r in duv_calls) or None
    row['bdd_call_ids'] = ",".join(str(r['id']) for r in bdd_calls) or None
    all_calls = list(duv_calls) + list(bdd_calls)
    # 本项目未记录 token 数，用字符数近似（列名沿用 tokens 便于后续替换）
    row['total_tokens_in'] = sum(r['prompt_chars'] or 0 for r in all_calls)
    row['total_tokens_out'] = sum(r['response_chars'] or 0 for r in all_calls)
    row['total_latency_ms'] = sum(r['latency_ms'] or 0 for r in all_calls)
    if row['model_effective'] is None and all_calls:
        try:
            row['model_effective'] = json.loads(all_calls[0]['extra'] or '{}').get('model_effective')
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
CSV_COLUMNS = ['run_id', 'provider', 'model_effective', 'module_type', 'seed',
               'step1_temp', 'step2_temp',
               'duv_success', 'duv_compile', 'duv_attempts',
               'bdd_success', 'bdd_parse_ok', 'bdd_attempts',
               'tb_success', 'tb_compile', 'sim_success',
               'sim_success_spec', 'sim_pass_rate', 'sim_pass_rate_spec',
               'oracle_error',
               'total_tokens_in', 'total_tokens_out', 'total_latency_ms',
               'failure_stage', 'failure_type']


def export_csv(batch, path=None):
    conn = _conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM baseline_runs WHERE batch = ? ORDER BY id",
                        (batch,)).fetchall()
    conn.close()
    out = Path(path or (PROJECT_ROOT / "output" / f"baseline_{batch}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: dict(r).get(k) for k in CSV_COLUMNS})
    return out, len(rows)


def report(batch, start_id, elapsed):
    conn = _conn()
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM baseline_runs WHERE batch = ?", (batch,)).fetchall()]
    conn.close()
    if not rows:
        print("没有数据")
        return

    print(f"\n{'=' * 76}")
    print(f"批次 {batch}  ·  {len(rows)} runs  ·  耗时 {elapsed/60:.1f} min")
    print(f"llm_calls 起点 id = {start_id}（本批数据为 id > {start_id}）")
    print('=' * 76)

    hdr = f"{'provider':10s}{'runs':>6s}{'DUV':>7s}{'BDD':>7s}{'TB':>7s}{'SIM':>7s}{'in chars':>12s}{'out chars':>12s}"
    print(hdr)
    print('-' * len(hdr))
    provs = sorted({r['provider'] for r in rows})
    for p in provs:
        rs = [r for r in rows if r['provider'] == p]
        n = len(rs)
        pct = lambda k: f"{sum(r[k] for r in rs) / n * 100:.0f}%"
        print(f"{p:10s}{n:>6d}{pct('duv_success'):>7s}{pct('bdd_success'):>7s}"
              f"{pct('tb_success'):>7s}{pct('sim_success'):>7s}"
              f"{sum(r['total_tokens_in'] for r in rs):>12,d}"
              f"{sum(r['total_tokens_out'] for r in rs):>12,d}")
    n = len(rows)
    pct = lambda k: f"{sum(r[k] for r in rows) / n * 100:.0f}%"
    print('-' * len(hdr))
    print(f"{'TOTAL':10s}{n:>6d}{pct('duv_success'):>7s}{pct('bdd_success'):>7s}"
          f"{pct('tb_success'):>7s}{pct('sim_success'):>7s}"
          f"{sum(r['total_tokens_in'] for r in rows):>12,d}"
          f"{sum(r['total_tokens_out'] for r in rows):>12,d}")

    # oracle 分解：把「BDD 期望值写错」从「DUV 实现错」里分离出来
    scored = [r for r in rows
              if r.get('sim_pass_rate') is not None
              and r.get('sim_pass_rate_spec') is not None]
    if scored:
        print(f"\noracle 分解（{len(scored)} 个 run 两臂都跑到了 Step 4）：")
        print(f"  {'provider':10s}{'spec 臂':>10s}{'bdd 臂':>10s}{'差值':>8s}"
              f"{'oracle 有误':>13s}")
        for p in sorted({r['provider'] for r in scored}):
            rs = [r for r in scored if r['provider'] == p]
            m = len(rs)
            spec = sum(r['sim_pass_rate_spec'] for r in rs) / m
            bdd = sum(r['sim_pass_rate'] for r in rs) / m
            orc = sum(r['oracle_error'] or 0 for r in rs)
            print(f"  {p:10s}{spec:>9.1f}%{bdd:>9.1f}%{spec - bdd:>7.1f}%"
                  f"{orc:>9d}/{m}")
        print("  spec 臂 = 期望值按规格重算时的测试通过率（失败只可能是 DUV 错）")
        print("  bdd 臂  = 期望值取自 BDD 时的测试通过率")
        print("  差值    = LLM 挑对了输入、却算错了期望值的那部分")

    fails = [r for r in rows if r['failure_stage']]
    if fails:
        print(f"\n失败分布（{len(fails)} / {n}）：")
        combo = {}
        for r in fails:
            k = (r['failure_stage'], r['failure_type'])
            combo[k] = combo.get(k, 0) + 1
        for (stage, typ), c in sorted(combo.items(), key=lambda kv: -kv[1]):
            print(f"  {stage:6s} {typ:14s} {c}")

    retried = [r for r in rows if (r['duv_attempts'] or 0) > 1 or (r['bdd_attempts'] or 0) > 1]
    print(f"\n发生过重试的 run: {len(retried)} / {n}")
    print(f"总延迟: {sum(r['total_latency_ms'] for r in rows) / 1000 / 60:.1f} min（LLM 调用累计）")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Collect a clean baseline dataset")
    ap.add_argument('--batch', required=True, help="批次标识，如 base01")
    ap.add_argument('--providers', default=None, help="逗号分隔，默认全部")
    ap.add_argument('--modules', default=None, help="逗号分隔，默认 alu,counter")
    ap.add_argument('--seeds', default=None, help="逗号分隔，默认 1,2,3,4,5")
    ap.add_argument('--resume', action='store_true', help="跳过已完成的 run")
    ap.add_argument('--export-only', action='store_true', help="只导出 CSV")
    args = ap.parse_args()

    if args.export_only:
        out, n = export_csv(args.batch)
        print(f"导出 {n} 行 -> {out}")
        report(args.batch, 0, 0)
        return

    providers = (args.providers.split(',') if args.providers else ALL_PROVIDERS)
    modules = (args.modules.split(',') if args.modules else MODULE_TYPES)
    seeds = ([int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS)

    usable, skipped = available_providers(providers)
    if skipped:
        print("跳过（无可用 API key 或构造失败）:")
        for k, v in skipped.items():
            print(f"  {k:10s} {v}")
    if not usable:
        sys.exit("没有可用的 provider")

    conn = _conn()
    start_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM llm_calls").fetchone()[0]
    conn.close()

    done = done_cells(args.batch) if args.resume else set()
    tasks = [(p, m, s) for p in usable for m in modules for s in seeds]
    todo = [t for t in tasks
            if f"{args.batch}_{t[0]}_{t[1]}_s{t[2]}" not in done]

    print(f"\n批次 {args.batch}: {len(usable)} providers × {len(modules)} modules "
          f"× {len(seeds)} seeds = {len(tasks)} runs"
          + (f"（跳过已完成 {len(tasks) - len(todo)}）" if args.resume else ""))
    print(f"llm_calls 起点 id = {start_id}\n")

    client = webapp.app.test_client()
    session_id = f"baseline-{args.batch}"
    t0 = time.time()
    for i, (p, m, s) in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {p} {m} seed={s}"
        print(f"{tag} ...", flush=True)
        try:
            row = run_one(client, args.batch, p, m, s, session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            row = {'created_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   'batch': args.batch,
                   'cell_key': f"{args.batch}_{p}_{m}_s{s}", 'run_id': None,
                   'provider': p, 'module_type': m, 'seed': s,
                   'failure_stage': 'harness', 'failure_type': 'api_error',
                   'notes': f"{type(e).__name__}: {e}"[:300]}
        save_run(row)
        print(f"  -> duv={row.get('duv_success')} bdd={row.get('bdd_success')} "
              f"tb={row.get('tb_success')} sim={row.get('sim_success')}"
              + (f"  FAIL@{row['failure_stage']}/{row['failure_type']}"
                 if row.get('failure_stage') else ""), flush=True)

    elapsed = time.time() - t0
    out, n = export_csv(args.batch)
    report(args.batch, start_id, elapsed)
    print(f"\nCSV: {out}（{n} 行）")


if __name__ == '__main__':
    main()
