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
import hashlib
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
# alu 与 alu_comb 是同一个设计任务的两种时序形态：alu 用 duv_alu/v1（寄存
# 输出），alu_comb 用 v2（纯组合）。除时序外两份 prompt 逐字相同，构成
# RQ2 的受控对照。BDD 侧两者完全一样。
MODULE_TYPES = ['alu', 'alu_comb', 'counter']

# module_type -> (后端识别的类型, DUV prompt 版本)
MODULE_SPEC = {
    'alu':      ('alu', 'v1'),
    'alu_comb': ('alu', 'v2'),
    'counter':  ('counter', 'v1'),
    'regfile':  ('regfile', 'v1'),
    'cpu':      ('cpu', 'v1'),
}
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
    # alu_comb 与 alu 共用同一段需求文本：BDD 描述的是行为，与设计是组合
    # 还是时序无关。共用同一个 lambda 而不是复制，避免两者日后走样。
    'counter': lambda bw: f"""{bw}-bit Counter with:
- UP mode (increment)
- DOWN mode (decrement)
- UP-DOWN mode (ping-pong)
- Load preset value
- Enable control
- Overflow flag
- Zero flag""",
    # regfile 与 alu/counter 不同：行是**有序有状态**的，写下去的值要被后面
    # 的读看见。所以需求文本必须说清 tag 词表和列名，否则解析器无从把一行
    # 认成读还是写——这和 ALU 必须说清有哪些 opcode 是同一个道理。
    'regfile': lambda bw: f"""32-entry x {bw}-bit Register File with:
- Two combinational read ports (raddr1/rdata1, raddr2/rdata2)
- One synchronous write port (wen, waddr, wdata), writes on the clock edge
- Register 0 is hardwired to zero: it always reads 0 and writes to it are ignored
- Active-low reset rst_n clears every register

Write the scenarios so that they form an ordered sequence: rows execute in
the order they appear and the register contents persist from one row to the
next. Tag each Scenario Outline with exactly one of:
- @write       the row writes data to addr
- @read        the row reads addr and checks the value
- @write_read  the row writes and then reads back the same addr
- @reset       the row asserts reset

Use these Examples columns: addr, data (for writes), expected (for reads).
Cover at least:
- A register written early and read back much later, after other writes
- Two different registers written in turn, then both read, to show one write
  does not disturb the other
- Overwriting a register and reading the new value
- Reading register 0, and attempting to write register 0 then reading it
- A reset followed by a read""",
}

MAX_RETRIES = 3
RETRY_BACKOFF = 15      # 秒；第 n 次失败后等 n * RETRY_BACKOFF

# 429 要等得久得多。main01 里 gemini 被限流 45 次，而 15/30/45s 的线性退避
# 完全不够——三次重试跑完仍在同一个配额窗口里。
RATE_LIMIT_BACKOFF = (60, 120, 240)


def _is_rate_limited(resp) -> bool:
    blob = json.dumps(resp or {}, ensure_ascii=False)[:1500].lower()
    return ('429' in blob or 'too many requests' in blob
            or 'rate limit' in blob or 'resource_exhausted' in blob)

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
        -- sim_success 只表示 vvp 退出码为 0（仿真跑起来了），断言失败时它
        -- 仍是 1。main01 里 104 格 sim_success=1 中只有 55 格断言全过，
        -- 差 49 格。要论文里报「通过率」必须用 sim_all_pass。
        sim_all_pass INTEGER, sim_all_pass_spec INTEGER,
        oracle_error INTEGER,
        total_tokens_in INTEGER, total_tokens_out INTEGER, total_latency_ms INTEGER,
        failure_stage TEXT, failure_type TEXT,
        duv_call_ids TEXT, bdd_call_ids TEXT,
        duv_path TEXT, bdd_path TEXT, tb_path TEXT,
        notes TEXT,
        -- RQ1 的自变量：这次 DUV 生成的 prompt 里到底有没有 BDD 块。
        -- 不从 workflow_mode 推断——两者理应一致，但一旦某处漏传
        -- bdd_filepath，推断出来的值就是错的，而实测值不会错。
        duv_prompt_has_bdd INTEGER,
        -- 请求的 model 与实际生效的 model 分开记：provider 可能忽略覆盖、
        -- 也可能把别名解析成别的名字，只记一个就分不出这两种情况。
        model_requested TEXT,
        -- 分阶段用量。合计值掩盖了「哪一步贵」，而 DUV 与 BDD 的 prompt
        -- 长度差一个数量级（spec-first 尤其）。
        duv_chars_in INTEGER, duv_chars_out INTEGER, duv_latency_ms INTEGER,
        bdd_chars_in INTEGER, bdd_chars_out INTEGER, bdd_latency_ms INTEGER,
        -- 产物指纹：用于事后确认某一行对应的确实是磁盘上那份文件
        duv_sha256 TEXT, bdd_sha256 TEXT, tb_sha256 TEXT,
        -- 分阶段的失败原文。notes 只留最后一次，跨阶段会互相覆盖。
        duv_error TEXT, bdd_error TEXT, tb_error TEXT, sim_error TEXT,
        -- impl-first 是 DUV -> BDD，spec-first 是 BDD -> DUV。两者的 run_id
        -- 起点、以及 DUV 是否看得到 BDD，都不同，所以必须随行记录。
        workflow_mode TEXT)""")

    # 迁移：表可能是旧版建的，CREATE TABLE IF NOT EXISTS 不会补列。
    # 旧批次的这些列留空即可——它们本来就没有跑过第二臂。
    # workflow_mode 同理：base01/base02 跑在支持它之前，全部是 implementation，
    # 但这里留空而不是回填，因为"当时没有这个概念"和"当时选了这个值"是两件事。
    have = {r[1] for r in conn.execute("PRAGMA table_info(baseline_runs)")}
    for col, typ in (('sim_success_spec', 'INTEGER'), ('sim_pass_rate', 'REAL'),
                     ('sim_pass_rate_spec', 'REAL'), ('oracle_error', 'INTEGER'),
                     ('workflow_mode', 'TEXT'), ('duv_prompt_has_bdd', 'INTEGER'),
                     ('prompt_version', 'TEXT'), ('model_requested', 'TEXT'),
                     ('duv_chars_in', 'INTEGER'), ('duv_chars_out', 'INTEGER'),
                     ('duv_latency_ms', 'INTEGER'), ('bdd_chars_in', 'INTEGER'),
                     ('bdd_chars_out', 'INTEGER'), ('bdd_latency_ms', 'INTEGER'),
                     ('duv_sha256', 'TEXT'), ('bdd_sha256', 'TEXT'),
                     ('tb_sha256', 'TEXT'), ('duv_error', 'TEXT'),
                     ('bdd_error', 'TEXT'), ('tb_error', 'TEXT'),
                     ('sim_error', 'TEXT'), ('sim_all_pass', 'INTEGER'),
                     ('sim_all_pass_spec', 'INTEGER')):
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
    """能否单独编译通过（不含 testbench）。返回 (0/1/None, 编译器输出)。

    iverilog 缺失时返回 (None, None)。

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
        # 同时返回编译器原文：Failure taxonomy 要按错误文本聚类，只留 0/1
        # 的话「wire 当 l-value」「端口位宽不符」「未声明标识符」全都塌成
        # 一个 duv_compile=0，事后无法区分。
        return (1 if r.returncode == 0 else 0), (r.stderr or '').strip()
    except FileNotFoundError:
        return None, None
    except Exception as e:
        return 0, f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------------------
# 单次 run
# ---------------------------------------------------------------------------
def _slug(provider, model):
    """cell_key 里代表一个 (provider, model) 组合的短标识。

    模型名里有 '/' 和 '.'，直接拼进 key 会难读也难在 shell 里用；
    取最后一段并把非字母数字压成 '-'，足以区分同一 provider 的不同模型。
    """
    if not model:
        return provider
    tail = model.rsplit('/', 1)[-1]
    tail = ''.join(ch if ch.isalnum() else '-' for ch in tail).strip('-')
    return f'{provider}~{tail}'


def _sha256(path):
    """产物文件的指纹。文件缺失时返回 None 而不是抛错。"""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _prompt_has_bdd(run_id):
    """这条 run 的 DUV prompt 里是否真的含有 BDD 块（实测，非推断）。

    workflow_mode 说的是「打算怎么跑」，这一列说的是「实际发出去的 prompt
    长什么样」。二者理应一致，但只要有一处漏传 bdd_filepath，前者就会骗人，
    而后者不会——RQ1 的自变量必须取后者。
    """
    if not run_id:
        return None
    conn = _conn()
    row = conn.execute(
        "SELECT prompt FROM llm_calls WHERE run_id=? AND task_type='web_duv_generation'"
        " ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return int('Scenario Outline:' in row[0] or 'BDD Specification Context' in row[0])


def _retrying_stage(client, url, payload, row, attempts_key):
    """调用一个生成阶段，带重试与兜底文本检测。

    两种 workflow 下 DUV 与 BDD 的先后不同，但每一步内部的重试、
    「success=True 但内容是兜底文本」的识别、失败时取回 run_id 的逻辑完全
    一样。抽出来共用，避免两个分支各写一遍而慢慢走样。

    返回 (成功的响应 或 None, 最后一次响应)。
    """
    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        row[attempts_key] = attempt
        try:
            resp = client.post(url, json=payload).get_json()
        except Exception as e:
            resp = {'success': False, 'error': f"{type(e).__name__}: {e}"}
        # 端点报 success 也可能是兜底文本，必须看内容
        if resp and resp.get('success') and not is_fallback(resp.get('full_content')):
            return resp, resp
        if resp and resp.get('success'):
            resp = dict(resp, success=False,
                        error='provider returned fallback text (API call failed silently)')
        # 失败时后端仍新建了 run_id，取回来以免这次调用的记录追踪不到
        if resp and resp.get('run_id') and not row.get('run_id'):
            row['run_id'] = resp['run_id']
        if attempt < MAX_RETRIES:
            if _is_rate_limited(resp):
                wait = RATE_LIMIT_BACKOFF[min(attempt - 1,
                                             len(RATE_LIMIT_BACKOFF) - 1)]
                print(f'    ⏳ 限流，等待 {wait}s 后重试 '
                      f'({attempt}/{MAX_RETRIES - 1})', flush=True)
            else:
                wait = attempt * RETRY_BACKOFF
            time.sleep(wait)
    return None, resp


def _duv_payload(common, module_type, seed, run_id, bdd_filepath):
    backend_type, prompt_version = MODULE_SPEC[module_type]
    p = dict(common, module_type=backend_type, bitwidth=BITWIDTH,
             prompt_version=prompt_version,
             sampling={'temperature': STEP1_TEMP, 'seed': seed})
    if run_id:
        p['run_id'] = run_id
    # spec-first：DUV 要看到先生成的 BDD，否则它跟规格没有任何关系
    if bdd_filepath:
        p['bdd_filepath'] = bdd_filepath
    return p


def _bdd_payload(common, module_type, seed, run_id):
    backend_type, _ = MODULE_SPEC[module_type]
    p = dict(common, module_type=backend_type,
             input=BDD_INPUT[backend_type](BITWIDTH),
             sampling={'temperature': STEP2_TEMP, 'seed': seed})
    if run_id:
        p['run_id'] = run_id
    return p


def run_one(client, batch, provider, module_type, seed, session_id,
            workflow='implementation', model=None):
    # 依赖链起点随 workflow 变：impl-first 是 DUV，spec-first 是 BDD。后端的
    # _chain_start 会据 workflow_mode 自行判断，所以这里只需按顺序调用，并把
    # 起点返回的 run_id 传给后一步。
    # 矩阵位置由 cell_key 标识，用于断点续跑；workflow 必须进 key，否则两种
    # 模式的同一格会互相当成「已完成」而被跳过。
    # model 必须进 cell_key：同一 provider 的两个模型是矩阵里的两格，
    # 不区分的话 --resume 会把第二个模型当成已完成而跳过。
    cell_key = f"{batch}_{workflow[:4]}_{_slug(provider, model)}_{module_type}_s{seed}"
    run_id = None
    row = {
        'created_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'batch': batch, 'cell_key': cell_key, 'run_id': None, 'provider': provider,
        'model_effective': None, 'model_requested': model,
        'module_type': module_type, 'seed': seed,
        'step1_temp': STEP1_TEMP, 'step2_temp': STEP2_TEMP,
        'duv_success': 0, 'duv_compile': None, 'duv_attempts': 0,
        'bdd_success': 0, 'bdd_parse_ok': 0, 'bdd_attempts': 0,
        'tb_success': 0, 'tb_compile': None, 'sim_success': 0,
        'sim_success_spec': None, 'sim_pass_rate': None,
        'sim_pass_rate_spec': None, 'oracle_error': None,
        'sim_all_pass': None, 'sim_all_pass_spec': None,
        'total_tokens_in': 0, 'total_tokens_out': 0, 'total_latency_ms': 0,
        'failure_stage': None, 'failure_type': None,
        'duv_call_ids': None, 'bdd_call_ids': None,
        'duv_path': None, 'bdd_path': None, 'tb_path': None, 'notes': None,
        'workflow_mode': workflow,
        'duv_prompt_has_bdd': None,
        'duv_chars_in': None, 'duv_chars_out': None, 'duv_latency_ms': None,
        'bdd_chars_in': None, 'bdd_chars_out': None, 'bdd_latency_ms': None,
        'duv_sha256': None, 'bdd_sha256': None, 'tb_sha256': None,
        'duv_error': None, 'bdd_error': None, 'tb_error': None,
        'sim_error': None,
        'prompt_version': MODULE_SPEC[module_type][1],
    }
    # batch 会被后端写进 llm_calls.extra，便于事后精确筛出本批数据
    common = {'llm': provider, 'session_id': session_id, 'batch': batch,
              'workflow_mode': workflow}
    if model:
        common['model'] = model

    def _fail(stage, resp):
        row['failure_stage'] = stage
        row['failure_type'] = classify_failure(stage, resp)
        err = str((resp or {}).get('error'))[:600]
        row['notes'] = err[:300]
        # 分阶段各存一份：notes 只有一格，跨阶段会互相覆盖
        if f'{stage}_error' in row:
            row[f'{stage}_error'] = err
        _attach_metrics(row, row.get('run_id'))
        return row

    def _record_duv(duv):
        row['duv_success'] = 1
        # 从落库的 prompt 实测，而不是照 workflow_mode 推断
        row['duv_prompt_has_bdd'] = _prompt_has_bdd(row.get('run_id'))
        row['duv_path'] = duv.get('filepath')
        row['model_effective'] = ((duv.get('call_meta') or {}).get('model_effective'))
        ok, cerr = iverilog_ok(duv.get('filepath'))
        row['duv_compile'] = ok
        if ok == 0 and cerr:
            row['duv_error'] = cerr[:2000]
        row['duv_sha256'] = _sha256(duv.get('filepath'))

    def _record_bdd(bdd):
        row['bdd_success'] = 1
        row['bdd_path'] = bdd.get('filepath')
        # 解析是否成功：生成的 feature 文件里至少要有一个 Scenario
        row['bdd_parse_ok'] = 1 if 'Scenario' in (bdd.get('full_content') or '') else 0
        row['bdd_sha256'] = _sha256(bdd.get('filepath'))

    if workflow == 'specification':
        # ---- spec-first: BDD 先行（依赖链起点），DUV 据它生成 ----
        bdd, resp = _retrying_stage(client, '/api/generate',
                                    _bdd_payload(common, module_type, seed, None),
                                    row, 'bdd_attempts')
        if not bdd:
            return _fail('bdd', resp)
        run_id = row['run_id'] = bdd.get('run_id')
        _record_bdd(bdd)

        duv, resp = _retrying_stage(
            client, '/api/generate-hardware',
            _duv_payload(common, module_type, seed, run_id, bdd.get('filepath')),
            row, 'duv_attempts')
        if not duv:
            return _fail('duv', resp)
        _record_duv(duv)
    else:
        # ---- impl-first: DUV 先行（依赖链起点），BDD 据它生成 ----
        duv, resp = _retrying_stage(
            client, '/api/generate-hardware',
            _duv_payload(common, module_type, seed, None, None),
            row, 'duv_attempts')
        if not duv:
            return _fail('duv', resp)
        run_id = row['run_id'] = duv.get('run_id')
        _record_duv(duv)

        bdd, resp = _retrying_stage(client, '/api/generate',
                                    _bdd_payload(common, module_type, seed, run_id),
                                    row, 'bdd_attempts')
        if not bdd:
            return _fail('bdd', resp)
        _record_bdd(bdd)

    # ---- Step 3: Testbench（确定性模板，不调 LLM）----
    # 不传 module_name：后端从 dut_filepath 读出真实模块名（单一事实来源）
    tb_req = {
        # 带上 run_id：Step 3/4 没有 LLM 调用，产物只能靠 run_artifacts 挂回本条链
        'run_id': run_id,
        'bdd_filepath': bdd.get('filepath'),
        'dut_filepath': duv.get('filepath'),
        # module_type 传 alu_comb 本身：TestbenchGenerator 认它，且能据此
        # 与 alu 区分开；testbench 要不要驱动时钟由 DUV 端口自动判断
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
    row['tb_sha256'] = _sha256(tb.get('filepath'))

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
    if sim.get('pass_rate') is not None:
        row['sim_all_pass'] = int(sim['pass_rate'] >= 100.0)
    if sim.get('success'):
        row['sim_success'] = 1
        row['tb_compile'] = 1
    else:
        row['tb_compile'] = 0 if 'compil' in json.dumps(sim).lower() else 1
        row['failure_stage'] = 'sim'
        row['failure_type'] = classify_failure('sim', sim)
        err = str(sim.get('error') or sim.get('output') or '')
        row['notes'] = err[:300]
        row['sim_error'] = err[:2000]

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
        if sim_spec.get('pass_rate') is not None:
            row['sim_all_pass_spec'] = int(sim_spec['pass_rate'] >= 100.0)
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
    for tag, calls in (('duv', duv_calls), ('bdd', bdd_calls)):
        row[f'{tag}_chars_in'] = sum(r['prompt_chars'] or 0 for r in calls) or None
        row[f'{tag}_chars_out'] = sum(r['response_chars'] or 0 for r in calls) or None
        row[f'{tag}_latency_ms'] = sum(r['latency_ms'] or 0 for r in calls) or None
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
CSV_COLUMNS = ['run_id', 'workflow_mode', 'prompt_version',
               'duv_prompt_has_bdd', 'provider', 'model_requested', 'model_effective', 'module_type', 'seed',
               'step1_temp', 'step2_temp',
               'duv_success', 'duv_compile', 'duv_attempts',
               'bdd_success', 'bdd_parse_ok', 'bdd_attempts',
               'tb_success', 'tb_compile', 'sim_success', 'sim_all_pass',
               'sim_success_spec', 'sim_pass_rate', 'sim_pass_rate_spec',
               'oracle_error',
               'total_tokens_in', 'total_tokens_out', 'total_latency_ms',
               'failure_stage', 'failure_type',
               'duv_chars_in', 'duv_chars_out', 'duv_latency_ms',
               'bdd_chars_in', 'bdd_chars_out', 'bdd_latency_ms',
               'duv_sha256', 'bdd_sha256', 'tb_sha256',
               'duv_error', 'bdd_error', 'tb_error', 'sim_error']


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
    ap.add_argument('--providers', default=None,
                    help="逗号分隔。可写 provider 或 provider:model，"
                         "例如 gemini:gemini-2.5-flash,openai:gpt-5.1")
    ap.add_argument('--modules', default=None, help="逗号分隔，默认 alu,counter")
    ap.add_argument('--seeds', default=None, help="逗号分隔，默认 1,2,3,4,5")
    ap.add_argument('--workflow', default='implementation',
                    choices=['implementation', 'specification'],
                    help="impl-first: DUV->BDD；spec-first: BDD->DUV")
    ap.add_argument('--resume', action='store_true', help="跳过已完成的 run")
    ap.add_argument('--export-only', action='store_true', help="只导出 CSV")
    args = ap.parse_args()

    if args.export_only:
        out, n = export_csv(args.batch)
        print(f"导出 {n} 行 -> {out}")
        report(args.batch, 0, 0)
        return

    # --providers 每项可写 "provider" 或 "provider:model"。同一 provider 的
    # 不同模型是矩阵里的不同格，所以用 (provider, model) 对而不是 provider 名
    # 来展开；探针仍按 provider 去重，避免同一家探测多次。
    raw = (args.providers.split(',') if args.providers else ALL_PROVIDERS)
    pairs = []
    for item in raw:
        item = item.strip()
        if not item:
            continue
        prov, _, mdl = item.partition(':')
        pairs.append((prov, mdl or None))
    modules = (args.modules.split(',') if args.modules else MODULE_TYPES)
    seeds = ([int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS)

    usable, skipped = available_providers([p for p, _ in pairs])
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
    # 探针按 provider 去重，所以用 usable 过滤 pairs；同一 provider 的多个
    # 模型各自成格（gemini 有两个模型，不能被压成一个）
    tasks = [(prov, mdl, m, s)
             for prov, mdl in pairs if prov in usable
             for m in modules for s in seeds]
    # cell_key 的构造必须与 run_one 内部完全一致，否则 --resume 会失效
    def _cell(p, mdl, m, s):
        return f"{args.batch}_{args.workflow[:4]}_{_slug(p, mdl)}_{m}_s{s}"
    todo = [t for t in tasks if _cell(*t) not in done]

    n_cfg = len([1 for prov, _ in pairs if prov in usable])
    print(f"\n批次 {args.batch} [{args.workflow}]: {n_cfg} model 配置 × {len(modules)} modules "
          f"× {len(seeds)} seeds = {len(tasks)} runs"
          + (f"（跳过已完成 {len(tasks) - len(todo)}）" if args.resume else ""))
    print(f"llm_calls 起点 id = {start_id}\n")

    client = webapp.app.test_client()
    session_id = f"baseline-{args.batch}"
    t0 = time.time()
    for i, (p, mdl, m, s) in enumerate(todo, 1):
        tag = f"[{i}/{len(todo)}] {_slug(p, mdl)} {m} seed={s}"
        print(f"{tag} ...", flush=True)
        try:
            row = run_one(client, args.batch, p, m, s, session_id,
                          workflow=args.workflow, model=mdl)
        except Exception as e:
            import traceback
            traceback.print_exc()
            row = {'created_at': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   'batch': args.batch,
                   'cell_key': _cell(p, mdl, m, s), 'run_id': None,
                   'provider': p, 'model_requested': mdl,
                   'module_type': m, 'seed': s,
                   'workflow_mode': args.workflow,
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
