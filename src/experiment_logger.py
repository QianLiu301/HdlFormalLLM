"""
Experiment Logger — 实验记录系统

自动记录每一次 LLM API 调用的完整信息到 SQLite 数据库，
为 Benchmark 实验和多 LLM 对比提供数据基础。

记录内容：
- provider / model / 调用方法（普通 or 流式）
- 完整 prompt / system_prompt / response
- 延迟（毫秒）、字符数、成功/失败、错误信息
- 实验上下文标签（task_type / run_id / module_name 等，由上层通过 call_context 设置）

使用方式：
1. 自动模式（无需改动 Provider 代码）：
   LLMProvider 基类的 __init_subclass__ 会调用 instrument_class()
   自动包装所有 _call_api* 方法。

2. 给调用打标签（在 Flask 路由或实验脚本中）：
   from src.experiment_logger import call_context
   with call_context(task_type='bdd_generation', run_id='exp001', module_name='alu_8bit'):
       ...  # 其中发生的所有 LLM 调用都会带上这些标签

3. 命令行查看：
   python -m src.experiment_logger stats     # 汇总统计
   python -m src.experiment_logger recent    # 最近 20 条
   python -m src.experiment_logger export out.jsonl  # 导出 JSONL
"""

import os
import json
import time
import sqlite3
import threading
import functools
import inspect
from contextlib import contextmanager
from typing import Optional, Dict, Any

# 数据库位置：output/experiments/experiments.db（可用环境变量覆盖）
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'experiments', 'experiments.db'
)
DB_PATH = os.environ.get('EXPERIMENT_DB_PATH', _DEFAULT_DB)

_lock = threading.Lock()
_local = threading.local()

# ---------------------------------------------------------------------------
# 单例自检
#
# 本仓库存在双重导入路径：main.py 把 src/ 注入 sys.path，因此模块既可能以
# `src.experiment_logger` 也可能以 `experiment_logger` 被加载。目前 llm_providers
# 优先尝试 `src.` 前缀，两条路径命中同一个模块对象，call_context 正常工作。
#
# 但一旦某种运行方式让 `src.` 导入失败并回退到裸名，就会出现两份模块、两个
# threading.local —— 此时 call_context 设的标签写在 A 副本，log_call 从 B 副本读，
# run_id/task_type 会静默变成 NULL。这里在导入期做一次自检，把该故障从
# 「静默失效」变成「立刻可见」。
#
# 哨兵挂在 sys 上：它是跨模块副本唯一保证单例的位置。
# ---------------------------------------------------------------------------
_SENTINEL = '_hdlformal_experiment_logger'
DUAL_LOAD_DETECTED = False

def _selfcheck_singleton():
    global DUAL_LOAD_DETECTED
    import sys as _sys
    prior = getattr(_sys, _SENTINEL, None)
    if prior is None:
        setattr(_sys, _SENTINEL, {'module': __name__, 'local': _local})
        return
    if prior['module'] == __name__:
        return  # 同名重复导入（reload），不是双重加载
    DUAL_LOAD_DETECTED = True
    print(f"⚠️  WARNING: experiment_logger loaded twice under different names: "
          f"'{prior['module']}' and '{__name__}'")
    if prior['local'] is not _local:
        print(f"❌ ERROR: experiment_logger loaded twice with distinct "
              f"threading.local; call_context will silently fail "
              f"(labels set in '{prior['module']}' are invisible to '{__name__}'). "
              f"Unify imports on 'src.experiment_logger'.", file=_sys.stderr)

_selfcheck_singleton()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,            -- ISO8601 UTC
    provider TEXT,                       -- Provider 类名，如 GroqProvider
    model TEXT,                          -- 模型名，如 llama-3.3-70b-versatile
    method TEXT,                         -- 被调用的方法名，如 _call_api / _call_api_stream
    streaming INTEGER DEFAULT 0,         -- 是否流式调用
    task_type TEXT,                      -- 上下文标签：bdd_generation / hardware_generation / testbench ...
    run_id TEXT,                         -- 实验批次 ID
    module_name TEXT,                    -- benchmark 模块名，如 alu_8bit
    prompt TEXT,
    system_prompt TEXT,
    response TEXT,
    prompt_chars INTEGER,
    response_chars INTEGER,
    max_tokens INTEGER,
    latency_ms INTEGER,
    success INTEGER DEFAULT 1,
    error TEXT,
    extra TEXT                           -- 预留 JSON 字段
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_calls_provider ON llm_calls(provider);
CREATE INDEX IF NOT EXISTS idx_calls_created ON llm_calls(created_at);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.executescript(_SCHEMA)
    return conn


def _utcnow() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ---------------------------------------------------------------------------
# 上下文标签：让上层代码给 LLM 调用打标签，无需层层传参
# ---------------------------------------------------------------------------

@contextmanager
def call_context(**labels):
    """
    with call_context(task_type='bdd_generation', run_id='exp001'):
        provider.generate_scenario_description(...)

    额外支持 extra=dict(...)：写入 llm_calls.extra（JSON）。
    嵌套时 extra 按键合并而非整体覆盖，内层可以只补自己关心的键。
    """
    old = getattr(_local, 'labels', {})
    merged = {**old, **labels}
    if old.get('extra') or labels.get('extra'):
        merged['extra'] = {**(old.get('extra') or {}), **(labels.get('extra') or {})}
    _local.labels = merged
    try:
        yield
    finally:
        _local.labels = old


def _current_labels() -> Dict[str, Any]:
    return getattr(_local, 'labels', {})


# ---------------------------------------------------------------------------
# 调用期事实登记
#
# call_context 记录的是调用方的**意图**（这属于哪次实验），而这里记录的是调用
# 过程中观测到的**事实**：实际发出的采样参数、API 返回的 finish_reason 等。
# provider 深处的代码无法把这些值顺着返回值传出来（_call_api 只返回字符串），
# 因此用 threading.local 传给埋点包装器，由它并入 extra。
# ---------------------------------------------------------------------------

def record_call_meta(**fields):
    """provider 在一次被埋点的调用内登记事实。不在此类调用内时静默忽略。"""
    cur = getattr(_local, 'call_meta', None)
    if cur is None:
        return
    cur.update({k: v for k, v in fields.items() if v is not None})


# ---------------------------------------------------------------------------
# 核心写入
# ---------------------------------------------------------------------------

def log_call(provider: str, model: str, method: str, prompt: str,
             response: str = None, system_prompt: str = None,
             max_tokens: int = None, latency_ms: int = None,
             success: bool = True, error: str = None,
             streaming: bool = False, extra: Dict = None):
    """记录一次 LLM 调用。写入失败不影响主流程（只打印警告）。"""
    labels = _current_labels()
    # 上下文级 extra（call_context 设置）与调用级 extra（埋点包装器计算）合并，
    # 同名键以调用级为准——包装器观测到的是实际发生的事实。
    ctx_extra = labels.get('extra') or {}
    if ctx_extra or extra:
        extra = {**ctx_extra, **(extra or {})}
    try:
        with _lock:
            conn = _connect()
            conn.execute(
                """INSERT INTO llm_calls
                   (created_at, provider, model, method, streaming,
                    task_type, run_id, module_name,
                    prompt, system_prompt, response,
                    prompt_chars, response_chars, max_tokens,
                    latency_ms, success, error, extra)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _utcnow(), provider, model, method, int(streaming),
                    labels.get('task_type'), labels.get('run_id'), labels.get('module_name'),
                    prompt, system_prompt, response,
                    len(prompt) if prompt else 0,
                    len(response) if response else 0,
                    max_tokens, latency_ms, int(success), error,
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                )
            )
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"⚠️  Experiment logger write failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# 自动埋点：包装 Provider 的 _call_api* 方法
# ---------------------------------------------------------------------------

# 方法名 -> api_path。用于 extra.api_path，让"这次调用走的哪条实现路径"可查。
_PATH_BY_METHOD = {
    '_call_api_stream': 'stream',
    '_call_api_sdk': 'sdk',
    '_call_api_rest': 'rest',
    '_call_api_with_json_mode': 'json_mode',
    '_call_api_text': 'text',
    '_call_api_completions': 'completions',
}


def _api_path(obj, method: str) -> str:
    """判定实际走的实现路径。`_call_api` 是转发型入口，需按 provider 状态推断。"""
    if method in _PATH_BY_METHOD:
        return _PATH_BY_METHOD[method]
    if method == '_call_api':
        if hasattr(obj, 'use_sdk'):                     # Gemini
            return 'sdk' if obj.use_sdk else 'rest'
        if type(obj).__name__ == 'OpenAIProvider':      # OpenAI 强制 JSON mode
            is_codex = getattr(obj, '_is_codex_model', None)
            try:
                if callable(is_codex) and is_codex(getattr(obj, 'model', '')):
                    return 'completions'
            except Exception:
                pass
            return 'json_mode'
        return 'default'
    return method


def _auto_extra(obj, cls_name: str, method: str, call_meta: Dict = None) -> Dict[str, Any]:
    """埋点包装器能自行观测到的事实（与调用方传入的意图区分开）。

    call_meta 是 provider 在调用期间通过 record_call_meta() 登记的内容：
    实际发出的采样参数、sampling_source/ignored、finish_reason 等。
    """
    path = _api_path(obj, method)
    info = {
        'api_path': path,
        'output_mode': 'json' if path == 'json_mode' else 'text',
        'model_effective': getattr(obj, 'model', None),
    }
    if call_meta:
        info.update(call_meta)
    # A1a 遗留项：Groq 的 _call_api_stream 曾因缩进错误不可达，此前所有 Groq
    # "流式"实际是非流式回退。标记原生流式，便于区分修复前后的数据。
    if cls_name == 'GroqProvider' and method == '_call_api_stream':
        info['groq_stream_native'] = True
    return info


def _reentrant() -> bool:
    """是否处于外层已记录的调用内部。

    Gemini 的 _call_api 转发给 _call_api_rest、OpenAI 的转发给
    _call_api_with_json_mode，而内层同样被埋点，一次逻辑调用会写两行
    （实测：外层带正确 max_tokens，内层因位置传参记成 NULL）。
    守卫保证"一次 API 调用 = 一行记录"，保留语义完整的外层。
    """
    return getattr(_local, 'in_call', False)


def _wrap_regular(cls_name, name, fn):
    @functools.wraps(fn)
    def wrapper(self, prompt, *args, **kwargs):
        if _reentrant():
            return fn(self, prompt, *args, **kwargs)   # 内层转发，不重复记录
        model = getattr(self, 'model', None)
        max_tokens = kwargs.get('max_tokens')
        system_prompt = kwargs.get('system_prompt')
        start = time.time()
        _local.in_call = True
        _local.call_meta = {}          # provider 在调用期间往里登记事实
        try:
            result = fn(self, prompt, *args, **kwargs)
            log_call(cls_name, model, name, prompt,
                     response=result if isinstance(result, str) else str(result),
                     system_prompt=system_prompt, max_tokens=max_tokens,
                     latency_ms=int((time.time() - start) * 1000), success=True,
                     extra=_auto_extra(self, cls_name, name, _local.call_meta))
            return result
        except Exception as e:
            log_call(cls_name, model, name, prompt,
                     system_prompt=system_prompt, max_tokens=max_tokens,
                     latency_ms=int((time.time() - start) * 1000),
                     success=False, error=f"{type(e).__name__}: {e}",
                     extra=_auto_extra(self, cls_name, name, _local.call_meta))
            raise
        finally:
            _local.in_call = False
            _local.call_meta = None
    wrapper._exp_logged = True
    return wrapper


def _wrap_stream(cls_name, name, fn):
    @functools.wraps(fn)
    def wrapper(self, prompt, *args, **kwargs):
        if _reentrant():
            return fn(self, prompt, *args, **kwargs)
        model = getattr(self, 'model', None)
        max_tokens = kwargs.get('max_tokens')
        system_prompt = kwargs.get('system_prompt')
        labels = _current_labels()  # 生成器可能在 context 退出后才被消费，先快照
        start = time.time()
        chunks = []

        def gen():
            success, error = True, None
            # 守卫在生成器体内设置：wrapper 只负责创建生成器，真正执行发生在被消费时。
            # 流式失败时 provider 会回退调用 self._call_api()，需一并抑制其重复记录。
            _local.in_call = True
            _local.call_meta = {}
            try:
                for chunk in fn(self, prompt, *args, **kwargs):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    yield chunk
            except Exception as e:
                success, error = False, f"{type(e).__name__}: {e}"
                raise
            finally:
                meta = _local.call_meta or {}
                _local.in_call = False
                _local.call_meta = None
                with call_context(**labels):
                    log_call(cls_name, model, name, prompt,
                             response=''.join(chunks),
                             system_prompt=system_prompt, max_tokens=max_tokens,
                             latency_ms=int((time.time() - start) * 1000),
                             success=success, error=error, streaming=True,
                             extra=_auto_extra(self, cls_name, name, meta))
        return gen()
    wrapper._exp_logged = True
    return wrapper


def instrument_class(cls):
    """包装类自身定义的所有 _call_api* 方法（不含继承来的，避免重复包装）。"""
    for name, fn in list(cls.__dict__.items()):
        if not name.startswith('_call_api') or not callable(fn):
            continue
        if getattr(fn, '_exp_logged', False):
            continue
        if inspect.isgeneratorfunction(fn):
            setattr(cls, name, _wrap_stream(cls.__name__, name, fn))
        else:
            setattr(cls, name, _wrap_regular(cls.__name__, name, fn))
    return cls


# ---------------------------------------------------------------------------
# 查询 / 导出
# ---------------------------------------------------------------------------

def get_stats() -> Dict:
    """按 provider 汇总：调用次数、成功率、平均延迟。"""
    conn = _connect()
    rows = conn.execute(
        """SELECT provider, model, COUNT(*),
                  SUM(success), AVG(latency_ms),
                  SUM(prompt_chars), SUM(response_chars)
           FROM llm_calls GROUP BY provider, model ORDER BY COUNT(*) DESC"""
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    conn.close()
    return {
        'total_calls': total,
        'by_provider': [
            {
                'provider': r[0], 'model': r[1], 'calls': r[2],
                'success_rate': round(r[3] / r[2], 3) if r[2] else 0,
                'avg_latency_ms': round(r[4]) if r[4] is not None else None,
                'prompt_chars': r[5], 'response_chars': r[6],
            } for r in rows
        ]
    }


def get_recent(limit: int = 20, run_id: str = None, include_untagged: bool = False) -> list:
    """最近的调用记录。

    默认排除 run_id 为空的行：那些是开发期打桩/探针产生的，没有归属的实验批次，
    混在输出里只会干扰。传 include_untagged=True 可以把它们带上——它们仍然是
    「这段时间做过什么」的痕迹，只是默认不展示。
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    sql = """SELECT id, created_at, provider, model, method, streaming,
                    task_type, run_id, module_name,
                    prompt_chars, response_chars, latency_ms, success, error,
                    extra
             FROM llm_calls"""
    params = []
    if run_id:
        sql += " WHERE run_id = ?"
        params.append(run_id)
    elif not include_untagged:
        sql += " WHERE run_id IS NOT NULL"
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = []
    for r in conn.execute(sql, params).fetchall():
        row = dict(r)
        # extra 以 JSON 文本存储，解析后输出，避免打印成一串转义字符
        if row.get('extra'):
            try:
                row['extra'] = json.loads(row['extra'])
            except (ValueError, TypeError):
                pass
        rows.append(row)
    conn.close()
    return rows


def export_jsonl(path: str, run_id: str = None) -> int:
    """导出完整记录（含 prompt/response 全文）为 JSONL，用于论文数据归档。"""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM llm_calls"
    params = []
    if run_id:
        sql += " WHERE run_id = ?"
        params.append(run_id)
    sql += " ORDER BY id"
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for row in conn.execute(sql, params):
            f.write(json.dumps(dict(row), ensure_ascii=False) + '\n')
            count += 1
    conn.close()
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stats'
    if cmd == 'stats':
        print(json.dumps(get_stats(), indent=2, ensure_ascii=False))
    elif cmd == 'recent':
        # recent [N] [run_id] —— get_recent 一直支持按 run_id 过滤，这里把它暴露出来，
        # 便于取出某一条 pipeline 依赖链的全部调用
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        rid = sys.argv[3] if len(sys.argv) > 3 else None
        # 第 4 个位置参数写 'all' 可以把未打标签的探针行也带出来
        untagged = len(sys.argv) > 4 and sys.argv[4] == 'all'
        print(json.dumps(get_recent(limit, run_id=rid, include_untagged=untagged),
                         indent=2, ensure_ascii=False))
    elif cmd == 'export':
        out = sys.argv[2] if len(sys.argv) > 2 else 'llm_calls.jsonl'
        n = export_jsonl(out)
        print(f"✅ Exported {n} records to {out}")
    else:
        print("Usage: python -m src.experiment_logger "
              "[stats | recent [N] [run_id] | export [file]]")
