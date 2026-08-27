"""
LLM Hardware Generator - Flask Backend
=======================================

Supports:
- ALU Verilog Generation
- Counter Verilog Generation
- Register File (Coming Soon)
- CPU (Coming Soon)
- BDD Test Scenario Generation
- Streaming output (SSE)
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from flask import send_file
import zipfile
import io

# ============================================================================
# Path Setup
# ============================================================================
from werkzeug.utils import secure_filename
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# 统一用 src. 前缀导入 experiment_logger：模块内有单例自检，若同时以裸名加载
# 会出现两个 threading.local，call_context 设的标签会静默丢失。
from src.experiment_logger import call_context, log_artifact  # noqa: E402
from src import prompt_store  # noqa: E402

# ============================================================================
# Configuration Loading
# ============================================================================
def load_config():
    config = {
        'proxy': {'enabled': False},
        'api_keys': {}
    }

    config_file = PROJECT_ROOT / 'config' / 'llm_config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
                print(f"✅ Config loaded from: {config_file}")
        except Exception as e:
            print(f"⚠️ Config file error: {e}")

    # Extract API keys from "providers" structure into flat "api_keys" dict
    providers_config = config.get('providers', {})
    for provider_name, provider_data in providers_config.items():
        if isinstance(provider_data, dict) and provider_data.get('api_key'):
            config['api_keys'][provider_name] = provider_data['api_key']

    env_keys = {
        'GROQ_API_KEY': 'groq',
        'DEEPSEEK_API_KEY': 'deepseek',
        'OPENAI_API_KEY': 'openai',
        'ANTHROPIC_API_KEY': 'anthropic',
        'GEMINI_API_KEY': 'gemini',
        'XAI_API_KEY': 'grok',
        'QWEN_API_KEY': 'qwen',
        'MISTRAL_API_KEY': 'mistral',
        'TOGETHER_API_KEY': 'together'
    }

    for env_var, provider in env_keys.items():
        if os.environ.get(env_var):
            config['api_keys'][provider] = os.environ[env_var]

    if os.environ.get('ENABLE_PROXY', '').lower() == 'false':
        config['proxy']['enabled'] = False

    return config

CONFIG = load_config()

# ============================================================================
# Proxy Setup
# ============================================================================
def setup_proxy():
    proxy_config = CONFIG.get('proxy', {})

    if not proxy_config.get('enabled', False):
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(key, None)
        print("🌐 Proxy: Disabled")
        return

    http_proxy = proxy_config.get('http_proxy', '')
    https_proxy = proxy_config.get('https_proxy', '')

    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
        os.environ['http_proxy'] = http_proxy
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
        os.environ['https_proxy'] = https_proxy
        print(f"🌐 Proxy: Enabled ({https_proxy})")

setup_proxy()

# ============================================================================
# Set API Keys
# ============================================================================
def setup_api_keys():
    api_keys = CONFIG.get('api_keys', {})

    key_mapping = {
        'groq': 'GROQ_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'claude': 'ANTHROPIC_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'grok': 'XAI_API_KEY',
        'qwen': 'QWEN_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'together': 'TOGETHER_API_KEY'
    }

    for provider, env_var in key_mapping.items():
        if api_keys.get(provider) and not api_keys[provider].startswith('your_'):
            os.environ[env_var] = api_keys[provider]

    # Display status with model info
    display_mapping = {
        'GROQ': 'GROQ_API_KEY',
        'DEEPSEEK': 'DEEPSEEK_API_KEY',
        'OPENAI': 'OPENAI_API_KEY',
        'CLAUDE': 'ANTHROPIC_API_KEY',
        'GEMINI': 'GEMINI_API_KEY',
        'GROK': 'XAI_API_KEY',
        'QWEN': 'QWEN_API_KEY',
        'MISTRAL': 'MISTRAL_API_KEY',
        'TOGETHER': 'TOGETHER_API_KEY'
    }

    # Build provider->model mapping from config
    providers_cfg = CONFIG.get('providers', {})
    print("🔑 API Keys status:")
    for display_name, env_var in display_mapping.items():
        status = "✅" if os.environ.get(env_var) else "❌"
        # Get model name from providers config
        provider_key = display_name.lower()
        model_name = ""
        if provider_key in providers_cfg and providers_cfg[provider_key].get('model'):
            model_name = f" (model: {providers_cfg[provider_key]['model']})"
        print(f"   {status} {display_name}{model_name}")

setup_api_keys()

# ============================================================================
# Import Modules
# ============================================================================
HAS_BDD_MODULE = False
HAS_ALU_MODULE = False
HAS_COUNTER_MODULE = False
HAS_REGFILE_MODULE = False
HAS_CPU_MODULE = False

try:
    from feature_generator_llm import FeatureGeneratorLLM
    from llm_providers import LLMFactory
    HAS_BDD_MODULE = True
    print("✅ BDD Generator module loaded")
except ImportError as e:
    print(f"⚠️ BDD module not available: {e}")

try:
    from alu_generator import ALUGenerator, DEFAULT_ALU_OPERATIONS
    HAS_ALU_MODULE = True
    print("✅ ALU Generator module loaded")
except ImportError as e:
    print(f"⚠️ ALU module not available: {e}")

try:
    from counter_generator import CounterGenerator
    HAS_COUNTER_MODULE = True
    print("✅ Counter Generator module loaded")
except ImportError as e:
    print(f"⚠️ Counter module not available: {e}")

try:
    from register_generator import RegFileGenerator
    HAS_REGFILE_MODULE = True
    print("✅ Register File Generator module loaded")
except ImportError as e:
    print(f"⚠️ Register File module not available: {e}")

try:
    from cpu_generator import CPUGenerator
    HAS_CPU_MODULE = True
    print("✅ CPU Generator module loaded")
except ImportError as e:
    print(f"⚠️ CPU module not available: {e}")

# DUT Interface Checker (Bug Detection)
HAS_DUT_CHECKER = False
try:
    from dut_interface_checker import InterfaceBugDetector, BugSeverity, CounterSpecification, RegFileSpecification, CPUSpecification
    HAS_DUT_CHECKER = True
    print("✅ DUT Interface Checker module loaded")
except ImportError as e:
    print(f"⚠️ DUT Interface Checker not available: {e}")

# 在其他模块导入之后添加
HAS_TESTBENCH_MODULE = False

try:
    from testbench_generator import TestbenchGenerator
    HAS_TESTBENCH_MODULE = True
    print("✅ Testbench Generator module loaded")
except ImportError as e:
    print(f"⚠️ Testbench Generator not available: {e}")

# Simulation Runner
HAS_SIMULATION_MODULE = False
try:
    from simulation_runner import WebSimulationRunner
    HAS_SIMULATION_MODULE = True
    print("✅ Simulation Runner module loaded")
except ImportError as e:
    print(f"⚠️ Simulation Runner not available: {e}")
# ============================================================================
# Flask App
# ============================================================================
app = Flask(__name__,
            static_folder=str(PROJECT_ROOT / 'static'),
            template_folder=str(PROJECT_ROOT / 'static'))
CORS(app)

# Store last generated files
last_generated_bdd = {'filename': None, 'filepath': None, 'llm': None}
last_generated_hw = {'filename': None, 'filepath': None, 'llm': None, 'module_type': None}
last_generated_tb = {'filename': None, 'filepath': None, 'bdd_source': None}

# Ensure output directories exist
(PROJECT_ROOT / 'output' / 'bdd').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'output' / 'dut').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'output' / 'testbench').mkdir(parents=True, exist_ok=True)

# ============================================================================
# Upload Configuration
# ============================================================================
UPLOAD_FOLDER = PROJECT_ROOT / 'output' / 'dut' / 'uploaded'
ALLOWED_EXTENSIONS = {'v', 'sv', 'vh'}
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize parser
try:
    from verilog_parser import VerilogParser

    verilog_parser = VerilogParser(upload_dir=str(UPLOAD_FOLDER))
    HAS_PARSER = True
    print("✅ Verilog Parser loaded")
except ImportError:
    verilog_parser = None
    HAS_PARSER = False
    print("⚠️ Verilog Parser not available")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# 三层 ID 体系：session_id > run_id > 单次 LLM 调用
#
#   session_id  浏览器会话（前端生成，存 localStorage）
#     └─ run_id     一条完整 artifact 依赖链：DUV -> BDD -> TB -> Sim
#          └─ 每次 LLM 调用一行 llm_calls 记录
#
# run_id 生命周期：Step 1（生成或上传 DUV）新建；Step 2/3/4 继承；
# Step 2 在 DUV 不变时重试则保持 run_id 并让 attempt 递增。
# DUV 变了整条链就失效，因此 Step 1 必须新建。
# ============================================================================

# Web 端只有 openai / gemini 真正接受 model 覆盖（见 /api/generate 与
# /api/generate-stream）；Step 1 的生成器完全不接收 model 参数。
# 本次不修改该行为，只如实记录，供前端提示与后续排查。
MODEL_OVERRIDE_PROVIDERS = ('openai', 'gemini')


def _new_run_id() -> str:
    import uuid
    return f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


SAMPLING_KEYS = ('temperature', 'top_p', 'seed', 'max_tokens')


def _parse_sampling(data) -> Optional[dict]:
    """从请求体取采样参数。空值一律丢弃——留空即“用 provider 默认值”，
    返回 None 时下游行为与引入本机制之前完全一致。"""
    raw = (data or {}).get('sampling') or {}
    out = {}
    for k in SAMPLING_KEYS:
        v = raw.get(k)
        if v is None or v == '':
            continue
        try:
            out[k] = int(v) if k in ('seed', 'max_tokens') else float(v)
        except (TypeError, ValueError):
            continue
    return out or None


def _last_call_meta(run_id):
    """取该 run 最新一条 llm_calls 的 extra + 用量，供前端显示元信息行。

    埋点是唯一知道“实际发出了什么”的地方（采样值、finish_reason 由 provider
    在调用期登记），所以从库里回读，而不是让端点再猜一遍。
    """
    try:
        import sqlite3
        from src.experiment_logger import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT model, extra, prompt_chars, response_chars, latency_ms
               FROM llm_calls WHERE run_id = ? ORDER BY id DESC LIMIT 1""",
            (run_id,)).fetchone()
        conn.close()
        if not row:
            return None
        meta = json.loads(row['extra']) if row['extra'] else {}
        meta['model_effective'] = meta.get('model_effective') or row['model']
        meta['prompt_chars'] = row['prompt_chars']
        meta['response_chars'] = row['response_chars']
        meta['latency_ms'] = row['latency_ms']
        # 各家 finish_reason 大小写与别名不统一（stop/STOP、length/MAX_TOKENS），
        # 归一成一个布尔量，前端不必再判别名
        fr = str(meta.get('finish_reason') or '').lower()
        meta['truncated'] = fr in ('length', 'max_tokens')
        return meta
    except Exception as e:
        print(f"⚠️  read call meta failed (non-fatal): {e}")
        return None


def _module_name_from_verilog(path):
    """从 Verilog 文件里读出顶层模块名；读不到返回 None。

    Step 3 生成 testbench 时需要知道实例化哪个模块，而这个名字只有 DUV 文件
    本身说了算——任何按约定拼出来的名字都可能与实际生成的不一致。
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / path
    if not p.is_file():
        return None
    try:
        import re
        text = p.read_text(encoding='utf-8', errors='replace')
        # 跳过注释里的 module 字样
        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
        m = re.search(r'^\s*module\s+(\w+)', text, re.M)
        return m.group(1) if m else None
    except Exception as e:
        print(f"⚠️  could not read module name from {path}: {e}")
        return None


def _apply_model_override(generator, llm_name, model) -> bool:
    """把用户选的 model 应用到已构造的 generator，返回是否真的应用了。

    生成器在 __init__ 里按 llm_provider 建好 self.llm，这里换成指定 model 的实例。
    此前这段逻辑只内联在流式端点里，非流式端点缺失，导致同一个 Step 1
    勾不勾 "Stream output" 会决定 model 选择是否生效。抽成函数以免两处再次漂移。
    """
    if not model or llm_name not in MODEL_OVERRIDE_PROVIDERS:
        return False
    try:
        from llm_providers import LLMFactory
        generator.llm = LLMFactory.create_provider(llm_name, model=model)
        print(f"🔷 [{llm_name.upper()}] Model overridden to: {model}")
        return True
    except Exception as e:
        print(f"⚠️  Failed to override model: {e}")
        return False


def _web_ctx(data, task_type, module_name=None, new_run=False, model_used=False):
    """从请求体取出三层 ID，组装 call_context 的参数。

    new_run=True   -> Step 1：无条件新建 run_id（DUV 变则依赖链失效）
    model_used     -> 该端点是否真的会把 model 传给 provider；
                      为 False 时只要请求带了 model 就记为被忽略
    返回 (run_id, ctx_kwargs)
    """
    data = data or {}
    run_id = None if new_run else (data.get('run_id') or '').strip() or None
    if not run_id:
        run_id = _new_run_id()

    llm_name = (data.get('llm') or '').strip().lower()
    model_requested = (data.get('model') or '').strip() or None
    ignored = bool(model_requested) and (
        not model_used or llm_name not in MODEL_OVERRIDE_PROVIDERS)

    try:
        attempt = max(0, int(data.get('attempt') or 0))
    except (TypeError, ValueError):
        attempt = 0

    ctx = {
        'task_type': task_type,
        'run_id': run_id,
        'module_name': module_name,
        'extra': {
            'session_id': (data.get('session_id') or '').strip() or None,
            'attempt': attempt,
            'model_requested': model_requested,
            'model_override_ignored': ignored,
            # 批量采集时带上批次标识，便于事后从 llm_calls 精确筛出某一批
            'batch': (data.get('batch') or '').strip() or None,
            # spec-first 会往 DUV prompt 里注入整份 BDD 规格，与 impl-first
            # 的 prompt 明显不同。不记这一栏，事后就无法从数据里区分两者。
            # 缺省按 implementation 记（与端点内部的取值口径一致），这样这一列
            # 永不为空——空值会让人分不清"是 impl-first"还是"这批数据没记"。
            'workflow_mode': (data.get('workflow_mode') or '').strip() or 'implementation',
        },
    }
    return run_id, ctx


def _read_bdd_context(data):
    """Specification-First：读取要作为设计依据的 BDD 规格。

    返回 (bdd_context, error)。error 非空时调用方必须中止：用户明确选了
    spec-first 并给了文件，读不到却照常生成，产出的是一份无视规格的硬件，
    而前端完全看不出来——这正是要消除的静默降级。
    """
    if (data.get('workflow_mode') or 'implementation') != 'specification':
        return None, None
    bdd_filepath = data.get('bdd_filepath')
    if not bdd_filepath:
        return None, None          # 还没生成 BDD，由前端把关先后顺序

    path = Path(bdd_filepath)
    if not path.exists():
        # 与 /api/generate-testbench 一致：相对路径按项目根目录再试一次
        path = PROJECT_ROOT / bdd_filepath
    if not path.exists():
        return None, f'BDD spec not found: {bdd_filepath}'
    try:
        text = path.read_text(encoding='utf-8')
    except Exception as e:
        return None, f'Failed to read BDD spec {bdd_filepath}: {e}'
    if not text.strip():
        return None, f'BDD spec is empty: {bdd_filepath}'

    print(f"📋 BDD-First mode: loaded BDD spec ({len(text)} chars)")
    return text, None


def _log_artifact(data, run_id, stage, path=None, meta=None):
    """把本步产物挂到依赖链上。

    testbench 与仿真没有 LLM 参与，在 llm_calls 里不会留下任何行；DUV/BDD 虽有
    调用记录，但那里存的是 prompt/response 而非输出文件路径。所以「某个 run_id
    产出了哪些文件、仿真结果如何」只能靠这张表回答。
    """
    data = data or {}
    log_artifact(run_id, stage, path=path,
                 session_id=(data.get('session_id') or '').strip() or None,
                 workflow_mode=(data.get('workflow_mode') or '').strip() or 'implementation',
                 meta=meta)


def _resolve_output_file(subdir, filename, remembered=None):
    """在 output/<subdir>/ 下按文件名定位产物，找不到返回 None。

    三个下载端点原本各写一份查找逻辑，硬件那份漏了扫子目录——而产物恰恰都存在
    output/<subdir>/<llm>/ 下，于是只要 last_generated_* 没指向目标就 404。
    那个"记住最近一次"的全局是模块级的、被所有请求共享，多开一个标签页或让
    仪表盘并行跑就会互相覆盖，因此兜底查找才是真正起作用的路径，必须正确。
    收敛成一处，避免再次各自漂移。
    """
    # filename 来自 URL 的 <filename> 转换器（不匹配斜杠），这里再挡一道，
    # 使本函数对任何调用方都安全
    if Path(filename).name != filename:
        return None

    if remembered:
        candidate = Path(remembered)
        if candidate.exists() and candidate.name == filename:
            return candidate

    base = PROJECT_ROOT / 'output' / subdir
    candidate = base / filename
    if candidate.exists():
        return candidate

    if base.exists():
        for sub in base.iterdir():
            if sub.is_dir():
                candidate = sub / filename
                if candidate.exists():
                    return candidate
    return None


def _duv_compile_check(path):
    """Step 1 生成后立刻单独编译一次，只报告不阻断。

    不这么做的话，一份语法不合法的 DUV 会一路通过 Step 1/2/3，直到 Step 4 才
    以 "Compilation failed" 暴露——失败被记在仿真头上，而真正的原因是 DUV 生成。
    做失败归因时这个错位会让统计失真。

    标准必须与 simulation_runner 用的一致（-g2012）：用更严格的标准检查，会把
    实际能仿真的设计误报成编译失败。
    """
    import subprocess, tempfile
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                ["iverilog", "-g2012", "-o", str(Path(tmp) / "a.out"), str(p)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60)
    except FileNotFoundError:
        return None          # 没装 iverilog：不报告，而不是谎报失败
    except Exception as e:
        return {'ok': None, 'error': f'{type(e).__name__}: {e}'}

    if r.returncode == 0:
        return {'ok': True, 'error': None}
    # 只取第一条错误：后面的多半是它的连锁反应
    first = next((ln.strip() for ln in (r.stderr or '').splitlines() if 'error' in ln.lower()
                  or 'syntax' in ln.lower()), (r.stderr or '').strip()[:200])
    return {'ok': False, 'error': first}


def _chain_start(data, step):
    """本次调用是否为依赖链起点（据此决定新建还是继承 run_id）。

    起点随工作流方向变：impl-first 是 DUV -> BDD -> TB，spec-first 是
    BDD -> DUV -> TB。把 DUV 一律当起点，会让 spec-first 下的 BDD 和据它
    生成的 DUV 落在两个 run_id 上，事后无法还原因果关系。
    """
    spec_first = (data.get('workflow_mode') or 'implementation') == 'specification'
    return (step == 'bdd') if spec_first else (step == 'duv')


# Initialize simulation runner
simulation_runner = None
if HAS_SIMULATION_MODULE:
    simulation_runner = WebSimulationRunner(project_root=str(PROJECT_ROOT))
    print(f"🔧 Simulation tools: {simulation_runner.get_tools_status()}")


# ============================================================================
# Upload DUT API
# ============================================================================
@app.route('/api/upload-dut', methods=['POST'])
def upload_dut():
    """Handle Verilog file upload and parsing with Bug Detection."""
    if not HAS_PARSER:
        return jsonify({'success': False, 'error': 'Verilog parser not available'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Allowed: .v, .sv, .vh'}), 400

    try:
        file_content = file.read()
        filename = secure_filename(file.filename)

        # ========== 保存文件到指定目录 ==========
        upload_dir = PROJECT_ROOT / 'generated' / 'uploaded'
        upload_dir.mkdir(parents=True, exist_ok=True)

        filepath = upload_dir / filename
        with open(filepath, 'wb') as f:
            f.write(file_content)

        # 解析文件 - 保留原有逻辑
        result = verilog_parser.parse_file(filename, file_content)

        # 更新 last_generated_hw 状态
        module_name = result.get('module', {}).get('name', 'unknown') if isinstance(result.get('module'),
                                                                                    dict) else result.get('name',
                                                                                                          'unknown')
        detected_module_type = result.get('module_type', 'custom')

        last_generated_hw['filename'] = filename
        last_generated_hw['filepath'] = str(filepath)
        last_generated_hw['llm'] = 'uploaded'
        last_generated_hw['module_type'] = detected_module_type

        content = file_content.decode('utf-8', errors='replace')

        result['filepath'] = str(filepath)
        result['full_content'] = content
        result['preview'] = content[:1000] + ('...' if len(content) > 1000 else '')
        result['llm'] = 'uploaded'

        # ========== DUT Bug 检测 (支持 ALU / Counter / RegFile / CPU) ==========
        if HAS_DUT_CHECKER and result.get('success'):
            try:
                modules = result.get('modules', [])
                bitwidth = 8
                detected_type = 'other'
                depth = 32

                if modules:
                    first_module = modules[0]
                    bitwidth = first_module.get('bitwidth', 8)
                    detected_type = first_module.get('detected_type', 'other')
                    depth = first_module.get('num_registers', 32)

                supported_types = ['alu', 'counter', 'regfile', 'cpu']

                if detected_type in supported_types:
                    detector = InterfaceBugDetector(
                        content, bitwidth,
                        module_type=detected_type,
                        depth=depth
                    )
                    bugs = detector.check_all()

                    bug_report = {
                        'checked': True,
                        'module_type': detected_type,
                        'total_bugs': len(bugs),
                        'has_critical': any(b.severity == BugSeverity.CRITICAL for b in bugs),
                        'has_warning': any(b.severity == BugSeverity.WARNING for b in bugs),
                        'should_block': any(b.severity == BugSeverity.CRITICAL for b in bugs),
                        'bugs': []
                    }

                    for bug in bugs:
                        bug_report['bugs'].append({
                            'type': bug.bug_type,
                            'severity': bug.severity.value,
                            'port': bug.port_name,
                            'line': bug.line_number,
                            'expected': bug.expected,
                            'actual': bug.actual,
                            'code': bug.raw_line,
                            'explanation': bug.explanation,
                            'impact': bug.impact
                        })

                    result['bug_detection'] = bug_report

                    if bug_report['has_critical']:
                        type_names = {'alu': 'ALU', 'counter': 'Counter', 'regfile': 'Register File', 'cpu': 'RISC-V CPU'}
                        result['validation_warning'] = (
                            f"Found {sum(1 for b in bugs if b.severity == BugSeverity.CRITICAL)} critical bugs!"
                            f"Validation reliability may be compromised by these bugs."
                        )
                else:
                    result['bug_detection'] = {
                        'checked': False,
                        'module_type': detected_type,
                        'message': f'Bug detection currently only supports ALU types; detected type: {detected_type}'
                    }

            except Exception as bug_check_error:
                print(f"⚠️ Error during bug detection: {bug_check_error}")
                result['bug_detection'] = {
                    'checked': False,
                    'error': str(bug_check_error)
                }

        # 上传 DUV 同样是依赖链的起点：新建 run_id 供 Step 2/3/4 继承。
        # 此端点不调用 LLM，因此只回传 ID，不需要 call_context。
        result['run_id'] = _new_run_id()
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500



@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum: 1MB'}), 413


# ============================================================================
# Upload BDD File API
# ============================================================================
@app.route('/api/upload-bdd', methods=['POST'])
def upload_bdd():
    """Handle BDD feature file upload."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    # Check file extension
    if not file.filename.lower().endswith('.feature'):
        return jsonify({'success': False, 'error': 'Invalid file type. Only .feature files allowed'}), 400

    try:
        file_content = file.read()
        filename = secure_filename(file.filename)

        # Save to uploaded BDD directory
        upload_dir = PROJECT_ROOT / 'output' / 'bdd' / 'uploaded'
        upload_dir.mkdir(parents=True, exist_ok=True)

        filepath = upload_dir / filename
        with open(filepath, 'wb') as f:
            f.write(file_content)

        # Parse BDD content for preview
        content = file_content.decode('utf-8', errors='replace')

        # Extract feature name and scenario count
        feature_name = ''
        scenario_count = 0
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Feature:'):
                feature_name = line.replace('Feature:', '').strip()
            if line.startswith('Scenario:') or line.startswith('Scenario Outline:'):
                scenario_count += 1

        # Update last_generated_bdd state
        last_generated_bdd['filename'] = filename
        last_generated_bdd['filepath'] = str(filepath)
        last_generated_bdd['llm'] = 'uploaded'

        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': str(filepath),
            'saved_path': str(filepath),
            'feature_name': feature_name,
            'scenario_count': scenario_count,
            'preview': content[:1000] + ('...' if len(content) > 1000 else ''),
            'full_content': content,
            'llm': 'uploaded'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


# ============================================================================
# Upload Testbench File API
# ============================================================================
@app.route('/api/upload-testbench', methods=['POST'])
def upload_testbench():
    """Handle Verilog testbench file upload."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    # Check file extension
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ['v', 'sv', 'vh']:
        return jsonify({'success': False, 'error': 'Invalid file type. Allowed: .v, .sv, .vh'}), 400

    try:
        file_content = file.read()
        filename = secure_filename(file.filename)

        # Save to uploaded testbench directory
        upload_dir = PROJECT_ROOT / 'output' / 'testbench' / 'uploaded'
        upload_dir.mkdir(parents=True, exist_ok=True)

        filepath = upload_dir / filename
        with open(filepath, 'wb') as f:
            f.write(file_content)

        # Parse testbench content
        content = file_content.decode('utf-8', errors='replace')

        # Extract module name and test count
        module_name = ''
        test_count = 0
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('module ') and '_tb' in line.lower():
                # Extract module name
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[1].split('(')[0].strip()
            # Count $display or test assertions
            if '$display' in line or 'assert' in line.lower():
                test_count += 1

        # Update last_generated_tb state
        last_generated_tb['filename'] = filename
        last_generated_tb['filepath'] = str(filepath)
        last_generated_tb['bdd_source'] = 'uploaded'

        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': str(filepath),
            'saved_path': str(filepath),
            'module_name': module_name,
            'test_count': test_count,
            'preview': content[:1000] + ('...' if len(content) > 1000 else ''),
            'full_content': content,
            'llm': 'uploaded'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

def make_sse_message(msg_type, **kwargs):
    """Helper to create SSE message"""
    data = {"type": msg_type}
    data.update(kwargs)
    return f"data: {json.dumps(data)}\n\n"


@app.route('/')
def index():
    return send_from_directory(str(PROJECT_ROOT / 'static'), 'bdd_generator.html')


@app.route('/api/health')
def health_check():
    sim_tools = simulation_runner.get_tools_status() if simulation_runner else {'can_simulate': False}
    return jsonify({
        'status': 'healthy',
        'bdd_module': HAS_BDD_MODULE,
        'alu_module': HAS_ALU_MODULE,
        'counter_module': HAS_COUNTER_MODULE,
        'regfile_module': HAS_REGFILE_MODULE,
        'cpu_module': HAS_CPU_MODULE,
        'testbench_module': HAS_TESTBENCH_MODULE,
        'simulation_module': HAS_SIMULATION_MODULE,
        'simulation_tools': sim_tools,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Experiment Logs API (实验记录查询)
# ============================================================================
@app.route('/api/experiment-stats')
def experiment_stats():
    """按 provider/model 汇总 LLM 调用统计（次数、成功率、平均延迟）"""
    try:
        from src.experiment_logger import get_stats
        return jsonify({'success': True, **get_stats()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _bench_runner():
    """按需导入 benchmark/run_experiments.py（不是包，用 sys.path 挂载）"""
    import sys as _sys
    if str(BENCHMARK_DIR) not in _sys.path:
        _sys.path.insert(0, str(BENCHMARK_DIR))
    import run_experiments
    return run_experiments


@app.route('/dashboard')
def dashboard():
    return send_from_directory(str(PROJECT_ROOT / 'static'), 'experiment_dashboard.html')


@app.route('/api/benchmark-modules')
def benchmark_modules():
    """题库模块清单（benchmark/index.json）"""
    try:
        with open(BENCHMARK_DIR / 'index.json', encoding='utf-8') as f:
            return jsonify({'success': True, **json.load(f)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/benchmark-run', methods=['POST'])
def benchmark_run():
    """后台启动一批实验；前端轮询 /api/benchmark-status 查看进度"""
    import threading
    data = request.json or {}
    llms = data.get('llms') or []
    reps = max(1, min(int(data.get('reps', 1)), 10))
    workers = max(1, min(int(data.get('workers', 3) or 3), 8))
    run_id = (data.get('run_id') or '').strip()
    if not run_id:
        # 自动 run_id 带上 LLM 名，方便在结果下拉框里区分（只用 '-'，保证 secure_filename 安全）
        tag = '-'.join(llms) if len(llms) <= 3 else f"{llms[0]}-and{len(llms) - 1}more"
        run_id = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    judge_name = (data.get('judge') or '').strip() or None
    wanted = set(data.get('modules') or [])

    if not llms:
        return jsonify({'success': False, 'error': 'no LLMs selected'}), 400
    if not wanted:
        # 必须显式选模块（防止误触全量真实 API 跑）；CLI 的 run_experiments.py 仍默认全跑
        return jsonify({'success': False, 'error': 'no modules selected — pick modules or click "Select all"'}), 400
    try:
        # 复用已有 run_id 会覆盖 artifacts（run_id/module/llm/repN 目录相同），拒绝
        import sqlite3 as _sq
        from src.experiment_logger import DB_PATH as _dbp
        _c = _sq.connect(_dbp)
        _n = _c.execute('SELECT COUNT(*) FROM benchmark_results WHERE run_id = ?', (run_id,)).fetchone()[0]
        _c.close()
        if _n:
            return jsonify({'success': False,
                            'error': f'run_id "{run_id}" already has {_n} results — pick a new one'}), 400
    except Exception:
        pass  # 查不到库不阻塞启动
    if any(r.get('status') == 'running' for r in _bench_runs.values()):
        return jsonify({'success': False, 'error': 'another run is already in progress'}), 409

    bench = _bench_runner()
    manifests = sorted(BENCHMARK_DIR.glob('*/*/manifest.json'))
    if wanted:
        manifests = [m for m in manifests if m.parent.name in wanted]
    if not manifests:
        return jsonify({'success': False, 'error': 'no matching modules'}), 400

    state = {'status': 'running', 'total': len(manifests) * len(llms) * reps,
             'done': 0, 'current': '', 'results': [], 'error': None}
    _bench_runs[run_id] = state

    def worker():
        from concurrent.futures import ThreadPoolExecutor, as_completed
        lock = threading.Lock()
        running = set()  # 正在跑的实验标签，用于进度条显示

        def run_task(mf, manifest, llm, rep, providers, judge):
            label = f"{manifest['name']} × {llm} rep{rep}"
            with lock:
                running.add(label)
                state['current'] = ' | '.join(sorted(running))
            try:
                row = bench.run_one(mf.parent, manifest, llm, providers[llm],
                                    rep, run_id, judge=judge, judge_name=judge_name)
            finally:
                with lock:
                    running.discard(label)
                    state['current'] = ' | '.join(sorted(running))
            with lock:
                state['done'] += 1
                state['results'].append({k: row[k] for k in (
                    'module', 'llm', 'rep', 'scenarios_count', 'tb_compiled',
                    'golden_passed', 'mutation_score', 'completeness', 'error',
                    'bdd_ms', 'tb_ms')})

        try:
            providers = {name: bench.make_provider(name) for name in llms}
            judge = bench.make_provider(judge_name) if judge_name and judge_name != 'mock' else None
            tasks = []
            for mf in manifests:
                manifest = json.loads(mf.read_text(encoding="utf-8"))
                for llm in llms:
                    for rep in range(1, reps + 1):
                        tasks.append((mf, manifest, llm, rep))
            # LLM 调用是 IO 等待，用线程池并行跑（call_context 是 threading.local，安全）
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_task, *t, providers, judge) for t in tasks]
                for f in as_completed(futures):
                    f.result()  # 让异常在这里抛出
            state['status'] = 'finished'
            state['current'] = ''
        except Exception as e:
            state['status'] = 'failed'
            state['error'] = f'{type(e).__name__}: {e}'

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'success': True, 'run_id': run_id, 'total': state['total']})


@app.route('/api/benchmark-status')
def benchmark_status():
    run_id = request.args.get('run_id', '')
    state = _bench_runs.get(run_id)
    if not state:
        return jsonify({'success': False, 'error': 'unknown run_id'}), 404
    return jsonify({'success': True, 'run_id': run_id, **state})


@app.route('/api/benchmark-results')
def benchmark_results():
    """历史结果：按 LLM 汇总 + 明细（从 SQLite benchmark_results 表读取）"""
    import sqlite3
    try:
        from src.experiment_logger import DB_PATH
        run_id = request.args.get('run_id')
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        runs = [dict(r) for r in conn.execute(
            '''SELECT run_id, GROUP_CONCAT(DISTINCT llm) AS llms, COUNT(*) AS n
               FROM benchmark_results GROUP BY run_id ORDER BY MAX(id) DESC''').fetchall()]
        where, params = ('WHERE run_id = ?', [run_id]) if run_id else ('', [])
        summary = [dict(r) for r in conn.execute(f'''
            SELECT llm, COUNT(*) AS runs,
                   AVG(tb_compiled) AS compile_rate,
                   AVG(golden_passed) AS golden_rate,
                   AVG(mutation_score) AS mutation_score,
                   AVG(completeness) AS completeness,
                   AVG(scenarios_count) AS avg_scenarios
            FROM benchmark_results {where}
            GROUP BY llm ORDER BY AVG(golden_passed) DESC''', params).fetchall()]
        detail = [dict(r) for r in conn.execute(f'''
            SELECT module, category, llm, rep, scenarios_count, tb_compiled,
                   golden_passed, mutants_total, mutants_detected,
                   mutation_score, completeness, error
            FROM benchmark_results {where}
            ORDER BY module, llm, rep''', params).fetchall()]
        conn.close()
        return jsonify({'success': True, 'runs': runs, 'summary': summary, 'detail': detail})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/benchmark-delete-run', methods=['POST'])
def benchmark_delete_run():
    """删除一个 run 的成绩行 + artifacts 目录（清理跑废的批次）"""
    import sqlite3
    import shutil
    run_id = ((request.json or {}).get('run_id') or '').strip()
    if not run_id:
        return jsonify({'success': False, 'error': 'run_id required'}), 400
    if _bench_runs.get(run_id, {}).get('status') == 'running':
        return jsonify({'success': False, 'error': 'run is still in progress'}), 409
    try:
        from src.experiment_logger import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute('DELETE FROM benchmark_results WHERE run_id = ?', (run_id,))
        conn.commit()
        conn.close()
        run_dir = PROJECT_ROOT / 'output' / 'benchmark_runs' / secure_filename(run_id)
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
        return jsonify({'success': True, 'deleted_rows': cur.rowcount})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


_fb_runs = {}  # run_id -> {'status', 'total', 'done', 'current', 'results': []}


def _fb_runner():
    """按需导入 benchmark/run_feedback.py（挂 sys.path，同 _bench_runner）"""
    import sys as _sys
    if str(BENCHMARK_DIR) not in _sys.path:
        _sys.path.insert(0, str(BENCHMARK_DIR))
    import run_feedback
    return run_feedback


@app.route('/api/feedback-run', methods=['POST'])
def feedback_run():
    """后台启动一批 Phase-2 反馈闭环任务；前端轮询 /api/feedback-status"""
    import threading
    data = request.json or {}
    llms = data.get('llms') or []
    reps = max(1, min(int(data.get('reps', 1) or 1), 5))
    iters = max(1, min(int(data.get('iters', 3) or 3), 6))
    workers = max(1, min(int(data.get('workers', 2) or 2), 8))
    arms = [a for a in (data.get('arms') or ['bdd', 'tb', 'bdd+']) if a in ('bdd', 'tb', 'bdd+')]
    wanted = set(data.get('modules') or [])
    run_id = (data.get('run_id') or '').strip()
    if not run_id:
        tag = '-'.join(llms) if len(llms) <= 3 else f"{llms[0]}-and{len(llms) - 1}more"
        run_id = f"fb_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not llms:
        return jsonify({'success': False, 'error': 'no LLMs selected'}), 400
    if not wanted:
        return jsonify({'success': False, 'error': 'no modules selected'}), 400
    if not arms:
        return jsonify({'success': False, 'error': 'select at least one arm'}), 400
    if any(r.get('status') == 'running' for r in _fb_runs.values()):
        return jsonify({'success': False, 'error': 'another feedback run is already in progress'}), 409

    fb = _fb_runner()
    conn = fb._fb_conn()
    existing = conn.execute('SELECT COUNT(*) FROM feedback_results WHERE run_id = ?',
                            (run_id,)).fetchone()[0]
    conn.close()
    if existing:
        return jsonify({'success': False,
                        'error': f'run_id "{run_id}" already has {existing} results — pick a new one'}), 400

    manifests = sorted(BENCHMARK_DIR.glob('*/*/manifest.json'))
    manifests = [m for m in manifests if m.parent.name in wanted]
    if not manifests:
        return jsonify({'success': False, 'error': 'no matching modules'}), 400

    state = {'status': 'running', 'total': len(manifests) * len(llms) * reps,
             'done': 0, 'current': '', 'results': [], 'error': None}
    _fb_runs[run_id] = state

    def worker():
        from concurrent.futures import ThreadPoolExecutor, as_completed
        lock = threading.Lock()
        running = set()

        def one(mod_dir, manifest, llm, rep, providers):
            label = f"{manifest['name']} × {llm} rep{rep}"
            with lock:
                running.add(label)
                state['current'] = ' | '.join(sorted(running))
            try:
                msg = fb.run_task(mod_dir, manifest, llm, providers[llm], rep,
                                  run_id, iters, arms)
            finally:
                with lock:
                    running.discard(label)
                    state['current'] = ' | '.join(sorted(running))
            with lock:
                state['done'] += 1
                state['results'].append(msg)

        try:
            providers = {name: fb.rx.make_provider(name) for name in llms}
            tasks = []
            for mf in manifests:
                manifest = json.loads(mf.read_text(encoding='utf-8'))
                for llm in llms:
                    for rep in range(1, reps + 1):
                        tasks.append((mf.parent, manifest, llm, rep))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(one, *t, providers) for t in tasks]
                for f in as_completed(futures):
                    f.result()
            state['status'] = 'finished'
            state['current'] = ''
        except Exception as e:
            state['status'] = 'failed'
            state['error'] = f'{type(e).__name__}: {e}'

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'success': True, 'run_id': run_id, 'total': state['total']})


@app.route('/api/feedback-status')
def feedback_status():
    run_id = request.args.get('run_id', '')
    state = _fb_runs.get(run_id)
    if not state:
        return jsonify({'success': False, 'error': 'unknown run_id'}), 404
    return jsonify({'success': True, 'run_id': run_id, **state})


@app.route('/api/feedback-export')
def feedback_export():
    """导出反馈闭环结果 CSV（Render 磁盘易失，跑完必须下载）"""
    import sqlite3
    import csv as _csv
    import io
    try:
        _fb_runner()._fb_conn().close()  # 确保表存在
        from src.experiment_logger import DB_PATH
        run_id = request.args.get('run_id')
        where, params = ('WHERE run_id = ?', [run_id]) if run_id else ('', [])
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            f'SELECT * FROM feedback_results {where} ORDER BY id', params).fetchall()]
        conn.close()
        buf = io.StringIO()
        if rows:
            w = _csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        tag = secure_filename(run_id) if run_id else 'all'
        return app.response_class(
            buf.getvalue(), mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=feedback_results_{tag}.csv'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/feedback-results')
def feedback_results():
    """反馈闭环结果：run 列表 + 臂×轮次透视 + 明细"""
    import sqlite3
    try:
        fb = _fb_runner()
        fb._fb_conn().close()  # 确保表存在
        from src.experiment_logger import DB_PATH
        run_id = request.args.get('run_id')
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        runs = [dict(r) for r in conn.execute(
            '''SELECT run_id, GROUP_CONCAT(DISTINCT llm) AS llms, COUNT(*) AS n
               FROM feedback_results GROUP BY run_id ORDER BY MAX(id) DESC''').fetchall()]
        where, params = ('WHERE run_id = ?', [run_id]) if run_id else ('', [])
        pivot = [dict(r) for r in conn.execute(f'''
            SELECT arm, iteration, COUNT(*) AS n,
                   AVG(tb_compiled) AS compile_rate,
                   AVG(golden_passed) AS golden_rate,
                   AVG(CASE WHEN golden_passed=1 THEN mutation_score END) AS mutation_score,
                   SUM(error IS NOT NULL) AS errors
            FROM feedback_results {where}
            GROUP BY arm, iteration ORDER BY arm, iteration''', params).fetchall()]
        detail = [dict(r) for r in conn.execute(f'''
            SELECT module, llm, rep, arm, iteration, feedback_kind, converged,
                   scenarios_count, tb_compiled, golden_passed, mutation_score, error
            FROM feedback_results {where}
            ORDER BY module, llm, rep, arm, iteration''', params).fetchall()]
        conn.close()
        return jsonify({'success': True, 'runs': runs, 'pivot': pivot, 'detail': detail})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/benchmark-export')
def benchmark_export():
    """导出实验成绩为 CSV 或 JSONL（论文数据分析用）"""
    import sqlite3
    import csv
    import io
    fmt = request.args.get('format', 'csv')
    run_id = request.args.get('run_id')
    try:
        from src.experiment_logger import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        where, params = ('WHERE run_id = ?', [run_id]) if run_id else ('', [])
        rows = [dict(r) for r in conn.execute(
            f'SELECT * FROM benchmark_results {where} ORDER BY id', params).fetchall()]
        conn.close()
        tag = run_id or 'all'
        if fmt == 'jsonl':
            body = '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)
            mime, fname = 'application/x-ndjson', f'benchmark_results_{tag}.jsonl'
        else:
            buf = io.StringIO()
            if rows:
                w = csv.DictWriter(buf, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            body, mime, fname = buf.getvalue(), 'text/csv', f'benchmark_results_{tag}.csv'
        return Response(body, mimetype=mime,
                        headers={'Content-Disposition': f'attachment; filename={fname}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/benchmark-download-artifacts')
def benchmark_download_artifacts():
    """打包下载某次实验的全部产物（BDD 场景、testbench、仿真日志）"""
    import io
    import zipfile
    run_id = request.args.get('run_id', '')
    run_dir = PROJECT_ROOT / 'output' / 'benchmark_runs' / secure_filename(run_id)
    if not run_id or not run_dir.is_dir():
        return jsonify({'success': False, 'error': f'no artifacts for run "{run_id}"'}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(run_dir.parent))
    buf.seek(0)
    return Response(buf.read(), mimetype='application/zip',
                    headers={'Content-Disposition':
                             f'attachment; filename=benchmark_artifacts_{secure_filename(run_id)}.zip'})


# ============================================================================
# Hardware Generator API (ALU, Counter, etc.)
# ============================================================================
@app.route('/api/generate-hardware', methods=['POST'])
def generate_hardware():
    """Generate hardware Verilog (non-streaming)"""
    data = request.json
    module_type = data.get('module_type', 'alu')
    llm_name = data.get('llm', 'groq')
    model = data.get('model')
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')
    workflow_mode = data.get('workflow_mode', 'implementation')
    bdd_filepath = data.get('bdd_filepath')

    # Parse natural language if provided
    if natural_input:
        parsed = parse_hardware_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']
        if parsed.get('module_type'):
            module_type = parsed['module_type']

    # Load BDD context for specification-first workflow
    bdd_context, bdd_error = _read_bdd_context(data)
    if bdd_error:
        return jsonify({'success': False, 'error': bdd_error}), 400

    sampling = _parse_sampling(data)
    prompt_version = (data.get('prompt_version') or '').strip() or 'v1'
    prompt_override = data.get('prompt_override') or None
    system_override = data.get('system_override') or None
    # impl-first 下 DUV 是依赖链起点，新建 run_id；spec-first 下起点是 BDD，
    # 这里改为继承前端传来的 run_id（详见 _chain_start）。
    # 与流式端点一样支持 model 覆盖（对 MODEL_OVERRIDE_PROVIDERS 生效）
    run_id, _ctx = _web_ctx(data, 'web_duv_generation', module_type,
                            new_run=_chain_start(data, 'duv'), model_used=True)
    if _ctx['extra']['model_override_ignored']:
        print(f"⚠️  model '{_ctx['extra']['model_requested']}' ignored for "
              f"provider '{llm_name}' (override wired only for "
              f"{'/'.join(MODEL_OVERRIDE_PROVIDERS)})")

    print(f"\n{'='*60}")
    print(f"🔧 Generating {bitwidth}-bit {module_type.upper()}")
    print(f"{'='*60}")
    print(f"   LLM: {llm_name.upper()}")
    print(f"   Module: {module_type}")
    print(f"   Bitwidth: {bitwidth}")
    print(f"   Workflow: {workflow_mode}")
    print(f"   Run ID: {run_id}")
    if bdd_context:
        print(f"   BDD Context: {len(bdd_context)} chars loaded")

    try:
        with call_context(**_ctx):
            if module_type == 'alu':
                if not HAS_ALU_MODULE:
                    return jsonify({'success': False, 'error': 'ALU module not available'}), 500

                generator = ALUGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=True
                )
                generator.bdd_context = bdd_context  # BDD-First: pass spec context
                generator.sampling = sampling        # None = 用 provider 默认值
                generator.prompt_version = prompt_version
                generator.prompt_override = prompt_override
                generator.system_override = system_override
                _apply_model_override(generator, llm_name, model)
                hw_path = generator.generate_alu(bitwidth=bitwidth, module_name='alu')

            elif module_type == 'counter':
                if not HAS_COUNTER_MODULE:
                    return jsonify({'success': False, 'error': 'Counter module not available'}), 500

                generator = CounterGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=True
                )
                generator.bdd_context = bdd_context  # BDD-First: pass spec context
                generator.sampling = sampling        # None = 用 provider 默认值
                generator.prompt_version = prompt_version
                generator.prompt_override = prompt_override
                generator.system_override = system_override
                _apply_model_override(generator, llm_name, model)
                hw_path = generator.generate_counter(bitwidth=bitwidth, module_name='counter')

            elif module_type == 'regfile':
                if not HAS_REGFILE_MODULE:
                    return jsonify({'success': False, 'error': 'Register File module not available'}), 500

                depth = data.get('depth', 32)  # Number of registers
                generator = RegFileGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=True
                )
                generator.bdd_context = bdd_context  # BDD-First: pass spec context
                generator.sampling = sampling        # None = 用 provider 默认值
                generator.prompt_version = prompt_version
                generator.prompt_override = prompt_override
                generator.system_override = system_override
                _apply_model_override(generator, llm_name, model)
                hw_path = generator.generate_regfile(bitwidth=bitwidth, depth=depth, module_name='regfile')

            elif module_type == 'cpu':
                if not HAS_CPU_MODULE:
                    return jsonify({'success': False, 'error': 'CPU module not available'}), 500

                pipeline_stages = data.get('pipeline_stages', 5)
                generator = CPUGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=True
                )
                generator.bdd_context = bdd_context  # BDD-First: pass spec context
                generator.sampling = sampling        # None = 用 provider 默认值
                generator.prompt_version = prompt_version
                generator.prompt_override = prompt_override
                generator.system_override = system_override
                _apply_model_override(generator, llm_name, model)
                hw_path = generator.generate_cpu(bitwidth=32, pipeline_stages=pipeline_stages, module_name='riscv_cpu')

            else:
                return jsonify({'success': False, 'error': f'Unknown module type: {module_type}'}), 400

        if not hw_path:
            return jsonify({'success': False, 'error': 'Generation failed'}), 500

        hw_path_obj = Path(hw_path)
        if not hw_path_obj.exists():
            return jsonify({'success': False, 'error': f'File was not created'}), 500

        with open(hw_path, 'r', encoding='utf-8') as f:
            content = f.read()

        last_generated_hw['filename'] = hw_path_obj.name
        last_generated_hw['filepath'] = str(hw_path)
        last_generated_hw['llm'] = llm_name
        last_generated_hw['module_type'] = module_type

        # 只报告不阻断：代码照样保存供查看，但立刻知道这份 DUV 是不是坏的
        duv_compile = _duv_compile_check(hw_path)

        _log_artifact(data, run_id, 'duv', hw_path,
                      meta={'llm': llm_name, 'module_type': module_type,
                            'bitwidth': bitwidth, 'streaming': False,
                            'compiles': (duv_compile or {}).get('ok')})

        return jsonify({
            'success': True,
            'duv_compile': duv_compile,
            'filename': hw_path_obj.name,
            'preview': content[:1000] + ('...' if len(content) > 1000 else ''),
            'full_content': content,
            'llm': llm_name,
            'bitwidth': bitwidth,
            'module_type': module_type,
            'filepath': str(hw_path),
            'run_id': run_id,
            'model_override_ignored': _ctx['extra']['model_override_ignored'],
            'model_requested': _ctx['extra']['model_requested'],
            'call_meta': _last_call_meta(run_id),
        })

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-hardware-stream', methods=['POST'])
def generate_hardware_stream():
    """Generate hardware Verilog with streaming output (SSE)"""
    data = request.json
    module_type = data.get('module_type', 'alu')
    llm_name = data.get('llm', 'groq')
    model = data.get('model')
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')
    workflow_mode = data.get('workflow_mode', 'implementation')

    # Load BDD context for specification-first workflow
    bdd_context, bdd_error = _read_bdd_context(data)
    if bdd_error:
        return jsonify({'success': False, 'error': bdd_error}), 400

    # Parse natural language if provided
    if natural_input:
        parsed = parse_hardware_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']
        if parsed.get('module_type'):
            module_type = parsed['module_type']

    sampling = _parse_sampling(data)
    prompt_version = (data.get('prompt_version') or '').strip() or 'v1'
    prompt_override = data.get('prompt_override') or None
    system_override = data.get('system_override') or None
    # impl-first 下 DUV 是依赖链起点；spec-first 下起点是 BDD，此处继承其 run_id。
    # 此端点（流式）在生成器构造后会替换 generator.llm 来应用 model 覆盖，
    # 对 gemini/openai 生效，故 model_used=True；非流式的同名端点没有这段逻辑。
    run_id, _ctx = _web_ctx(data, 'web_duv_generation', module_type,
                            new_run=_chain_start(data, 'duv'), model_used=True)

    def _prep(gen):
        """把 Prompt 面板的选择交给生成器——_create_*_prompt 从这些属性上取。

        此前这三个值在本端点被解析后就再没用过，于是 Stream 打开时（默认打开）
        改 prompt、切版本全部无声失效，而非流式路径是生效的。
        """
        gen.prompt_version = prompt_version
        gen.prompt_override = prompt_override
        gen.system_override = system_override
        return gen

    def _generate_inner():
        try:
            yield make_sse_message("start", llm=llm_name, bitwidth=bitwidth,
                                   module_type=module_type, run_id=run_id,
                                   model_override_ignored=_ctx['extra']['model_override_ignored'],
                                   model_requested=_ctx['extra']['model_requested'])
            yield make_sse_message("info", message=f"Initializing {bitwidth}-bit {module_type.upper()} generator...")

            # Variable to store module name for later use
            module_name = None

            # Select generator based on module type
            if module_type == 'alu':
                if not HAS_ALU_MODULE:
                    yield make_sse_message("error", message="ALU module not available")
                    return
                generator = _prep(ALUGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                ))
                # 与非流式路径共用同一张表。这里原本另抄了一份，两处一旦漂移，
                # 开不开 Stream 就会生成操作集不同的硬件。
                operations = dict(DEFAULT_ALU_OPERATIONS)
                module_name = f"alu_{bitwidth}bit"
                prompt = generator._create_alu_prompt(bitwidth, operations, module_name)

            elif module_type == 'counter':
                if not HAS_COUNTER_MODULE:
                    yield make_sse_message("error", message="Counter module not available")
                    return
                generator = _prep(CounterGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                ))
                modes = ['up', 'down', 'updown']
                module_name = f"counter_{bitwidth}bit"
                prompt = generator._create_counter_prompt(bitwidth, modes, module_name)

            elif module_type == 'regfile':
                if not HAS_REGFILE_MODULE:
                    yield make_sse_message("error", message="Register File module not available")
                    return
                depth = data.get('depth', 32)
                generator = _prep(RegFileGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                ))
                module_name = f"regfile_{bitwidth}bit"
                prompt = generator._create_regfile_prompt(bitwidth, depth, module_name)

            elif module_type == 'cpu':
                if not HAS_CPU_MODULE:
                    yield make_sse_message("error", message="CPU module not available")
                    return
                pipeline_stages = data.get('pipeline_stages', 5)
                generator = _prep(CPUGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                ))
                module_name = "riscv_cpu"
                prompt = generator._create_cpu_prompt(32, pipeline_stages, module_name)

            else:
                yield make_sse_message("error", message=f"Unknown module type: {module_type}")
                return

            # BDD-First: append BDD specification context to prompt
            # 与 4 个 generator 走同一份文本，避免同一功能因 Stream 开关而 prompt 不同
            if bdd_context:
                prompt += prompt_store.bdd_context_block(bdd_context)
                yield make_sse_message("info", message="📋 BDD spec context injected into prompt...")

            # Override model if specified (for Gemini/OpenAI model selection)
            _apply_model_override(generator, llm_name, model)

            yield make_sse_message("info", message=f"Calling {llm_name.upper()} API...")

            # Get LLM and stream
            llm = generator.llm
            full_content = ""

            # Dynamic token budget based on module complexity
            if module_type == 'cpu':
                max_tokens = 20000
            elif module_type == 'regfile':
                max_tokens = 8000
            elif module_type == 'alu':
                max_tokens = 5000 + (bitwidth // 16) * 1000
            else:
                max_tokens = 6000 + (bitwidth // 16) * 500

            # 模板（或用户覆盖）渲染出的 system prompt 由 _create_*_prompt 暂存在
            # 生成器上。此前流式路径没有取用，于是同一个 stage 在 Stream 开/关时
            # 发出的 system prompt 并不相同，且面板里的 system 编辑框无效。
            system_prompt = (getattr(generator, '_rendered_system', None)
                             or "You are an expert Verilog hardware designer.")

            if hasattr(llm, '_call_api_stream'):
                for chunk in llm._call_api_stream(prompt, max_tokens=max_tokens,
                                                  system_prompt=system_prompt,
                                                  sampling=sampling):
                    if chunk:
                        full_content += chunk
                        yield make_sse_message("chunk", content=chunk)
            else:
                yield make_sse_message("info", message="Using standard mode...")
                response = llm._call_api(
                    prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )
                if response:
                    full_content = response
                    chunk_size = 100
                    for i in range(0, len(full_content), chunk_size):
                        chunk = full_content[i:i+chunk_size]
                        yield make_sse_message("chunk", content=chunk)

            if not full_content:
                yield make_sse_message("error", message="LLM returned empty response")
                return

            # Extract verilog code
            verilog_code = generator._extract_verilog(full_content)
            if not verilog_code:
                verilog_code = full_content

            # Guard: LLM API 失败时会降级返回模板文本，绝不能把非 Verilog 内容存成 .v
            if 'module' not in verilog_code:
                yield make_sse_message(
                    "error",
                    message=f"{llm_name.upper()} did not return Verilog code — the API call "
                            f"likely failed (check server console for proxy/API-key errors). "
                            f"Response preview: {full_content[:200]}")
                return

            # Fix module name and save based on module type
            if module_type == 'alu':
                verilog_code = generator._fix_module_name(verilog_code, module_name)
                hw_path = generator._save_alu(verilog_code, module_name, bitwidth)
            elif module_type == 'counter':
                if hasattr(generator, '_fix_module_name'):
                    verilog_code = generator._fix_module_name(verilog_code, module_name)
                hw_path = generator._save_counter(verilog_code, module_name, bitwidth, ['up', 'down', 'updown'])
            elif module_type == 'regfile':
                depth = data.get('depth', 32)
                if hasattr(generator, '_fix_module_name'):
                    verilog_code = generator._fix_module_name(verilog_code, module_name)
                hw_path = generator._save_regfile(verilog_code, module_name, bitwidth, depth)
            elif module_type == 'cpu':
                pipeline_stages = data.get('pipeline_stages', 5)
                if hasattr(generator, '_fix_module_name'):
                    verilog_code = generator._fix_module_name(verilog_code, module_name)
                hw_path = generator._save_cpu(verilog_code, module_name, 32, pipeline_stages)

            filename = Path(hw_path).name

            last_generated_hw['filename'] = filename
            last_generated_hw['filepath'] = str(hw_path)
            last_generated_hw['llm'] = llm_name
            last_generated_hw['module_type'] = module_type

            duv_compile = _duv_compile_check(hw_path)

            _log_artifact(data, run_id, 'duv', hw_path,
                          meta={'llm': llm_name, 'module_type': module_type,
                                'bitwidth': bitwidth, 'streaming': True,
                                'compiles': (duv_compile or {}).get('ok')})

            # 到这里流式生成器已耗尽，其 finally 已写入 llm_calls，可安全回读
            yield make_sse_message("complete", filename=filename, filepath=str(hw_path),
                                   duv_compile=duv_compile,
                                   call_meta=_last_call_meta(run_id))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield make_sse_message("error", message=str(e))

    def generate():
        # call_context 必须在生成器体内。SSE 响应是惰性的：路由函数返回时
        # 上下文早已退出，真正执行发生在 Flask 消费 body 时。用 yield from
        # 让上下文在整个迭代期间保持有效（生成器挂起在 with 内部）。
        with call_context(**_ctx):
            yield from _generate_inner()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# ============================================================================
# Yosys Analysis API
# ============================================================================
@app.route('/api/analyze-dut', methods=['POST'])
def analyze_dut():
    """Analyze DUT with Yosys: synthesize, get stats, generate circuit diagram."""
    try:
        from yosys_analyzer import YosysAnalyzer

        data = request.json
        filename = data.get('filename')
        llm_name = data.get('llm', 'default')
        module_name = data.get('module_name')

        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400

        # Find the DUT file
        dut_path = None

        # Check LLM-specific directory first
        candidate = PROJECT_ROOT / 'output' / 'dut' / llm_name / filename
        if candidate.exists():
            dut_path = candidate

        # Check uploaded directory
        if not dut_path:
            candidate = PROJECT_ROOT / 'output' / 'dut' / 'uploaded' / filename
            if candidate.exists():
                dut_path = candidate

        # Search all DUT subdirectories
        if not dut_path:
            dut_base = PROJECT_ROOT / 'output' / 'dut'
            if dut_base.exists():
                for sub in dut_base.iterdir():
                    if sub.is_dir():
                        candidate = sub / filename
                        if candidate.exists():
                            dut_path = candidate
                            break

        if not dut_path:
            return jsonify({'success': False, 'error': f'DUT file not found: {filename}'}), 404

        print(f"\n{'='*60}")
        print(f"🔬 Yosys Analysis: {dut_path.name}")
        print(f"{'='*60}")

        analyzer = YosysAnalyzer(project_root=str(PROJECT_ROOT))
        result = analyzer.analyze(str(dut_path), module_name=module_name)

        if result['success']:
            print(f"   ✅ Synthesis successful")
            print(f"   📊 Cells: {result['stats'].get('cell_total', 0)}")
            print(f"   📊 Wires: {result['stats'].get('wires', 0)}")

        return jsonify(result)

    except ImportError:
        return jsonify({
            'success': False,
            'error': 'Yosys analyzer module not available'
        }), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-yosys/<path:filepath>')
def download_yosys(filepath):
    """Download Yosys output file (DOT, SVG, JSON)."""
    import re as _re
    # Sanitize: only allow files under output/yosys/
    if '..' in filepath or not _re.match(r'^[\w\-/\\.]+\.(dot|svg|json|txt)$', filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    full_path = PROJECT_ROOT / 'output' / 'yosys' / filepath
    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(
        str(full_path.parent.absolute()), full_path.name,
        as_attachment=True, download_name=full_path.name
    )


@app.route('/api/download-hardware/<filename>')
def download_hardware(filename):
    """Download generated hardware file"""
    print(f"\n📥 Hardware Download request: {filename}")

    file_path = _resolve_output_file('dut', filename, last_generated_hw['filepath'])
    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True,
        download_name=filename
    )


def parse_hardware_natural_language(input_text):
    """Parse natural language input for hardware generation"""
    import re
    input_lower = input_text.lower()

    result = {'bitwidth': 16, 'llm': None, 'module_type': None}

    # Parse bitwidth
    bitwidth_patterns = [
        (r'(\d+)\s*-?\s*bit', lambda m: int(m.group(1))),
        (r'(\d+)\s*位', lambda m: int(m.group(1))),
    ]

    for pattern, extract_func in bitwidth_patterns:
        match = re.search(pattern, input_lower)
        if match:
            bw = extract_func(match)
            if bw in [8, 16, 32, 64]:
                result['bitwidth'] = bw
            break

    # Parse LLM provider
    llm_keywords = {
        'groq': ['groq'],
        'deepseek': ['deepseek', 'deep seek'],
        'openai': ['openai', 'gpt', 'chatgpt'],
        'claude': ['claude', 'anthropic'],
        'gemini': ['gemini', 'google'],
        'grok': ['grok', 'xai'],
        'qwen': ['qwen', 'tongyi', 'alibaba'],
        'mistral': ['mistral', 'codestral'],
        'together': ['together'],
    }

    for provider, keywords in llm_keywords.items():
        for keyword in keywords:
            if keyword in input_lower:
                result['llm'] = provider
                break

    # Parse module type
    module_keywords = {
        'alu': ['alu', 'arithmetic', 'logic unit'],
        'counter': ['counter', 'count', '计数器'],
        'regfile': ['register file', 'regfile', 'register bank', '寄存器'],
        'cpu': ['cpu', 'processor', '处理器'],
    }

    for module, keywords in module_keywords.items():
        for keyword in keywords:
            if keyword in input_lower:
                result['module_type'] = module
                break

    return result


# ============================================================================
# Legacy ALU API (backward compatibility)
# ============================================================================
# ============================================================================
# BDD Generator API
# ============================================================================
@app.route('/api/generate-stream', methods=['POST'])
def generate_bdd_stream():
    """Generate BDD Feature file with streaming output (SSE)"""
    if not HAS_BDD_MODULE:
        return jsonify({'success': False, 'error': 'BDD Generator module not available'}), 500

    data = request.json
    llm_name = data.get('llm', 'groq')
    model = data.get('model')
    user_input = data.get('input', '')

    if not user_input:
        return jsonify({'success': False, 'error': 'Please enter your requirements'}), 400

    sampling = _parse_sampling(data)
    prompt_version = (data.get('prompt_version') or '').strip() or 'v1'
    prompt_override = data.get('prompt_override') or None
    system_override = data.get('system_override') or None
    # impl-first 下 BDD 继承 DUV 的 run_id；spec-first 下 BDD 才是依赖链起点，
    # 此时新建（详见 _chain_start）。此端点确实会对 openai/gemini 应用 model
    # 覆盖，故 model_used=True
    run_id, _ctx = _web_ctx(data, 'web_bdd_generation',
                            (data.get('module_type') or '').strip() or None,
                            new_run=_chain_start(data, 'bdd'), model_used=True)

    def _generate_inner():
        try:
            yield make_sse_message("start", llm=llm_name, run_id=run_id,
                                   model_override_ignored=_ctx['extra']['model_override_ignored'],
                                   model_requested=_ctx['extra']['model_requested'])

            generator = FeatureGeneratorLLM(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=False
            )
            generator.sampling = sampling   # None = 用 provider 默认值
            generator.prompt_version = prompt_version
            generator.prompt_override = prompt_override
            generator.system_override = system_override

            if model and llm_name in ('openai', 'gemini'):
                try:
                    llm = LLMFactory.create_provider(llm_name, model=model)
                    generator.llm = llm
                    print(f"🔷 [{llm_name.upper()}] Model overridden to: {model}")
                except Exception as e:
                    yield make_sse_message("error", message=str(e))
                    return

            requirements = generator.parse_user_input(user_input)
            bitwidth = requirements.get("bitwidth", "?")
            ops_count = len(requirements.get("operations", []))

            yield make_sse_message("info", message=f"Parsed: {bitwidth}-bit ALU with {ops_count} operations")

            prompt = generator._create_prompt(requirements)
            yield make_sse_message("info", message="Calling LLM API...")

            llm = generator.llm
            full_content = ""

            # Match the non-streaming branch below, which goes through
            # FeatureGeneratorLLM._call_llm() with a hard-coded 4000.
            # Without this, each provider fell back to its own _call_api_stream
            # default: 4000 for eight of them, but 8192 for Gemini.
            max_tokens = 4000

            if hasattr(llm, '_call_api_stream'):
                for chunk in llm._call_api_stream(prompt, max_tokens=max_tokens,
                                                  sampling=sampling):
                    if chunk:
                        full_content += chunk
                        yield make_sse_message("chunk", content=chunk)
            else:
                yield make_sse_message("info", message="Using standard mode...")
                full_content = generator._call_llm(prompt)
                if full_content:
                    for i in range(0, len(full_content), 50):
                        yield make_sse_message("chunk", content=full_content[i:i+50])

            if not full_content:
                yield make_sse_message("error", message="LLM returned empty response")
                return

            full_content = generator._clean_response(full_content)
            feature_path = generator._save_feature(full_content, requirements)
            filename = Path(feature_path).name

            last_generated_bdd['filename'] = filename
            last_generated_bdd['filepath'] = str(feature_path)
            last_generated_bdd['llm'] = llm_name

            _log_artifact(data, run_id, 'bdd', feature_path,
                          meta={'llm': llm_name, 'streaming': True})

            yield make_sse_message("complete", filename=filename,
                                   filepath=str(feature_path), run_id=run_id,
                                   call_meta=_last_call_meta(run_id))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield make_sse_message("error", message=str(e))

    def generate():
        # 同 Step 1 流式端点：上下文必须包在生成器体内，否则 run_id 会静默丢失
        with call_context(**_ctx):
            yield from _generate_inner()

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'
    })


@app.route('/api/generate', methods=['POST'])
def generate_bdd():
    """Generate BDD Feature file (non-streaming)"""
    if not HAS_BDD_MODULE:
        return jsonify({'success': False, 'error': 'BDD Generator module not available'}), 500

    try:
        data = request.json
        llm_name = data.get('llm', 'groq')
        model = data.get('model')
        user_input = data.get('input', '')

        if not user_input:
            return jsonify({'success': False, 'error': 'Please enter your requirements'}), 400

        sampling = _parse_sampling(data)
        prompt_version = (data.get('prompt_version') or '').strip() or 'v1'
        prompt_override = data.get('prompt_override') or None
        system_override = data.get('system_override') or None
        # impl-first 下 BDD 继承 DUV 的 run_id；spec-first 下 BDD 是链起点，新建
        run_id, _ctx = _web_ctx(data, 'web_bdd_generation',
                                (data.get('module_type') or '').strip() or None,
                                new_run=_chain_start(data, 'bdd'), model_used=True)

        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )
        generator.sampling = sampling   # None = 用 provider 默认值
        generator.prompt_version = prompt_version
        generator.prompt_override = prompt_override
        generator.system_override = system_override

        if model and llm_name in ('openai', 'gemini'):
            try:
                llm = LLMFactory.create_provider(llm_name, model=model)
                generator.llm = llm
                print(f"🔷 [{llm_name.upper()}] Model overridden to: {model}")
            except Exception as e:
                print(f"⚠️  Failed to override model: {e}")
        elif _ctx['extra']['model_override_ignored']:
            print(f"⚠️  model '{model}' ignored for provider '{llm_name}' "
                  f"(override only wired for {'/'.join(MODEL_OVERRIDE_PROVIDERS)})")

        with call_context(**_ctx):
            feature_path = generator.generate_feature(user_input)

        if not feature_path:
            return jsonify({'success': False, 'error': 'Generation failed'}), 500

        feature_path_obj = Path(feature_path)
        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()

        last_generated_bdd['filename'] = feature_path_obj.name
        last_generated_bdd['filepath'] = str(feature_path)
        last_generated_bdd['llm'] = llm_name

        _log_artifact(data, run_id, 'bdd', feature_path,
                      meta={'llm': llm_name, 'streaming': False})

        return jsonify({
            'success': True,
            'filename': feature_path_obj.name,
            'preview': content[:500] + ('...' if len(content) > 500 else ''),
            'full_content': content,
            'llm': llm_name,
            'model': model or llm_name,
            'filepath': str(feature_path),
            'run_id': run_id,
            'model_override_ignored': _ctx['extra']['model_override_ignored'],
            'model_requested': _ctx['extra']['model_requested'],
            'call_meta': _last_call_meta(run_id),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_bdd(filename):
    """Download generated BDD file"""
    file_path = _resolve_output_file('bdd', filename, last_generated_bdd['filepath'])
    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(str(file_path.parent.absolute()), file_path.name, as_attachment=True, download_name=filename)


@app.route('/api/prompts/<stage>')
def get_prompt_templates(stage):
    """某 stage 的可用 prompt 模板版本与内容。

    stage 取值见 prompts/ 目录：duv_alu / duv_counter / duv_register /
    duv_cpu / bdd_alu / bdd_counter / bdd_regfile / bdd_cpu
    """
    try:
        from src import prompt_store as ps
    except ImportError:
        import prompt_store as ps

    versions = ps.available(stage)
    if not versions:
        return jsonify({'success': False, 'error': f'unknown stage: {stage}'}), 404

    # available() 只是 glob 目录，缺 PyYAML 时照样列得出版本，但 load() 全部
    # 返回 None——面板于是显示一个有版本、却没有内容的空壳。把原因说出来。
    if not ps.TEMPLATES_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'PyYAML is not installed on the server, so prompt '
                     'templates cannot be read. Generation still works using '
                     'each generator\'s built-in prompt, but template version '
                     'selection and prompt overrides have no effect. '
                     'Fix: pip install PyYAML',
            'stage': stage, 'versions': versions,
        }), 503

    out = {}
    for v in versions:
        doc = ps.load(stage, v) or {}
        out[v] = {
            'system': doc.get('system'),
            'user': doc.get('user'),
            'variables': doc.get('variables') or [],
            'source': doc.get('source'),
            'user_sha256': ps.sha256(doc.get('user')),
            'system_sha256': ps.sha256(doc.get('system')),
        }
    return jsonify({'success': True, 'stage': stage,
                    'versions': versions, 'default': 'v1', 'templates': out})


@app.route('/api/prompts')
def list_prompt_stages():
    """列出全部 stage 及其版本，供前端把 Step/模块类型映射到 stage。"""
    try:
        from src import prompt_store as ps
    except ImportError:
        import prompt_store as ps
    stages = {}
    if ps.PROMPTS_DIR.is_dir():
        for d in sorted(p for p in ps.PROMPTS_DIR.iterdir() if p.is_dir()):
            vs = ps.available(d.name)
            if vs:
                stages[d.name] = vs
    return jsonify({'success': True, 'stages': stages})


@app.route('/api/sampling-info')
def sampling_info():
    """各 provider 支持哪些采样参数、默认值是多少。

    前端据此决定控件置灰与 placeholder。从 provider 类直接读取，
    避免在 JS 里另抄一份会与后端漂移的表。
    """
    info = {}
    for name in ('groq', 'gemini', 'deepseek', 'openai', 'claude',
                 'grok', 'qwen', 'mistral', 'together'):
        cls = _provider_class(name)
        if cls is None:
            continue
        defaults = getattr(cls, 'DEFAULT_SAMPLING', {}) or {}
        # Gemini 非流式路径叫 rest，其余叫 default——统一成前端易用的两个键
        nonstream = defaults.get('default') or defaults.get('rest') or {}
        stream = defaults.get('stream') or {}
        info[name] = {
            'supported': sorted(getattr(cls, 'SUPPORTED_SAMPLING', set())),
            'defaults': {'nonstream': nonstream, 'stream': stream},
            'field_map': getattr(cls, 'SAMPLING_FIELD_MAP', {}),
            'ranges': getattr(cls, 'SAMPLING_RANGES', {}),
        }
    # OpenAI 的 GPT-5 系列强制 temperature=1，前端需据此置灰
    info.setdefault('openai', {})['forced_note'] = (
        'GPT-5 series forces temperature=1; the value cannot be overridden')
    return jsonify({'success': True, 'providers': info})


def _provider_class(name):
    """按名字拿到 provider 类（不实例化，因此不需要 api_key）。"""
    try:
        import llm_providers as lp
    except ImportError:
        import src.llm_providers as lp
    mapping = {
        'groq': 'GroqProvider', 'gemini': 'GeminiProvider',
        'deepseek': 'DeepSeekProvider', 'openai': 'OpenAIProvider',
        'claude': 'ClaudeProvider', 'grok': 'GrokProvider',
        'qwen': 'QwenProvider', 'mistral': 'MistralProvider',
        'together': 'TogetherProvider',
    }
    return getattr(lp, mapping.get(name, ''), None)


# ============================================================================
# Testbench Generator API
# ============================================================================
@app.route('/api/generate-testbench', methods=['POST'])
def generate_testbench():
    """Generate Verilog testbench from BDD file"""
    if not HAS_TESTBENCH_MODULE:
        return jsonify({'success': False, 'error': 'Testbench Generator module not available'}), 500

    try:
        data = request.json
        bdd_filepath = data.get('bdd_filepath')
        dut_info = data.get('dut_info', {})

        if not bdd_filepath:
            return jsonify({'success': False, 'error': 'No BDD file specified'}), 400

        # Verify BDD file exists
        bdd_path = Path(bdd_filepath)
        if not bdd_path.is_absolute():
            bdd_path = PROJECT_ROOT / bdd_filepath

        if not bdd_path.exists():
            return jsonify({'success': False, 'error': f'BDD file not found: {bdd_filepath}'}), 404

        # DUV 文件是模块名的唯一事实来源。调用方以前自己拼
        # f"{type}_{bitwidth}bit"，但 Step 1 生成的是 `module alu`，两者对不上时
        # testbench 会实例化一个不存在的模块，仿真必然 elaboration 失败。
        derived = _module_name_from_verilog(data.get('dut_filepath'))
        if derived:
            given = dut_info.get('module_name')
            if given and given != derived:
                print(f"⚠️  module_name '{given}' from request overridden by "
                      f"'{derived}' read from the DUV file")
            dut_info['module_name'] = derived

        # Initialize generator
        generator = TestbenchGenerator(
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        # Generate testbench
        # oracle 的来源是受控变量：'bdd'（默认）用 BDD 写的期望值，
        # 'spec' 让生成器按规格重算，只保留 BDD 提供的激励。
        oracle_source = (data.get('oracle_source') or 'bdd').strip().lower()
        if oracle_source not in ('bdd', 'spec'):
            return jsonify({'success': False,
                            'error': f"oracle_source must be 'bdd' or 'spec', "
                                     f"got {oracle_source!r}"}), 400

        result = generator.generate_single(
            bdd_filepath=str(bdd_path),
            dut_info=dut_info,
            oracle_source=oracle_source
        )

        if not result['success']:
            return jsonify(result), 500

        # Store last generated info
        last_generated_tb['filename'] = result['filename']
        last_generated_tb['filepath'] = result['filepath']
        last_generated_tb['bdd_source'] = bdd_filepath

        # Step 3 没有 LLM 参与，llm_calls 里不会留下任何行；产物挂到链上是
        # 「这条 run 的 testbench 是哪个文件」的唯一记录来源。
        run_id = (data.get('run_id') or '').strip() or None
        _log_artifact(data, run_id, 'testbench', result['filepath'],
                      meta={'bdd_source': bdd_filepath,
                            'module_name': dut_info.get('module_name'),
                            'test_count': result.get('test_count'),
                            'generator': result.get('generator'),
                            'source_bdd_llm': result.get('source_bdd_llm'),
                            # 决定这份 testbench 衡量的是 BDD 整体质量还是仅激励
                            'oracle_source': result.get('oracle_source'),
                            # BDD 里有多少内容工具读不懂——本身就是质量信号
                            'parse_stats': result.get('parse_stats')})

        return jsonify({
            'success': True,
            'run_id': run_id,
            'filename': result['filename'],
            'filepath': result['filepath'],
            'content': result['content'],
            'full_content': result['full_content'],
            'quality_summary': result['quality_summary'],
            'test_count': result['test_count'],
            # Step 3 无 LLM 参与，只回传 BDD 的来源与生成方式
            'source_bdd_llm': result.get('source_bdd_llm'),
            'generator': result.get('generator', 'deterministic-template'),
            'oracle_source': result.get('oracle_source'),
            'parse_stats': result.get('parse_stats'),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-testbench-batch', methods=['POST'])
def generate_testbench_batch():
    """Batch generate testbenches for all BDD files"""
    if not HAS_TESTBENCH_MODULE:
        return jsonify({'success': False, 'error': 'Testbench Generator module not available'}), 500

    try:
        data = request.json
        dut_info = data.get('dut_info', {})

        # Initialize generator
        generator = TestbenchGenerator(
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        # Batch generate
        generated_by_llm = generator.generate_all()

        if not generated_by_llm:
            return jsonify({
                'success': False,
                'error': 'No .feature files found in output/bdd/'
            }), 404

        # Prepare response
        results = []
        total_files = 0
        for llm_name, files in generated_by_llm.items():
            total_files += len(files)
            results.append({
                'llm': llm_name,
                'count': len(files),
                'files': [f.name for f in files]
            })

        return jsonify({
            'success': True,
            'total_files': total_files,
            'llm_count': len(generated_by_llm),
            'results': results,
            'output_dir': str(PROJECT_ROOT / 'output' / 'testbench'),
            'quality_report_dir': str(PROJECT_ROOT / 'output' / 'quality_reports')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-testbench-zip')
def download_testbench_zip():
    """Download all testbench files as ZIP"""
    try:
        testbench_dir = PROJECT_ROOT / 'output' / 'testbench'

        if not testbench_dir.exists():
            return jsonify({'error': 'Testbench directory not found'}), 404

        # Create ZIP in memory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in testbench_dir.rglob('*.v'):
                # Get relative path for ZIP structure
                arcname = file_path.relative_to(testbench_dir)
                zf.write(file_path, arcname)

        memory_file.seek(0)

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'testbenches_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-quality-zip')
def download_quality_zip():
    """Download all quality reports as ZIP"""
    try:
        quality_dir = PROJECT_ROOT / 'output' / 'quality_reports'

        if not quality_dir.exists():
            return jsonify({'error': 'Quality reports directory not found'}), 404

        # Create ZIP in memory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in quality_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(quality_dir)
                    zf.write(file_path, arcname)

        memory_file.seek(0)

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'quality_reports_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quality-comparison')
def get_quality_comparison():
    """Get quality comparison data for display"""
    try:
        quality_dir = PROJECT_ROOT / 'output' / 'quality_reports'
        comparison_file = quality_dir / 'quality_comparison.txt'

        if not comparison_file.exists():
            return jsonify({'success': False, 'error': 'No comparison report found'}), 404

        # Parse comparison file
        with open(comparison_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract data from the report
        results = []
        lines = content.split('\n')

        in_scores_section = False
        for line in lines:
            if 'Overall Quality Scores' in line:
                in_scores_section = True
                continue
            if in_scores_section and line.startswith('-'):
                continue
            if in_scores_section and line.strip() == '':
                in_scores_section = False
                continue

            if in_scores_section and line.strip():
                # Parse line: "groq           4            79.8%   85.0%  72.0%"
                parts = line.split()
                if len(parts) >= 5 and parts[0] not in ['LLM', '=']:
                    try:
                        llm_name = parts[0]
                        count = int(parts[1])
                        avg_score = float(parts[2].replace('%', ''))
                        best_score = float(parts[3].replace('%', ''))
                        worst_score = float(parts[4].replace('%', ''))

                        results.append({
                            'llm': llm_name,
                            'count': count,
                            'avg_score': avg_score,
                            'best_score': best_score,
                            'worst_score': worst_score
                        })
                    except (ValueError, IndexError):
                        continue

        # Sort by average score (descending)
        results.sort(key=lambda x: x['avg_score'], reverse=True)

        # Add rank
        for i, result in enumerate(results):
            result['rank'] = i + 1

        return jsonify({
            'success': True,
            'results': results,
            'raw_content': content
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-testbench/<filename>')
def download_testbench(filename):
    """Download generated testbench file"""
    file_path = _resolve_output_file('testbench', filename, last_generated_tb['filepath'])
    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True,
        download_name=filename
    )


# ============================================================================
# Simulation API
# ============================================================================
@app.route('/api/check-simulation-tools')
def check_simulation_tools():
    """Check if simulation tools are available"""
    if simulation_runner:
        return jsonify(simulation_runner.get_tools_status())
    return jsonify({
        'can_simulate': False,
        'tools': {'iverilog': False, 'vvp': False},
        'error': 'Simulation module not loaded'
    })


@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """Run simulation for a single testbench"""
    if not simulation_runner or not simulation_runner.can_run_simulation():
        return jsonify({
            'success': False,
            'error': 'Simulation tools not available on server',
            'tools_available': simulation_runner.get_tools_status() if simulation_runner else {}
        }), 503

    try:
        data = request.json
        testbench_path = data.get('testbench_path')
        dut_path = data.get('dut_path')

        if not testbench_path or not dut_path:
            return jsonify({'success': False, 'error': 'Missing testbench_path or dut_path'}), 400

        # Convert relative paths to absolute
        tb_full = PROJECT_ROOT / testbench_path
        dut_full = PROJECT_ROOT / dut_path

        result = simulation_runner.run_single(str(tb_full), str(dut_full))

        # Step 4 同样没有 LLM 参与。仿真结果本身不落文件，记的是结论。
        run_id = (data.get('run_id') or '').strip() or None
        _log_artifact(data, run_id, 'simulation', testbench_path,
                      meta={'dut_path': dut_path,
                            'success': bool(result.get('success')),
                            'pass_rate': result.get('pass_rate'),
                            'total_tests': result.get('total_tests'),
                            'passed_tests': result.get('passed_tests')})

        return jsonify({**result, 'run_id': run_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run-simulation-batch', methods=['POST'])
def run_simulation_batch():
    """Run simulations for all testbenches"""
    if not simulation_runner or not simulation_runner.can_run_simulation():
        return jsonify({
            'success': False,
            'error': 'Simulation tools not available on server'
        }), 503

    try:
        result = simulation_runner.run_batch()
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# @app.route('/api/download-vcd/<path:filepath>')
# def download_vcd(filepath):
#     """Download VCD file"""
#     file_path = PROJECT_ROOT / filepath
#     if not file_path.exists():
#         return jsonify({'error': 'File not found'}), 404
#
#     return send_from_directory(
#         str(file_path.parent.absolute()),
#         file_path.name,
#         as_attachment=True
#     )
@app.route('/api/download-vcd/<path:filepath>')
def download_vcd(filepath):
    """Download VCD file"""
    # Fix Windows path separators
    filepath = filepath.replace('\\', '/')

    file_path = PROJECT_ROOT / filepath

    if not file_path.exists():
        return jsonify({'error': f'File not found: {filepath}'}), 404

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True,
        mimetype='application/x-vcd'  # VCD mimetype for GTKWave association
    )

@app.route('/api/download-all-waveforms')
def download_all_waveforms():
    """Download all VCD files from all LLMs as a zip"""
    waveform_dir = PROJECT_ROOT / 'output' / 'waveform'

    if not waveform_dir.exists():
        return jsonify({'error': 'No waveform directory found'}), 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for llm_dir in waveform_dir.iterdir():
            if llm_dir.is_dir():
                for vcd_file in llm_dir.glob('*.vcd'):
                    # Store as llm_name/filename.vcd
                    zf.write(vcd_file, f'{llm_dir.name}/{vcd_file.name}')

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='waveforms_all.zip'
    )


@app.route('/api/download-simulation-log/<path:filepath>')
def download_simulation_log(filepath):
    """Download simulation log file"""
    file_path = PROJECT_ROOT / filepath
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True
    )
# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print()
    print("=" * 60)
    print("🤖 LLM Hardware Generator")
    print("=" * 60)
    print(f"📁 Project: {PROJECT_ROOT}")
    print(f"📡 Server: http://localhost:{port}")
    print()
    print("📋 Available Modules:")
    print(f"   {'✅' if HAS_ALU_MODULE else '❌'} ALU Generator")
    print(f"   {'✅' if HAS_COUNTER_MODULE else '❌'} Counter Generator")
    print(f"   {'✅' if HAS_REGFILE_MODULE else '❌'} Register File Generator")
    print(f"   {'✅' if HAS_CPU_MODULE else '❌'} RISC-V CPU Generator")
    print(f"   {'✅' if HAS_BDD_MODULE else '❌'} BDD Generator")
    print(f"   {'✅' if HAS_TESTBENCH_MODULE else '❌'} Testbench Generator")
    print()
    print("=" * 60)
    print()

    app.run(debug=debug, host='0.0.0.0', port=port)