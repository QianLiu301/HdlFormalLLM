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

PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

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
    from alu_generator import ALUGenerator
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
# Hardware Generator API (ALU, Counter, etc.)
# ============================================================================
@app.route('/api/generate-hardware', methods=['POST'])
def generate_hardware():
    """Generate hardware Verilog (non-streaming)"""
    data = request.json
    module_type = data.get('module_type', 'alu')
    llm_name = data.get('llm', 'groq')
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')

    # Parse natural language if provided
    if natural_input:
        parsed = parse_hardware_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']
        if parsed.get('module_type'):
            module_type = parsed['module_type']

    print(f"\n{'='*60}")
    print(f"🔧 Generating {bitwidth}-bit {module_type.upper()}")
    print(f"{'='*60}")
    print(f"   LLM: {llm_name.upper()}")
    print(f"   Module: {module_type}")
    print(f"   Bitwidth: {bitwidth}")

    try:
        if module_type == 'alu':
            if not HAS_ALU_MODULE:
                return jsonify({'success': False, 'error': 'ALU module not available'}), 500

            generator = ALUGenerator(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=True
            )
            hw_path = generator.generate_alu(bitwidth=bitwidth, module_name='alu')

        elif module_type == 'counter':
            if not HAS_COUNTER_MODULE:
                return jsonify({'success': False, 'error': 'Counter module not available'}), 500

            generator = CounterGenerator(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=True
            )
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

        return jsonify({
            'success': True,
            'filename': hw_path_obj.name,
            'preview': content[:1000] + ('...' if len(content) > 1000 else ''),
            'full_content': content,
            'llm': llm_name,
            'bitwidth': bitwidth,
            'module_type': module_type,
            'filepath': str(hw_path)
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
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')

    # Parse natural language if provided
    if natural_input:
        parsed = parse_hardware_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']
        if parsed.get('module_type'):
            module_type = parsed['module_type']

    def generate():
        try:
            yield make_sse_message("start", llm=llm_name, bitwidth=bitwidth, module_type=module_type)
            yield make_sse_message("info", message=f"Initializing {bitwidth}-bit {module_type.upper()} generator...")

            # Variable to store module name for later use
            module_name = None

            # Select generator based on module type
            if module_type == 'alu':
                if not HAS_ALU_MODULE:
                    yield make_sse_message("error", message="ALU module not available")
                    return
                generator = ALUGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                )
                operations = {
                    "ADD": {"opcode": "0000", "description": "Addition (A + B)"},
                    "SUB": {"opcode": "0001", "description": "Subtraction (A - B)"},
                    "AND": {"opcode": "0010", "description": "Bitwise AND (A & B)"},
                    "OR": {"opcode": "0011", "description": "Bitwise OR (A | B)"},
                }
                module_name = f"alu_{bitwidth}bit"
                prompt = generator._create_alu_prompt(bitwidth, operations, module_name)

            elif module_type == 'counter':
                if not HAS_COUNTER_MODULE:
                    yield make_sse_message("error", message="Counter module not available")
                    return
                generator = CounterGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                )
                modes = ['up', 'down', 'updown']
                module_name = f"counter_{bitwidth}bit"
                prompt = generator._create_counter_prompt(bitwidth, modes, module_name)

            elif module_type == 'regfile':
                if not HAS_REGFILE_MODULE:
                    yield make_sse_message("error", message="Register File module not available")
                    return
                depth = data.get('depth', 32)
                generator = RegFileGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                )
                module_name = f"regfile_{bitwidth}bit"
                prompt = generator._create_regfile_prompt(bitwidth, depth, module_name)

            elif module_type == 'cpu':
                if not HAS_CPU_MODULE:
                    yield make_sse_message("error", message="CPU module not available")
                    return
                pipeline_stages = data.get('pipeline_stages', 5)
                generator = CPUGenerator(
                    llm_provider=llm_name,
                    project_root=str(PROJECT_ROOT),
                    debug=False
                )
                module_name = "riscv_cpu"
                prompt = generator._create_cpu_prompt(32, pipeline_stages, module_name)

            else:
                yield make_sse_message("error", message=f"Unknown module type: {module_type}")
                return

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

            if hasattr(llm, '_call_api_stream'):
                for chunk in llm._call_api_stream(prompt, max_tokens=max_tokens):
                    if chunk:
                        full_content += chunk
                        yield make_sse_message("chunk", content=chunk)
            else:
                yield make_sse_message("info", message="Using standard mode...")
                response = llm._call_api(
                    prompt,
                    max_tokens=max_tokens,
                    system_prompt="You are an expert Verilog hardware designer."
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

            yield make_sse_message("complete", filename=filename, filepath=str(hw_path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield make_sse_message("error", message=str(e))

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


@app.route('/api/yosys-status')
def yosys_status():
    """Check if Yosys is available on the system."""
    try:
        from yosys_analyzer import YosysAnalyzer
        analyzer = YosysAnalyzer(project_root=str(PROJECT_ROOT))
        return jsonify(analyzer.get_tool_info())
    except Exception as e:
        return jsonify({
            'yosys_available': False,
            'graphviz_available': False,
            'yosys_path': None,
            'graphviz_path': None,
            'error': str(e)
        })


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

    file_path = None

    if last_generated_hw['filepath']:
        candidate = Path(last_generated_hw['filepath'])
        if candidate.exists() and candidate.name == filename:
            file_path = candidate

    if not file_path:
        dut_dir = PROJECT_ROOT / 'output' / 'dut'
        candidate = dut_dir / filename
        if candidate.exists():
            file_path = candidate

    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/list-hardware')
def list_hardware_files():
    """List generated hardware files"""
    dut_dir = PROJECT_ROOT / 'output' / 'dut'
    files = []

    if dut_dir.exists():
        for f in dut_dir.glob('*.v'):
            module_type = 'unknown'
            if 'alu' in f.name.lower():
                module_type = 'alu'
            elif 'counter' in f.name.lower():
                module_type = 'counter'
            elif 'regfile' in f.name.lower() or 'register' in f.name.lower():
                module_type = 'regfile'
            elif 'cpu' in f.name.lower():
                module_type = 'cpu'

            files.append({
                'filename': f.name,
                'module_type': module_type,
                'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })

    files.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify({'files': files})


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
@app.route('/api/generate-alu', methods=['POST'])
def generate_alu():
    """Generate ALU Verilog (non-streaming) - Legacy"""
    data = dict(request.json) if request.json else {}
    data['module_type'] = 'alu'

    # Manually call generate_hardware logic
    module_type = 'alu'
    llm_name = data.get('llm', 'groq')
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')

    if natural_input:
        parsed = parse_hardware_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']

    if not HAS_ALU_MODULE:
        return jsonify({'success': False, 'error': 'ALU module not available'}), 500

    try:
        generator = ALUGenerator(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )
        hw_path = generator.generate_alu(bitwidth=bitwidth, module_name='alu')

        if not hw_path:
            return jsonify({'success': False, 'error': 'Generation failed'}), 500

        hw_path_obj = Path(hw_path)
        with open(hw_path, 'r', encoding='utf-8') as f:
            content = f.read()

        last_generated_hw['filename'] = hw_path_obj.name
        last_generated_hw['filepath'] = str(hw_path)
        last_generated_hw['llm'] = llm_name
        last_generated_hw['module_type'] = 'alu'

        return jsonify({
            'success': True,
            'filename': hw_path_obj.name,
            'full_content': content,
            'llm': llm_name,
            'bitwidth': bitwidth,
            'module_type': 'alu',
            'filepath': str(hw_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-alu-stream', methods=['POST'])
def generate_alu_stream():
    """Generate ALU with streaming - Legacy"""
    data = dict(request.json) if request.json else {}
    module_type = 'alu'
    llm_name = data.get('llm', 'groq')
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')

    if natural_input:
        parsed = parse_hardware_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']

    def generate():
        try:
            yield make_sse_message("start", llm=llm_name, bitwidth=bitwidth, module_type='alu')
            yield make_sse_message("info", message=f"Initializing {bitwidth}-bit ALU generator...")

            if not HAS_ALU_MODULE:
                yield make_sse_message("error", message="ALU module not available")
                return

            generator = ALUGenerator(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=False
            )
            operations = {
                "ADD": {"opcode": "0000", "description": "Addition (A + B)"},
                "SUB": {"opcode": "0001", "description": "Subtraction (A - B)"},
                "AND": {"opcode": "0010", "description": "Bitwise AND (A & B)"},
                "OR": {"opcode": "0011", "description": "Bitwise OR (A | B)"},
            }
            prompt = generator._create_alu_prompt(bitwidth, operations, "alu")

            yield make_sse_message("info", message=f"Calling {llm_name.upper()} API...")

            llm = generator.llm
            full_content = ""

            # Dynamic token budget for ALU
            max_tokens = 5000 + (bitwidth // 16) * 1000 + len(operations) * 200

            if hasattr(llm, '_call_api_stream'):
                for chunk in llm._call_api_stream(prompt, max_tokens=max_tokens):
                    if chunk:
                        full_content += chunk
                        yield make_sse_message("chunk", content=chunk)
            else:
                yield make_sse_message("info", message="Using standard mode...")
                response = llm._call_api(prompt, max_tokens=max_tokens,
                                         system_prompt="You are an expert Verilog hardware designer.")
                if response:
                    full_content = response
                    for i in range(0, len(full_content), 100):
                        yield make_sse_message("chunk", content=full_content[i:i+100])

            if not full_content:
                yield make_sse_message("error", message="LLM returned empty response")
                return

            verilog_code = generator._extract_verilog(full_content)
            if not verilog_code:
                verilog_code = full_content

            hw_path = generator._save_alu(verilog_code, "alu", bitwidth)
            filename = Path(hw_path).name

            last_generated_hw['filename'] = filename
            last_generated_hw['filepath'] = str(hw_path)
            last_generated_hw['llm'] = llm_name
            last_generated_hw['module_type'] = 'alu'

            yield make_sse_message("complete", filename=filename, filepath=str(hw_path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield make_sse_message("error", message=str(e))

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'
    })


@app.route('/api/download-alu/<filename>')
def download_alu(filename):
    return download_hardware(filename)


@app.route('/api/list-alu')
def list_alu_files():
    return list_hardware_files()


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

    def generate():
        try:
            yield make_sse_message("start", llm=llm_name)

            generator = FeatureGeneratorLLM(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=False
            )

            if model and llm_name == 'openai':
                try:
                    llm = LLMFactory.create_provider('openai', model=model)
                    generator.llm = llm
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

            if hasattr(llm, '_call_api_stream'):
                for chunk in llm._call_api_stream(prompt):
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

            yield make_sse_message("complete", filename=filename, filepath=str(feature_path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield make_sse_message("error", message=str(e))

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

        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        if model and llm_name == 'openai':
            try:
                llm = LLMFactory.create_provider('openai', model=model)
                generator.llm = llm
            except:
                pass

        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            return jsonify({'success': False, 'error': 'Generation failed'}), 500

        feature_path_obj = Path(feature_path)
        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()

        last_generated_bdd['filename'] = feature_path_obj.name
        last_generated_bdd['filepath'] = str(feature_path)
        last_generated_bdd['llm'] = llm_name

        return jsonify({
            'success': True,
            'filename': feature_path_obj.name,
            'preview': content[:500] + ('...' if len(content) > 500 else ''),
            'full_content': content,
            'llm': llm_name,
            'model': model or llm_name,
            'filepath': str(feature_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_bdd(filename):
    """Download generated BDD file"""
    file_path = None

    if last_generated_bdd['filepath']:
        candidate = Path(last_generated_bdd['filepath'])
        if candidate.exists() and candidate.name == filename:
            file_path = candidate

    if not file_path:
        base_dir = PROJECT_ROOT / 'output' / 'bdd'
        if base_dir.exists():
            for llm_dir in base_dir.iterdir():
                if llm_dir.is_dir():
                    candidate = llm_dir / filename
                    if candidate.exists():
                        file_path = candidate
                        break

    if not file_path:
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(str(file_path.parent.absolute()), file_path.name, as_attachment=True, download_name=filename)


@app.route('/api/llm-list')
def get_llm_list():
    """Get available LLM providers and module types"""
    return jsonify({
        'llms': [
            {'id': 'groq', 'name': 'Groq', 'description': 'Fast & Free'},
            {'id': 'deepseek', 'name': 'DeepSeek', 'description': 'Chinese LLM'},
            {'id': 'openai', 'name': 'OpenAI', 'description': 'GPT-5 Series'},
            {'id': 'claude', 'name': 'Claude', 'description': 'Anthropic'},
            {'id': 'gemini', 'name': 'Gemini', 'description': 'Google'},
            {'id': 'qwen', 'name': 'Qwen', 'description': 'Alibaba Tongyi'},
            {'id': 'together', 'name': 'Together', 'description': 'Together AI'},
            {'id': 'mistral', 'name': 'Mistral', 'description': 'Mistral/Codestral'},
            {'id': 'grok', 'name': 'Grok', 'description': 'xAI'}
        ],
        'openai_models': [
            {'id': 'gpt-5-mini', 'name': 'GPT-5 Mini'},
            {'id': 'gpt-5', 'name': 'GPT-5'},
            {'id': 'gpt-5.1', 'name': 'GPT-5.1'}
        ],
        'bitwidths': [8, 16, 32, 64],
        'module_types': [
            {'id': 'alu', 'name': 'ALU', 'description': 'Arithmetic Logic Unit', 'available': HAS_ALU_MODULE},
            {'id': 'counter', 'name': 'Counter', 'description': 'Up/Down Counter', 'available': HAS_COUNTER_MODULE},
            {'id': 'regfile', 'name': 'Register File', 'description': 'Multi-port Register Bank', 'available': HAS_REGFILE_MODULE},
            {'id': 'cpu', 'name': 'CPU', 'description': 'RISC-V 5-Stage Pipelined Processor', 'available': HAS_CPU_MODULE}
        ]
    })

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

        # Initialize generator
        generator = TestbenchGenerator(
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        # Generate testbench
        result = generator.generate_single(
            bdd_filepath=str(bdd_path),
            dut_info=dut_info
        )

        if not result['success']:
            return jsonify(result), 500

        # Store last generated info
        last_generated_tb['filename'] = result['filename']
        last_generated_tb['filepath'] = result['filepath']
        last_generated_tb['bdd_source'] = bdd_filepath

        return jsonify({
            'success': True,
            'filename': result['filename'],
            'filepath': result['filepath'],
            'content': result['content'],
            'full_content': result['full_content'],
            'quality_summary': result['quality_summary'],
            'test_count': result['test_count'],
            'llm': result['llm']
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
    file_path = None

    if last_generated_tb['filepath']:
        candidate = Path(last_generated_tb['filepath'])
        if candidate.exists() and candidate.name == filename:
            file_path = candidate

    if not file_path:
        base_dir = PROJECT_ROOT / 'output' / 'testbench'
        if base_dir.exists():
            for llm_dir in base_dir.iterdir():
                if llm_dir.is_dir():
                    candidate = llm_dir / filename
                    if candidate.exists():
                        file_path = candidate
                        break

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

        return jsonify(result)

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

@app.route('/api/download-waveforms/<llm_name>')
def download_waveforms_batch(llm_name):
    """Download all VCD files for an LLM as a zip"""
    waveform_dir = PROJECT_ROOT / 'output' / 'waveform' / llm_name

    if not waveform_dir.exists():
        return jsonify({'error': 'No waveform directory found'}), 404

    vcd_files = list(waveform_dir.glob('*.vcd'))
    if not vcd_files:
        return jsonify({'error': 'No VCD files found'}), 404

    # Create zip in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for vcd_file in vcd_files:
            zf.write(vcd_file, vcd_file.name)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'waveforms_{llm_name}.zip'
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