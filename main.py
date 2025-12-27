"""
LLM Hardware Generator - Flask Backend
=======================================

Supports:
- ALU Verilog Generation
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

# ============================================================================
# Path Setup
# ============================================================================
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

    env_keys = {
        'GROQ_API_KEY': 'groq',
        'DEEPSEEK_API_KEY': 'deepseek',
        'OPENAI_API_KEY': 'openai',
        'ANTHROPIC_API_KEY': 'anthropic',
        'GEMINI_API_KEY': 'gemini'
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
        'gemini': 'GEMINI_API_KEY'
    }

    for provider, env_var in key_mapping.items():
        if api_keys.get(provider) and not api_keys[provider].startswith('your_'):
            os.environ[env_var] = api_keys[provider]

    print("🔑 API Keys status:")
    for provider, env_var in key_mapping.items():
        status = "✅" if os.environ.get(env_var) else "❌"
        print(f"   {status} {provider.upper()}")

setup_api_keys()

# ============================================================================
# Import Modules
# ============================================================================
HAS_BDD_MODULE = False
HAS_ALU_MODULE = False

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

# ============================================================================
# Flask App
# ============================================================================
app = Flask(__name__,
            static_folder=str(PROJECT_ROOT / 'static'),
            template_folder=str(PROJECT_ROOT / 'static'))
CORS(app)

# Store last generated files
last_generated_bdd = {'filename': None, 'filepath': None, 'llm': None}
last_generated_alu = {'filename': None, 'filepath': None, 'llm': None}

# Ensure output directories exist
(PROJECT_ROOT / 'output' / 'bdd').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'output' / 'dut').mkdir(parents=True, exist_ok=True)


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
    return jsonify({
        'status': 'healthy',
        'bdd_module': HAS_BDD_MODULE,
        'alu_module': HAS_ALU_MODULE,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# ALU Generator API
# ============================================================================
@app.route('/api/generate-alu', methods=['POST'])
def generate_alu():
    """Generate ALU Verilog (non-streaming)"""
    if not HAS_ALU_MODULE:
        return jsonify({
            'success': False,
            'error': 'ALU Generator module not available'
        }), 500

    try:
        data = request.json
        llm_name = data.get('llm', 'groq')
        bitwidth = data.get('bitwidth', 16)
        natural_input = data.get('input', '')

        # Parse natural language if provided
        if natural_input:
            parsed = parse_alu_natural_language(natural_input)
            bitwidth = parsed.get('bitwidth', bitwidth)
            if parsed.get('llm'):
                llm_name = parsed['llm']

        print(f"\n{'='*60}")
        print(f"🔧 Generating {bitwidth}-bit ALU")
        print(f"{'='*60}")
        print(f"   LLM: {llm_name.upper()}")
        print(f"   Bitwidth: {bitwidth}")

        generator = ALUGenerator(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        alu_path = generator.generate_alu(
            bitwidth=bitwidth,
            module_name='alu'
        )

        if not alu_path:
            return jsonify({
                'success': False,
                'error': 'ALU generation failed'
            }), 500

        alu_path_obj = Path(alu_path)
        if not alu_path_obj.exists():
            return jsonify({
                'success': False,
                'error': f'File was not created: {alu_path}'
            }), 500

        with open(alu_path, 'r', encoding='utf-8') as f:
            content = f.read()

        last_generated_alu['filename'] = alu_path_obj.name
        last_generated_alu['filepath'] = str(alu_path)
        last_generated_alu['llm'] = llm_name

        print(f"\n✅ Success! File: {alu_path_obj.name}")

        return jsonify({
            'success': True,
            'filename': alu_path_obj.name,
            'preview': content[:1000] + ('...' if len(content) > 1000 else ''),
            'full_content': content,
            'llm': llm_name,
            'bitwidth': bitwidth,
            'filepath': str(alu_path)
        })

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate-alu-stream', methods=['POST'])
def generate_alu_stream():
    """Generate ALU Verilog with streaming output (SSE)"""
    if not HAS_ALU_MODULE:
        return jsonify({
            'success': False,
            'error': 'ALU Generator module not available'
        }), 500

    data = request.json
    llm_name = data.get('llm', 'groq')
    bitwidth = data.get('bitwidth', 16)
    natural_input = data.get('input', '')

    # Parse natural language if provided
    if natural_input:
        parsed = parse_alu_natural_language(natural_input)
        bitwidth = parsed.get('bitwidth', bitwidth)
        if parsed.get('llm'):
            llm_name = parsed['llm']

    def generate():
        try:
            yield make_sse_message("start", llm=llm_name, bitwidth=bitwidth)
            yield make_sse_message("info", message=f"Initializing {bitwidth}-bit ALU generator...")

            generator = ALUGenerator(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=False
            )

            yield make_sse_message("info", message=f"Calling {llm_name.upper()} API...")

            # Get LLM and check for streaming support
            llm = generator.llm

            # Create prompt
            operations = {
                "ADD": {"opcode": "0000", "description": "Addition (A + B)"},
                "SUB": {"opcode": "0001", "description": "Subtraction (A - B)"},
                "AND": {"opcode": "0010", "description": "Bitwise AND (A & B)"},
                "OR": {"opcode": "0011", "description": "Bitwise OR (A | B)"},
            }
            prompt = generator._create_alu_prompt(bitwidth, operations, "alu")

            full_content = ""

            if hasattr(llm, '_call_api_stream'):
                # Use streaming
                for chunk in llm._call_api_stream(prompt, max_tokens=3000):
                    if chunk:
                        full_content += chunk
                        yield make_sse_message("chunk", content=chunk)
            else:
                # Fallback to non-streaming
                yield make_sse_message("info", message="Using standard mode...")
                response = llm._call_api(
                    prompt,
                    max_tokens=3000,
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

            # Extract and validate Verilog
            verilog_code = generator._extract_verilog(full_content)
            if not verilog_code:
                verilog_code = full_content

            # Save file
            alu_path = generator._save_alu(verilog_code, "alu", bitwidth)
            filename = Path(alu_path).name

            last_generated_alu['filename'] = filename
            last_generated_alu['filepath'] = str(alu_path)
            last_generated_alu['llm'] = llm_name

            yield make_sse_message("complete", filename=filename, filepath=str(alu_path))

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


@app.route('/api/download-alu/<filename>')
def download_alu(filename):
    """Download generated ALU file"""
    print(f"\n📥 ALU Download request: {filename}")

    file_path = None

    if last_generated_alu['filepath']:
        candidate = Path(last_generated_alu['filepath'])
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


@app.route('/api/list-alu')
def list_alu_files():
    """List generated ALU files"""
    dut_dir = PROJECT_ROOT / 'output' / 'dut'
    files = []

    if dut_dir.exists():
        for f in dut_dir.glob('*.v'):
            files.append({
                'filename': f.name,
                'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })

    files.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify({'files': files})


def parse_alu_natural_language(input_text):
    """Parse natural language input for ALU generation"""
    import re
    input_lower = input_text.lower()

    result = {'bitwidth': 16, 'llm': None}

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
    }

    for provider, keywords in llm_keywords.items():
        for keyword in keywords:
            if keyword in input_lower:
                result['llm'] = provider
                break

    return result


# ============================================================================
# BDD Generator API (existing)
# ============================================================================
@app.route('/api/generate-stream', methods=['POST'])
def generate_bdd_stream():
    """Generate BDD Feature file with streaming output (SSE)"""
    if not HAS_BDD_MODULE:
        return jsonify({
            'success': False,
            'error': 'BDD Generator module not available'
        }), 500

    data = request.json
    llm_name = data.get('llm', 'groq')
    model = data.get('model')
    user_input = data.get('input', '')

    if not user_input:
        return jsonify({
            'success': False,
            'error': 'Please enter your requirements'
        }), 400

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
                    chunk_size = 50
                    for i in range(0, len(full_content), chunk_size):
                        chunk = full_content[i:i+chunk_size]
                        yield make_sse_message("chunk", content=chunk)

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

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/generate', methods=['POST'])
def generate_bdd():
    """Generate BDD Feature file (non-streaming)"""
    if not HAS_BDD_MODULE:
        return jsonify({
            'success': False,
            'error': 'BDD Generator module not available'
        }), 500

    try:
        data = request.json
        llm_name = data.get('llm', 'groq')
        model = data.get('model')
        user_input = data.get('input', '')

        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Please enter your requirements'
            }), 400

        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        if model and llm_name == 'openai':
            try:
                llm = LLMFactory.create_provider('openai', model=model)
                generator.llm = llm
            except Exception as e:
                pass

        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            return jsonify({
                'success': False,
                'error': 'Generation failed'
            }), 500

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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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

    return send_from_directory(
        str(file_path.parent.absolute()),
        file_path.name,
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/llm-list')
def get_llm_list():
    """Get available LLM providers"""
    return jsonify({
        'llms': [
            {'id': 'groq', 'name': 'Groq', 'description': 'Fast & Free'},
            {'id': 'deepseek', 'name': 'DeepSeek', 'description': 'Chinese LLM'},
            {'id': 'openai', 'name': 'OpenAI', 'description': 'GPT-5 Series'},
            {'id': 'claude', 'name': 'Claude', 'description': 'Anthropic'},
            {'id': 'gemini', 'name': 'Gemini', 'description': 'Google'}
        ],
        'openai_models': [
            {'id': 'gpt-5-mini', 'name': 'GPT-5 Mini'},
            {'id': 'gpt-5', 'name': 'GPT-5'},
            {'id': 'gpt-5.1', 'name': 'GPT-5.1'},
            {'id': 'gpt-5.1-codex', 'name': 'GPT-5.1 Codex'}
        ],
        'bitwidths': [8, 16, 32, 64]
    })


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
    print("📋 Available Generators:")
    print(f"   {'✅' if HAS_ALU_MODULE else '❌'} ALU Generator (Verilog)")
    print(f"   {'✅' if HAS_BDD_MODULE else '❌'} BDD Generator (Test Scenarios)")
    print()
    print("=" * 60)
    print()

    app.run(debug=debug, host='0.0.0.0', port=port)