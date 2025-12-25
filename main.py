"""
LLM BDD Generator - Flask Backend with Streaming Support
=========================================================

Supports Server-Sent Events (SSE) for real-time streaming output.
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
try:
    from feature_generator_llm import FeatureGeneratorLLM
    from llm_providers import LLMFactory
    HAS_MODULES = True
    print("✅ Modules imported from src/")
except ImportError as e:
    print(f"⚠️ Warning: Could not import modules: {e}")
    HAS_MODULES = False

# ============================================================================
# Flask App
# ============================================================================
app = Flask(__name__,
            static_folder=str(PROJECT_ROOT / 'static'),
            template_folder=str(PROJECT_ROOT / 'static'))
CORS(app)

last_generated = {
    'filename': None,
    'filepath': None,
    'llm': None
}

output_dir = PROJECT_ROOT / 'output' / 'bdd'
output_dir.mkdir(parents=True, exist_ok=True)


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
        'modules_loaded': HAS_MODULES,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# Streaming API Endpoint (SSE)
# ============================================================================
@app.route('/api/generate-stream', methods=['POST'])
def generate_stream():
    """Generate BDD Feature file with streaming output (SSE)"""
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported.'
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
            # Send start event
            yield make_sse_message("start", llm=llm_name)

            # Create generator
            generator = FeatureGeneratorLLM(
                llm_provider=llm_name,
                project_root=str(PROJECT_ROOT),
                debug=False
            )

            # For OpenAI with specific model
            if model and llm_name == 'openai':
                try:
                    llm = LLMFactory.create_provider('openai', model=model)
                    generator.llm = llm
                except Exception as e:
                    yield make_sse_message("error", message=str(e))
                    return

            # Parse requirements
            requirements = generator.parse_user_input(user_input)
            bitwidth = requirements.get("bitwidth", "?")
            ops_count = len(requirements.get("operations", []))

            yield make_sse_message("info", message=f"Parsed: {bitwidth}-bit ALU with {ops_count} operations")

            # Create prompt
            prompt = generator._create_prompt(requirements)

            yield make_sse_message("info", message="Calling LLM API...")

            # Check if provider supports streaming
            llm = generator.llm
            full_content = ""

            if hasattr(llm, '_call_api_stream'):
                # Use streaming
                for chunk in llm._call_api_stream(prompt):
                    if chunk:
                        full_content += chunk
                        yield make_sse_message("chunk", content=chunk)
            else:
                # Fallback to non-streaming
                yield make_sse_message("info", message="Streaming not supported, using standard mode...")

                full_content = generator._call_llm(prompt)
                if full_content:
                    # Send in chunks to simulate streaming
                    chunk_size = 50
                    for i in range(0, len(full_content), chunk_size):
                        chunk = full_content[i:i+chunk_size]
                        yield make_sse_message("chunk", content=chunk)

            if not full_content:
                yield make_sse_message("error", message="LLM returned empty response")
                return

            # Clean content
            full_content = generator._clean_response(full_content)

            # Save file
            feature_path = generator._save_feature(full_content, requirements)
            filename = Path(feature_path).name

            # Update last_generated
            last_generated['filename'] = filename
            last_generated['filepath'] = str(feature_path)
            last_generated['llm'] = llm_name

            # Send completion event
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


# ============================================================================
# Original Non-Streaming Endpoint
# ============================================================================
@app.route('/api/generate', methods=['POST'])
def generate_bdd():
    """Generate BDD Feature file using LLM (non-streaming)"""
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported.'
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

        print(f"\n{'='*60}")
        print(f"🚀 Generating BDD Feature")
        print(f"{'='*60}")
        print(f"   LLM: {llm_name.upper()}")
        if model:
            print(f"   Model: {model}")
        print(f"   Input: {user_input[:50]}...")
        print()

        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        if model and llm_name == 'openai':
            try:
                llm = LLMFactory.create_provider('openai', model=model)
                generator.llm = llm
                print(f"   ✅ OpenAI model set to: {model}")
            except Exception as e:
                print(f"   ⚠️ Failed to set model {model}: {e}")

        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            return jsonify({
                'success': False,
                'error': 'LLM returned empty response. Please try again.'
            }), 500

        feature_path_obj = Path(feature_path)
        if not feature_path_obj.exists():
            return jsonify({
                'success': False,
                'error': f'File was not created: {feature_path}'
            }), 500

        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()

        last_generated['filename'] = feature_path_obj.name
        last_generated['filepath'] = str(feature_path)
        last_generated['llm'] = llm_name

        print(f"\n{'='*60}")
        print(f"✅ Success! File: {feature_path_obj.name}")
        print(f"{'='*60}\n")

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
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated BDD file"""
    print(f"\n📥 Download request: {filename}")

    file_path = None

    if last_generated['filepath']:
        candidate = Path(last_generated['filepath'])
        if candidate.exists() and candidate.name == filename:
            file_path = candidate
            print(f"   Found via last_generated: {file_path}")

    if not file_path:
        base_dir = PROJECT_ROOT / 'output' / 'bdd'

        if base_dir.exists():
            for llm_dir in base_dir.iterdir():
                if llm_dir.is_dir():
                    candidate = llm_dir / filename
                    if candidate.exists():
                        file_path = candidate
                        print(f"   Found in {llm_dir.name}/: {file_path}")
                        break

    if not file_path:
        print(f"❌ File not found: {filename}")
        return jsonify({'error': 'File not found'}), 404

    print(f"✅ Sending file: {file_path}")

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
            {'id': 'gpt-5-mini', 'name': 'GPT-5 Mini (Recommended)'},
            {'id': 'gpt-5', 'name': 'GPT-5'},
            {'id': 'gpt-5.1', 'name': 'GPT-5.1'},
            {'id': 'gpt-5.1-codex', 'name': 'GPT-5.1 Codex'}
        ]
    })


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print()
    print("=" * 60)
    print("🤖 LLM BDD Generator (with Streaming Support)")
    print("=" * 60)
    print(f"📁 Project: {PROJECT_ROOT}")
    print(f"📡 Server: http://localhost:{port}")
    print(f"🔧 Debug: {debug}")
    print()
    print("📋 Endpoints:")
    print("   POST /api/generate        - Standard generation")
    print("   POST /api/generate-stream - Streaming generation (SSE)")
    print()
    print("=" * 60)
    print()

    app.run(debug=debug, host='0.0.0.0', port=port)