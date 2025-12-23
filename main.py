"""
LLM BDD Generator - Flask Backend
==================================

Project Structure:
├── config/llm_config.json   # Local API keys (gitignored)
├── src/                     # Source code
├── static/                  # Frontend files
├── output/bdd/              # Generated files
└── main.py                  # This file

Run: python main.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# Path Setup - Add src/ to Python path
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# ============================================================================
# Configuration Loading
# ============================================================================
def load_config():
    """Load configuration from config file or environment variables"""
    config = {
        'proxy': {'enabled': False},
        'api_keys': {}
    }

    # Try to load from config file (local development)
    config_file = PROJECT_ROOT / 'config' / 'llm_config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
                print(f"✅ Config loaded from: {config_file}")
        except Exception as e:
            print(f"⚠️ Config file error: {e}")

    # Environment variables override config file (production)
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

    # Check proxy setting from environment
    if os.environ.get('ENABLE_PROXY', '').lower() == 'false':
        config['proxy']['enabled'] = False

    return config

CONFIG = load_config()

# ============================================================================
# Proxy Setup
# ============================================================================
def setup_proxy():
    """Setup proxy based on configuration"""
    proxy_config = CONFIG.get('proxy', {})

    if not proxy_config.get('enabled', False):
        # Clear any existing proxy settings
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
# Set API Keys as Environment Variables (for llm_providers.py)
# ============================================================================
def setup_api_keys():
    """Set API keys as environment variables for llm_providers.py to use"""
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

    # Print status
    print("🔑 API Keys status:")
    for provider, env_var in key_mapping.items():
        status = "✅" if os.environ.get(env_var) else "❌"
        print(f"   {status} {provider.upper()}")

setup_api_keys()

# ============================================================================
# Import Modules from src/
# ============================================================================
try:
    from feature_generator_llm import FeatureGeneratorLLM
    from llm_providers import LLMFactory
    HAS_MODULES = True
    print("✅ Modules imported from src/")
except ImportError as e:
    print(f"⚠️ Warning: Could not import modules: {e}")
    print(f"   Make sure feature_generator_llm.py and llm_providers.py are in src/")
    HAS_MODULES = False

# ============================================================================
# Flask App
# ============================================================================
app = Flask(__name__,
            static_folder=str(PROJECT_ROOT / 'static'),
            template_folder=str(PROJECT_ROOT / 'static'))
CORS(app)

# Store last generated file info
last_generated = {
    'filename': None,
    'filepath': None,
    'llm': None
}

# Ensure output directory exists
output_dir = PROJECT_ROOT / 'output' / 'bdd'
output_dir.mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(str(PROJECT_ROOT / 'static'), 'bdd_generator.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'modules_loaded': HAS_MODULES,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/generate', methods=['POST'])
def generate_bdd():
    """
    Generate BDD Feature file using LLM

    Request:
    {
        "llm": "groq|deepseek|openai|claude|gemini",
        "model": "gpt-5-mini",  // only for openai
        "input": "16-bit ALU with ADD, SUB"
    }
    """
    if not HAS_MODULES:
        return jsonify({
            'success': False,
            'error': 'Required modules not imported. Check if src/ contains the required files.'
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

        # Create generator with project root for correct output path
        generator = FeatureGeneratorLLM(
            llm_provider=llm_name,
            project_root=str(PROJECT_ROOT),
            debug=True
        )

        # For OpenAI with specific model, override
        if model and llm_name == 'openai':
            try:
                llm = LLMFactory.create_provider('openai', model=model)
                generator.llm = llm
                print(f"   ✅ OpenAI model set to: {model}")
            except Exception as e:
                print(f"   ⚠️ Failed to set model {model}: {e}")

        # Generate feature
        feature_path = generator.generate_feature(user_input)

        if not feature_path:
            return jsonify({
                'success': False,
                'error': 'LLM returned empty response. Please try again.'
            }), 500

        # Verify file
        feature_path_obj = Path(feature_path)
        if not feature_path_obj.exists():
            return jsonify({
                'success': False,
                'error': f'File was not created: {feature_path}'
            }), 500

        # Read content for preview
        with open(feature_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Store last generated info
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

    # Method 1: Try last_generated filepath first
    if last_generated['filepath']:
        candidate = Path(last_generated['filepath'])
        if candidate.exists() and candidate.name == filename:
            file_path = candidate
            print(f"   Found via last_generated: {file_path}")

    # Method 2: Search in output/bdd
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
    print("🤖 LLM BDD Generator")
    print("=" * 60)
    print(f"📁 Project: {PROJECT_ROOT}")
    print(f"📡 Server: http://localhost:{port}")
    print(f"🔧 Debug: {debug}")
    print()
    print("=" * 60)
    print()

    app.run(debug=debug, host='0.0.0.0', port=port)