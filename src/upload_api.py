"""
Upload DUT API - Add this to your main.py
==========================================

This code provides the /api/upload-dut endpoint for handling
Verilog file uploads.

Authors: Rolf Drechsler, Qian Liu
Paper: https://arxiv.org/abs/2512.17814

USAGE:
1. Copy the VerilogParser class to src/verilog_parser.py
2. Add the following imports and route to your main.py
"""

# ============================================================
# Add these imports to main.py
# ============================================================
"""
from flask import request, jsonify
from werkzeug.utils import secure_filename
import os

# Import the parser
from verilog_parser import VerilogParser
"""

# ============================================================
# Add this configuration to main.py
# ============================================================
"""
# Upload configuration
UPLOAD_FOLDER = 'output/dut/uploaded'
ALLOWED_EXTENSIONS = {'v', 'sv', 'vh'}
MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize parser
verilog_parser = VerilogParser(upload_dir=UPLOAD_FOLDER)
"""

# ============================================================
# Add this route to main.py
# ============================================================

# --- Start of route code ---

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# This is a standalone example - integrate into your main.py

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'output/dut/uploaded'
ALLOWED_EXTENSIONS = {'v', 'sv', 'vh'}
MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Import parser (adjust path as needed)
try:
    from verilog_parser import VerilogParser

    verilog_parser = VerilogParser(upload_dir=UPLOAD_FOLDER)
    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False
    print("⚠️  VerilogParser not found")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload-dut', methods=['POST'])
def upload_dut():
    """
    Handle Verilog file upload and parsing.

    Returns:
        JSON with parsed module information or error
    """
    # Check if parser is available
    if not HAS_PARSER:
        return jsonify({
            'success': False,
            'error': 'Verilog parser not available'
        }), 500

    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    # Check extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: .v, .sv, .vh'
        }), 400

    try:
        # Read file content
        file_content = file.read()

        # Check file size
        if len(file_content) > MAX_CONTENT_LENGTH:
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum: 1MB'
            }), 400

        # Secure the filename
        filename = secure_filename(file.filename)

        # Parse the file
        result = verilog_parser.parse_file(filename, file_content)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/download-uploaded/<filename>')
def download_uploaded(filename):
    """Download an uploaded file."""
    from flask import send_from_directory
    return send_from_directory(
        UPLOAD_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# Error handler for file too large
# ============================================================
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size: 1MB'
    }), 413


# --- End of route code ---


if __name__ == '__main__':
    print("=" * 60)
    print("Upload DUT API Test Server")
    print("=" * 60)
    print(f"Parser available: {HAS_PARSER}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Max file size: {MAX_CONTENT_LENGTH / 1024 / 1024}MB")
    print("=" * 60)

    app.run(debug=True, port=5001)