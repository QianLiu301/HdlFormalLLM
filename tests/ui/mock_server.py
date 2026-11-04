"""
The web app with the trial runner's mock model (src/xdb/trial.MockLLM): the
model menus of Steps 0-2 offer "Mock (no API key)", which answers with the
canned samples of src/xdb/samples.py. For the browser checks and for trying
the website without an API key. Port: $PORT, default 5000.

    python tests/ui/mock_server.py              # canned answers
    python tests/ui/mock_server.py compile      # the first design of each type does not compile
    python tests/ui/mock_server.py interface    # ... names its ports differently from the testbench
"""
import contextlib
import io
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    import main  # noqa: E402
from src.xdb import trial as TR  # noqa: E402

TR.install_mock()
TR.MockLLM.first_fault = sys.argv[1] if len(sys.argv) > 1 else None
OPTION = '<option value="mock">Mock (no API key)</option>'


def index():
    html = (ROOT / 'static/bdd_generator.html').read_text(encoding='utf-8')
    for sel in ('req-llm', 'hw-llm', 'bdd-llm'):
        html = re.sub(rf'(<select id="{sel}"[^>]*>)', lambda m: m.group(1) + OPTION, html, count=1)
    resp = main.app.response_class(html, mimetype='text/html')
    resp.cache_control.no_store = True          # never mix this page up with the real one
    return resp


main.app.view_functions['index'] = index
port = int(os.environ.get('PORT', 5000))
print(f'mock model (first fault: {TR.MockLLM.first_fault}) on http://localhost:{port}', flush=True)
main.app.run(host='127.0.0.1', port=port, debug=False)
