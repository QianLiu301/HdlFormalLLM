"""
Mask credentials in text that can leave the machine: error messages shown in
the browser or stored in the databases, trial outputs that are shared through
git, and staged changes checked by the pre-commit hook (tools/check_secrets.py).

Two layers:
  1. the secrets this installation actually uses (API keys from the
     environment, config/llm_config.json and .env*, the DATABASE_URL
     password) are replaced wherever they appear, whatever their format;
  2. patterns of common credential formats (Google "AIza…", "sk-…", "gsk_…",
     "xai-…", bearer tokens, "?key=…" in URLs, passwords in database URLs)
     catch keys that are not configured here.

No third-party imports: used by llm_providers, main, the trial runner and the
git hook.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
MASK = '***'

# values that look like credentials but are placeholders
_PLACEHOLDERS = {'', 'password', 'pass', 'user', 'xxxx', 'changeme', MASK}

PATTERNS = [
    # Google API keys (Gemini, Maps, ...)
    (re.compile(r'AIza[0-9A-Za-z_\-]{30,}'), 'AIza' + MASK),
    # OpenAI / DeepSeek / Qwen / Anthropic ("sk-ant-…"), Groq, xAI, Together, GitHub
    (re.compile(r'\b(sk-(?:ant-|proj-)?|gsk_|xai-|tgp_v1_|ghp_|github_pat_)[A-Za-z0-9_\-]{16,}'), r'\1' + MASK),
    (re.compile(r'(\bBearer\s+)[A-Za-z0-9._~+/\-]{16,}=*', re.I), r'\1' + MASK),
    # keys in query strings: ...?key=..., &api_key=...
    (re.compile(r'([?&](?:key|api_key|apikey|access_token|token|secret)=)[^&\s\'"<>]+', re.I), r'\1' + MASK),
    # passwords in connection URLs: postgresql://user:password@host
    (re.compile(r'(\b[a-z][a-z0-9+.\-]*://[^:/\s@]+:)(?!\*\*\*@)[^@\s/]+@', re.I), r'\1' + MASK + '@'),
]


def _url_password(url: str) -> Optional[str]:
    m = re.match(r'^[a-z][a-z0-9+.\-]*://[^:/\s@]+:([^@\s/]+)@', (url or '').strip(), re.I)
    return m.group(1) if m else None


def _env_file_values(path: Path) -> List[str]:
    out = []
    try:
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                v = line.split('=', 1)[1].strip().strip('"\'')
                out += [v, _url_password(v) or '']
    except OSError:
        pass
    return out


def known_secrets(root: Path = ROOT) -> List[str]:
    """The credentials configured on this machine (longest first)."""
    vals: List[str] = []
    for k, v in os.environ.items():
        if re.search(r'(API_KEY|_TOKEN|_SECRET|PASSWORD)$', k) or k == 'DATABASE_URL':
            vals += [v, _url_password(v) or '']
    try:
        cfg = json.loads((root / 'config' / 'llm_config.json').read_text(encoding='utf-8'))
        for p in (cfg.get('providers') or {}).values():
            if isinstance(p, dict):
                vals.append(str(p.get('api_key') or ''))
    except (OSError, ValueError, AttributeError):
        pass
    for f in root.glob('.env*'):
        if f.name != '.env.example' and f.is_file():
            vals += _env_file_values(f)
    keep = {v for v in vals if len(v) >= 8 and v.lower() not in _PLACEHOLDERS and 'xxxx' not in v.lower()}
    return sorted(keep, key=len, reverse=True)


def redact_secrets(text: Any, secrets: Optional[List[str]] = None) -> Any:
    """text with every credential masked; non-strings are returned unchanged."""
    if not isinstance(text, str) or not text:
        return text
    for s in (known_secrets() if secrets is None else secrets):
        if s in text:
            text = text.replace(s, MASK)
    for rx, repl in PATTERNS:
        text = rx.sub(repl, text)
    return text


def redact_obj(obj: Any, secrets: Optional[List[str]] = None) -> Any:
    """redact_secrets over every string of a JSON-like structure."""
    secrets = known_secrets() if secrets is None else secrets
    if isinstance(obj, str):
        return redact_secrets(obj, secrets)
    if isinstance(obj, dict):
        return {k: redact_obj(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, secrets) for v in obj]
    return obj
