"""
Refuse to commit credentials (pre-commit hook; see .githooks/pre-commit).

Checks what would be committed:
  * files that hold secrets by name (.env, .env.*, config/llm_config.json, *.pem, id_rsa ...);
  * every added line, for the keys configured on this machine (exact match,
    whatever their format) and for common key formats (src/redact.py:
    Google "AIza…", "sk-…", "gsk_…", "xai-…", bearer tokens, "?key=…",
    passwords in connection URLs other than localhost test databases).

    python tools/check_secrets.py            # staged changes (what the hook runs)
    python tools/check_secrets.py --all      # every tracked file

One-time setup in a clone:  git config core.hooksPath .githooks
Exit status 1 and a list of findings (values masked) when something is found.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.redact import PATTERNS, known_secrets, redact_secrets  # noqa: E402

SECRET_FILES = re.compile(r'(^|/)(\.env(\..+)?|llm_config\.json|.*\.pem|id_rsa|id_ed25519|.*\.p12|credentials\.json)$')
ALLOWED_FILES = {'.env.example', 'config/llm_config.example.json'}
# test databases on this machine are not secrets
LOCAL_URL = re.compile(r'://[^:/\s@]+:[^@\s/]+@(localhost|127\.0\.0\.1)\b')
# documentation placeholders
PLACEHOLDER = re.compile(r'((?i:user:pass(?:word)?@)|xxxx|<[^>]+>|\{[^}]*\}|\*\*\*|…|\.\.\.)')
# files that define the key patterns or hold deliberately fake keys: only the
# check against the keys configured on this machine applies to them
PATTERN_EXEMPT = {'src/redact.py', 'tools/check_secrets.py', 'tests/test_secrets.py'}


def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace').stdout


def staged():
    """(path, [(line number, text)]) of every added line in the index."""
    names = [n for n in _git('diff', '--cached', '--name-only', '--diff-filter=ACMR').splitlines() if n]
    diff = _git('diff', '--cached', '-U0', '--no-color', '--diff-filter=ACMR')
    lines, path, n = {}, None, 0
    for ln in diff.splitlines():
        if ln.startswith('+++ '):
            path = ln[6:] if ln.startswith('+++ b/') else None
        elif ln.startswith('@@'):
            m = re.search(r'\+(\d+)', ln)
            n = int(m.group(1)) if m else 0
        elif ln.startswith('+') and path:
            lines.setdefault(path, []).append((n, ln[1:]))
            n += 1
    return names, lines


def tracked():
    names = [n for n in _git('ls-files').splitlines() if n]
    lines = {}
    for name in names:
        p = ROOT / name
        try:
            if p.stat().st_size > 5_000_000:
                continue
            text = p.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        lines[name] = list(enumerate(text.splitlines(), 1))
    return names, lines


def findings(names, lines, secrets):
    out = []
    for name in names:
        if SECRET_FILES.search(name) and name not in ALLOWED_FILES:
            out.append(f'{name}: this file holds secrets and must not be committed')
    for name, rows in lines.items():
        for n, text in rows:
            hit = next((s for s in secrets if s in text), None)
            what = 'a key configured on this machine' if hit else None
            if not what and name not in PATTERN_EXEMPT:
                probe = LOCAL_URL.sub('://local@', text)
                for rx, _ in PATTERNS:
                    m = rx.search(probe)
                    if m and not PLACEHOLDER.search(m.group(0)):
                        what = 'something that looks like a key or password'
                        break
            if what:
                out.append(f'{name}:{n}: {what}: {redact_secrets(text.strip(), secrets)[:140]}')
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    names, lines = tracked() if '--all' in argv else staged()
    found = findings(names, lines, known_secrets(ROOT))
    if not found:
        return 0
    print('Refusing to commit: credentials found (values masked).', file=sys.stderr)
    for f in found:
        print('  ' + f, file=sys.stderr)
    print('Remove them (git restore --staged <file>, or edit the line). A file that holds secrets\n'
          'belongs in .gitignore. If this is a false alarm: git commit --no-verify', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
