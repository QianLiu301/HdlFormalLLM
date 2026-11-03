"""
Explainability database (PostgreSQL + pgvector).

Design: docs/explainability_db_design.md. Everything here is opt-in: when
DATABASE_URL is not set, `enabled()` is False and the website behaves exactly
as before. Nothing in this package may raise into a web request; callers use
`ingest.ingest_run_safe`.

    DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
    python -m src.xdb upgrade          # create / migrate tables
    python -m src.xdb stats            # row counts
"""

import os
from contextlib import contextmanager
from functools import lru_cache

ENV_VAR = 'DATABASE_URL'


_DOTENV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), '.env')


def _from_dotenv():
    """DATABASE_URL from the project's .env (gitignored), so the password
    never has to live in a shell profile. Only this one key is read."""
    import re
    try:
        with open(_DOTENV, encoding='utf-8-sig') as f:      # -sig: tolerate a BOM
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        s = line.strip()
        if s.startswith('export '):
            s = s[len('export '):].strip()
        k, sep, v = s.partition('=')
        if sep and k.strip() == ENV_VAR:
            v = v.strip()
            # Neon's "psql" snippet: psql 'postgresql://...'
            m = re.search(r"postgres(?:ql)?://[^\s'\"]+", v)
            return m.group(0) if m else v.strip('"').strip("'")
    return None


def database_url():
    if 'XDB_DISABLE_DOTENV' in os.environ:          # tests
        url = (os.environ.get(ENV_VAR) or '').strip()
    else:
        url = (os.environ.get(ENV_VAR) or _from_dotenv() or '').strip()
    if not url:
        return None
    # Neon / Heroku style URLs; SQLAlchemy wants an explicit driver-less scheme
    if url.startswith('postgres://'):
        url = 'postgresql://' + url[len('postgres://'):]
    return url


def enabled() -> bool:
    return database_url() is not None


@lru_cache(maxsize=4)
def _engine_for(url):
    from sqlalchemy import create_engine
    # pool_pre_ping: Neon suspends idle computes, stale pooled connections
    # must be detected instead of failing the first query after a pause.
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5,
                         future=True)


def get_engine(url=None):
    url = url or database_url()
    if not url:
        raise RuntimeError(f'{ENV_VAR} is not set')
    return _engine_for(url)


@contextmanager
def session_scope(url=None):
    """One transaction: commit on success, roll back on any error."""
    from sqlalchemy.orm import Session
    with Session(get_engine(url), future=True) as s:
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
