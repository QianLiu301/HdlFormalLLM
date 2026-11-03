"""
CLI for the explainability database.

    python -m src.xdb upgrade              create / migrate tables (needs DATABASE_URL)
    python -m src.xdb current              show the applied migration
    python -m src.xdb stats                row count per table
    python -m src.xdb ingest --testbench output/testbench/.../x_tb.v --log output/.../x.log
                             [--bdd ...] [--duv ...] [--requirements ...] [--run-id ...]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url=None):
    from alembic.config import Config
    cfg = Config()
    cfg.set_main_option('script_location', str(Path(__file__).parent / 'migrations'))
    if url:
        cfg.set_main_option('sqlalchemy.url', url.replace('%', '%%'))
    return cfg


def upgrade(url=None, revision='head'):
    from alembic import command
    command.upgrade(alembic_config(url), revision)


def main(argv=None):
    from src import xdb
    ap = argparse.ArgumentParser(prog='python -m src.xdb')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('upgrade')
    sub.add_parser('current')
    sub.add_parser('stats')
    rev = sub.add_parser('revision', help='autogenerate a migration (developers)')
    rev.add_argument('-m', '--message', required=True)
    ing = sub.add_parser('ingest')
    for k in ('testbench', 'log', 'bdd', 'duv', 'requirements', 'run-id'):
        ing.add_argument(f'--{k}')
    a = ap.parse_args(argv)

    if not xdb.enabled():
        sys.exit(f'{xdb.ENV_VAR} is not set')
    from alembic import command
    if a.cmd == 'upgrade':
        upgrade()
        print('database is at head')
    elif a.cmd == 'current':
        command.current(alembic_config(), verbose=True)
    elif a.cmd == 'revision':
        command.revision(alembic_config(), message=a.message, autogenerate=True)
    elif a.cmd == 'stats':
        from src.xdb.queries import storage_usage, table_counts
        for t, n in table_counts().items():
            print(f'{t:22s} {n}')
        u = storage_usage()
        print(f"\nstorage: database {u['database_mb']} MB of {u['limit_mb']:.0f} MB "
              f"({u['used_pct']}%), of which files {u['artifacts_mb']} MB, "
              f"LLM prompts/responses {u['llm_text_mb']} MB")
    elif a.cmd == 'ingest':
        from src.xdb.ingest import ingest_run
        paths = {'testbench': a.testbench, 'log': a.log, 'bdd': a.bdd,
                 'duv': a.duv, 'requirements': a.requirements}
        print(ingest_run(PROJECT_ROOT, paths, run_id=getattr(a, 'run_id')))


if __name__ == '__main__':
    main()
