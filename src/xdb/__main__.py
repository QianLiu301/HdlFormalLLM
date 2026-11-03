"""
CLI for the explainability database.

    python -m src.xdb upgrade              create / migrate tables (needs DATABASE_URL)
    python -m src.xdb current              show the applied migration
    python -m src.xdb stats                row count per table
    python -m src.xdb ingest --testbench output/testbench/.../x_tb.v --log output/.../x.log
                             [--bdd ...] [--duv ...] [--requirements ...] [--run-id ...]
    python -m src.xdb remap                map steps of stored scenarios to canonical steps
    python -m src.xdb runs                 recent simulations (ids for --run)
    python -m src.xdb explain ALU-SUB-001/r7 [--run N]       local: one check
    python -m src.xdb explain --failures [--run N]           local: every failing check
    python -m src.xdb explain --global overflow=True [--feature N]
    python -m src.xdb explain --requirement REQ-2 [--session N]
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
    sub.add_parser('remap')
    sub.add_parser('runs')
    ex = sub.add_parser('explain')
    ex.add_argument('tag', nargs='?')
    ex.add_argument('--run', type=int)
    ex.add_argument('--session', type=int)
    ex.add_argument('--feature', type=int)
    ex.add_argument('--global', dest='global_', metavar='PORT=VALUE')
    ex.add_argument('--requirement')
    ex.add_argument('--failures', action='store_true')
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
    elif a.cmd == 'remap':
        from src.xdb.mapping import remap_all
        with xdb.session_scope() as s:
            print(remap_all(s))
    elif a.cmd == 'runs':
        from src.xdb.queries import recent_runs
        for r in recent_runs():
            print(f"run #{r['sim_run_id']:<4} session {r['session_id']:<4} feature {r['feature_id']}  "
                  f"{r['design']:<14} {r['created_at']:%Y-%m-%d %H:%M}  {r['verdicts']}")
    elif a.cmd == 'explain':
        _explain(a)
    elif a.cmd == 'ingest':
        from src.xdb.ingest import ingest_run
        paths = {'testbench': a.testbench, 'log': a.log, 'bdd': a.bdd,
                 'duv': a.duv, 'requirements': a.requirements}
        print(ingest_run(PROJECT_ROOT, paths, run_id=getattr(a, 'run_id')))


def _explain(a):
    from sqlalchemy import select
    from src import xdb
    from src.xdb import explain as E
    from src.xdb import models as M
    if a.global_:
        port, _, value = a.global_.partition('=')
        feature = a.feature
        if feature is None:
            with xdb.session_scope() as s:
                feature = s.scalar(select(M.Testbench.feature_id).join(M.SimRun)
                                   .order_by(M.SimRun.id.desc()).limit(1))
        print(E.explain_global(feature, port.strip(), value.strip())['text'])
    elif a.requirement:
        with xdb.session_scope() as s:
            q = select(M.Requirement.id).where(M.Requirement.req_key == a.requirement,
                                               M.Requirement.is_current)
            if a.session:
                q = q.where(M.Requirement.session_id == a.session)
            rid = s.scalar(q.order_by(M.Requirement.id.desc()).limit(1))
        if rid is None:
            sys.exit(f'requirement {a.requirement} not found')
        print(E.explain_requirement(rid, sim_run_id=a.run)['text'])
    elif a.failures:
        with xdb.session_scope() as s:
            run = a.run or s.scalar(select(M.SimRun.id).order_by(M.SimRun.id.desc()).limit(1))
            tags = s.scalars(select(M.CheckItem.tag).join(M.CheckResult)
                             .where(M.CheckResult.sim_run_id == run, M.CheckResult.status == 'fail')
                             .order_by(M.CheckItem.test_no)).all()
        if not tags:
            print(f'simulation #{run}: no failing checks')
        for tg in tags:
            print(E.explain_check(tg, sim_run_id=run)['text'], end='\n\n')
    elif a.tag:
        out = E.explain_check(a.tag, sim_run_id=a.run, session_id=a.session)
        print(out['text'] if out else f'no simulated check {a.tag} found')
    else:
        sys.exit('give a check tag, --failures, --global PORT=VALUE or --requirement REQ-n')


if __name__ == '__main__':
    main()
