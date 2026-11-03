"""Alembic environment for the explainability database.

The URL comes from DATABASE_URL (or -x url=...), never from a file in git.
"""
from alembic import context

from src.xdb import database_url
from src.xdb.models import Base

config = context.config
target_metadata = Base.metadata


def _url():
    return context.get_x_argument(as_dictionary=True).get('url') or \
        config.get_main_option('sqlalchemy.url') or database_url()


def run_migrations_offline():
    context.configure(url=_url(), target_metadata=target_metadata,
                      literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    from sqlalchemy import create_engine
    engine = create_engine(_url(), future=True)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata,
                          compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
