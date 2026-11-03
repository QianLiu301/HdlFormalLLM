"""flag probe link

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-03 14:52:55.838999
"""
from alembic import op
import sqlalchemy as sa


revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('sim_run', sa.Column('probe_of', sa.BigInteger(), nullable=True))
    op.create_foreign_key('fk_sim_run_probe_of', 'sim_run', 'sim_run', ['probe_of'], ['id'])


def downgrade():
    op.drop_constraint('fk_sim_run_probe_of', 'sim_run', type_='foreignkey')
    op.drop_column('sim_run', 'probe_of')
