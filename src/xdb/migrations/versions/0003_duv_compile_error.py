"""duv compile error

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03 13:24:04.478258
"""
from alembic import op
import sqlalchemy as sa


revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('duv', sa.Column('compile_error', sa.Text(), nullable=True))


def downgrade():
    op.drop_column('duv', 'compile_error')
