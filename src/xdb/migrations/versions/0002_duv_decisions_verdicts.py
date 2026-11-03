"""duv, decisions, verdicts

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03 13:03:04.530939
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('decision',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('session_id', sa.BigInteger(), nullable=False),
    sa.Column('question_key', sa.String(length=16), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('options', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('default_index', sa.Integer(), nullable=True),
    sa.Column('answer', sa.Text(), nullable=False),
    sa.Column('answer_source', sa.String(length=16), nullable=True),
    sa.Column('why', sa.Text(), nullable=True),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_decision_prov_kind'),
    sa.ForeignKeyConstraint(['session_id'], ['session.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id', 'question_key', name='uq_decision_session_q')
    )
    op.create_table('duv',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('session_id', sa.BigInteger(), nullable=True),
    sa.Column('design_id', sa.BigInteger(), nullable=False),
    sa.Column('artifact', sa.String(length=64), nullable=False),
    sa.Column('module_name', sa.String(length=128), nullable=True),
    sa.Column('origin', sa.String(length=12), nullable=False),
    sa.Column('parent_id', sa.BigInteger(), nullable=True),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('workflow_mode', sa.String(length=16), nullable=True),
    sa.Column('prompt_has_bdd', sa.Boolean(), nullable=True),
    sa.Column('based_on_feature_id', sa.BigInteger(), nullable=True),
    sa.Column('prompt_has_decisions', sa.Boolean(), nullable=True),
    sa.Column('compiles', sa.Boolean(), nullable=True),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_duv_prov_kind'),
    sa.ForeignKeyConstraint(['artifact'], ['artifact.sha256'], ),
    sa.ForeignKeyConstraint(['based_on_feature_id'], ['feature.id'], ),
    sa.ForeignKeyConstraint(['design_id'], ['design.id'], ),
    sa.ForeignKeyConstraint(['parent_id'], ['duv.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['session.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id', 'artifact', name='uq_duv_session_artifact')
    )
    op.create_index('ix_duv_session', 'duv', ['session_id'], unique=False)
    op.add_column('artifact', sa.Column('encoding', sa.String(length=16), server_default='identity', nullable=False))
    op.add_column('check_item', sa.Column('seq', sa.Integer(), server_default='1', nullable=False))
    op.add_column('check_item', sa.Column('test_no', sa.Integer(), nullable=True))
    op.add_column('check_item', sa.Column('port', sa.String(length=64), nullable=True))
    op.add_column('check_item', sa.Column('width', sa.Integer(), nullable=True))
    op.add_column('check_item', sa.Column('operation', sa.String(length=32), nullable=True))
    op.add_column('check_item', sa.Column('stimulus', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('check_item', sa.Column('expected_bdd', sa.Numeric(precision=40, scale=0), nullable=True))
    op.add_column('check_item', sa.Column('expected_ref', sa.Numeric(precision=40, scale=0), nullable=True))
    op.add_column('check_item', sa.Column('expected_used', sa.Numeric(precision=40, scale=0), nullable=True))
    op.add_column('check_item', sa.Column('oracle', sa.String(length=8), nullable=True))
    op.drop_constraint('uq_check_item_tb_tag', 'check_item', type_='unique')
    op.create_unique_constraint('uq_check_item_tb_tag_seq', 'check_item', ['testbench_id', 'tag', 'seq'])
    op.add_column('check_result', sa.Column('observed_value', sa.Numeric(precision=40, scale=0), nullable=True))
    op.add_column('check_result', sa.Column('verdict', sa.String(length=16), nullable=True))
    op.create_index('ix_check_result_verdict', 'check_result', ['verdict'], unique=False)
    op.create_check_constraint('ck_check_result_verdict', 'check_result',
        "verdict IS NULL OR verdict IN ('pass', 'scenario_wrong', 'design_wrong', "
        "'both_wrong', 'consistent_wrong', 'undecided')")
    op.add_column('explanation', sa.Column('structure', postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                                           server_default=sa.text("'{}'::jsonb")))
    op.add_column('explanation_target', sa.Column('feature_id', sa.BigInteger(), nullable=True))
    op.add_column('explanation_target', sa.Column('sim_run_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key('fk_explanation_target_sim_run', 'explanation_target', 'sim_run', ['sim_run_id'], ['id'])
    op.create_foreign_key('fk_explanation_target_feature', 'explanation_target', 'feature', ['feature_id'], ['id'])
    op.add_column('feature', sa.Column('parent_id', sa.BigInteger(), nullable=True))
    op.add_column('feature', sa.Column('reason', sa.Text(), nullable=True))
    op.add_column('feature', sa.Column('workflow_mode', sa.String(length=16), nullable=True))
    op.add_column('feature', sa.Column('prompt_has_decisions', sa.Boolean(), nullable=True))
    op.create_foreign_key('fk_feature_parent', 'feature', 'feature', ['parent_id'], ['id'])
    op.add_column('sim_run', sa.Column('duv_id', sa.BigInteger(), nullable=True))
    op.create_index('ix_sim_run_duv', 'sim_run', ['duv_id'], unique=False)
    op.create_foreign_key('fk_sim_run_duv', 'sim_run', 'duv', ['duv_id'], ['id'])


def downgrade():
    op.drop_constraint('fk_sim_run_duv', 'sim_run', type_='foreignkey')
    op.drop_index('ix_sim_run_duv', table_name='sim_run')
    op.drop_column('sim_run', 'duv_id')
    op.drop_constraint('fk_feature_parent', 'feature', type_='foreignkey')
    op.drop_column('feature', 'prompt_has_decisions')
    op.drop_column('feature', 'workflow_mode')
    op.drop_column('feature', 'reason')
    op.drop_column('feature', 'parent_id')
    op.drop_constraint('fk_explanation_target_sim_run', 'explanation_target', type_='foreignkey')
    op.drop_constraint('fk_explanation_target_feature', 'explanation_target', type_='foreignkey')
    op.drop_column('explanation_target', 'sim_run_id')
    op.drop_column('explanation_target', 'feature_id')
    op.drop_column('explanation', 'structure')
    op.drop_constraint('ck_check_result_verdict', 'check_result', type_='check')
    op.drop_index('ix_check_result_verdict', table_name='check_result')
    op.drop_column('check_result', 'verdict')
    op.drop_column('check_result', 'observed_value')
    op.drop_constraint('uq_check_item_tb_tag_seq', 'check_item', type_='unique')
    op.create_unique_constraint('uq_check_item_tb_tag', 'check_item', ['testbench_id', 'tag'])
    op.drop_column('check_item', 'oracle')
    op.drop_column('check_item', 'expected_used')
    op.drop_column('check_item', 'expected_ref')
    op.drop_column('check_item', 'expected_bdd')
    op.drop_column('check_item', 'stimulus')
    op.drop_column('check_item', 'operation')
    op.drop_column('check_item', 'width')
    op.drop_column('check_item', 'port')
    op.drop_column('check_item', 'test_no')
    op.drop_column('check_item', 'seq')
    op.drop_column('artifact', 'encoding')
    op.drop_index('ix_duv_session', table_name='duv')
    op.drop_table('duv')
    op.drop_table('decision')
