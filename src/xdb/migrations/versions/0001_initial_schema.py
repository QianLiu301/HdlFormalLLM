"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-11-03 10:12:31.057284
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # pgvector is used from v2 on (semantic search); enabling it now keeps
    # later migrations free of superuser-only statements.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('artifact',
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=24), nullable=False),
    sa.Column('content', sa.LargeBinary(), nullable=False),
    sa.Column('orig_name', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('sha256')
    )
    op.create_table('canonical_step',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('module_type', sa.String(length=32), nullable=False),
    sa.Column('role', sa.String(length=8), nullable=False),
    sa.Column('template', sa.Text(), nullable=False),
    sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('predicate', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('module_type', 'template', 'version', name='uq_canonical_step')
    )
    op.create_table('design',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('module_type', sa.String(length=32), nullable=False),
    sa.Column('bitwidth', sa.Integer(), nullable=True),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source', 'name', name='uq_design_source_name')
    )
    op.create_table('feedback',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('subject_type', sa.String(length=24), nullable=False),
    sa.Column('subject_id', sa.BigInteger(), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('body', sa.Text(), nullable=True),
    sa.Column('proposed', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('status', sa.String(length=10), nullable=False),
    sa.Column('result_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('port',
    sa.Column('design_id', sa.BigInteger(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('direction', sa.String(length=8), nullable=False),
    sa.Column('width', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['design_id'], ['design.id'], ),
    sa.PrimaryKeyConstraint('design_id', 'name')
    )
    op.create_table('session',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('run_key', sa.String(length=128), nullable=False),
    sa.Column('run_id', sa.String(length=128), nullable=True),
    sa.Column('design_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['design_id'], ['design.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_key')
    )
    op.create_table('feature',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('session_id', sa.BigInteger(), nullable=False),
    sa.Column('artifact', sa.String(length=64), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_feature_prov_kind'),
    sa.ForeignKeyConstraint(['artifact'], ['artifact.sha256'], ),
    sa.ForeignKeyConstraint(['session_id'], ['session.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id', 'artifact', name='uq_feature_session_artifact')
    )
    op.create_table('llm_call',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('source_ref', sa.String(length=64), nullable=True),
    sa.Column('session_id', sa.BigInteger(), nullable=True),
    sa.Column('task_type', sa.String(length=64), nullable=True),
    sa.Column('provider', sa.String(length=64), nullable=True),
    sa.Column('model_requested', sa.String(length=128), nullable=True),
    sa.Column('model_served', sa.String(length=128), nullable=True),
    sa.Column('prompt_name', sa.String(length=64), nullable=True),
    sa.Column('prompt_version', sa.String(length=32), nullable=True),
    sa.Column('system_prompt', sa.Text(), nullable=True),
    sa.Column('prompt', sa.Text(), nullable=True),
    sa.Column('response', sa.Text(), nullable=True),
    sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['session.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_ref')
    )
    op.create_index('ix_llm_call_session', 'llm_call', ['session_id'], unique=False)
    op.create_table('requirement',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('supersedes', sa.BigInteger(), nullable=True),
    sa.Column('design_id', sa.BigInteger(), nullable=True),
    sa.Column('session_id', sa.BigInteger(), nullable=True),
    sa.Column('req_key', sa.String(length=32), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('origin', sa.String(length=16), nullable=False),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('uid', sa.UUID(), nullable=False),
    sa.Column('rev', sa.Integer(), nullable=False),
    sa.Column('is_current', sa.Boolean(), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_requirement_prov_kind'),
    sa.ForeignKeyConstraint(['design_id'], ['design.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['session.id'], ),
    sa.ForeignKeyConstraint(['supersedes'], ['requirement.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_requirement_session', 'requirement', ['session_id'], unique=False)
    op.create_index('uq_requirement_current', 'requirement', ['uid'], unique=True, postgresql_where=sa.text('is_current'))
    op.create_table('explanation_target',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('entry', sa.String(length=8), nullable=False),
    sa.Column('question', sa.Text(), nullable=True),
    sa.Column('subject_type', sa.String(length=24), nullable=True),
    sa.Column('subject_id', sa.BigInteger(), nullable=True),
    sa.Column('p_t', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('q_t', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('scope', sa.String(length=8), nullable=False),
    sa.Column('mode', sa.String(length=16), nullable=False),
    sa.Column('mapping_llm_call', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['mapping_llm_call'], ['llm_call.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('scenario',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('supersedes', sa.BigInteger(), nullable=True),
    sa.Column('feature_id', sa.BigInteger(), nullable=False),
    sa.Column('scenario_key', sa.String(length=64), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('tags', postgresql.ARRAY(sa.Text()), nullable=True),
    sa.Column('is_outline', sa.Boolean(), nullable=False),
    sa.Column('gherkin', sa.Text(), nullable=True),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('uid', sa.UUID(), nullable=False),
    sa.Column('rev', sa.Integer(), nullable=False),
    sa.Column('is_current', sa.Boolean(), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_scenario_prov_kind'),
    sa.ForeignKeyConstraint(['feature_id'], ['feature.id'], ),
    sa.ForeignKeyConstraint(['supersedes'], ['scenario.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_scenario_feature', 'scenario', ['feature_id'], unique=False)
    op.create_index('uq_scenario_current', 'scenario', ['uid'], unique=True, postgresql_where=sa.text('is_current'))
    op.create_table('testbench',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('session_id', sa.BigInteger(), nullable=False),
    sa.Column('feature_id', sa.BigInteger(), nullable=True),
    sa.Column('artifact', sa.String(length=64), nullable=False),
    sa.Column('generator', sa.String(length=16), nullable=False),
    sa.Column('oracle_source', sa.String(length=8), nullable=True),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_testbench_prov_kind'),
    sa.ForeignKeyConstraint(['artifact'], ['artifact.sha256'], ),
    sa.ForeignKeyConstraint(['feature_id'], ['feature.id'], ),
    sa.ForeignKeyConstraint(['session_id'], ['session.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id', 'artifact', name='uq_testbench_session_artifact')
    )
    op.create_table('example_row',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('supersedes', sa.BigInteger(), nullable=True),
    sa.Column('scenario_id', sa.BigInteger(), nullable=False),
    sa.Column('row_no', sa.Integer(), nullable=False),
    sa.Column('vals', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('prov_kind', sa.String(length=8), nullable=False),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('uid', sa.UUID(), nullable=False),
    sa.Column('rev', sa.Integer(), nullable=False),
    sa.Column('is_current', sa.Boolean(), nullable=False),
    sa.CheckConstraint("prov_kind IN ('human', 'llm', 'rule', 'import')", name='ck_example_row_prov_kind'),
    sa.ForeignKeyConstraint(['scenario_id'], ['scenario.id'], ),
    sa.ForeignKeyConstraint(['supersedes'], ['example_row.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_example_row_scenario', 'example_row', ['scenario_id'], unique=False)
    op.create_index('ix_example_row_vals', 'example_row', ['vals'], unique=False, postgresql_using='gin')
    op.create_index('uq_example_row_current', 'example_row', ['uid'], unique=True, postgresql_where=sa.text('is_current'))
    op.create_table('explanation',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('supersedes', sa.BigInteger(), nullable=True),
    sa.Column('target_id', sa.BigInteger(), nullable=False),
    sa.Column('algorithm', sa.String(length=32), nullable=False),
    sa.Column('selected', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('text_rule', sa.Text(), nullable=False),
    sa.Column('text_polished', sa.Text(), nullable=True),
    sa.Column('polish_llm_call', sa.BigInteger(), nullable=True),
    sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('auto_checks', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('uid', sa.UUID(), nullable=False),
    sa.Column('rev', sa.Integer(), nullable=False),
    sa.Column('is_current', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['polish_llm_call'], ['llm_call.id'], ),
    sa.ForeignKeyConstraint(['supersedes'], ['explanation.id'], ),
    sa.ForeignKeyConstraint(['target_id'], ['explanation_target.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('uq_explanation_current', 'explanation', ['uid'], unique=True, postgresql_where=sa.text('is_current'))
    op.create_table('scenario_requirement',
    sa.Column('scenario_id', sa.BigInteger(), nullable=False),
    sa.Column('requirement_id', sa.BigInteger(), nullable=False),
    sa.Column('method', sa.String(length=16), nullable=False),
    sa.Column('confidence', sa.REAL(), nullable=True),
    sa.Column('prov_ref', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['requirement_id'], ['requirement.id'], ),
    sa.ForeignKeyConstraint(['scenario_id'], ['scenario.id'], ),
    sa.PrimaryKeyConstraint('scenario_id', 'requirement_id', 'method')
    )
    op.create_table('scenario_step',
    sa.Column('scenario_id', sa.BigInteger(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('keyword', sa.String(length=8), nullable=False),
    sa.Column('role', sa.String(length=1), nullable=False),
    sa.Column('surface_text', sa.Text(), nullable=False),
    sa.Column('from_background', sa.Boolean(), nullable=False),
    sa.CheckConstraint("role IN ('P', 'Q')", name='ck_scenario_step_role'),
    sa.ForeignKeyConstraint(['scenario_id'], ['scenario.id'], ),
    sa.PrimaryKeyConstraint('scenario_id', 'position')
    )
    op.create_table('sim_run',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('testbench_id', sa.BigInteger(), nullable=False),
    sa.Column('dut_artifact', sa.String(length=64), nullable=True),
    sa.Column('target', sa.String(length=64), nullable=False),
    sa.Column('simulator', sa.String(length=64), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('log_artifact', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['dut_artifact'], ['artifact.sha256'], ),
    sa.ForeignKeyConstraint(['log_artifact'], ['artifact.sha256'], ),
    sa.ForeignKeyConstraint(['testbench_id'], ['testbench.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('testbench_id', 'dut_artifact', 'log_artifact', name='uq_sim_run_inputs')
    )
    op.create_table('check_item',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('testbench_id', sa.BigInteger(), nullable=False),
    sa.Column('tag', sa.String(length=96), nullable=False),
    sa.Column('label', sa.Text(), nullable=True),
    sa.Column('scenario_id', sa.BigInteger(), nullable=True),
    sa.Column('example_row_id', sa.BigInteger(), nullable=True),
    sa.Column('strength', sa.String(length=8), nullable=False),
    sa.Column('weak_reason', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['example_row_id'], ['example_row.id'], ),
    sa.ForeignKeyConstraint(['scenario_id'], ['scenario.id'], ),
    sa.ForeignKeyConstraint(['testbench_id'], ['testbench.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('testbench_id', 'tag', name='uq_check_item_tb_tag')
    )
    op.create_index('ix_check_item_scenario', 'check_item', ['scenario_id'], unique=False)
    op.create_table('step_mapping',
    sa.Column('scenario_id', sa.BigInteger(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('method', sa.String(length=16), nullable=False),
    sa.Column('canonical_step_id', sa.BigInteger(), nullable=True),
    sa.Column('params', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('method_ref', sa.Text(), nullable=True),
    sa.Column('confidence', sa.REAL(), nullable=True),
    sa.ForeignKeyConstraint(['canonical_step_id'], ['canonical_step.id'], ),
    sa.ForeignKeyConstraint(['scenario_id', 'position'], ['scenario_step.scenario_id', 'scenario_step.position'], ),
    sa.PrimaryKeyConstraint('scenario_id', 'position', 'method')
    )
    op.create_table('trace_event',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('sim_run_id', sa.BigInteger(), nullable=False),
    sa.Column('sim_time', sa.BigInteger(), nullable=False),
    sa.Column('port', sa.String(length=64), nullable=False),
    sa.Column('value', sa.Text(), nullable=False),
    sa.Column('canonical_step_id', sa.BigInteger(), nullable=True),
    sa.ForeignKeyConstraint(['canonical_step_id'], ['canonical_step.id'], ),
    sa.ForeignKeyConstraint(['sim_run_id'], ['sim_run.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_trace_event_run', 'trace_event', ['sim_run_id'], unique=False)
    op.create_table('check_result',
    sa.Column('sim_run_id', sa.BigInteger(), nullable=False),
    sa.Column('check_id', sa.BigInteger(), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('expected', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('observed', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('sim_time', sa.BigInteger(), nullable=True),
    sa.Column('message', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['check_id'], ['check_item.id'], ),
    sa.ForeignKeyConstraint(['sim_run_id'], ['sim_run.id'], ),
    sa.PrimaryKeyConstraint('sim_run_id', 'check_id')
    )


def downgrade():
    op.drop_table('check_result')
    op.drop_index('ix_trace_event_run', table_name='trace_event')
    op.drop_table('trace_event')
    op.drop_table('step_mapping')
    op.drop_index('ix_check_item_scenario', table_name='check_item')
    op.drop_table('check_item')
    op.drop_table('sim_run')
    op.drop_table('scenario_step')
    op.drop_table('scenario_requirement')
    op.drop_index('uq_explanation_current', table_name='explanation', postgresql_where=sa.text('is_current'))
    op.drop_table('explanation')
    op.drop_index('uq_example_row_current', table_name='example_row', postgresql_where=sa.text('is_current'))
    op.drop_index('ix_example_row_vals', table_name='example_row', postgresql_using='gin')
    op.drop_index('ix_example_row_scenario', table_name='example_row')
    op.drop_table('example_row')
    op.drop_table('testbench')
    op.drop_index('uq_scenario_current', table_name='scenario', postgresql_where=sa.text('is_current'))
    op.drop_index('ix_scenario_feature', table_name='scenario')
    op.drop_table('scenario')
    op.drop_table('explanation_target')
    op.drop_index('uq_requirement_current', table_name='requirement', postgresql_where=sa.text('is_current'))
    op.drop_index('ix_requirement_session', table_name='requirement')
    op.drop_table('requirement')
    op.drop_index('ix_llm_call_session', table_name='llm_call')
    op.drop_table('llm_call')
    op.drop_table('feature')
    op.drop_table('session')
    op.drop_table('port')
    op.drop_table('feedback')
    op.drop_table('design')
    op.drop_table('canonical_step')
    op.drop_table('artifact')
