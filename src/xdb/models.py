"""
ORM models for the explainability database (design doc §5, v1 tables).

Conventions
- Revisioned tables (requirement, scenario, example_row, explanation) carry
  uid / rev / supersedes / is_current. Rows are never edited in place except
  to clear is_current when a newer revision is added.
- Generated rows carry prov_kind ∈ {human, llm, rule, import} and prov_ref.
- Artifacts are content-addressed (SHA-256) and stored in the database.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (BigInteger, Boolean, CheckConstraint, DateTime,
                        ForeignKey, ForeignKeyConstraint, Index, Integer,
                        LargeBinary, Numeric, REAL as Real, String, Text,
                        UniqueConstraint, text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PROV_KINDS = ('human', 'llm', 'rule', 'import')

# Who is wrong when a check is evaluated (design doc §0, D14). O = observed,
# B = expected value written in the BDD scenario, R = reference model.
VERDICTS = ('pass',              # O = B = R
            'scenario_wrong',    # O != B, O = R
            'design_wrong',      # O != B, B = R
            'both_wrong',        # O, B, R all different
            'consistent_wrong',  # O = B != R: test passes, both deviate the same way
            'undecided')         # no reference model (or a value is missing)

# Exact integers wider than 64 bits (values are stored unsigned, masked to the port)
WideInt = Numeric(40, 0)


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


def _created():
    return mapped_column(DateTime(timezone=True), nullable=False, default=_now,
                         server_default=text('now()'))


def _prov_check(table):
    return CheckConstraint(f"prov_kind IN {PROV_KINDS}", name=f'ck_{table}_prov_kind')


class RevisionMixin:
    uid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False,
                                           default=uuid.uuid4)
    rev: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


# ----------------------------------------------------------------- core

class Design(Base):
    """What is being designed (spec level), e.g. alu_32bit. Concrete Verilog
    implementations of it are rows of `duv`."""
    __tablename__ = 'design'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    module_type: Mapped[str] = mapped_column(String(32), nullable=False)
    bitwidth: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16), nullable=False)   # web | benchmark
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at = _created()
    __table_args__ = (UniqueConstraint('source', 'name', name='uq_design_source_name'),)


class Port(Base):
    __tablename__ = 'port'
    design_id: Mapped[int] = mapped_column(ForeignKey('design.id'), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)


class Artifact(Base):
    __tablename__ = 'artifact'
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # identity = stored as is; gzip later when storage gets tight (D18).
    # Readers must go through artifact_text()/artifact_bytes().
    encoding: Mapped[str] = mapped_column(String(16), nullable=False, default='identity',
                                          server_default='identity')
    orig_name: Mapped[str | None] = mapped_column(Text)
    created_at = _created()


class Session(Base):
    """One pass through the website's Steps 0-4. No personal data."""
    __tablename__ = 'session'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    run_id: Mapped[str | None] = mapped_column(String(128))       # experiment_logger run_id
    design_id: Mapped[int | None] = mapped_column(ForeignKey('design.id'))
    created_at = _created()
    note: Mapped[str | None] = mapped_column(Text)


class LlmCall(Base):
    __tablename__ = 'llm_call'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_ref: Mapped[str | None] = mapped_column(String(64), unique=True)  # sqlite:<id>
    session_id: Mapped[int | None] = mapped_column(ForeignKey('session.id'))
    task_type: Mapped[str | None] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(64))
    model_requested: Mapped[str | None] = mapped_column(String(128))
    model_served: Mapped[str | None] = mapped_column(String(128))
    prompt_name: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    prompt: Mapped[str | None] = mapped_column(Text)
    response: Mapped[str | None] = mapped_column(Text)
    params: Mapped[dict | None] = mapped_column(JSONB)
    usage: Mapped[dict | None] = mapped_column(JSONB)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at = _created()


# ----------------------------------------------------------------- model M

class Requirement(RevisionMixin, Base):
    __tablename__ = 'requirement'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    supersedes: Mapped[int | None] = mapped_column(ForeignKey('requirement.id'))
    design_id: Mapped[int | None] = mapped_column(ForeignKey('design.id'))
    session_id: Mapped[int | None] = mapped_column(ForeignKey('session.id'))
    req_key: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    origin: Mapped[str] = mapped_column(String(16), nullable=False)  # spec_fp | req_step | user
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (_prov_check('requirement'),)


class Feature(Base):
    __tablename__ = 'feature'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('session.id'), nullable=False)
    artifact: Mapped[str] = mapped_column(ForeignKey('artifact.sha256'), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    # File-level versions (D16): a regenerated or corrected feature is a new
    # row pointing at the one it replaces, with the reason.
    parent_id: Mapped[int | None] = mapped_column(ForeignKey('feature.id'))
    reason: Mapped[str | None] = mapped_column(Text)       # generated | regenerated | feedback:<id>
    workflow_mode: Mapped[str | None] = mapped_column(String(16))   # implementation | specification
    prompt_has_decisions: Mapped[bool | None] = mapped_column(Boolean)
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (UniqueConstraint('session_id', 'artifact', name='uq_feature_session_artifact'),
                      _prov_check('feature'))


class Decision(Base):
    """Step 0: an open point of the description and the answer given. The same
    answers are added to the DUV and the BDD prompts, so they explain choices
    both sides made (e.g. which way a counter counts after a load)."""
    __tablename__ = 'decision'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('session.id'), nullable=False)
    question_key: Mapped[str] = mapped_column(String(16), nullable=False)    # Q1
    question: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list | None] = mapped_column(JSONB)
    default_index: Mapped[int | None] = mapped_column(Integer)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    answer_source: Mapped[str | None] = mapped_column(String(16))           # default | user | custom
    why: Mapped[str | None] = mapped_column(Text)
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (UniqueConstraint('session_id', 'question_key', name='uq_decision_session_q'),
                      _prov_check('decision'))


class Duv(Base):
    """One concrete Verilog implementation (generated, uploaded, golden or mutant)."""
    __tablename__ = 'duv'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey('session.id'))
    design_id: Mapped[int] = mapped_column(ForeignKey('design.id'), nullable=False)
    artifact: Mapped[str] = mapped_column(ForeignKey('artifact.sha256'), nullable=False)
    module_name: Mapped[str | None] = mapped_column(String(128))
    origin: Mapped[str] = mapped_column(String(12), nullable=False)   # llm | upload | golden | mutant | unknown
    parent_id: Mapped[int | None] = mapped_column(ForeignKey('duv.id'))
    reason: Mapped[str | None] = mapped_column(Text)                 # generated | regenerated | feedback:<id>
    workflow_mode: Mapped[str | None] = mapped_column(String(16))
    prompt_has_bdd: Mapped[bool | None] = mapped_column(Boolean)
    based_on_feature_id: Mapped[int | None] = mapped_column(ForeignKey('feature.id'))
    prompt_has_decisions: Mapped[bool | None] = mapped_column(Boolean)
    compiles: Mapped[bool | None] = mapped_column(Boolean)
    # first compiler error of the Step-1 check (iverilog -g2012), e.g.
    # "alu.v:70: syntax error" -- the explanation when nothing can be simulated
    compile_error: Mapped[str | None] = mapped_column(Text)
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (UniqueConstraint('session_id', 'artifact', name='uq_duv_session_artifact'),
                      _prov_check('duv'))


class Scenario(RevisionMixin, Base):
    __tablename__ = 'scenario'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    supersedes: Mapped[int | None] = mapped_column(ForeignKey('scenario.id'))
    feature_id: Mapped[int] = mapped_column(ForeignKey('feature.id'), nullable=False)
    scenario_key: Mapped[str] = mapped_column(String(64), nullable=False)  # ALU-ADD-001
    title: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    is_outline: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gherkin: Mapped[str | None] = mapped_column(Text)      # verbatim block from the feature file
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (_prov_check('scenario'),)


class ExampleRow(RevisionMixin, Base):
    __tablename__ = 'example_row'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    supersedes: Mapped[int | None] = mapped_column(ForeignKey('example_row.id'))
    scenario_id: Mapped[int] = mapped_column(ForeignKey('scenario.id'), nullable=False)
    row_no: Mapped[int] = mapped_column(Integer, nullable=False)   # ALU-ADD-001/r3 -> 3
    vals: Mapped[dict] = mapped_column(JSONB, nullable=False)
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (_prov_check('example_row'),)


class CanonicalStep(Base):
    """Alphabet Σ (design doc §4). Filled by phase 3."""
    __tablename__ = 'canonical_step'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    module_type: Mapped[str] = mapped_column(String(32), nullable=False)
    role: Mapped[str] = mapped_column(String(8), nullable=False)    # context | action | outcome
    template: Mapped[str] = mapped_column(Text, nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    predicate: Mapped[dict | None] = mapped_column(JSONB)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __table_args__ = (UniqueConstraint('module_type', 'template', 'version',
                                       name='uq_canonical_step'),)


class ScenarioStep(Base):
    __tablename__ = 'scenario_step'
    scenario_id: Mapped[int] = mapped_column(ForeignKey('scenario.id'), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(8), nullable=False)   # Given When Then And But *
    role: Mapped[str] = mapped_column(String(1), nullable=False)      # P (Given/When) | Q (Then)
    surface_text: Mapped[str] = mapped_column(Text, nullable=False)
    from_background: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    __table_args__ = (CheckConstraint("role IN ('P', 'Q')", name='ck_scenario_step_role'),)


class StepMapping(Base):
    __tablename__ = 'step_mapping'
    scenario_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String(16), primary_key=True)  # rule | embedding | llm | human
    canonical_step_id: Mapped[int | None] = mapped_column(ForeignKey('canonical_step.id'))
    params: Mapped[dict | None] = mapped_column(JSONB)
    method_ref: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Real)
    __table_args__ = (ForeignKeyConstraint(['scenario_id', 'position'],
                                           ['scenario_step.scenario_id', 'scenario_step.position']),)


class ScenarioRequirement(Base):
    __tablename__ = 'scenario_requirement'
    scenario_id: Mapped[int] = mapped_column(ForeignKey('scenario.id'), primary_key=True)
    requirement_id: Mapped[int] = mapped_column(ForeignKey('requirement.id'), primary_key=True)
    # llm_tag = @req_N tag written by the model that generated the scenario (its claim)
    method: Mapped[str] = mapped_column(String(16), primary_key=True)  # llm_tag | rule | judge_llm | human
    confidence: Mapped[float | None] = mapped_column(Real)
    prov_ref: Mapped[str | None] = mapped_column(Text)


# ----------------------------------------------------------------- tests, belief B

class Testbench(Base):
    __tablename__ = 'testbench'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey('session.id'), nullable=False)
    feature_id: Mapped[int | None] = mapped_column(ForeignKey('feature.id'))
    artifact: Mapped[str] = mapped_column(ForeignKey('artifact.sha256'), nullable=False)
    generator: Mapped[str] = mapped_column(String(16), nullable=False)    # template | llm | probe
    oracle_source: Mapped[str | None] = mapped_column(String(8))         # bdd | spec
    prov_kind: Mapped[str] = mapped_column(String(8), nullable=False)
    prov_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()
    __table_args__ = (UniqueConstraint('session_id', 'artifact', name='uq_testbench_session_artifact'),
                      _prov_check('testbench'))


class CheckItem(Base):
    __tablename__ = 'check_item'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    testbench_id: Mapped[int] = mapped_column(ForeignKey('testbench.id'), nullable=False)
    tag: Mapped[str] = mapped_column(String(96), nullable=False)        # BDD:ALU-ADD-001/r3 | GEN:RF-R0
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default='1')
    label: Mapped[str | None] = mapped_column(Text)
    # What this assertion actually does (testbench index v2); NULL for older testbenches
    test_no: Mapped[int | None] = mapped_column(Integer)
    port: Mapped[str | None] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer)
    operation: Mapped[str | None] = mapped_column(String(32))
    stimulus: Mapped[dict | None] = mapped_column(JSONB)
    expected_bdd = mapped_column(WideInt)        # value written in the scenario
    expected_ref = mapped_column(WideInt)        # reference model; NULL = none exists
    expected_used = mapped_column(WideInt)       # what the testbench compares against
    oracle: Mapped[str | None] = mapped_column(String(8))                # bdd | spec
    scenario_id: Mapped[int | None] = mapped_column(ForeignKey('scenario.id'))
    example_row_id: Mapped[int | None] = mapped_column(ForeignKey('example_row.id'))
    strength: Mapped[str] = mapped_column(String(8), nullable=False, default='normal')  # normal | weak
    weak_reason: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (UniqueConstraint('testbench_id', 'tag', 'seq', name='uq_check_item_tb_tag_seq'),)


class SimRun(Base):
    __tablename__ = 'sim_run'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    testbench_id: Mapped[int] = mapped_column(ForeignKey('testbench.id'), nullable=False)
    duv_id: Mapped[int | None] = mapped_column(ForeignKey('duv.id'))
    # set on a flag-probe run (src/xdb/probe.py): the main simulation it observes
    probe_of: Mapped[int | None] = mapped_column(ForeignKey('sim_run.id'))
    dut_artifact: Mapped[str | None] = mapped_column(ForeignKey('artifact.sha256'))
    target: Mapped[str] = mapped_column(String(64), nullable=False)   # generated_duv | golden | mutant:<file>
    simulator: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False)   # ok | compile_error | timeout
    log_artifact: Mapped[str] = mapped_column(ForeignKey('artifact.sha256'), nullable=False)
    created_at = _created()
    __table_args__ = (UniqueConstraint('testbench_id', 'dut_artifact', 'log_artifact',
                                       name='uq_sim_run_inputs'),)


class CheckResult(Base):
    __tablename__ = 'check_result'
    sim_run_id: Mapped[int] = mapped_column(ForeignKey('sim_run.id'), primary_key=True)
    check_id: Mapped[int] = mapped_column(ForeignKey('check_item.id'), primary_key=True)
    # pass | fail | not_tested (skipped / never reported) | setup
    status: Mapped[str] = mapped_column(String(12), nullable=False)
    expected: Mapped[dict | None] = mapped_column(JSONB)
    observed: Mapped[dict | None] = mapped_column(JSONB)       # raw text from the log, e.g. {"result": "x"}
    observed_value = mapped_column(WideInt)                    # O, when it is a number
    verdict: Mapped[str | None] = mapped_column(String(16))    # see VERDICTS; NULL when not run
    sim_time: Mapped[int | None] = mapped_column(BigInteger)
    message: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (CheckConstraint(f"verdict IS NULL OR verdict IN {VERDICTS}",
                                      name='ck_check_result_verdict'),)


class TraceEvent(Base):
    __tablename__ = 'trace_event'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sim_run_id: Mapped[int] = mapped_column(ForeignKey('sim_run.id'), nullable=False)
    sim_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    port: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_step_id: Mapped[int | None] = mapped_column(ForeignKey('canonical_step.id'))


# ----------------------------------------------------------------- explanations

class ExplanationTarget(Base):
    __tablename__ = 'explanation_target'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    entry: Mapped[str] = mapped_column(String(8), nullable=False)          # click | nl
    question: Mapped[str | None] = mapped_column(Text)
    subject_type: Mapped[str | None] = mapped_column(String(24))
    subject_id: Mapped[int | None] = mapped_column(BigInteger)
    # context the explanation is computed in: which specification, which run
    feature_id: Mapped[int | None] = mapped_column(ForeignKey('feature.id'))
    sim_run_id: Mapped[int | None] = mapped_column(ForeignKey('sim_run.id'))
    p_t: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    q_t: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    scope: Mapped[str] = mapped_column(String(8), nullable=False)          # global | local
    # complete | contrastive (other observed rows) | counterfactual (solver, v2)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default='complete')
    mapping_llm_call: Mapped[int | None] = mapped_column(ForeignKey('llm_call.id'))
    created_at = _created()


class Explanation(RevisionMixin, Base):
    __tablename__ = 'explanation'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    supersedes: Mapped[int | None] = mapped_column(ForeignKey('explanation.id'))
    target_id: Mapped[int] = mapped_column(ForeignKey('explanation_target.id'), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False)
    selected: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Structured explanation (D19); text is rendered from it at display time,
    # text_rule keeps the rendering that was current when it was computed.
    structure: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    text_rule: Mapped[str] = mapped_column(Text, nullable=False)
    text_polished: Mapped[str | None] = mapped_column(Text)
    polish_llm_call: Mapped[int | None] = mapped_column(ForeignKey('llm_call.id'))
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    auto_checks: Mapped[dict | None] = mapped_column(JSONB)
    created_at = _created()


class Feedback(Base):
    __tablename__ = 'feedback'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(24), nullable=False)
    subject_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)   # correction | error_report | rating | comment
    body: Mapped[str | None] = mapped_column(Text)
    proposed: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default='open')
    result_ref: Mapped[str | None] = mapped_column(Text)
    created_at = _created()


# One current revision per uid, and the lookups the explanation queries use.
for _t in ('requirement', 'scenario', 'example_row', 'explanation'):
    Index(f'uq_{_t}_current', Base.metadata.tables[_t].c.uid, unique=True,
          postgresql_where=text('is_current'))
Index('ix_scenario_feature', Scenario.feature_id)
Index('ix_example_row_scenario', ExampleRow.scenario_id)
Index('ix_check_item_scenario', CheckItem.scenario_id)
Index('ix_trace_event_run', TraceEvent.sim_run_id)
Index('ix_requirement_session', Requirement.session_id)
Index('ix_llm_call_session', LlmCall.session_id)
Index('ix_example_row_vals', ExampleRow.vals, postgresql_using='gin')
Index('ix_duv_session', Duv.session_id)
Index('ix_sim_run_duv', SimRun.duv_id)
Index('ix_check_result_verdict', CheckResult.verdict)
