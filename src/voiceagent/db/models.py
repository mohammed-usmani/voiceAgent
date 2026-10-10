import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voiceagent.db.base import Base
from voiceagent.settings import DEFAULT_VOICE, EMBEDDING_DIM


class Timestamps:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Job(Timestamps, Base):
    """A role being screened for. Status: generating → criteria_ready → active → closed,
    or criteria_failed."""

    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200))
    company_name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    recruiter_notes: Mapped[str] = mapped_column(Text, default="")
    agent_name: Mapped[str] = mapped_column(String(60), default="Aria")
    timezone: Mapped[str] = mapped_column(String(60), default="Asia/Kolkata")
    calling_hours: Mapped[dict] = mapped_column(
        JSONB, default=lambda: {"start": "10:00", "end": "19:00"}
    )
    auto_send_invites: Mapped[bool] = mapped_column(Boolean, default=True)
    response_days: Mapped[int] = mapped_column(Integer, default=3)
    voice: Mapped[str] = mapped_column(
        String(100), default=DEFAULT_VOICE, server_default=DEFAULT_VOICE
    )
    # True when the description or notes changed after criteria were generated.
    criteria_stale: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    status: Mapped[str] = mapped_column(String(30), default="generating")


class CriteriaVersion(Timestamps, Base):
    """Screening criteria for a job. Status: draft → approved; old ones → superseded.
    Approved versions are never edited — an edit creates a new draft."""

    __tablename__ = "criteria_versions"
    __table_args__ = (UniqueConstraint("job_id", "version"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    requirements: Mapped[list] = mapped_column(JSONB)
    job_facts: Mapped[dict] = mapped_column(JSONB)
    pass_threshold: Mapped[float] = mapped_column(Float, default=0.6)
    warnings: Mapped[list] = mapped_column(JSONB, default=list)
    model: Mapped[str] = mapped_column(String(120), default="")
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Knowledge(Timestamps, Base):
    """Retrievable examples: approved criteria (kind='criteria'), later recruiter
    overrides (kind='override')."""

    __tablename__ = "knowledge"
    __table_args__ = (
        Index(
            "ix_knowledge_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class Task(Timestamps, Base):
    """Background work. Status: queued → running → done, or failed after max_attempts."""

    __tablename__ = "tasks"
    __table_args__ = (Index("ix_tasks_claim", "status", "run_at"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)


class Event(Base):
    """Append-only feed for the frontend; ordered by id."""

    __tablename__ = "events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    entity: Mapped[str] = mapped_column(String(30))
    entity_id: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(50))
    data: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# A call in one of these states still owns the candidate: no second call may start.
ACTIVE_CALL_STATUSES = ("queued", "ringing", "in_progress")
# A candidate in one of these states is done with calls; calling again needs force.
FINISHED_CANDIDATE_STATUSES = ("screened", "decided", "notified", "declined_consent", "withdrawn")
# Calls that never became a conversation. After MAX_UNANSWERED of them the candidate is
# `unreachable` and campaigns stop calling them (a recruiter can still call by hand).
UNANSWERED_CALL_STATUSES = ("no_answer", "voicemail", "failed")
MAX_UNANSWERED = 3


class Candidate(Timestamps, Base):
    """Someone to screen for a job. Status: pending → calling → screened → decided →
    notified, or no_answer, callback, declined_consent, withdrawn, unreachable (MAX_UNANSWERED
    calls nobody answered). A recruiter can stop calling them (`excluded_at`) without
    changing their status."""

    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint("job_id", "email"),
        CheckConstraint(
            "phone IS NOT NULL OR sip_address IS NOT NULL", name="candidates_reachable"
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(320))
    phone: Mapped[str | None] = mapped_column(String(20))
    sip_address: Mapped[str | None] = mapped_column(String(200))
    resume_text: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="pending")
    # Set while a recruiter has stopped calling them: campaigns skip them and a manual call
    # is refused. Their status and history are untouched.
    excluded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Call(Timestamps, Base):
    """One call to a candidate. Status: queued → ringing → in_progress → completed, or
    no_answer, voicemail, failed. Deleting the candidate deletes their calls and transcripts.
    At most one active call per candidate, enforced by the database too."""

    __tablename__ = "calls"
    __table_args__ = (
        Index(
            "uq_calls_one_active_per_candidate",
            "candidate_id",
            unique=True,
            postgresql_where=text(
                "status IN (" + ", ".join(f"'{s}'" for s in ACTIVE_CALL_STATUSES) + ")"
            ),
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    criteria_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("criteria_versions.id"))
    room: Mapped[str] = mapped_column(String(100), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    # completed · declined_consent · callback_requested · not_interested (the agent's
    # end_call), candidate_hung_up · voicemail · error (set by code)
    end_reason: Mapped[str | None] = mapped_column(String(30))
    callback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consent_given: Mapped[bool | None] = mapped_column(Boolean)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # [{"role": "agent" | "candidate", "text": str, "at": ISO 8601}]
    transcript: Mapped[list] = mapped_column(JSONB, default=list)
    error: Mapped[str | None] = mapped_column(Text)
    # The candidate's status when this call was queued, restored if the call never becomes
    # a conversation (a failed re-call of a screened candidate keeps them screened).
    prior_candidate_status: Mapped[str | None] = mapped_column(String(30))


class Campaign(Timestamps, Base):
    """Background calling of a job's candidates (worker.campaign.tick). Status: active ⇄
    paused; → done once no candidate can be called again. Starting it again reopens it."""

    __tablename__ = "campaigns"
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(20), default="active")
    # Calls of this job in progress at once (queued, ringing or talking), manual ones included.
    max_concurrent: Mapped[int] = mapped_column(Integer, default=2, server_default="2")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Answer(Timestamps, Base):
    """What the candidate said for one requirement, extracted from the transcript after the
    call. `value` is in the requirement's unit; null when there was no usable answer."""

    __tablename__ = "answers"
    __table_args__ = (UniqueConstraint("call_id", "requirement_id"),)
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"))
    requirement_id: Mapped[str] = mapped_column(String(60))
    value: Mapped[bool | float | str | None] = mapped_column(JSONB)
    declined: Mapped[bool] = mapped_column(Boolean, default=False)
    quote: Mapped[str] = mapped_column(Text, default="")
    # Below 0.6 a hard requirement counts as unanswered; a quote not found in the
    # candidate's own words caps it at 0.3.
    confidence: Mapped[float] = mapped_column(Float)
    inconsistency: Mapped[str | None] = mapped_column(Text)


class Decision(Timestamps, Base):
    """The assessment of one call. `result` is the agent's; `final_result` the recruiter's
    when they override it. Status: pending_review → approved → sent, or → overridden →
    approved → sent. One decision per call: re-running the assessment never adds another."""

    __tablename__ = "decisions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), unique=True
    )
    criteria_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("criteria_versions.id"))
    result: Mapped[str] = mapped_column(String(20))  # suitable · not_suitable · needs_review
    score: Mapped[float | None] = mapped_column(Float)
    hard_failures: Mapped[list] = mapped_column(JSONB, default=list)  # requirement ids
    # [{"requirement_id", "verdict", "score", "quote", "confidence", "note"}]
    reasons: Mapped[list] = mapped_column(JSONB, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(20), default="pending_review")
    final_result: Mapped[str | None] = mapped_column(String(20))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    # The reviewer agent's verdict: {"agree", "notes", "challenged", "model"}; null if it
    # wasn't asked.
    agent_review: Mapped[dict | None] = mapped_column(JSONB)
    # Every step of the assessment run: [{"step", ...inputs and outputs}]
    trace: Mapped[list] = mapped_column(JSONB, default=list)


class Message(Timestamps, Base):
    """The outcome email for a decision. Status: draft → queued → sending → sent, or
    failed (retry with :approve). `issues` are checker objections the writer couldn't fix:
    a message with issues is never sent without a recruiter."""

    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"), unique=True
    )
    kind: Mapped[str] = mapped_column(String(20))  # invite · rejection
    to: Mapped[str] = mapped_column(String(320))
    subject: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    issues: Mapped[list] = mapped_column(JSONB, default=list)
    model: Mapped[str] = mapped_column(String(300), default="")
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
