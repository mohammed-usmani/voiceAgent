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


class Candidate(Timestamps, Base):
    """Someone to screen for a job. Status: pending → calling → screened → decided →
    notified, or no_answer, callback, declined_consent, withdrawn."""

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


class Call(Timestamps, Base):
    """One call to a candidate. Status: queued → ringing → in_progress → completed, or
    no_answer, failed. Deleting the candidate deletes their calls and transcripts."""

    __tablename__ = "calls"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    criteria_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("criteria_versions.id"))
    room: Mapped[str] = mapped_column(String(100), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    # completed · declined_consent · callback_requested · not_interested (the agent's
    # end_call), candidate_hung_up · error (set by code)
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
