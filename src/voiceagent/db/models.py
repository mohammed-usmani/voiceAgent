import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
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
from voiceagent.settings import EMBEDDING_DIM


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
