"""Retrievable examples in pgvector: approved criteria of past jobs (and, later,
recruiter overrides) that ground and calibrate the agents."""

import uuid

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from voiceagent.criteria.schema import Criteria
from voiceagent.db.models import Job, Knowledge
from voiceagent.llm import Models


def add_knowledge(
    session: Session,
    models: Models,
    kind: str,
    entries: list[tuple[uuid.UUID | None, str, dict]],
) -> None:
    if not entries:
        return
    vectors = models.embed([text for _, text, _ in entries])
    for (job_id, text, data), vector in zip(entries, vectors, strict=True):
        session.add(Knowledge(kind=kind, job_id=job_id, text=text, data=data, embedding=vector))


def search_knowledge(
    session: Session,
    models: Models,
    kind: str,
    query: str,
    k: int = 8,
    exclude_job_id: uuid.UUID | None = None,
) -> list[Knowledge]:
    vector = models.embed([query])[0]
    stmt = select(Knowledge).where(Knowledge.kind == kind)
    if exclude_job_id is not None:
        stmt = stmt.where(or_(Knowledge.job_id.is_(None), Knowledge.job_id != exclude_job_id))
    stmt = stmt.order_by(Knowledge.embedding.cosine_distance(vector)).limit(k)
    return list(session.scalars(stmt))


def index_criteria(session: Session, models: Models, job: Job, criteria: Criteria) -> None:
    """One entry per requirement, so retrieval finds individual checks of similar roles.
    Replaces the job's earlier entries, so only its approved criteria are retrievable."""
    session.execute(
        delete(Knowledge).where(Knowledge.kind == "criteria", Knowledge.job_id == job.id)
    )
    entries = [
        (
            job.id,
            f"{job.title}: {r.label}. {r.question}",
            {"job_title": job.title, **r.model_dump(mode="json")},
        )
        for r in criteria.requirements
    ]
    add_knowledge(session, models, "criteria", entries)


def _fmt(v) -> str:
    # Pydantic stores 30 as 30.0; show whole numbers without the ".0".
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _describe(data: dict) -> str:
    if data.get("hard") and data.get("rule"):
        rule = data["rule"]
        unit = f" {data['unit']}" if data.get("unit") else ""
        kind = f"must-have: {rule['op']} {_fmt(rule['value'])}{unit}"
    else:
        kind = "weighted"
    return f"{data['job_title']}: {data['label']} ({kind}) — asked as: {data['question']}"


def similar_criteria(
    session: Session,
    models: Models,
    query: str,
    exclude_job_id: uuid.UUID | None = None,
    k: int = 8,
) -> list[str]:
    hits = search_knowledge(session, models, "criteria", query, k=k, exclude_job_id=exclude_job_id)
    return [_describe(h.data) for h in hits]
