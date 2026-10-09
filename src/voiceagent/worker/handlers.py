import uuid

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from voiceagent.agents.criteria import run_criteria_agent
from voiceagent.db.models import CriteriaVersion, Job
from voiceagent.events import emit
from voiceagent.knowledge.store import similar_criteria
from voiceagent.llm import Models


def generate_criteria(session: Session, models: Models, payload: dict) -> None:
    job = session.get(Job, uuid.UUID(payload["job_id"]))
    if job is None:  # deleted while queued
        return
    result = run_criteria_agent(
        models,
        title=job.title,
        description=job.description,
        notes=job.recruiter_notes,
        similar_lookup=lambda q: similar_criteria(session, models, q, exclude_job_id=job.id),
    )
    session.execute(
        update(CriteriaVersion)
        .where(CriteriaVersion.job_id == job.id, CriteriaVersion.status == "draft")
        .values(status="superseded")
    )
    latest = session.scalar(
        select(func.max(CriteriaVersion.version)).where(CriteriaVersion.job_id == job.id)
    )
    version = (latest or 0) + 1
    c = result.criteria
    session.add(
        CriteriaVersion(
            job_id=job.id,
            version=version,
            status="draft",
            requirements=[r.model_dump(mode="json") for r in c.requirements],
            job_facts=c.job_facts.model_dump(mode="json"),
            pass_threshold=c.pass_threshold,
            warnings=result.warnings,
            model=models.name,
        )
    )
    if job.status in ("generating", "criteria_failed"):
        job.status = "criteria_ready"
    emit(session, "job", job.id, "criteria_ready", {"version": version, "rounds": result.rounds})
    session.commit()


def generate_criteria_failed(session: Session, payload: dict, error: str) -> None:
    job = session.get(Job, uuid.UUID(payload["job_id"]))
    if job is None:
        return
    if job.status == "generating":
        job.status = "criteria_failed"
    emit(session, "job", job.id, "criteria_failed", {"error": error[:500]})
    session.commit()


HANDLERS = {
    "generate_criteria": (generate_criteria, generate_criteria_failed),
}
