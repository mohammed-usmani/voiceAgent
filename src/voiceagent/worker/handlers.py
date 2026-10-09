"""Task handlers. A handler leaves its writes uncommitted: the worker commits them together
with the task's completion (or, for `on_failure`, its failure), so a crash in between can't
leave a result without a finished task."""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from voiceagent.agents.criteria import run_criteria_agent
from voiceagent.db.models import CriteriaVersion, Job, Task
from voiceagent.events import emit
from voiceagent.knowledge.store import similar_criteria
from voiceagent.llm import Models
from voiceagent.worker.queue import heartbeat


def generate_criteria(session: Session, models: Models, task: Task) -> None:
    job_id = uuid.UUID(task.payload["job_id"])
    job = session.get(Job, job_id)
    if job is None:  # deleted while queued
        return
    title, description, notes = job.title, job.description, job.recruiter_notes
    session.rollback()  # hold no transaction (or locks) through the slow agent run

    def similar(query: str) -> list[str]:
        try:
            return similar_criteria(session, models, query, exclude_job_id=job_id)
        finally:
            session.rollback()

    result = run_criteria_agent(
        models,
        title=title,
        description=description,
        notes=notes,
        similar_lookup=similar,
        heartbeat=lambda: heartbeat(session, task),
    )

    # Lock the job, as the API's criteria endpoints do, so versions and status can't race.
    job = session.get(Job, job_id, with_for_update=True, populate_existing=True)
    if job is None:  # deleted while generating
        return
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
    if job.status in ("generating", "criteria_failed"):  # never demote a job approved meanwhile
        job.status = "criteria_ready"
    emit(session, "job", job.id, "criteria_ready", {"version": version, "rounds": result.rounds})


def generate_criteria_failed(session: Session, payload: dict, error: str) -> None:
    job = session.get(Job, uuid.UUID(payload["job_id"]), with_for_update=True)
    if job is None:
        return
    if job.status == "generating":
        job.status = "criteria_failed"
    emit(session, "job", job.id, "criteria_failed", {"error": error[:500]})


HANDLERS = {
    "generate_criteria": (generate_criteria, generate_criteria_failed),
}
