import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_session
from voiceagent.api.schemas import JobCreate, JobOut, JobUpdate
from voiceagent.db.models import Job
from voiceagent.events import emit
from voiceagent.worker.queue import enqueue

router = APIRouter()
DB = Annotated[Session, Depends(get_session)]


def get_job(session: Session, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post("/jobs", status_code=201, response_model=JobOut)
def create_job(body: JobCreate, session: DB) -> Job:
    job = Job(**body.model_dump(), status="generating")
    session.add(job)
    session.flush()
    enqueue(session, "generate_criteria", {"job_id": str(job.id)})
    emit(session, "job", job.id, "created", {"title": job.title})
    session.commit()
    return job


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    session: DB,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Job]:
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
    if status:
        stmt = stmt.where(Job.status == status)
    return list(session.scalars(stmt))


@router.get("/jobs/{job_id}", response_model=JobOut)
def read_job(job_id: uuid.UUID, session: DB) -> Job:
    return get_job(session, job_id)


@router.patch("/jobs/{job_id}", response_model=JobOut)
def update_job(job_id: uuid.UUID, body: JobUpdate, session: DB) -> Job:
    job = get_job(session, job_id)
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in changes.items():
        setattr(job, field, value)
    emit(session, "job", job.id, "updated", {"fields": sorted(changes)})
    session.commit()
    return job
