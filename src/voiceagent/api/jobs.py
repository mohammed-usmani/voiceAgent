import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_models, get_session
from voiceagent.api.schemas import CriteriaOut, JobCreate, JobOut, JobUpdate
from voiceagent.criteria.schema import Criteria, check_criteria, normalize_weights
from voiceagent.db.models import CriteriaVersion, Job, Task
from voiceagent.events import emit
from voiceagent.knowledge.store import index_criteria
from voiceagent.llm import Models
from voiceagent.worker.queue import enqueue

router = APIRouter()
DB = Annotated[Session, Depends(get_session)]


def get_job(session: Session, job_id: uuid.UUID, *, lock: bool = False) -> Job:
    """`lock=True` takes the job row lock that serializes everything that changes a job's
    criteria (these endpoints and the worker), so versions and status can't race."""
    job = session.get(Job, job_id, with_for_update=lock or None, populate_existing=lock)
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


def latest_criteria(session: Session, job_id: uuid.UUID) -> CriteriaVersion | None:
    return session.scalars(
        select(CriteriaVersion)
        .where(CriteriaVersion.job_id == job_id, CriteriaVersion.status != "superseded")
        .order_by(CriteriaVersion.version.desc())
        .limit(1)
    ).first()


def as_criteria(cv: CriteriaVersion) -> Criteria:
    return Criteria(
        requirements=cv.requirements, job_facts=cv.job_facts, pass_threshold=cv.pass_threshold
    )


def criteria_out(cv: CriteriaVersion) -> CriteriaOut:
    errors, _ = check_criteria(as_criteria(cv))
    return CriteriaOut(
        id=cv.id,
        version=cv.version,
        status=cv.status,
        requirements=cv.requirements,
        job_facts=cv.job_facts,
        pass_threshold=cv.pass_threshold,
        warnings=cv.warnings,
        errors=errors,
        model=cv.model,
        approved_at=cv.approved_at,
    )


@router.get("/jobs/{job_id}/criteria", response_model=CriteriaOut)
def read_criteria(job_id: uuid.UUID, session: DB) -> CriteriaOut:
    job = get_job(session, job_id)
    cv = latest_criteria(session, job.id)
    if cv is None:
        raise HTTPException(status_code=404, detail=f"no criteria yet (job is {job.status})")
    return criteria_out(cv)


@router.patch("/jobs/{job_id}/criteria", response_model=CriteriaOut)
def edit_criteria(job_id: uuid.UUID, body: Criteria, session: DB) -> CriteriaOut:
    job = get_job(session, job_id, lock=True)
    cv = latest_criteria(session, job.id)
    if cv is None:
        raise HTTPException(status_code=409, detail=f"no criteria to edit (job is {job.status})")
    c = normalize_weights(body)
    fields = dict(
        requirements=[r.model_dump(mode="json") for r in c.requirements],
        job_facts=c.job_facts.model_dump(mode="json"),
        pass_threshold=c.pass_threshold,
        warnings=check_criteria(c)[1],
        model="recruiter edit",
    )
    if cv.status == "draft":
        for name, value in fields.items():
            setattr(cv, name, value)
    else:  # approved versions are immutable: start a new draft
        cv = CriteriaVersion(job_id=job.id, version=cv.version + 1, status="draft", **fields)
        session.add(cv)
    session.flush()
    emit(session, "job", job.id, "criteria_edited", {"version": cv.version})
    session.commit()
    return criteria_out(cv)


@router.post("/jobs/{job_id}/criteria:regenerate", status_code=202, response_model=JobOut)
def regenerate_criteria(job_id: uuid.UUID, session: DB) -> Job:
    job = get_job(session, job_id, lock=True)
    pending = session.scalar(
        select(Task.id).where(
            Task.kind == "generate_criteria",
            Task.status.in_(("queued", "running")),
            Task.payload["job_id"].astext == str(job.id),
        )
    )
    if pending is not None:
        raise HTTPException(status_code=409, detail="criteria are already being generated")
    if job.status != "active":
        job.status = "generating"
    enqueue(session, "generate_criteria", {"job_id": str(job.id)})
    emit(session, "job", job.id, "criteria_regenerating")
    session.commit()
    return job


@router.post("/jobs/{job_id}/criteria:approve", response_model=CriteriaOut)
def approve_criteria(
    job_id: uuid.UUID, session: DB, models: Annotated[Models, Depends(get_models)]
) -> CriteriaOut:
    job = get_job(session, job_id, lock=True)
    cv = latest_criteria(session, job.id)
    if cv is None or cv.status != "draft":
        raise HTTPException(status_code=409, detail="no draft criteria to approve")
    c = as_criteria(cv)
    errors, _ = check_criteria(c)
    if not c.requirements:
        errors.append("add at least one requirement")
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    session.execute(
        update(CriteriaVersion)
        .where(CriteriaVersion.job_id == job.id, CriteriaVersion.status == "approved")
        .values(status="superseded")
    )
    cv.status = "approved"
    cv.approved_at = datetime.now(UTC)
    job.status = "active"
    try:
        index_criteria(session, models, job, c)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=502, detail=f"couldn't index criteria: {e}") from e
    emit(session, "job", job.id, "criteria_approved", {"version": cv.version})
    session.commit()
    return criteria_out(cv)
