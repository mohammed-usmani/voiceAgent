"""Candidates for a job and their calls."""

import csv
import io
import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_session
from voiceagent.api.jobs import get_job
from voiceagent.api.schemas import CallOut, CandidateCreate, CandidateOut, ImportResult, RowError
from voiceagent.db.models import ACTIVE_CALL_STATUSES, Call, Candidate, CriteriaVersion, Job
from voiceagent.events import emit
from voiceagent.worker.queue import enqueue

router = APIRouter()
DB = Annotated[Session, Depends(get_session)]
CSV_COLUMNS = {"name", "email", "phone", "sip_address", "resume_text"}
MAX_IMPORT_ROWS = 1000


def get_candidate(session: Session, candidate_id: uuid.UUID, *, lock: bool = False) -> Candidate:
    candidate = session.get(
        Candidate, candidate_id, with_for_update=lock or None, populate_existing=lock
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return candidate


def _messages(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        msg = err["msg"].removeprefix("Value error, ")
        out.append(f"{err['loc'][0]}: {msg}" if err["loc"] else msg)
    return out


def parse_candidates_csv(text: str) -> tuple[list[tuple[int, CandidateCreate]], list[RowError]]:
    """Valid rows with their row numbers, and the errors of the rest.

    Rows are numbered as a spreadsheet shows them (the header is row 1; a quoted
    multi-line résumé is still one row). A bad row is reported, never fatal; a file
    without the needed columns raises ValueError.
    """
    reader = csv.DictReader(io.StringIO(text))
    reader.fieldnames = [(h or "").strip().lower() for h in reader.fieldnames or []]
    columns = set(reader.fieldnames)
    if not {"name", "email"} <= columns or not columns & {"phone", "sip_address"}:
        raise ValueError("the CSV needs columns name, email, and phone or sip_address")
    rows: list[tuple[int, CandidateCreate]] = []
    errors: list[RowError] = []
    for n, raw in enumerate(reader, start=2):
        if n - 1 > MAX_IMPORT_ROWS:
            raise ValueError(f"at most {MAX_IMPORT_ROWS} candidates per file")
        fields = {k: v or "" for k, v in raw.items() if k in CSV_COLUMNS}
        if not any(v.strip() for v in fields.values()):
            continue
        try:
            rows.append((n, CandidateCreate(**fields)))
        except ValidationError as e:
            errors.append(RowError(row=n, errors=_messages(e)))
    return rows, errors


def _already_added(email: str) -> str:
    return f"email: {email} was already added to this job"


@router.post("/jobs/{job_id}/candidates", status_code=201, response_model=CandidateOut)
def add_candidate(job_id: uuid.UUID, body: CandidateCreate, session: DB) -> Candidate:
    job = get_job(session, job_id, lock=True)  # serializes with imports: emails stay unique
    taken = session.scalar(
        select(Candidate.id).where(Candidate.job_id == job.id, Candidate.email == body.email)
    )
    if taken is not None:
        raise HTTPException(status_code=409, detail=_already_added(body.email))
    candidate = Candidate(job_id=job.id, **body.model_dump())
    session.add(candidate)
    session.flush()
    emit(session, "candidate", candidate.id, "created", {"job_id": str(job.id)})
    session.commit()
    return candidate


@router.post("/jobs/{job_id}/candidates:import", response_model=ImportResult)
def import_candidates(
    job_id: uuid.UUID,
    body: Annotated[bytes, Body(media_type="text/csv")],
    session: DB,
) -> ImportResult:
    """Send the file as the request body with `Content-Type: text/csv`. Columns: name,
    email, phone and/or sip_address, optional resume_text; others are ignored."""
    job = get_job(session, job_id, lock=True)
    try:
        rows, errors = parse_candidates_csv(body.decode("utf-8-sig"))  # -sig: Excel's BOM
    except (ValueError, csv.Error) as e:  # UnicodeDecodeError is a ValueError
        raise HTTPException(status_code=422, detail=str(e)) from e
    taken = set(session.scalars(select(Candidate.email).where(Candidate.job_id == job.id)))
    added = []
    for n, row in rows:
        if row.email in taken:
            errors.append(RowError(row=n, errors=[_already_added(row.email)]))
            continue
        taken.add(row.email)
        added.append(Candidate(job_id=job.id, **row.model_dump()))
    session.add_all(added)
    emit(
        session,
        "job",
        job.id,
        "candidates_imported",
        {"created": len(added), "errors": len(errors)},
    )
    session.commit()
    return ImportResult(created=len(added), errors=sorted(errors, key=lambda e: e.row))


@router.get("/jobs/{job_id}/candidates", response_model=list[CandidateOut])
def list_candidates(
    job_id: uuid.UUID,
    session: DB,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Candidate]:
    job = get_job(session, job_id)
    stmt = (
        select(Candidate)
        .where(Candidate.job_id == job.id)
        .order_by(Candidate.created_at, Candidate.id)
        .limit(limit)
        .offset(offset)
    )
    if status:
        stmt = stmt.where(Candidate.status == status)
    return list(session.scalars(stmt))


@router.get("/candidates/{candidate_id}", response_model=CandidateOut)
def read_candidate(candidate_id: uuid.UUID, session: DB) -> Candidate:
    return get_candidate(session, candidate_id)


@router.delete("/candidates/{candidate_id}", status_code=204)
def delete_candidate(candidate_id: uuid.UUID, session: DB) -> Response:
    """Delete the candidate; their calls and transcripts go with them (ON DELETE CASCADE).
    Answers and decisions (milestone 3) will cascade the same way."""
    candidate = get_candidate(session, candidate_id, lock=True)
    emit(session, "candidate", candidate.id, "deleted", {"job_id": str(candidate.job_id)})
    session.delete(candidate)
    session.commit()
    return Response(status_code=204)


@router.post("/candidates/{candidate_id}/call", status_code=202, response_model=CallOut)
def call_candidate(candidate_id: uuid.UUID, session: DB) -> Call:
    """Queue a call now. Calling hours, retries and campaigns arrive in milestone 5."""
    candidate = get_candidate(session, candidate_id, lock=True)  # one request at a time
    job = session.get(Job, candidate.job_id)
    criteria = session.scalars(
        select(CriteriaVersion).where(
            CriteriaVersion.job_id == job.id, CriteriaVersion.status == "approved"
        )
    ).first()
    if job.status != "active" or criteria is None:
        raise HTTPException(
            status_code=409, detail=f"job is {job.status}: approve its criteria before calling"
        )
    active = session.scalar(
        select(Call.id).where(
            Call.candidate_id == candidate.id, Call.status.in_(ACTIVE_CALL_STATUSES)
        )
    )
    if active is not None:
        raise HTTPException(status_code=409, detail="a call to this candidate is already active")
    previous = session.scalar(
        select(func.count()).select_from(Call).where(Call.candidate_id == candidate.id)
    )
    call = Call(
        candidate_id=candidate.id,
        criteria_version_id=criteria.id,
        attempt=previous + 1,
        status="queued",
    )
    session.add(call)
    session.flush()
    candidate.status = "calling"
    enqueue(session, "dial", {"call_id": str(call.id)})
    emit(
        session,
        "call",
        call.id,
        "queued",
        {"candidate_id": str(candidate.id), "attempt": call.attempt},
    )
    session.commit()
    return call


@router.get("/candidates/{candidate_id}/calls", response_model=list[CallOut])
def list_calls(candidate_id: uuid.UUID, session: DB) -> list[Call]:
    candidate = get_candidate(session, candidate_id)
    stmt = select(Call).where(Call.candidate_id == candidate.id).order_by(Call.attempt)
    return list(session.scalars(stmt))


@router.get("/calls/{call_id}", response_model=CallOut)
def read_call(call_id: uuid.UUID, session: DB) -> Call:
    call = session.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")
    return call
