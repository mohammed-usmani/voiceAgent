"""Reads and writes for a screening call, shared by the worker (dialing) and the agent
process (the conversation)."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from sqlalchemy import String, cast, exists, func, select
from sqlalchemy.orm import Session, sessionmaker

from voiceagent.criteria.schema import Criteria
from voiceagent.db.base import make_engine, make_sessionmaker
from voiceagent.db.models import (
    ACTIVE_CALL_STATUSES,
    FINISHED_CANDIDATE_STATUSES,
    MAX_UNANSWERED,
    UNANSWERED_CALL_STATUSES,
    Call,
    Candidate,
    CriteriaVersion,
    Job,
    Task,
)
from voiceagent.events import emit
from voiceagent.settings import get_settings
from voiceagent.worker.queue import enqueue


@dataclass(frozen=True)
class CallOutcome:
    """How the agent ended the call (its end_call, or "voicemail" when a machine answered),
    as stored on the call row."""

    reason: str
    callback_at: datetime | None = None


# A call nobody has written to for this long has lost its agent (never dispatched, killed,
# or its final save failed): no screening call runs this long. Well above the dial task's
# lease (worker.queue.LEASE), so a dial that is merely slow is never swept.
# ponytail: a call that really runs past this loses its transcript (finish_call skips a
# failed call); add an agent heartbeat if long calls become real.
MAX_CALL_MINUTES = 30
STALE_ERROR = "the agent never reported the end of the call"


def lock_call(session: Session, call_id: uuid.UUID) -> Call | None:
    """Lock a call for writing, with its candidate.

    Lock order, everywhere: the candidate, then its call. DELETE /candidates/{id} locks
    the candidate and then (ON DELETE CASCADE) its calls; taking the call first can
    deadlock against it.
    """
    candidate_id = session.scalar(select(Call.candidate_id).where(Call.id == call_id))
    if candidate_id is None:
        return None
    session.get(Candidate, candidate_id, with_for_update=True, populate_existing=True)
    return session.get(Call, call_id, with_for_update=True, populate_existing=True)


def close_stale_calls(session: Session, candidate_id: uuid.UUID | None = None) -> int:
    """Fail active calls nobody has written to for MAX_CALL_MINUTES (all of them, or one
    candidate's), so the candidate can be called again. Returns how many. Not committed.

    A call whose dial task is still waiting (a backlog, or no worker running) is left to
    that task: it hasn't rung yet, so failing it would burn an attempt for nothing."""
    cutoff = datetime.now(UTC) - timedelta(minutes=MAX_CALL_MINUTES)
    dial_pending = exists().where(
        Task.kind == "dial",
        Task.status.in_(("queued", "running")),
        Task.payload["call_id"].astext == cast(Call.id, String),
    )
    stale = select(Call.id).where(
        Call.status.in_(ACTIVE_CALL_STATUSES), Call.updated_at < cutoff, ~dial_pending
    )
    if candidate_id is not None:
        stale = stale.where(Call.candidate_id == candidate_id)
    closed = 0
    for call_id in session.scalars(stale).all():
        call = lock_call(session, call_id)
        if call is not None and call.status in ACTIVE_CALL_STATUSES and call.updated_at < cutoff:
            stop_call(session, call, "failed", STALE_ERROR)
            closed += 1
    return closed


def stop_call(session: Session, call: Call, status: str, error: str) -> None:
    """End a call that never became a conversation: nobody answered, a machine did, the
    dial failed, or the agent couldn't load it. The candidate goes back to what they were
    before this call (or no_answer), so they can be called again, unless this was their
    MAX_UNANSWERED-th such call: then they are `unreachable`. Not committed."""
    call.status = status
    call.end_reason = {"failed": "error", "voicemail": "voicemail"}.get(status)
    call.error = error[:2000]
    call.ended_at = datetime.now(UTC)
    prior = call.prior_candidate_status
    if prior in (None, "calling"):
        prior = "pending"
    candidate = session.get(Candidate, call.candidate_id)
    unanswered = session.scalar(
        select(func.count())
        .select_from(Call)
        .where(Call.candidate_id == candidate.id, Call.status.in_(UNANSWERED_CALL_STATUSES))
    )
    if prior in FINISHED_CANDIDATE_STATUSES:
        candidate.status = prior  # e.g. a failed re-call of a screened candidate
    elif unanswered >= MAX_UNANSWERED:
        candidate.status = "unreachable"
        emit(session, "candidate", candidate.id, "unreachable", {"unanswered_calls": unanswered})
    elif status in ("no_answer", "voicemail"):
        candidate.status = "no_answer"
    else:
        candidate.status = prior
    emit(session, "call", call.id, status, {"error": error[:500]})


def queue_call(
    session: Session, candidate: Candidate, criteria_version_id: uuid.UUID, **event: str
) -> Call:
    """Queue a call to the candidate now, with its dial task; the candidate is `calling`.
    The caller holds the candidate's lock and has checked that no call is active. `event`
    is added to the call's `queued` event. Not committed."""
    previous = session.scalar(
        select(func.count()).select_from(Call).where(Call.candidate_id == candidate.id)
    )
    call = Call(
        candidate_id=candidate.id,
        criteria_version_id=criteria_version_id,
        attempt=previous + 1,
        status="queued",
        prior_candidate_status=candidate.status,
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
        {"candidate_id": str(candidate.id), "attempt": call.attempt} | event,
    )
    return call


CANDIDATE_STATUS = {
    "completed": "screened",
    "candidate_hung_up": "screened",
    "declined_consent": "declined_consent",
    "callback_requested": "callback",
    "not_interested": "withdrawn",
}


@dataclass(frozen=True)
class LoadedCall:
    job: Job
    candidate: Candidate
    criteria: Criteria


@lru_cache
def default_sessionmaker() -> sessionmaker:
    return make_sessionmaker(make_engine(get_settings().database_url))


def load_call(sm: sessionmaker, call_id: uuid.UUID) -> LoadedCall:
    """Everything the agent needs for the call; marks it in progress."""
    with sm() as session:
        call = lock_call(session, call_id)
        if call is None:
            raise LookupError(f"call {call_id} not found")
        if call.status not in ACTIVE_CALL_STATUSES:
            raise LookupError(f"call {call_id} is already {call.status}")
        candidate = session.get(Candidate, call.candidate_id)
        job = session.get(Job, candidate.job_id)
        cv = session.get(CriteriaVersion, call.criteria_version_id)
        criteria = Criteria(
            requirements=cv.requirements, job_facts=cv.job_facts, pass_threshold=cv.pass_threshold
        )
        call.status = "in_progress"
        call.started_at = call.started_at or datetime.now(UTC)
        emit(session, "call", call.id, "in_progress")
        session.commit()
        return LoadedCall(job=job, candidate=candidate, criteria=criteria)


def finish_call(
    sm: sessionmaker,
    call_id: uuid.UUID,
    outcome: CallOutcome | None,
    turns: list[dict],
    error: str | None = None,
) -> None:
    """Save how the call ended and what was said, then queue the assessment.

    No outcome means the agent never called end_call: the candidate hung up, unless the
    agent's side failed (`error`): then the call failed and the candidate wasn't screened.
    A hang-up before the agent said anything (e.g. while it listened for a machine) is
    retried like an unanswered call.
    """
    reason = outcome.reason if outcome else "candidate_hung_up"
    with sm() as session:
        call = lock_call(session, call_id)
        if call is None or call.status not in ACTIVE_CALL_STATUSES:
            return  # the candidate was deleted, or this call was already saved
        if outcome is None and error is not None:
            call.transcript = turns
            stop_call(session, call, "failed", error)
            session.commit()
            return
        if outcome is None and not any(t["role"] == "agent" for t in turns):
            call.transcript = turns
            stop_call(session, call, "no_answer", "hung up before the greeting")
            session.commit()
            return
        if reason == "voicemail":  # nobody to screen: retried like an unanswered call
            call.transcript = turns
            stop_call(session, call, "voicemail", "a machine answered; no message was left")
            session.commit()
            return
        call.status = "completed"
        call.end_reason = reason
        call.callback_at = outcome.callback_at if outcome else None
        call.consent_given = {"completed": True, "declined_consent": False}.get(reason)
        call.transcript = turns
        call.ended_at = datetime.now(UTC)
        session.get(Candidate, call.candidate_id).status = CANDIDATE_STATUS[reason]
        enqueue(session, "postcall", {"call_id": str(call.id)})
        emit(session, "call", call.id, "ended", {"end_reason": reason, "turns": len(turns)})
        session.commit()


def fail_call(sm: sessionmaker, call_id: uuid.UUID, error: str) -> None:
    """The agent couldn't run the call (e.g. it couldn't load it)."""
    with sm() as session:
        call = lock_call(session, call_id)
        if call is not None and call.status in ACTIVE_CALL_STATUSES:
            stop_call(session, call, "failed", error)
            session.commit()
