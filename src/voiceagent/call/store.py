"""Reads and writes for a screening call, shared by the worker (dialing) and the agent
process (the conversation)."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from voiceagent.criteria.schema import Criteria
from voiceagent.db.base import make_engine, make_sessionmaker
from voiceagent.db.models import ACTIVE_CALL_STATUSES, Call, Candidate, CriteriaVersion, Job
from voiceagent.events import emit
from voiceagent.settings import get_settings
from voiceagent.worker.queue import enqueue


@dataclass(frozen=True)
class CallOutcome:
    """How the agent ended the call (its end_call), as stored on the call row."""

    reason: str
    callback_at: datetime | None = None


def stop_call(session: Session, call: Call, status: str, error: str) -> None:
    """End a call that never became a conversation: nobody answered, the dial failed, or
    the agent couldn't load it. The candidate can be called again. Not committed."""
    call.status = status
    call.end_reason = "error" if status == "failed" else None
    call.error = error[:2000]
    call.ended_at = datetime.now(UTC)
    candidate = session.get(Candidate, call.candidate_id)
    candidate.status = "no_answer" if status == "no_answer" else "pending"
    emit(session, "call", call.id, status, {"error": error[:500]})


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
        call = session.get(Call, call_id, with_for_update=True)
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
    sm: sessionmaker, call_id: uuid.UUID, outcome: CallOutcome | None, turns: list[dict]
) -> None:
    """Save how the call ended and what was said, then queue the assessment.

    No outcome means the agent never called end_call: the candidate hung up.
    """
    reason = outcome.reason if outcome else "candidate_hung_up"
    with sm() as session:
        call = session.get(Call, call_id, with_for_update=True)
        if call is None or call.status not in ACTIVE_CALL_STATUSES:
            return  # the candidate was deleted, or this call was already saved
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
        call = session.get(Call, call_id, with_for_update=True)
        if call is not None and call.status in ACTIVE_CALL_STATUSES:
            stop_call(session, call, "failed", error)
            session.commit()
