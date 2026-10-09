"""Reads and writes for a screening call, shared by the worker (dialing) and the agent
process (the conversation)."""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from voiceagent.db.models import Call, Candidate
from voiceagent.events import emit


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
