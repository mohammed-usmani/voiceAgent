"""The `send_message` task: email a decision's message to the candidate.

A send is never retried automatically: a failure marks the message `failed`, and the
recruiter retries with POST /decisions/{id}:approve.
"""

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from voiceagent.db.models import Candidate, Decision, Message, Task
from voiceagent.events import emit
from voiceagent.llm import Models
from voiceagent.outcomes.email import Email, Mailer, default_mailer
from voiceagent.settings import get_settings
from voiceagent.worker.queue import enqueue

logger = logging.getLogger("voiceagent.worker")

# Tests replace it with a fake; None means "from settings" on the first send.
MAILER: Mailer | None = None


def mailer() -> Mailer:
    global MAILER
    if MAILER is None:
        MAILER = default_mailer(get_settings())
    return MAILER


def queue_send(session: Session, decision: Decision, message: Message) -> None:
    """Approve the decision and queue its message for sending. Not committed."""
    message.status = "queued"
    message.error = None
    decision.status = "approved"
    enqueue(session, "send_message", {"message_id": str(message.id)})
    emit(session, "message", message.id, "queued", {"decision_id": str(decision.id)})


def lock_message(session: Session, message_id: uuid.UUID) -> Message | None:
    """Lock a message for writing with its candidate and decision.

    Lock order, everywhere: candidate, decision, message — the order DELETE
    /candidates/{id} takes by cascade and the review endpoints follow (decision, then
    message), so none of them can deadlock with a send.
    """
    row = session.execute(
        select(Decision.candidate_id, Decision.id)
        .join(Message, Message.decision_id == Decision.id)
        .where(Message.id == message_id)
    ).first()
    if row is None:
        return None
    session.get(Candidate, row.candidate_id, with_for_update=True, populate_existing=True)
    session.get(Decision, row.id, with_for_update=True, populate_existing=True)
    return session.get(Message, message_id, with_for_update=True, populate_existing=True)


def _failed(session: Session, message: Message, error: str) -> None:
    message.status = "failed"
    message.error = error[:2000]
    emit(session, "message", message.id, "send_failed", {"error": error[:500]})


def send_message(session: Session, models: Models, task: Task) -> None:
    message_id = uuid.UUID(task.payload["message_id"])
    message = lock_message(session, message_id)
    if message is None:  # deleted with its candidate
        return
    if message.status == "sending":
        # A rerun after the worker died mid-send: the email may already have gone out, and
        # sending again could reach the candidate twice.
        _failed(session, message, "the worker stopped while sending; check it wasn't received")
        return
    if message.status != "queued":
        return
    message.status = "sending"
    email = Email(to=message.to, subject=message.subject, body=message.body)
    session.commit()  # visible before the send, so a rerun can tell it was attempted

    try:
        mailer().send(email)
    except Exception as e:
        logger.warning("message %s not sent: %s", message_id, e)
        message = lock_message(session, message_id)
        if message is not None:
            _failed(session, message, str(e) or repr(e))
        return
    message = lock_message(session, message_id)
    if message is None:
        return
    message.status = "sent"
    message.sent_at = datetime.now(UTC)
    message.error = None
    decision = session.get(Decision, message.decision_id)
    decision.status = "sent"
    session.get(Candidate, decision.candidate_id).status = "notified"
    data = {"decision_id": str(decision.id)}
    if not getattr(mailer(), "delivers", True):
        data["delivered"] = False  # development: only logged
    emit(session, "message", message.id, "sent", data)


def send_message_failed(session: Session, payload: dict, error: str) -> None:
    message = lock_message(session, uuid.UUID(payload["message_id"]))
    if message is not None and message.status in ("queued", "sending"):
        _failed(session, message, error)
