"""The `dial` task: ring the candidate, and once they answer, the agent takes over.

ponytail: one worker blocks while a phone rings (up to the SIP ringing timeout), so a
campaign's calls ring one after another per worker; run several workers to ring in
parallel, or make dialing async if that is too slow.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from livekit import api
from sqlalchemy import update
from sqlalchemy.orm import Session

from voiceagent import outbound
from voiceagent.call.store import lock_call, stop_call
from voiceagent.db.models import ACTIVE_CALL_STATUSES, Call, Candidate, Job, Task
from voiceagent.events import emit
from voiceagent.llm import Models

logger = logging.getLogger("voiceagent.worker")

# (address, room, display_name, metadata) -> returns once answered and the agent is sent in
Dialer = Callable[[str, str, str, str], None]

# SIP answers that mean nobody picked up, as opposed to a broken number or trunk:
# 408 timeout, 480 unavailable, 486 busy, 487 cancelled after ringing, 600 busy
# everywhere, 603 declined.
NO_ANSWER_SIP = {408, 480, 486, 487, 600, 603}


def livekit_dialer(address: str, room: str, display_name: str, metadata: str) -> None:
    asyncio.run(outbound.dial(address, room, display_name=display_name, metadata=metadata))


DIALER: Dialer = livekit_dialer


def dial_outcome(error: Exception) -> str:
    if isinstance(error, api.SipCallError) and error.sip_status_code in NO_ANSWER_SIP:
        return "no_answer"
    return "failed"


def dial(session: Session, models: Models, task: Task) -> None:
    call = lock_call(session, uuid.UUID(task.payload["call_id"]))
    if call is None:  # the candidate was deleted while the call was queued
        return
    if call.status != "queued":
        # A rerun after the worker died mid-dial. The candidate may already have been
        # rung (or be talking to the agent); ringing again is worse than a retry button.
        if call.status == "ringing":
            stop_call(session, call, "failed", "the worker stopped while dialing; not redialed")
        return
    candidate = session.get(Candidate, call.candidate_id)
    job = session.get(Job, candidate.job_id)
    call.status = "ringing"
    call.room = f"screen-{call.id.hex[:12]}"
    emit(session, "call", call.id, "ringing")
    session.commit()  # visible before the slow dial, so a rerun can tell it already rang

    address = candidate.phone or candidate.sip_address
    metadata = json.dumps({"call_id": str(call.id)})
    try:
        DIALER(address, call.room, f"{job.company_name} Hiring", metadata)
    except Exception as e:
        logger.warning("call %s not answered: %s", call.id, e)
        call = lock_call(session, call.id)
        if call is not None and call.status in ACTIVE_CALL_STATUSES:
            stop_call(session, call, dial_outcome(e), str(e))
        return
    # The agent marks the call in_progress itself when it loads it, and may already have
    # finished a short call; only a call still ringing moves on.
    answered = session.execute(
        update(Call)
        .where(Call.id == call.id, Call.status == "ringing")
        .values(status="in_progress", started_at=datetime.now(UTC))
    )
    if answered.rowcount:
        emit(session, "call", call.id, "answered")


def dial_failed(session: Session, payload: dict, error: str) -> None:
    call = lock_call(session, uuid.UUID(payload["call_id"]))
    if call is not None and call.status in ("queued", "ringing"):
        stop_call(session, call, "failed", error)
