"""Calls across candidates, and watching a live one."""

import os
import uuid
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from livekit import api
from sqlalchemy import select
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_session
from voiceagent.api.schemas import CallListItem, ViewerToken
from voiceagent.db.models import ACTIVE_CALL_STATUSES, Call, Candidate, Job

router = APIRouter()
DB = Annotated[Session, Depends(get_session)]
VIEWER_TTL = timedelta(hours=1)
LIVEKIT_SETTINGS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


@router.get("/calls", response_model=list[CallListItem])
def list_all_calls(
    session: DB,
    status: Annotated[list[str] | None, Query()] = None,
    job_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[CallListItem]:
    """Calls, newest first, with their candidate and job: `?status=in_progress` finds the
    live ones (repeat `status` for several)."""
    stmt = (
        select(Call, Candidate.name, Job.id, Job.title)
        .join(Candidate, Call.candidate_id == Candidate.id)
        .join(Job, Candidate.job_id == Job.id)
        .order_by(Call.created_at.desc(), Call.id)
        .limit(limit)
    )
    if status:
        stmt = stmt.where(Call.status.in_(status))
    if job_id is not None:
        stmt = stmt.where(Job.id == job_id)
    return [
        CallListItem(
            id=call.id,
            candidate_id=call.candidate_id,
            candidate_name=name,
            job_id=jid,
            job_title=title,
            room=call.room,
            status=call.status,
            attempt=call.attempt,
            started_at=call.started_at,
            created_at=call.created_at,
        )
        for call, name, jid, title in session.execute(stmt)
    ]


@router.post("/calls/{call_id}/viewer-token", response_model=ViewerToken)
def viewer_token(call_id: uuid.UUID, session: DB) -> ViewerToken:
    """A LiveKit token to watch a ringing or live call: a hidden participant that can
    subscribe but never publish audio or data. Valid for an hour."""
    call = session.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")
    # A queued call has no room until its dial task rings the candidate.
    if call.status not in ACTIVE_CALL_STATUSES or not call.room:
        raise HTTPException(
            status_code=409,
            detail=f"call is {call.status}: only a ringing or live call can be watched",
        )
    url, key, secret = (os.environ.get(name, "") for name in LIVEKIT_SETTINGS)
    if not (url and key and secret):
        raise HTTPException(
            status_code=503,
            detail="LiveKit isn't configured: set " + ", ".join(LIVEKIT_SETTINGS),
        )
    token = (
        api.AccessToken(key, secret)
        .with_identity(f"viewer-{uuid.uuid4().hex[:8]}")
        .with_name("Viewer")
        .with_ttl(VIEWER_TTL)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=call.room,
                can_subscribe=True,
                can_publish=False,
                can_publish_data=False,
                hidden=True,
            )
        )
        .to_jwt()
    )
    return ViewerToken(url=url, token=token, room=call.room)
