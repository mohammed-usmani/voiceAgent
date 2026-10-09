from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_session
from voiceagent.api.schemas import EventOut
from voiceagent.db.models import Event

router = APIRouter()


@router.get("/events", response_model=list[EventOut])
def list_events(
    session: Annotated[Session, Depends(get_session)],
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[Event]:
    stmt = select(Event).where(Event.id > after).order_by(Event.id).limit(limit)
    return list(session.scalars(stmt))
