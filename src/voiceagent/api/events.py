import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker as SessionMaker

from voiceagent.api.deps import get_session
from voiceagent.api.schemas import EventOut
from voiceagent.db.models import Event

router = APIRouter()
POLL_S = 1.0
HEARTBEAT_S = 15.0  # a comment line this often keeps proxies from closing a quiet stream
BATCH = 500


@router.get("/events", response_model=list[EventOut])
def list_events(
    session: Annotated[Session, Depends(get_session)],
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    last: Annotated[int | None, Query(ge=1, le=500)] = None,
) -> list[Event]:
    """Events after `after`, oldest first; or with `last`, the newest `last` events (still
    oldest first), which is what a feed opening on recent history wants."""
    if last is not None:
        newest = select(Event).order_by(Event.id.desc()).limit(last)
        return list(reversed(session.scalars(newest).all()))
    stmt = select(Event).where(Event.id > after).order_by(Event.id).limit(limit)
    return list(session.scalars(stmt))


def events_after(sm: SessionMaker, after: int) -> list[EventOut]:
    with sm() as session:
        stmt = select(Event).where(Event.id > after).order_by(Event.id).limit(BATCH)
        return [EventOut.model_validate(e) for e in session.scalars(stmt)]


async def event_stream(
    sm: SessionMaker,
    after: int,
    disconnected: Callable[[], Awaitable[bool]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Server-sent event frames for every event after `after`, as they are written, until
    `disconnected()` says the client left; a heartbeat comment after HEARTBEAT_S of quiet.

    ponytail: polls every POLL_S; LISTEN/NOTIFY if the delay or the queries ever matter.
    Ids come from a sequence at insert, so an event whose transaction commits after a later
    one's can be passed over (as with GET /events); fine for refreshing views.
    """
    quiet_since = clock()
    while not await disconnected():
        events = await asyncio.to_thread(events_after, sm, after)
        for e in events:
            yield f"id: {e.id}\ndata: {e.model_dump_json()}\n\n"
            after = e.id
        if events:
            quiet_since = clock()
        elif clock() - quiet_since >= HEARTBEAT_S:
            yield ": heartbeat\n\n"
            quiet_since = clock()
        if len(events) < BATCH:  # a full batch means more are waiting: read on at once
            await sleep(POLL_S)


async def opened(frames: AsyncIterator[str]) -> AsyncIterator[str]:
    """`frames` after an opening comment. The response headers go out with the first chunk,
    so without it a client of a quiet stream saw nothing, not even the connection, until
    the first heartbeat."""
    yield ": open\n\n"
    async for frame in frames:
        yield frame


def stream_start(sm: SessionMaker, after: int | None, last_event_id: int | None) -> int:
    """The id to stream after: the later of `after` and `last_event_id`, or the newest
    event when neither is given."""
    if after is None and last_event_id is None:
        with sm() as session:
            return session.scalar(select(func.max(Event.id))) or 0
    return max(after or 0, last_event_id or 0)


@router.get("/events/stream", response_class=StreamingResponse)
async def stream_events(
    request: Request,
    after: Annotated[int | None, Query(ge=0)] = None,
    last_event_id: Annotated[int | None, Header()] = None,
) -> StreamingResponse:
    """Every event after `after` (or after the Last-Event-ID a reconnecting EventSource
    sends) as server-sent events, as they are written. With neither, only new events:
    a page that just loaded fetched the current state already, and replaying the whole
    history would flood its activity feed."""
    sm = request.app.state.sessionmaker
    return StreamingResponse(
        opened(event_stream(sm, stream_start(sm, after, last_event_id), request.is_disconnected)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
