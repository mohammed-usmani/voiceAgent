"""GET /events/stream: the event feed as server-sent events."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from starlette.requests import Request

from tests.fakes import FakeModels
from voiceagent.api import events as events_api
from voiceagent.api.app import create_app
from voiceagent.api.events import HEARTBEAT_S, event_stream, stream_start
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Event
from voiceagent.events import emit

KEY = {"X-API-Key": "test-key"}


def write(sm, *types: str) -> None:
    with sm() as s:
        for t in types:
            emit(s, "job", "j1", t)
        s.commit()


def until(polls: int):
    """disconnected() for a client that stays for `polls` polls, then leaves."""
    checks = iter([False] * polls + [True])

    async def disconnected() -> bool:
        return next(checks)

    return disconnected


async def no_sleep(_seconds: float) -> None:
    pass


async def collect(stream) -> list[str]:
    return [frame async for frame in stream]


def ids(frames) -> list[int]:
    return [int(f.split("\n")[0].removeprefix("id: ")) for f in frames if f.startswith("id: ")]


async def test_streams_the_events_after_the_given_id(engine, session):
    sm = make_sessionmaker(engine)
    write(sm, "created", "criteria_ready", "criteria_approved")
    frames = await collect(event_stream(sm, 1, until(1), sleep=no_sleep))
    assert ids(frames) == [2, 3]
    data = json.loads(frames[0].split("data: ", 1)[1])
    assert (data["id"], data["entity"], data["entity_id"], data["type"]) == (
        2,
        "job",
        "j1",
        "criteria_ready",
    )


async def test_events_written_while_streaming_arrive(engine, session):
    sm = make_sessionmaker(engine)

    async def a_call_is_answered(_seconds: float) -> None:
        write(sm, "answered")

    frames = await collect(event_stream(sm, 0, until(2), sleep=a_call_is_answered))
    assert ids(frames) == [1]


async def test_a_quiet_stream_sends_a_heartbeat(engine, session):
    sm = make_sessionmaker(engine)
    now = [0.0]

    async def a_long_wait(_seconds: float) -> None:
        now[0] += HEARTBEAT_S

    def clock() -> float:
        return now[0]

    quiet = await collect(event_stream(sm, 0, until(1), sleep=a_long_wait, clock=clock))
    assert quiet == []  # nothing yet: 15 s haven't passed
    now[0] = 0.0
    frames = await collect(event_stream(sm, 0, until(2), sleep=a_long_wait, clock=clock))
    assert frames == [": heartbeat\n\n"]


async def test_stops_when_the_client_leaves(engine, session):
    sm = make_sessionmaker(engine)
    write(sm, "created")
    assert await collect(event_stream(sm, 0, until(0), sleep=no_sleep)) == []


@pytest.fixture
def client(engine, session, monkeypatch):
    monkeypatch.setattr(events_api, "POLL_S", 0)
    checks = iter([False, True])  # the client stays for one poll

    async def disconnected(self) -> bool:
        return next(checks)

    monkeypatch.setattr(Request, "is_disconnected", disconnected)
    app = create_app(
        sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="test-key"
    )
    return TestClient(app)


def test_the_stream_needs_the_api_key(client):
    assert client.get("/events/stream").status_code == 401


def test_the_stream_resumes_after_last_event_id(client, engine):
    # Review focus 1: a reconnecting EventSource sends the last id it saw.
    write(make_sessionmaker(engine), "created", "criteria_ready", "criteria_approved")
    r = client.get("/events/stream?after=1", headers={**KEY, "Last-Event-ID": "2"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert ids(r.text.split("\n\n")) == [3]


def test_a_stream_without_after_starts_at_the_newest_event(engine, session):
    sm = make_sessionmaker(engine)
    write(sm, "created")
    newest = session.scalar(select(func.max(Event.id)))
    assert stream_start(sm, after=None, last_event_id=None) == newest
    assert stream_start(sm, after=3, last_event_id=None) == 3
    assert stream_start(sm, after=3, last_event_id=7) == 7


def test_the_stream_opens_at_once_even_when_quiet(client):
    r = client.get("/events/stream", headers={"X-API-Key": "test-key"})
    assert r.text.startswith(": open\n\n")
