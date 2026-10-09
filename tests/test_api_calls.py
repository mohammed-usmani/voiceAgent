from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from tests.fakes import FakeModels
from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.api.app import create_app
from voiceagent.call.store import MAX_CALL_MINUTES
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Candidate, Event, Task

KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client(engine, session):
    app = create_app(
        sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="test-key"
    )
    return TestClient(app)


def test_call_queues_a_dial_with_the_approved_criteria(client, session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY)
    assert r.status_code == 202
    call = r.json()
    assert (call["status"], call["attempt"], call["criteria_version_id"]) == (
        "queued",
        1,
        str(cv.id),
    )
    task = session.scalars(select(Task)).one()
    assert (task.kind, task.payload) == ("dial", {"call_id": call["id"]})
    session.expire_all()
    assert session.get(Candidate, candidate.id).status == "calling"
    assert [e.type for e in session.scalars(select(Event))] == ["queued"]


@pytest.mark.parametrize(
    ("job_status", "criteria_status"),
    [("criteria_ready", "draft"), ("generating", "draft"), ("closed", "approved")],
)
def test_call_needs_an_active_job_with_approved_criteria(
    client, session, job_status, criteria_status
):
    job, _ = seed_job(session, status=job_status, criteria_status=criteria_status)
    candidate = seed_candidate(session, job)
    assert client.post(f"/candidates/{candidate.id}/call", headers=KEY).status_code == 409
    assert session.scalars(select(Task)).all() == []


@pytest.mark.parametrize("active", ["queued", "ringing", "in_progress"])
def test_second_call_while_one_is_active_is_refused(client, session, active):
    # Review focus 1: a double-click must not ring the candidate twice.
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    seed_call(session, candidate, cv, status=active)
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY)
    assert r.status_code == 409 and "already" in r.json()["detail"]
    assert session.scalars(select(Task)).all() == []


def test_calling_again_after_no_answer_is_attempt_two(client, session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="no_answer")
    seed_call(session, candidate, cv, status="no_answer")
    assert client.post(f"/candidates/{candidate.id}/call", headers=KEY).json()["attempt"] == 2


def test_read_calls(client, session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    first = seed_call(session, candidate, cv, status="no_answer")
    turns = [{"role": "agent", "text": "Hi", "at": "2026-10-09T09:00:00+00:00"}]
    second = seed_call(session, candidate, cv, status="completed", attempt=2, transcript=turns)
    calls = client.get(f"/candidates/{candidate.id}/calls", headers=KEY).json()
    assert [c["id"] for c in calls] == [str(first.id), str(second.id)]
    one = client.get(f"/calls/{second.id}", headers=KEY).json()
    assert (one["status"], one["transcript"]) == ("completed", turns)
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/calls/{missing}", headers=KEY).status_code == 404
    assert client.post(f"/candidates/{missing}/call", headers=KEY).status_code == 404


def test_a_stale_active_call_does_not_block_a_new_one(client, session):
    # I1: an agent that never reported back must not lock the candidate out forever.
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="calling")
    stale = seed_call(session, candidate, cv, status="in_progress")
    session.execute(
        update(Call)
        .where(Call.id == stale.id)
        .values(updated_at=datetime.now(UTC) - timedelta(minutes=MAX_CALL_MINUTES + 1))
    )
    session.commit()
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY)
    assert r.status_code == 202 and r.json()["attempt"] == 2
    session.expire_all()
    assert session.get(Call, stale.id).status == "failed"
