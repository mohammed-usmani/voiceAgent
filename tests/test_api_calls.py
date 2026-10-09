import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from tests.fakes import FakeModels
from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.api.app import create_app
from voiceagent.call.store import MAX_CALL_MINUTES, stop_call
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


@pytest.mark.parametrize(
    "status", ["withdrawn", "declined_consent", "screened", "decided", "notified"]
)
def test_finished_candidates_are_not_called_again_by_accident(client, session, status):
    # M6: someone who declined, withdrew or was already screened needs an explicit force.
    job, _ = seed_job(session)
    candidate = seed_candidate(session, job, status=status)
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY)
    assert r.status_code == 409 and "force" in r.json()["detail"]
    assert session.scalars(select(Task)).all() == []
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY, json={"force": True})
    assert r.status_code == 202


@pytest.mark.parametrize(("outcome", "status"), [("failed", "screened"), ("no_answer", "screened")])
def test_a_failed_recall_keeps_what_the_candidate_was(client, session, outcome, status):
    # M6: a forced re-call that fails must not wipe the record of the screening.
    job, _ = seed_job(session)
    candidate = seed_candidate(session, job, status="screened")
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY, json={"force": True})
    session.expire_all()
    call = session.get(Call, uuid.UUID(r.json()["id"]))
    stop_call(session, call, outcome, "boom")
    session.commit()
    session.expire_all()
    assert session.get(Candidate, candidate.id).status == status


def test_a_failed_call_after_a_callback_request_keeps_the_callback(client, session):
    job, _ = seed_job(session)
    candidate = seed_candidate(session, job, status="callback")
    r = client.post(f"/candidates/{candidate.id}/call", headers=KEY)
    session.expire_all()
    stop_call(session, session.get(Call, uuid.UUID(r.json()["id"])), "failed", "boom")
    session.commit()
    session.expire_all()
    assert session.get(Candidate, candidate.id).status == "callback"
