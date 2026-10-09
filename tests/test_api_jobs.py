import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fakes import FakeModels
from voiceagent.api.app import create_app
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import CriteriaVersion, Event, Task
from voiceagent.settings import DEFAULT_VOICE

KEY = {"X-API-Key": "test-key"}
JOB = {
    "title": "Backend Engineer",
    "company_name": "Acme",
    "description": "We build APIs. Requirements: 3+ years of Python experience. Pune, hybrid.",
}


@pytest.fixture
def client(engine, session):
    app = create_app(
        sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="test-key"
    )
    return TestClient(app)


def test_health_needs_no_key(client):
    assert client.get("/health").json() == {"ok": True}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_endpoints_need_the_key(client, headers):
    assert client.get("/jobs", headers=headers).status_code == 401


def test_create_job_queues_criteria_generation(client, session):
    r = client.post("/jobs", json=JOB, headers=KEY)
    assert r.status_code == 201
    job = r.json()
    assert job["status"] == "generating" and job["agent_name"] == "Aria"
    task = session.scalars(select(Task)).one()
    assert (task.kind, task.payload) == ("generate_criteria", {"job_id": job["id"]})
    events = client.get("/events", headers=KEY).json()
    assert [e["type"] for e in events] == ["created"]


@pytest.mark.parametrize(
    "patch",
    [
        {"description": "too short"},
        {"description": "x" * 20001},
        {"timezone": "Mars/Base"},
        {"response_days": 0},
    ],
)
def test_create_job_validates(client, patch):
    assert client.post("/jobs", json=JOB | patch, headers=KEY).status_code == 422


def test_get_list_and_update_job(client):
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    assert client.get(f"/jobs/{job['id']}", headers=KEY).json()["title"] == "Backend Engineer"
    assert [j["id"] for j in client.get("/jobs", headers=KEY).json()] == [job["id"]]
    r = client.patch(
        f"/jobs/{job['id']}", json={"agent_name": "Meera", "response_days": 5}, headers=KEY
    )
    assert (r.json()["agent_name"], r.json()["response_days"]) == ("Meera", 5)


def test_unknown_job_is_404(client):
    assert client.get("/jobs/00000000-0000-0000-0000-000000000000", headers=KEY).status_code == 404


def test_events_paginate_after_an_id(client):
    client.post("/jobs", json=JOB, headers=KEY)
    client.post("/jobs", json=JOB, headers=KEY)
    first = client.get("/events", params={"limit": 1}, headers=KEY).json()
    rest = client.get("/events", params={"after": first[0]["id"]}, headers=KEY).json()
    assert len(first) == 1 and len(rest) == 1 and rest[0]["id"] > first[0]["id"]


def test_app_refuses_to_start_without_a_key(engine):
    with pytest.raises(RuntimeError):
        create_app(sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="")


def test_non_ascii_key_is_rejected_not_a_server_error(client):
    assert client.get("/jobs", headers={"X-API-Key": "clé".encode()}).status_code == 401


def test_new_job_defaults_voice_hours_and_fresh_criteria(client):
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    assert job["voice"] == DEFAULT_VOICE
    assert job["calling_hours"] == {"start": "10:00", "end": "19:00"}
    assert job["criteria_stale"] is False


def test_voice_and_calling_hours_are_settable(client):
    body = JOB | {"voice": "voice-123", "calling_hours": {"start": "09:30", "end": "18:00"}}
    job = client.post("/jobs", json=body, headers=KEY).json()
    assert (job["voice"], job["calling_hours"]) == ("voice-123", {"start": "09:30", "end": "18:00"})
    r = client.patch(
        f"/jobs/{job['id']}",
        json={"calling_hours": {"start": "11:00", "end": "17:00"}},
        headers=KEY,
    )
    assert r.json()["calling_hours"] == {"start": "11:00", "end": "17:00"}
    assert r.json()["voice"] == "voice-123"


@pytest.mark.parametrize(
    "hours",
    [
        {"start": "19:00", "end": "10:00"},
        {"start": "10:00", "end": "10:00"},
        {"start": "9:00", "end": "17:00"},
        {"start": "24:00", "end": "25:00"},
        {"start": "10:00"},
    ],
)
def test_calling_hours_are_validated(client, hours):
    assert client.post("/jobs", json=JOB | {"calling_hours": hours}, headers=KEY).status_code == 422
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    r = client.patch(f"/jobs/{job['id']}", json={"calling_hours": hours}, headers=KEY)
    assert r.status_code == 422


def job_with_criteria(client, session) -> str:
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    session.add(
        CriteriaVersion(
            job_id=uuid.UUID(job["id"]), version=1, requirements=[], job_facts={"summary": "APIs"}
        )
    )
    session.commit()
    return job["id"]


@pytest.mark.parametrize(
    "patch",
    [
        {"description": JOB["description"] + " Now fully on-site."},
        {"recruiter_notes": "Must also know Kafka."},
    ],
)
def test_editing_the_text_marks_criteria_stale(client, session, patch):
    job_id = job_with_criteria(client, session)
    r = client.patch(f"/jobs/{job_id}", json=patch, headers=KEY)
    assert r.json()["criteria_stale"] is True
    assert "criteria_stale" in [e.type for e in session.scalars(select(Event))]


@pytest.mark.parametrize("patch", [{"agent_name": "Meera"}, {"description": JOB["description"]}])
def test_edits_that_keep_the_text_leave_criteria_fresh(client, session, patch):
    job_id = job_with_criteria(client, session)
    assert (
        client.patch(f"/jobs/{job_id}", json=patch, headers=KEY).json()["criteria_stale"] is False
    )


def test_editing_before_any_criteria_exist_is_not_stale(client):
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    r = client.patch(f"/jobs/{job['id']}", json={"recruiter_notes": "Kafka too."}, headers=KEY)
    assert r.json()["criteria_stale"] is False
