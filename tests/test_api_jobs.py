import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fakes import FakeModels
from voiceagent.api.app import create_app
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Task

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
