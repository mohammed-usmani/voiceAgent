import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fakes import FakeModels
from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.api.app import create_app
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Campaign, Event

KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client(engine, session):
    app = create_app(
        sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="test-key"
    )
    return TestClient(app)


def test_starting_a_campaign(client, session):
    job, _ = seed_job(session)
    r = client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert (body["status"], body["max_concurrent"], body["in_flight"]) == ("active", 2, 0)
    assert session.get(Campaign, job.id).status == "active"
    assert [e.type for e in session.scalars(select(Event))] == ["campaign_started"]


@pytest.mark.parametrize(
    ("job_status", "criteria_status"),
    [("criteria_ready", "draft"), ("generating", "draft"), ("closed", "approved")],
)
def test_a_campaign_needs_an_active_job_with_approved_criteria(
    client, session, job_status, criteria_status
):
    job, _ = seed_job(session, status=job_status, criteria_status=criteria_status)
    r = client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    assert r.status_code == 409
    assert "approve its criteria" in r.json()["detail"]
    assert session.get(Campaign, job.id) is None


def test_max_concurrent_is_bounded(client, session):
    job, _ = seed_job(session)
    r = client.post(f"/jobs/{job.id}/campaign", json={"max_concurrent": 0}, headers=KEY)
    assert r.status_code == 422
    r = client.post(f"/jobs/{job.id}/campaign", json={"max_concurrent": 5}, headers=KEY)
    assert r.json()["max_concurrent"] == 5


def test_pausing_and_resuming(client, session):
    job, _ = seed_job(session)
    client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    r = client.post(f"/jobs/{job.id}/campaign:pause", headers=KEY)
    assert (r.status_code, r.json()["status"]) == (200, "paused")
    assert r.json()["paused_at"] is not None
    assert client.post(f"/jobs/{job.id}/campaign:pause", headers=KEY).status_code == 200
    r = client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    assert (r.json()["status"], r.json()["paused_at"]) == ("active", None)
    types = [e.type for e in session.scalars(select(Event).order_by(Event.id))]
    assert types == ["campaign_started", "campaign_paused", "campaign_started"]


def test_a_done_campaign_cannot_be_paused_but_can_restart(client, session):
    job, _ = seed_job(session)
    client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    session.get(Campaign, job.id).status = "done"
    session.commit()
    assert client.post(f"/jobs/{job.id}/campaign:pause", headers=KEY).status_code == 409
    r = client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    assert (r.json()["status"], r.json()["finished_at"]) == ("active", None)


def test_progress_counts_candidates_by_status_and_calls_in_flight(client, session):
    job, cv = seed_job(session)
    seed_candidate(session, job, email="a@example.com")
    seed_candidate(session, job, email="b@example.com")
    talking = seed_candidate(session, job, email="c@example.com", status="calling")
    seed_call(session, talking, cv, status="in_progress")
    client.post(f"/jobs/{job.id}/campaign", headers=KEY)
    body = client.get(f"/jobs/{job.id}/campaign", headers=KEY).json()
    assert body["candidates"] == {"pending": 2, "calling": 1}
    assert body["in_flight"] == 1


def test_no_campaign_or_no_job_is_404(client, session):
    job, _ = seed_job(session)
    assert client.get(f"/jobs/{job.id}/campaign", headers=KEY).status_code == 404
    assert client.post(f"/jobs/{job.id}/campaign:pause", headers=KEY).status_code == 404
    assert client.post(f"/jobs/{uuid.uuid4()}/campaign", headers=KEY).status_code == 404
