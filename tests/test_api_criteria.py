import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fakes import FakeModels
from voiceagent.api.app import create_app
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import CriteriaVersion, Job, Knowledge, Task

KEY = {"X-API-Key": "test-key"}
JOB = {
    "title": "Backend Engineer",
    "company_name": "Acme",
    "description": "We build APIs. Requirements: 3+ years of Python experience. Pune, hybrid.",
}
REQ = {
    "id": "python_years",
    "label": "Python experience",
    "kind": "number",
    "unit": "years",
    "question": "How many years of Python?",
    "hard": True,
    "rule": {"op": ">=", "value": 3},
    "source": "3+ years of Python experience",
}
CRITERIA = {"requirements": [REQ], "job_facts": {"summary": "APIs"}, "pass_threshold": 0.6}


def make_client(engine, models=None):
    return TestClient(
        create_app(
            sessionmaker=make_sessionmaker(engine),
            models=models or FakeModels(),
            api_key="test-key",
        )
    )


@pytest.fixture
def client(engine, session):
    return make_client(engine)


def job_with_draft(client, session, requirements=(REQ,)):
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    j = session.get(Job, job["id"])
    session.add(
        CriteriaVersion(
            job_id=j.id,
            version=1,
            status="draft",
            requirements=list(requirements),
            job_facts={"summary": "APIs"},
            warnings=["notice_days: removed (illegal: age)"],
            model="fake",
        )
    )
    j.status = "criteria_ready"
    for t in session.scalars(select(Task)):  # the worker already ran the create task
        t.status = "done"
    session.commit()
    return job["id"]


def test_criteria_not_ready_yet(client):
    job = client.post("/jobs", json=JOB, headers=KEY).json()
    r = client.get(f"/jobs/{job['id']}/criteria", headers=KEY)
    assert r.status_code == 404 and "generating" in r.json()["detail"]


def test_read_draft_with_warnings_and_no_errors(client, session):
    job_id = job_with_draft(client, session)
    body = client.get(f"/jobs/{job_id}/criteria", headers=KEY).json()
    assert body["status"] == "draft" and body["version"] == 1
    assert body["errors"] == [] and body["warnings"]


def test_approve_activates_job_and_indexes_knowledge(client, session):
    job_id = job_with_draft(client, session)
    r = client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    assert r.status_code == 200 and r.json()["status"] == "approved"
    session.expire_all()
    assert session.get(Job, job_id).status == "active"
    assert [k.data["id"] for k in session.scalars(select(Knowledge))] == ["python_years"]
    again = client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    assert again.status_code == 409


def test_edit_draft_in_place_normalizes_weights(client, session):
    job_id = job_with_draft(client, session)
    soft = {
        "id": "fastapi",
        "label": "FastAPI",
        "kind": "text",
        "question": "Tell me about it",
        "weight": 0.4,
        "rubric": "Built a production API",
        "source": "We build APIs",
    }
    r = client.patch(
        f"/jobs/{job_id}/criteria", json=CRITERIA | {"requirements": [REQ, soft]}, headers=KEY
    )
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1 and body["requirements"][1]["weight"] == 1.0
    assert body["warnings"] == []


def test_editing_approved_criteria_creates_a_new_draft(client, session):
    job_id = job_with_draft(client, session)
    client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    r = client.patch(f"/jobs/{job_id}/criteria", json=CRITERIA, headers=KEY)
    assert (r.json()["version"], r.json()["status"]) == (2, "draft")
    session.expire_all()
    assert session.get(Job, job_id).status == "active"  # v1 still governs calls
    client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    session.expire_all()
    statuses = {v.version: v.status for v in session.scalars(select(CriteriaVersion))}
    assert statuses == {1: "superseded", 2: "approved"}


def test_invalid_edit_blocks_approval(client, session):
    job_id = job_with_draft(client, session)
    bad = REQ | {"rule": {"op": ">=", "value": "three"}}
    r = client.patch(
        f"/jobs/{job_id}/criteria", json=CRITERIA | {"requirements": [bad]}, headers=KEY
    )
    assert r.status_code == 200 and r.json()["errors"]
    r = client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    assert r.status_code == 422 and r.json()["detail"]["errors"]


def test_empty_criteria_cannot_be_approved(client, session):
    job_id = job_with_draft(client, session, requirements=())
    assert client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY).status_code == 422


def test_approve_rolls_back_when_embedding_fails(engine, session):
    client = make_client(engine, FakeModels(embed_error=RuntimeError("embeddings down")))
    job_id = job_with_draft(client, session)
    r = client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    assert r.status_code == 502
    session.expire_all()
    assert session.scalars(select(CriteriaVersion)).one().status == "draft"
    assert session.get(Job, job_id).status == "criteria_ready"
    assert session.scalars(select(Knowledge)).all() == []


def test_regenerate_queues_a_task_and_refuses_while_generating(client, session):
    job_id = job_with_draft(client, session)
    assert client.post(f"/jobs/{job_id}/criteria:regenerate", headers=KEY).status_code == 202
    session.expire_all()
    assert session.get(Job, job_id).status == "generating"
    assert len(session.scalars(select(Task)).all()) == 2  # create + regenerate
    assert client.post(f"/jobs/{job_id}/criteria:regenerate", headers=KEY).status_code == 409


def test_regenerating_an_active_job_keeps_it_active(client, session):
    job_id = job_with_draft(client, session)
    client.post(f"/jobs/{job_id}/criteria:approve", headers=KEY)
    assert client.post(f"/jobs/{job_id}/criteria:regenerate", headers=KEY).status_code == 202
    session.expire_all()
    assert session.get(Job, job_id).status == "active"
