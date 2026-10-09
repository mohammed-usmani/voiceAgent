import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fakes import FakeModels
from tests.seed import seed_decision, seed_finished_call, seed_message
from voiceagent.agents.messages import Check, Letter
from voiceagent.api.app import create_app
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Answer, Candidate, Decision, Event, Knowledge, Message, Task
from voiceagent.worker.main import run_once

KEY = {"X-API-Key": "test-key"}
INVITE = Letter(subject="Next steps at Acme", body="Hi Priya, we'd like to move forward.")
REJECTION = Letter(subject="Your application to Acme", body="Hi Priya, thank you.")


@pytest.fixture
def client(engine, session):
    app = create_app(
        sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="test-key"
    )
    return TestClient(app)


def decided(session, *, message=True, **decision_kw):
    """A screened candidate with a decision (not_suitable by default) and its draft."""
    call = seed_finished_call(session)
    session.get(Candidate, call.candidate_id).status = "decided"
    session.add(
        Answer(
            call_id=call.id,
            requirement_id="expected_salary",
            value=30.0,
            quote="30",
            confidence=0.9,
        )
    )
    session.add(
        Answer(call_id=call.id, requirement_id="python_years", value=2.0, quote="2", confidence=0.9)
    )
    decision = seed_decision(session, call, **decision_kw)
    if message:
        seed_message(session, decision)
    return decision


def drain(engine, models):
    sm = make_sessionmaker(engine)
    while run_once(sm, models):
        pass


def test_list_filters_by_job_and_status(client, session):
    a = decided(session)
    b = decided(session, status="sent")
    r = client.get("/decisions", params={"status": "pending_review"}, headers=KEY)
    assert [d["id"] for d in r.json()] == [str(a.id)]
    job_b = session.get(Candidate, b.candidate_id).job_id
    r = client.get("/decisions", params={"job_id": str(job_b)}, headers=KEY)
    assert [(d["id"], d["candidate_name"]) for d in r.json()] == [(str(b.id), "Priya Sharma")]
    assert len(client.get("/decisions", params={"limit": 1}, headers=KEY).json()) == 1
    assert client.get("/decisions", params={"limit": 1, "offset": 2}, headers=KEY).json() == []


def test_read_shows_answers_reasons_review_and_the_draft(client, session):
    d = decided(
        session,
        agent_review={"agree": True, "notes": "clear", "challenged": "not_suitable", "model": "r"},
    )
    body = client.get(f"/decisions/{d.id}", headers=KEY).json()
    assert (body["result"], body["status"], body["hard_failures"]) == (
        "not_suitable",
        "pending_review",
        ["python_years"],
    )
    assert [a["requirement_id"] for a in body["answers"]] == ["python_years", "expected_salary"]
    assert body["reasons"][0]["quote"] == "I am expecting around 30 lakhs per annum."
    assert body["agent_review"]["notes"] == "clear"
    assert (body["message"]["kind"], body["message"]["status"]) == ("rejection", "draft")


def test_unknown_ids_are_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/decisions/{missing}", headers=KEY).status_code == 404
    assert client.post(f"/decisions/{missing}:approve", headers=KEY).status_code == 404
    assert client.get(f"/messages/{missing}", headers=KEY).status_code == 404


def test_approve_sends_the_drafted_rejection(client, engine, session, mailer):
    d = decided(session)
    r = client.post(f"/decisions/{d.id}:approve", headers=KEY)
    assert r.status_code == 200
    assert (r.json()["status"], r.json()["message"]["status"]) == ("approved", "queued")
    drain(engine, FakeModels())
    session.expire_all()
    assert session.get(Decision, d.id).status == "sent"
    assert session.get(Candidate, d.candidate_id).status == "notified"
    assert [e.subject for e in mailer.sent] == ["Your application to Acme"]


def test_approving_twice_sends_one_email(client, engine, session, mailer):
    d = decided(session)
    assert client.post(f"/decisions/{d.id}:approve", headers=KEY).status_code == 200
    second = client.post(f"/decisions/{d.id}:approve", headers=KEY)
    assert (second.status_code, second.json()["detail"]) == (409, "the message is already queued")
    drain(engine, FakeModels())
    assert len(mailer.sent) == 1
    assert client.post(f"/decisions/{d.id}:approve", headers=KEY).status_code == 409  # sent
    assert [t.kind for t in session.scalars(select(Task))] == ["send_message"]


def test_needs_review_cannot_be_approved_without_a_result(client, session):
    d = decided(session, result="needs_review", message=False)
    r = client.post(f"/decisions/{d.id}:approve", headers=KEY)
    assert r.status_code == 409 and ":override" in r.json()["detail"]


def test_a_failed_send_is_retried_by_approving_again(client, engine, session, mailer):
    d = decided(session)
    mailer.error = ConnectionRefusedError("smtp down")
    client.post(f"/decisions/{d.id}:approve", headers=KEY)
    drain(engine, FakeModels())
    session.expire_all()
    assert session.scalars(select(Message)).one().status == "failed"
    mailer.error = None
    assert client.post(f"/decisions/{d.id}:approve", headers=KEY).status_code == 200
    drain(engine, FakeModels())
    session.expire_all()
    assert session.scalars(select(Message)).one().status == "sent"
    assert len(mailer.sent) == 1


def test_override_redrafts_indexes_and_waits_for_approval(client, engine, session, mailer):
    d = decided(session)
    r = client.post(
        f"/decisions/{d.id}:override",
        json={"result": "suitable", "note": "30 LPA is fine for this hire"},
        headers=KEY,
    )
    assert r.status_code == 200
    body = r.json()
    assert (body["result"], body["final_result"], body["status"]) == (
        "not_suitable",
        "suitable",
        "overridden",
    )
    assert body["message"] is None  # the rejection is gone; an invite is being drafted
    assert body["reviewer_note"] == "30 LPA is fine for this hire"
    k = session.scalars(select(Knowledge)).one()
    assert (k.kind, k.data["recruiter_result"], k.data["requirement_id"]) == (
        "override",
        "suitable",
        "expected_salary",
    )
    assert client.post(f"/decisions/{d.id}:approve", headers=KEY).status_code == 409

    drain(engine, FakeModels({Letter: [INVITE], Check: [Check()]}))
    session.expire_all()
    m = session.scalars(select(Message)).one()
    assert (m.kind, m.status) == ("invite", "draft")  # an override is never auto-sent
    assert mailer.sent == []
    assert client.post(f"/decisions/{d.id}:approve", headers=KEY).status_code == 200
    drain(engine, FakeModels())
    assert [e.subject for e in mailer.sent] == ["Next steps at Acme"]


def test_overriding_back_and_forth_sends_only_the_final_result(client, engine, session, mailer):
    # Recruiter flips to suitable, then back to not_suitable before the invite is drafted:
    # whatever runs in between, the candidate gets one rejection and never an invite.
    d = decided(session)
    for result in ("suitable", "not_suitable"):
        r = client.post(
            f"/decisions/{d.id}:override", json={"result": result, "note": "n"}, headers=KEY
        )
        assert r.status_code == 200
    drain(engine, FakeModels({Letter: [REJECTION], Check: [Check()]}))
    session.expire_all()
    m = session.scalars(select(Message)).one()
    assert (m.kind, m.status) == ("rejection", "draft")
    client.post(f"/decisions/{d.id}:approve", headers=KEY)
    drain(engine, FakeModels())
    assert [e.subject for e in mailer.sent] == ["Your application to Acme"]


def test_override_keeps_a_draft_of_the_same_kind(client, session):
    d = decided(session)
    client.post(
        f"/decisions/{d.id}:override",
        json={"result": "not_suitable", "note": "agreed"},
        headers=KEY,
    )
    session.expire_all()
    assert session.scalars(select(Message)).one().kind == "rejection"
    assert [t.kind for t in session.scalars(select(Task))] == []


@pytest.mark.parametrize(
    "body", [{"result": "needs_review", "note": "x"}, {"result": "suitable", "note": ""}, {}]
)
def test_override_needs_a_final_result_and_a_note(client, session, body):
    d = decided(session)
    assert client.post(f"/decisions/{d.id}:override", json=body, headers=KEY).status_code == 422


def test_a_sent_decision_cannot_be_overridden(client, session):
    d = decided(session, status="sent")
    r = client.post(
        f"/decisions/{d.id}:override", json={"result": "suitable", "note": "x"}, headers=KEY
    )
    assert r.status_code == 409


def test_a_failed_index_leaves_the_decision_unchanged(engine, session):
    app = create_app(
        sessionmaker=make_sessionmaker(engine),
        models=FakeModels(embed_error=RuntimeError("embedder down")),
        api_key="test-key",
    )
    d = decided(session)
    r = TestClient(app).post(
        f"/decisions/{d.id}:override", json={"result": "suitable", "note": "x"}, headers=KEY
    )
    assert r.status_code == 502
    session.expire_all()
    assert (session.get(Decision, d.id).status, session.scalars(select(Message)).one().kind) == (
        "pending_review",
        "rejection",
    )


def test_messages_can_be_edited_until_queued(client, session):
    d = decided(session)
    m = session.scalars(select(Message)).one()
    r = client.patch(f"/messages/{m.id}", json={"body": "Dear Priya, thank you."}, headers=KEY)
    assert (r.status_code, r.json()["body"], r.json()["subject"]) == (
        200,
        "Dear Priya, thank you.",
        "Your application to Acme",
    )
    assert client.get(f"/messages/{m.id}", headers=KEY).json()["body"] == "Dear Priya, thank you."
    client.post(f"/decisions/{d.id}:approve", headers=KEY)
    assert client.patch(f"/messages/{m.id}", json={"body": "x"}, headers=KEY).status_code == 409
    assert client.patch(f"/messages/{m.id}", json={"subject": ""}, headers=KEY).status_code == 422
    assert ("message", "edited") in [(e.entity, e.type) for e in session.scalars(select(Event))]
