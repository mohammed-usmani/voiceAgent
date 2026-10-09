import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.fakes import FakeModels
from tests.seed import (
    CRITERIA,
    seed_call,
    seed_candidate,
    seed_decision,
    seed_finished_call,
    seed_job,
)
from tests.test_postcall import drain, replies
from voiceagent.api.app import create_app
from voiceagent.api.candidates import MAX_IMPORT_BYTES, parse_candidates_csv
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Candidate, Decision, Event, Job, Knowledge
from voiceagent.knowledge.store import index_override
from voiceagent.worker.queue import enqueue

KEY = {"X-API-Key": "test-key"}
CSV_HEADERS = {**KEY, "Content-Type": "text/csv"}
PRIYA = {
    "name": "Priya Sharma",
    "email": "Priya@Example.com",
    "phone": "+91 98765 43210",
    "resume_text": "4 years at Infosys.",
}
CSV_FILE = (
    "name,email,phone,sip_address,resume_text\n"
    'Priya Sharma,priya@example.com,+919876543210,,"4 years at Infosys,\nmostly Django."\n'
    "Arjun Rao,arjun@example.com,,arjun@sip.linphone.org,\n"
    "\n"
    "Bad Row,bad-email,12345,,\n"
)


@pytest.fixture
def client(engine, session):
    app = create_app(
        sessionmaker=make_sessionmaker(engine), models=FakeModels(), api_key="test-key"
    )
    return TestClient(app)


@pytest.fixture
def job(session):
    return seed_job(session)[0]


def test_add_candidate_normalises_contact_details(client, job):
    r = client.post(f"/jobs/{job.id}/candidates", json=PRIYA, headers=KEY)
    assert r.status_code == 201
    c = r.json()
    assert (c["email"], c["phone"], c["sip_address"], c["status"]) == (
        "priya@example.com",
        "+919876543210",
        None,
        "pending",
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"phone": "98765"},
        {"phone": None},
        {"email": "not-an-email"},
        {"name": ""},
        {"phone": None, "sip_address": "no-at-sign"},
    ],
)
def test_add_candidate_validates(client, job, patch):
    assert (
        client.post(f"/jobs/{job.id}/candidates", json=PRIYA | patch, headers=KEY).status_code
        == 422
    )


def test_same_email_twice_is_409(client, job):
    client.post(f"/jobs/{job.id}/candidates", json=PRIYA, headers=KEY)
    r = client.post(f"/jobs/{job.id}/candidates", json=PRIYA | {"name": "P. Sharma"}, headers=KEY)
    assert r.status_code == 409


def test_unknown_job_is_404(client):
    url = "/jobs/00000000-0000-0000-0000-000000000000/candidates"
    assert client.post(url, json=PRIYA, headers=KEY).status_code == 404
    assert client.get(url, headers=KEY).status_code == 404


def test_list_paginates_and_get_reads_one(client, job):
    for email in ("a@example.com", "b@example.com", "c@example.com"):
        client.post(f"/jobs/{job.id}/candidates", json=PRIYA | {"email": email}, headers=KEY)
    everyone = client.get(f"/jobs/{job.id}/candidates", headers=KEY).json()
    page = client.get(
        f"/jobs/{job.id}/candidates", params={"limit": 1, "offset": 1}, headers=KEY
    ).json()
    assert len(everyone) == 3 and [c["id"] for c in page] == [everyone[1]["id"]]
    one = client.get(f"/candidates/{everyone[0]['id']}", headers=KEY).json()
    assert one["email"] == everyone[0]["email"]
    filtered = client.get(f"/jobs/{job.id}/candidates", params={"status": "screened"}, headers=KEY)
    assert filtered.json() == []


def test_import_keeps_good_rows_and_reports_bad_ones(client, session, job):
    r = client.post(f"/jobs/{job.id}/candidates:import", content=CSV_FILE, headers=CSV_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["created"] == 2
    [bad] = body["errors"]
    assert bad["row"] == 4 and any("email" in e for e in bad["errors"])
    assert any("phone" in e for e in bad["errors"])
    priya = session.scalars(select(Candidate).where(Candidate.email == "priya@example.com")).one()
    assert priya.resume_text == "4 years at Infosys,\nmostly Django."
    assert "candidates_imported" in [e.type for e in session.scalars(select(Event))]


def test_import_handles_excel_bom_capitalised_headers_and_extra_columns(client, job):
    text = "﻿Name,Email,Phone,Notes\nPriya Sharma,priya@example.com,+919876543210,met at fair\n"
    r = client.post(f"/jobs/{job.id}/candidates:import", content=text.encode(), headers=CSV_HEADERS)
    assert r.json() == {"created": 1, "errors": []}


@pytest.mark.parametrize(
    "content",
    [
        b"name,email\nPriya,priya@example.com\n",  # no phone or sip_address column
        b"",
        b"\xff\xfe\x00n\x00a\x00m\x00e",  # UTF-16, not UTF-8
    ],
)
def test_unreadable_files_are_422(client, job, content):
    r = client.post(f"/jobs/{job.id}/candidates:import", content=content, headers=CSV_HEADERS)
    assert r.status_code == 422


def test_reimporting_a_file_reports_duplicates_instead_of_adding_them(client, session, job):
    # Review focus 2: a second upload (or a repeated email inside one file) must not
    # create a second candidate, or the same person gets called twice.
    client.post(f"/jobs/{job.id}/candidates:import", content=CSV_FILE, headers=CSV_HEADERS)
    repeat = CSV_FILE + "Priya Again,PRIYA@example.com,+919811111111,,\n"
    body = client.post(
        f"/jobs/{job.id}/candidates:import", content=repeat, headers=CSV_HEADERS
    ).json()
    assert body["created"] == 0
    assert [e["row"] for e in body["errors"]] == [2, 3, 4, 5]
    assert "already added" in body["errors"][0]["errors"][0]
    assert len(session.scalars(select(Candidate)).all()) == 2


def test_delete_purges_the_candidate_and_their_calls(client, session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    seed_call(session, candidate, cv, status="completed", transcript=[{"role": "agent"}])
    candidate_id = candidate.id  # read before expire_all: the row is about to go
    assert client.delete(f"/candidates/{candidate_id}", headers=KEY).status_code == 204
    session.expire_all()
    assert session.scalars(select(Candidate)).all() == []
    assert session.scalars(select(Call)).all() == []
    deleted = [e for e in session.scalars(select(Event)) if e.type == "deleted"]
    assert deleted[0].data == {"job_id": str(job.id)}
    assert client.get(f"/candidates/{candidate_id}", headers=KEY).status_code == 404


def test_deleting_a_candidate_leaves_none_of_their_words_in_other_decisions(
    client, engine, session
):
    # A's override calibrates B's scoring; once A is deleted, B's decision (and its trace,
    # served by the API) must hold nothing A said or the recruiter wrote about A.
    a_call = seed_finished_call(session)
    quote = "Forty lakhs and not a rupee less."
    reason = {"requirement_id": "expected_salary", "score": 0.2, "quote": quote}
    a = seed_decision(session, a_call, reasons=[reason])
    job = session.get(Job, session.get(Candidate, a_call.candidate_id).job_id)
    index_override(
        session,
        FakeModels(),
        job=job,
        decision=a,
        requirements=CRITERIA.requirements,
        final_result="suitable",
        note="Anita negotiates hard",
    )
    b_call = seed_finished_call(session)
    enqueue(session, "postcall", {"call_id": str(b_call.id)})
    session.commit()
    models = FakeModels(replies())
    drain(engine, models)
    assert quote in models.prompts[1][2]  # B's scorer did see the example
    assert client.delete(f"/candidates/{a_call.candidate_id}", headers=KEY).status_code == 204
    session.expire_all()
    b = session.scalars(select(Decision)).one()
    stored = json.dumps([b.trace, b.reasons, b.summary, b.agent_review])
    assert "rupee" not in stored and "Anita" not in stored
    assert "rupee" not in client.get(f"/decisions/{b.id}", headers=KEY).text


def test_parse_numbers_rows_as_a_spreadsheet_does():
    # No database: the multi-line résumé is still row 2, the blank line isn't a row.
    rows, errors = parse_candidates_csv(CSV_FILE)
    assert [(n, c.email) for n, c in rows] == [(2, "priya@example.com"), (3, "arjun@example.com")]
    assert [e.row for e in errors] == [4]


def test_parse_refuses_a_file_without_the_needed_columns():
    with pytest.raises(ValueError, match="name, email, and phone or sip_address"):
        parse_candidates_csv("full name,mail\nPriya,priya@example.com\n")


def test_import_refuses_files_over_the_byte_cap(client, session, job):
    # M8: the body is capped before it is decoded or parsed.
    big = CSV_FILE.encode() + b"x" * MAX_IMPORT_BYTES
    r = client.post(f"/jobs/{job.id}/candidates:import", content=big, headers=CSV_HEADERS)
    assert r.status_code == 413
    assert session.scalars(select(Candidate)).all() == []


def test_import_explains_a_non_utf8_file(client, job):
    # M8: Excel's plain "CSV" is cp1252; say how to fix it instead of a codec error.
    text = "name,email,phone\nRené Dupont,rene@example.com,+919876543210\n"
    r = client.post(
        f"/jobs/{job.id}/candidates:import", content=text.encode("cp1252"), headers=CSV_HEADERS
    )
    assert r.status_code == 422
    assert "UTF-8" in r.json()["detail"] and "codec" not in r.json()["detail"]


def test_delete_removes_the_candidates_quotes_from_calibration(client, session):
    call = seed_finished_call(session)
    decision = seed_decision(session, call, result="not_suitable")
    job = session.get(Job, session.get(Candidate, call.candidate_id).job_id)
    index_override(
        session,
        FakeModels(),
        job=job,
        decision=decision,
        requirements=CRITERIA.requirements,
        final_result="suitable",
        note="fine",
    )
    session.commit()
    assert session.scalars(select(Knowledge)).all() != []
    assert client.delete(f"/candidates/{call.candidate_id}", headers=KEY).status_code == 204
    session.expire_all()
    assert session.scalars(select(Knowledge)).all() == []
    assert session.scalars(select(Decision)).all() == []
