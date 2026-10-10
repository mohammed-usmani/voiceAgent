import json
from datetime import UTC, datetime, timedelta

import pytest
from livekit import api
from sqlalchemy import select

from tests.fakes import FakeModels
from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Candidate, Event, Task
from voiceagent.worker import dial as dial_module
from voiceagent.worker.main import run_once
from voiceagent.worker.queue import LEASE, enqueue


def queued_call(session, **candidate):
    job, cv = seed_job(session)
    person = seed_candidate(session, job, status="calling", **candidate)
    call = seed_call(session, person, cv)
    enqueue(session, "dial", {"call_id": str(call.id)})
    session.commit()
    return call


class FakeDialer:
    def __init__(self, error: Exception | None = None, during=None):
        self.calls = []
        self.error = error
        self.during = during

    def __call__(self, address, room, display_name, metadata):
        self.calls.append((address, room, display_name, metadata))
        if self.during:
            self.during()
        if self.error:
            raise self.error


def run(engine, monkeypatch, dialer):
    monkeypatch.setattr(dial_module, "DIALER", dialer)
    assert run_once(make_sessionmaker(engine), FakeModels()) is True


def test_answered_call_is_in_progress(engine, session, monkeypatch):
    call = queued_call(session)
    dialer = FakeDialer()
    run(engine, monkeypatch, dialer)
    session.expire_all()
    c = session.get(Call, call.id)
    assert (c.status, c.room) == ("in_progress", f"screen-{call.id.hex[:12]}")
    assert c.started_at is not None
    assert dialer.calls == [
        ("+919876543210", c.room, "Acme Hiring", json.dumps({"call_id": str(call.id)}))
    ]
    assert session.scalars(select(Task)).one().status == "done"
    events = [e.type for e in session.scalars(select(Event).order_by(Event.id))]
    assert events == ["ringing", "answered"]


def test_sip_address_is_dialled_when_there_is_no_phone(engine, session, monkeypatch):
    queued_call(session, phone=None, sip_address="priya@sip.linphone.org")
    dialer = FakeDialer()
    run(engine, monkeypatch, dialer)
    assert dialer.calls[0][0] == "priya@sip.linphone.org"


def sip_error(code: int, reason: str) -> api.SipCallError:
    meta = {"sip_status_code": str(code), "sip_status": reason}
    return api.SipCallError("unavailable", reason, status=503, metadata=meta)


@pytest.mark.parametrize(
    ("error", "status", "candidate_status"),
    [
        (sip_error(486, "Busy Here"), "no_answer", "no_answer"),
        (sip_error(408, "Request Timeout"), "no_answer", "no_answer"),
        (sip_error(603, "Decline"), "no_answer", "no_answer"),
        (sip_error(404, "Not Found"), "failed", "pending"),
        (RuntimeError("set SIP_OUTBOUND_TRUNK_ID to call phone numbers"), "failed", "pending"),
    ],
)
def test_unanswered_or_failed_dial(engine, session, monkeypatch, error, status, candidate_status):
    call = queued_call(session)
    run(engine, monkeypatch, FakeDialer(error=error))
    session.expire_all()
    c = session.get(Call, call.id)
    assert (c.status, c.error) == (status, str(error))
    assert c.ended_at is not None
    assert c.end_reason == ("error" if status == "failed" else None)
    assert session.get(Candidate, c.candidate_id).status == candidate_status
    # No retries until campaigns (milestone 5): the outcome is final for this attempt.
    assert session.scalars(select(Task)).one().status == "done"


def test_rerun_after_a_crash_mid_dial_never_rings_twice(engine, session, monkeypatch):
    # Review focus 4: the worker died after marking the call ringing; its lease ran out
    # and the task came back. The candidate may already have picked up.
    call = queued_call(session)
    call.status = "ringing"
    session.commit()
    dialer = FakeDialer()
    run(engine, monkeypatch, dialer)
    session.expire_all()
    assert dialer.calls == []
    c = session.get(Call, call.id)
    assert c.status == "failed" and "not redialed" in c.error
    assert session.get(Candidate, c.candidate_id).status == "pending"


def test_answer_never_overwrites_a_call_the_agent_already_finished(engine, session, monkeypatch):
    # Review focus 5: a short call can end in the agent process before the dial returns.
    call = queued_call(session)

    def agent_finishes():
        with make_sessionmaker(engine)() as s:
            s.get(Call, call.id).status = "completed"
            s.commit()

    run(engine, monkeypatch, FakeDialer(during=agent_finishes))
    session.expire_all()
    assert session.get(Call, call.id).status == "completed"


def test_call_deleted_while_queued_is_skipped(engine, session, monkeypatch):
    call = queued_call(session)
    session.delete(session.get(Candidate, call.candidate_id))
    session.commit()
    dialer = FakeDialer()
    run(engine, monkeypatch, dialer)
    assert dialer.calls == []
    assert session.scalars(select(Task)).one().status == "done"


def test_dial_task_out_of_attempts_marks_the_call_failed(engine, session, monkeypatch):
    call = queued_call(session)
    call.status = "ringing"
    task = session.scalars(select(Task)).one()
    task.status, task.attempts = "running", task.max_attempts
    task.started_at = datetime.now(UTC) - LEASE - timedelta(minutes=1)
    session.commit()
    run(engine, monkeypatch, FakeDialer())
    session.expire_all()
    assert session.get(Call, call.id).status == "failed"


@pytest.mark.parametrize(
    ("trunk", "dialled"), [("", "priya@sip.linphone.org"), ("ST_pstn", "+919876543210")]
)
def test_with_both_a_phone_and_a_sip_address_the_phone_needs_a_trunk(
    engine, session, monkeypatch, trunk, dialled
):
    # A phone number can only be rung through an outbound trunk: without one, the SIP
    # address is rung instead of failing the call.
    monkeypatch.setenv("SIP_OUTBOUND_TRUNK_ID", trunk)
    queued_call(session, phone="+919876543210", sip_address="priya@sip.linphone.org")
    dialer = FakeDialer()
    run(engine, monkeypatch, dialer)
    assert dialer.calls[0][0] == dialled
