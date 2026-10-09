from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from tests.fakes import FakeModels
from tests.seed import seed_call, seed_decision, seed_finished_call, seed_message
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Candidate, CriteriaVersion, Decision, Event, Message, Task
from voiceagent.outcomes.email import LogMailer
from voiceagent.worker.main import run_once
from voiceagent.worker.queue import LEASE
from voiceagent.worker.send import queue_send


def queued_message(session, **message_kw):
    call = seed_finished_call(session)
    session.get(Candidate, call.candidate_id).status = "decided"
    decision = seed_decision(session, call)
    message = seed_message(session, decision, **message_kw)
    queue_send(session, decision, message)
    session.commit()
    return decision, message


def test_queue_send_approves_and_enqueues(session):
    decision, message = queued_message(session)
    assert (decision.status, message.status) == ("approved", "queued")
    task = session.scalars(select(Task)).one()
    assert (task.kind, task.payload) == ("send_message", {"message_id": str(message.id)})


def test_sends_and_marks_everything_notified(engine, session, mailer):
    decision, message = queued_message(session)
    assert run_once(make_sessionmaker(engine), FakeModels()) is True
    session.expire_all()
    m = session.get(Message, message.id)
    assert (m.status, m.error) == ("sent", None) and m.sent_at is not None
    assert session.get(Decision, decision.id).status == "sent"
    assert session.get(Candidate, decision.candidate_id).status == "notified"
    (email,) = mailer.sent
    assert (email.to, email.subject) == ("priya@example.com", "Your application to Acme")
    assert session.scalars(select(Task)).one().status == "done"
    assert [e.type for e in session.scalars(select(Event).order_by(Event.id))] == [
        "queued",
        "sent",
    ]


def test_a_send_without_smtp_says_it_was_not_delivered(engine, session, monkeypatch):
    monkeypatch.setattr("voiceagent.worker.send.MAILER", LogMailer())
    decision, message = queued_message(session)
    run_once(make_sessionmaker(engine), FakeModels())
    session.expire_all()
    sent = session.scalars(select(Event).where(Event.type == "sent")).one()
    assert sent.data == {"decision_id": str(decision.id), "delivered": False}


def test_a_failed_send_is_marked_for_a_manual_retry(engine, session, mailer):
    mailer.error = ConnectionRefusedError("smtp down")
    decision, message = queued_message(session)
    run_once(make_sessionmaker(engine), FakeModels())
    session.expire_all()
    m = session.get(Message, message.id)
    assert (m.status, m.error) == ("failed", "smtp down")
    assert session.get(Decision, decision.id).status == "approved"
    assert session.get(Candidate, decision.candidate_id).status == "decided"
    assert session.scalars(select(Task)).one().status == "done"  # not retried by itself


def test_rerun_after_a_crash_mid_send_never_emails_twice(engine, session, mailer):
    # The worker died after marking the message "sending"; the lease ran out and the task
    # came back. The email may already be out: fail it rather than send it again.
    decision, message = queued_message(session)
    message.status = "sending"
    task = session.scalars(select(Task)).one()
    task.status, task.attempts = "running", 1
    task.started_at = datetime.now(UTC) - LEASE - timedelta(seconds=1)
    session.commit()
    run_once(make_sessionmaker(engine), FakeModels())
    session.expire_all()
    m = session.get(Message, message.id)
    assert m.status == "failed" and "check it wasn't received" in m.error
    assert mailer.sent == []


def test_only_queued_messages_are_sent(engine, session, mailer):
    decision, message = queued_message(session)
    message.status = "draft"  # e.g. a stale task for a message being edited again
    session.commit()
    run_once(make_sessionmaker(engine), FakeModels())
    assert mailer.sent == []


def test_lease_lost_on_the_last_attempt_fails_the_message(engine, session, mailer):
    decision, message = queued_message(session)
    task = session.scalars(select(Task)).one()
    task.status, task.attempts = "running", task.max_attempts
    task.started_at = datetime.now(UTC) - LEASE - timedelta(seconds=1)
    session.commit()
    run_once(make_sessionmaker(engine), FakeModels())
    session.expire_all()
    assert session.get(Message, message.id).status == "failed"
    assert mailer.sent == []


def seed_finished_call_for(session, decision):
    first = session.get(Call, decision.call_id)
    return seed_call(
        session,
        session.get(Candidate, first.candidate_id),
        session.get(CriteriaVersion, first.criteria_version_id),
        status="completed",
    )


def test_an_auto_send_is_refused_when_the_candidate_was_already_emailed(engine, session, mailer):
    decision, message = queued_message(session, kind="invite")
    task = session.scalars(select(Task)).one()
    task.payload = {**task.payload, "auto": True}
    old_call = seed_finished_call_for(session, decision)
    old = seed_decision(session, old_call)
    seed_message(session, old, status="sent")
    run_once(make_sessionmaker(engine), FakeModels())
    session.expire_all()
    assert session.get(Message, message.id).status == "draft"
    d = session.get(Decision, decision.id)
    assert (d.status, "already emailed" in d.summary) == ("pending_review", True)
    assert session.get(Candidate, decision.candidate_id).status == "decided"
    assert mailer.sent == []
