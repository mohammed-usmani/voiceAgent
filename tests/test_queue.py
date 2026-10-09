from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from voiceagent.db.models import Event, Task
from voiceagent.events import emit
from voiceagent.worker.queue import LEASE, claim, complete, enqueue, fail


def test_claim_takes_due_tasks_in_order(session):
    later = enqueue(session, "b", {}, run_at=datetime.now(UTC) + timedelta(hours=1))
    first = enqueue(session, "a", {"x": 1})
    session.commit()
    task = claim(session)
    assert task.id == first.id
    assert (task.status, task.attempts) == ("running", 1)
    assert claim(session) is None  # the other one isn't due
    assert later.status == "queued"


def test_complete_and_retry_then_fail(session):
    enqueue(session, "a", {})
    session.commit()
    task = claim(session)
    assert fail(session, task, "boom") is False
    assert task.status == "queued" and task.run_at > datetime.now(UTC)
    task.run_at = datetime.now(UTC)
    session.commit()
    task = claim(session)
    complete(session, task)
    assert task.status == "done" and task.last_error is None


def test_out_of_attempts_fails(session):
    task = enqueue(session, "a", {})
    task.attempts = task.max_attempts = 3
    task.status = "running"
    session.commit()
    assert fail(session, task, "boom") is True
    assert task.status == "failed" and task.last_error == "boom"


def test_stale_running_task_is_reclaimed(session):
    task = enqueue(session, "a", {})
    task.status, task.attempts = "running", 1
    task.started_at = datetime.now(UTC) - LEASE - timedelta(seconds=1)
    session.commit()
    again = claim(session)
    assert again.id == task.id and again.attempts == 2


def test_reclaimed_task_out_of_attempts_fails(session):
    task = enqueue(session, "a", {})
    task.status, task.attempts, task.max_attempts = "running", 3, 3
    task.started_at = datetime.now(UTC) - LEASE - timedelta(seconds=1)
    session.commit()
    assert claim(session).id == task.id  # returned failed, for the caller's failure hook
    session.commit()
    session.refresh(task)
    assert (task.status, task.attempts) == ("failed", 3) and "lease" in task.last_error


def test_emit_appends_ordered_events(session):
    emit(session, "job", "123", "created", {"title": "x"})
    emit(session, "job", "123", "criteria_ready")
    session.commit()
    events = list(session.scalars(select(Event).order_by(Event.id)))
    assert [e.type for e in events] == ["created", "criteria_ready"]
    assert events[1].data == {}


def test_enqueue_defaults(session):
    task = enqueue(session, "generate_criteria", {"job_id": "j"})
    session.commit()
    assert session.get(Task, task.id).payload == {"job_id": "j"}
