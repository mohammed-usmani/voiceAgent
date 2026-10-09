"""A task queue in Postgres: workers claim with FOR UPDATE SKIP LOCKED, so several can
run at once without taking the same task."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from voiceagent.db.models import Task

# A task still "running" this long after it started (or last renewed its lease with
# `heartbeat`) belongs to a worker that died. Handlers renew between steps; one step is
# at most one structured() call, i.e. 3 x llm.MODEL_TIMEOUT_S, well under the lease.
LEASE = timedelta(minutes=10)
BACKOFF_S = 30


def enqueue(session: Session, kind: str, payload: dict, run_at: datetime | None = None) -> Task:
    task = Task(kind=kind, payload=payload, run_at=run_at or datetime.now(UTC))
    session.add(task)
    session.flush()
    return task


def claim(session: Session) -> Task | None:
    """Take the next due task and mark it running.

    A task whose lease ran out on its last attempt comes back already `failed`, uncommitted
    and still locked, so the caller can run the handler's failure hook in the same commit.
    """
    now = datetime.now(UTC)
    stmt = (
        select(Task)
        .where(
            or_(
                and_(Task.status == "queued", Task.run_at <= now),
                and_(Task.status == "running", Task.started_at < now - LEASE),
            )
        )
        .order_by(Task.run_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    task = session.scalars(stmt).first()
    if task is None:
        session.commit()
        return None
    if task.status == "running" and task.attempts >= task.max_attempts:
        task.status = "failed"
        task.last_error = "worker lost the task (lease expired) and it is out of attempts"
        return task
    task.status = "running"
    task.attempts += 1
    task.started_at = now
    session.commit()
    return task


def heartbeat(session: Session, task: Task) -> None:
    """Renew a running task's lease, in its own short transaction."""
    session.execute(update(Task).where(Task.id == task.id).values(started_at=datetime.now(UTC)))
    session.commit()


def complete(session: Session, task: Task) -> None:
    task.status = "done"
    task.last_error = None
    session.commit()


def fail(session: Session, task: Task, error: str) -> bool:
    """Record a failure. Returns True if the task is out of attempts and won't run again."""
    task.last_error = error[:2000]
    if task.attempts >= task.max_attempts:
        task.status = "failed"
        session.commit()
        return True
    task.status = "queued"
    task.run_at = datetime.now(UTC) + timedelta(seconds=BACKOFF_S * 2 ** (task.attempts - 1))
    session.commit()
    return False
