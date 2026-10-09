from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from tests.fakes import FakeModels
from voiceagent.agents.criteria import Critique, Draft
from voiceagent.api.app import create_app
from voiceagent.criteria.schema import Criteria, JobFacts, Requirement, Rule
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import CriteriaVersion, Event, Job, Task
from voiceagent.llm import StructuredOutputError
from voiceagent.worker import main as worker_main
from voiceagent.worker.main import run_once
from voiceagent.worker.queue import LEASE, enqueue

JD = "Backend Engineer. Requirements: 3+ years of Python experience. Remote-friendly team."
PYTHON = Requirement(
    id="python_years",
    label="Python experience",
    kind="number",
    unit="years",
    question="How many years of Python?",
    hard=True,
    rule=Rule(op=">=", value=3),
    source="3+ years of Python experience",
)
GOOD = FakeModels(
    {
        Draft: [
            Draft(criteria=Criteria(requirements=[PYTHON], job_facts=JobFacts(summary="APIs")))
        ],
        Critique: [Critique()],
    }
)


def new_job(session):
    job = Job(title="Backend Engineer", company_name="Acme", description=JD)
    session.add(job)
    session.flush()
    enqueue(session, "generate_criteria", {"job_id": str(job.id)})
    session.commit()
    return job


def test_generates_a_draft_version(engine, session):
    job = new_job(session)
    assert run_once(make_sessionmaker(engine), GOOD) is True
    session.expire_all()
    cv = session.scalars(select(CriteriaVersion)).one()
    assert (cv.version, cv.status, cv.model) == (1, "draft", "fake")
    assert cv.requirements[0]["id"] == "python_years"
    assert session.get(Job, job.id).status == "criteria_ready"
    assert session.scalars(select(Task)).one().status == "done"
    assert [e.type for e in session.scalars(select(Event))] == ["criteria_ready"]


def test_regeneration_supersedes_old_draft(engine, session):
    job = new_job(session)
    sm = make_sessionmaker(engine)
    run_once(sm, GOOD)
    enqueue(session, "generate_criteria", {"job_id": str(job.id)})
    session.commit()
    run_once(sm, GOOD)
    session.expire_all()
    versions = {v.version: v.status for v in session.scalars(select(CriteriaVersion))}
    assert versions == {1: "superseded", 2: "draft"}


def test_failure_retries_then_marks_job(engine, session):
    job = new_job(session)
    broken = FakeModels({Draft: [StructuredOutputError("both models failed")]})
    sm = make_sessionmaker(engine)
    run_once(sm, broken)
    session.expire_all()
    task = session.scalars(select(Task)).one()
    assert (task.status, task.attempts) == ("queued", 1)
    assert session.get(Job, job.id).status == "generating"


def test_failed_generation_marks_job(engine, session):
    job = new_job(session)
    task = session.scalars(select(Task)).one()
    task.attempts = task.max_attempts - 1
    session.commit()
    broken = FakeModels({Draft: [StructuredOutputError("both models failed")]})
    run_once(make_sessionmaker(engine), broken)
    session.expire_all()
    assert session.scalars(select(Task)).one().status == "failed"
    assert session.get(Job, job.id).status == "criteria_failed"
    assert "criteria_failed" in [e.type for e in session.scalars(select(Event))]


def test_unknown_task_kind_fails_permanently(engine, session):
    enqueue(session, "nope", {})
    session.commit()
    run_once(make_sessionmaker(engine), GOOD)
    session.expire_all()
    task = session.scalars(select(Task)).one()
    assert task.status == "failed" and "unknown task kind" in task.last_error


def test_empty_queue_returns_false(engine, session):
    assert run_once(make_sessionmaker(engine), GOOD) is False


class Hooked(FakeModels):
    """FakeModels that calls `hook(schema)` before every structured reply."""

    def __init__(self, replies, hook):
        super().__init__(replies)
        self.hook = hook

    def structured(self, schema, system, user):
        self.hook(schema)
        return super().structured(schema, system, user)


GOOD_REPLIES = {
    Draft: [Draft(criteria=Criteria(requirements=[PYTHON], job_facts=JobFacts(summary="APIs")))],
    Critique: [Critique()],
}


def test_approval_during_generation_keeps_the_job_active(engine, session):
    job = new_job(session)
    job_id = job.id
    session.add(
        CriteriaVersion(
            job_id=job_id,
            version=1,
            status="draft",
            requirements=[PYTHON.model_dump(mode="json")],
            job_facts={"summary": "APIs"},
        )
    )
    session.commit()
    sm = make_sessionmaker(engine)
    api = TestClient(create_app(sessionmaker=sm, models=FakeModels(), api_key="k"))
    approved = []

    def approve_mid_run(schema):
        if schema is Critique and not approved:
            r = api.post(f"/jobs/{job_id}/criteria:approve", headers={"X-API-Key": "k"})
            approved.append(r.status_code)

    run_once(sm, Hooked(GOOD_REPLIES, approve_mid_run))
    session.expire_all()
    assert approved == [200]
    assert session.get(Job, job_id).status == "active"
    versions = {v.version: v.status for v in session.scalars(select(CriteriaVersion))}
    assert versions == {1: "approved", 2: "draft"}
    assert session.scalars(select(Task)).one().status == "done"


def test_lease_lost_on_the_last_attempt_marks_the_job_failed(engine, session):
    job = new_job(session)
    task = session.scalars(select(Task)).one()
    task.status, task.attempts = "running", task.max_attempts
    task.started_at = datetime.now(UTC) - LEASE - timedelta(seconds=1)
    session.commit()
    assert run_once(make_sessionmaker(engine), GOOD) is True
    session.expire_all()
    task = session.scalars(select(Task)).one()
    assert task.status == "failed" and "lease" in task.last_error
    assert session.get(Job, job.id).status == "criteria_failed"
    assert [e.type for e in session.scalars(select(Event))] == ["criteria_failed"]
    assert session.scalars(select(CriteriaVersion)).all() == []  # the handler didn't run


def test_crash_before_completion_leaves_no_half_written_result(engine, session, monkeypatch):
    job = new_job(session)

    def crash(session, task):
        raise RuntimeError("worker died")

    monkeypatch.setattr(worker_main, "complete", crash)
    with pytest.raises(RuntimeError):
        run_once(make_sessionmaker(engine), GOOD)
    session.expire_all()
    assert session.scalars(select(CriteriaVersion)).all() == []
    assert session.scalars(select(Task)).one().status == "running"
    assert session.get(Job, job.id).status == "generating"


def test_no_transaction_stays_open_while_the_agent_runs(engine, session):
    new_job(session)
    session.close()
    idle_in_transaction = []

    def probe(schema):
        with engine.connect() as c:
            idle_in_transaction.append(
                c.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname = "
                        "current_database() AND state LIKE 'idle in transaction%'"
                    )
                ).scalar()
            )

    run_once(make_sessionmaker(engine), Hooked(GOOD_REPLIES, probe))
    assert idle_in_transaction and set(idle_in_transaction) == {0}


def test_worker_loop_survives_database_errors(monkeypatch):
    class Stop(BaseException):
        pass

    calls = []

    def flaky_run_once(sessionmaker, models):
        calls.append(1)
        if len(calls) == 1:
            raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))
        raise Stop

    monkeypatch.setattr(worker_main, "get_settings", lambda: SimpleNamespace(database_url=""))
    monkeypatch.setattr(worker_main, "make_engine", lambda url: None)
    monkeypatch.setattr(worker_main, "make_sessionmaker", lambda engine: None)
    monkeypatch.setattr(worker_main, "default_models", lambda settings: None)
    monkeypatch.setattr(worker_main, "run_once", flaky_run_once)
    monkeypatch.setattr(worker_main, "sweep_calls", lambda sessionmaker: None)
    monkeypatch.setattr(worker_main.time, "sleep", lambda s: None)
    with pytest.raises(Stop):
        worker_main.main()
    assert len(calls) == 2


def test_worker_loop_sweeps_stale_calls_once_a_minute(monkeypatch):
    # I1: a call the agent never reported back is closed by the worker, on a cadence.
    class Stop(BaseException):
        pass

    clock = iter([0.0, 0.0, 30.0, 61.0, 61.0, 90.0])
    sweeps, runs = [], []

    def run_once(sessionmaker, models):
        runs.append(1)
        if len(runs) == 4:
            raise Stop
        return True

    monkeypatch.setattr(worker_main, "get_settings", lambda: SimpleNamespace(database_url=""))
    monkeypatch.setattr(worker_main, "make_engine", lambda url: None)
    monkeypatch.setattr(worker_main, "make_sessionmaker", lambda engine: "sm")
    monkeypatch.setattr(worker_main, "default_models", lambda settings: None)
    monkeypatch.setattr(worker_main, "run_once", run_once)
    monkeypatch.setattr(worker_main, "sweep_calls", sweeps.append)
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: next(clock))
    with pytest.raises(Stop):
        worker_main.main()
    assert sweeps == ["sm", "sm"]  # at t=0 and t=61, not at t=30 or t=90


def test_long_runs_renew_their_lease(engine, session):
    new_job(session)
    task_id = session.scalars(select(Task)).one().id
    session.close()
    sm = make_sessionmaker(engine)
    started = []

    def age_then_read_lease(schema):
        with sm() as s:
            task = s.get(Task, task_id)
            if schema is Draft:  # pretend the run has already taken longer than the lease
                task.started_at = datetime.now(UTC) - LEASE - timedelta(minutes=1)
                s.commit()
            else:
                started.append(task.started_at)

    run_once(sm, Hooked(GOOD_REPLIES, age_then_read_lease))
    assert started and started[0] > datetime.now(UTC) - timedelta(minutes=1)


def test_new_draft_clears_stale_criteria(engine, session):
    job = new_job(session)
    job.criteria_stale = True
    session.commit()
    run_once(make_sessionmaker(engine), GOOD)
    session.expire_all()
    assert session.get(Job, job.id).criteria_stale is False


def test_description_edited_during_generation_leaves_criteria_stale(engine, session):
    # The draft was written from the old text, so it is stale the moment it lands.
    job = new_job(session)
    edited = []

    def edit_midway(schema):
        if not edited:
            with make_sessionmaker(engine)() as s:
                s.get(Job, job.id).description = JD + " Now fully on-site."
                s.commit()
            edited.append(True)

    run_once(make_sessionmaker(engine), Hooked(GOOD_REPLIES, edit_midway))
    session.expire_all()
    assert session.get(Job, job.id).criteria_stale is True
