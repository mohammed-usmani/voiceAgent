from sqlalchemy import select

from tests.fakes import FakeModels
from voiceagent.agents.criteria import Critique, Draft
from voiceagent.criteria.schema import Criteria, JobFacts, Requirement, Rule
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import CriteriaVersion, Event, Job, Task
from voiceagent.llm import StructuredOutputError
from voiceagent.worker.main import run_once
from voiceagent.worker.queue import enqueue

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
