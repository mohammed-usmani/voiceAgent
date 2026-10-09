from datetime import UTC, datetime

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from tests.conftest import TEST_URL
from tests.seed import (
    seed_call,
    seed_candidate,
    seed_decision,
    seed_finished_call,
    seed_job,
    seed_message,
)
from voiceagent.db.models import Answer, Call, Campaign, Candidate, Decision, Job, Message


def test_migration_creates_tables_and_vector_extension(session):
    tables = set(
        session.scalars(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
    )
    assert {"jobs", "criteria_versions", "knowledge", "tasks", "events"} <= tables
    assert session.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")) == 1


def test_job_defaults(session):
    job = Job(title="Backend Engineer", company_name="Acme", description="x" * 60)
    session.add(job)
    session.commit()
    assert job.status == "generating"
    assert job.agent_name == "Aria"
    assert job.calling_hours == {"start": "10:00", "end": "19:00"}
    assert job.created_at is not None


def test_models_match_the_migrations(engine):
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", TEST_URL)
    command.check(cfg)  # raises if autogenerate would emit any operation


def test_candidates_and_calls_tables_exist(session):
    tables = set(
        session.scalars(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
    )
    assert {"candidates", "calls"} <= tables


def test_candidate_needs_a_phone_or_sip_address(session):
    job, _ = seed_job(session)
    with pytest.raises(IntegrityError):
        seed_candidate(session, job, phone=None, sip_address=None)


def test_candidate_email_is_unique_per_job(session):
    job, _ = seed_job(session)
    seed_candidate(session, job)
    with pytest.raises(IntegrityError):
        seed_candidate(session, job)


def test_call_defaults_and_cascade_on_candidate_delete(session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    call = seed_call(session, candidate, cv)
    assert (call.attempt, call.transcript, call.room, call.end_reason) == (1, [], "", None)
    session.delete(candidate)
    session.commit()
    assert session.scalars(select(Call)).all() == []


def test_connections_give_up_quickly_when_postgres_is_unreachable(monkeypatch):
    # I1/I2: the agent loads its call while the candidate waits on the line; an unreachable
    # host must fail in seconds, not after the OS's TCP timeout.
    from voiceagent.db import base

    seen = {}
    monkeypatch.setattr(base, "create_engine", lambda url, **kw: seen.update(kw))
    base.make_engine("postgresql+psycopg://x@10.255.255.1/db")
    assert seen["connect_args"] == {"connect_timeout": 5}


def test_one_decision_per_call_and_one_message_per_decision(session):
    call = seed_finished_call(session)
    decision = seed_decision(session, call)
    seed_message(session, decision)
    with pytest.raises(IntegrityError):
        seed_decision(session, call)
    session.rollback()
    with pytest.raises(IntegrityError):
        seed_message(session, decision)


def test_deleting_a_candidate_deletes_answers_decisions_and_messages(session):
    call = seed_finished_call(session)
    session.add(Answer(call_id=call.id, requirement_id="python_years", value=5.0, confidence=0.9))
    decision = seed_decision(session, call)
    seed_message(session, decision)
    session.delete(session.get(Candidate, call.candidate_id))
    session.commit()
    for model in (Answer, Decision, Message):
        assert session.scalars(select(model)).all() == []


def test_answer_and_decision_defaults(session):
    call = seed_finished_call(session)
    answer = Answer(call_id=call.id, requirement_id="python_years", value=None, confidence=0.0)
    session.add(answer)
    decision = seed_decision(session, call)
    session.commit()
    assert (answer.declined, answer.quote, answer.inconsistency) == (False, "", None)
    assert (decision.status, decision.final_result, decision.agent_review) == (
        "pending_review",
        None,
        None,
    )
    assert decision.trace == []


def test_a_candidate_can_have_only_one_active_call(session):
    # Review focus 2: whatever races past the application's checks, the database refuses
    # a second queued/ringing/in-progress call for the same candidate.
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    seed_call(session, candidate, cv, status="in_progress")
    with pytest.raises(IntegrityError):
        seed_call(session, candidate, cv, attempt=2, status="queued")
    session.rollback()
    seed_call(session, candidate, cv, attempt=2, status="no_answer")  # finished calls: any number


def test_campaign_defaults_and_cascade_on_job_delete(session):
    job, _ = seed_job(session)
    session.add(Campaign(job_id=job.id, started_at=datetime.now(UTC)))
    session.commit()
    campaign = session.get(Campaign, job.id)
    assert (campaign.status, campaign.max_concurrent) == ("active", 2)
    session.delete(job)
    session.commit()
    assert session.scalars(select(Campaign)).all() == []
