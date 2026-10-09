import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from tests.conftest import TEST_URL
from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.db.models import Call, Job


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
