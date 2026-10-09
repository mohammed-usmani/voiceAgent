from sqlalchemy import text

from voiceagent.db.models import Job


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
