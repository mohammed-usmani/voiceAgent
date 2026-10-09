import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from tests.fakes import FakeMailer
from voiceagent.db.base import make_sessionmaker

TEST_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/screening_test"
)
TABLES = (
    "jobs, criteria_versions, knowledge, tasks, events, candidates, calls, answers, decisions, "
    "messages, campaigns"
)


@pytest.fixture(scope="session")
def engine():
    server, name = TEST_URL.rsplit("/", 1)
    admin = create_engine(f"{server}/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    except OperationalError:
        pytest.skip("Postgres not reachable; set TEST_DATABASE_URL")
    finally:
        admin.dispose()
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", TEST_URL)
    command.upgrade(cfg, "head")
    eng = create_engine(TEST_URL)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    s = make_sessionmaker(engine)()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def mailer(monkeypatch):
    """No test ever sends real email: every send goes to this fake."""
    fake = FakeMailer()
    monkeypatch.setattr("voiceagent.worker.send.MAILER", fake)
    return fake
