"""screening-worker: claims tasks from Postgres and runs their handlers."""

import logging
import time

from sqlalchemy.orm import sessionmaker as SessionMaker

from voiceagent.db.base import make_engine, make_sessionmaker
from voiceagent.llm import Models, default_models
from voiceagent.settings import get_settings
from voiceagent.worker.handlers import HANDLERS
from voiceagent.worker.queue import claim, complete, fail

logger = logging.getLogger("voiceagent.worker")
IDLE_SLEEP_S = 1.0


def run_once(sessionmaker: SessionMaker, models: Models) -> bool:
    """Run one task. Returns False when there was nothing to do."""
    with sessionmaker() as session:
        task = claim(session)
        if task is None:
            return False
        handler = HANDLERS.get(task.kind)
        if handler is None:
            task.attempts = task.max_attempts
            fail(session, task, f"unknown task kind {task.kind!r}")
            return True
        run, on_failure = handler
        try:
            run(session, models, task.payload)
        except Exception as e:
            session.rollback()
            logger.exception("task %s (%s) failed", task.id, task.kind)
            if fail(session, task, repr(e)) and on_failure is not None:
                on_failure(session, task.payload, repr(e))
        else:
            complete(session, task)
        return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    sessionmaker = make_sessionmaker(make_engine(settings.database_url))
    models = default_models(settings)
    logger.info("worker started")
    while True:
        if not run_once(sessionmaker, models):
            time.sleep(IDLE_SLEEP_S)
