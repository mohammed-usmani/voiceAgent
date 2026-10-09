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
ERROR_SLEEP_S = 5.0


def run_once(sessionmaker: SessionMaker, models: Models) -> bool:
    """Run one task. Returns False when there was nothing to do."""
    with sessionmaker() as session:
        task = claim(session)
        if task is None:
            return False
        run, on_failure = HANDLERS.get(task.kind, (None, None))
        if task.status == "failed":  # its worker died on the last attempt
            logger.error("task %s (%s): %s", task.id, task.kind, task.last_error)
            if on_failure is not None:
                on_failure(session, task.payload, task.last_error)
            session.commit()
            return True
        if run is None:
            task.attempts = task.max_attempts
            fail(session, task, f"unknown task kind {task.kind!r}")
            return True
        try:
            run(session, models, task)
        except Exception as e:
            session.rollback()
            logger.exception("task %s (%s) failed", task.id, task.kind)
            if fail(session, task, repr(e)) and on_failure is not None:
                on_failure(session, task.payload, repr(e))
                session.commit()
        else:
            complete(session, task)  # commits the handler's writes too
        return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    sessionmaker = make_sessionmaker(make_engine(settings.database_url))
    models = default_models(settings)
    logger.info("worker started")
    while True:
        try:
            busy = run_once(sessionmaker, models)
        except Exception:  # e.g. Postgres restarting; the task's lease brings it back
            logger.exception("worker loop failed; retrying in %ss", ERROR_SLEEP_S)
            time.sleep(ERROR_SLEEP_S)
            continue
        if not busy:
            time.sleep(IDLE_SLEEP_S)
