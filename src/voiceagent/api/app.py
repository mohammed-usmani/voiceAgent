"""screening-api: the REST API the recruiter frontend uses."""

import logging
import os

from fastapi import Depends, FastAPI
from sqlalchemy.orm import sessionmaker as SessionMaker

from voiceagent.api import candidates, events, jobs
from voiceagent.api.deps import require_api_key
from voiceagent.db.base import make_engine, make_sessionmaker
from voiceagent.llm import Models, default_models
from voiceagent.settings import get_settings


def create_app(*, sessionmaker: SessionMaker, models: Models, api_key: str) -> FastAPI:
    if not api_key:
        raise RuntimeError("SCREENING_API_KEY must be set")
    app = FastAPI(title="Screening API")
    app.state.sessionmaker = sessionmaker
    app.state.models = models
    app.state.api_key = api_key
    auth = [Depends(require_api_key)]
    app.include_router(jobs.router, dependencies=auth)
    app.include_router(candidates.router, dependencies=auth)
    app.include_router(events.router, dependencies=auth)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    app = create_app(
        sessionmaker=make_sessionmaker(make_engine(settings.database_url)),
        models=default_models(settings),
        api_key=settings.api_key,
    )
    uvicorn.run(
        app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000"))
    )
