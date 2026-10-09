import secrets
from collections.abc import Iterator
from typing import Annotated

from fastapi import Header, HTTPException, Request
from sqlalchemy.orm import Session

from voiceagent.llm import Models


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.sessionmaker() as session:
        yield session


def get_models(request: Request) -> Models:
    return request.app.state.models


def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.api_key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")
