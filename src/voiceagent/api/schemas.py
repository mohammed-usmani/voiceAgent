import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from voiceagent.criteria.schema import JobFacts, Requirement


def _valid_timezone(v: str | None) -> str | None:
    if v is None:
        return v
    try:
        ZoneInfo(v)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"unknown timezone {v!r}") from None
    return v


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    company_name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=50, max_length=20000)
    recruiter_notes: str = Field(default="", max_length=5000)
    agent_name: str = Field(default="Aria", min_length=1, max_length=60)
    timezone: str = "Asia/Kolkata"
    auto_send_invites: bool = True
    response_days: int = Field(default=3, ge=1, le=30)

    _tz = field_validator("timezone")(_valid_timezone)


class JobUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=2, max_length=200)
    company_name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, min_length=50, max_length=20000)
    recruiter_notes: str | None = Field(default=None, max_length=5000)
    agent_name: str | None = Field(default=None, min_length=1, max_length=60)
    timezone: str | None = None
    auto_send_invites: bool | None = None
    response_days: int | None = Field(default=None, ge=1, le=30)

    _tz = field_validator("timezone")(_valid_timezone)


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    company_name: str
    description: str
    recruiter_notes: str
    agent_name: str
    timezone: str
    auto_send_invites: bool
    response_days: int
    status: str
    created_at: datetime


class CriteriaOut(BaseModel):
    id: uuid.UUID
    version: int
    status: str
    requirements: list[Requirement]
    job_facts: JobFacts
    pass_threshold: float
    warnings: list[str]
    errors: list[str]
    model: str
    approved_at: datetime | None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    entity: str
    entity_id: str
    type: str
    data: dict
    created_at: datetime
