import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from voiceagent.criteria.schema import JobFacts, Requirement
from voiceagent.settings import DEFAULT_VOICE


def _valid_timezone(v: str | None) -> str | None:
    if v is None:
        return v
    try:
        ZoneInfo(v)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"unknown timezone {v!r}") from None
    return v


HH_MM = r"^([01]\d|2[0-3]):[0-5]\d$"


class CallingHours(BaseModel):
    """Local times (the job's timezone) between which candidates may be called."""

    start: str = Field(pattern=HH_MM, examples=["10:00"])
    end: str = Field(pattern=HH_MM, examples=["19:00"])

    @model_validator(mode="after")
    def _start_before_end(self) -> "CallingHours":
        if self.start >= self.end:  # zero-padded HH:MM compares correctly as text
            raise ValueError("calling_hours start must be before end")
        return self


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    company_name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=50, max_length=20000)
    recruiter_notes: str = Field(default="", max_length=5000)
    agent_name: str = Field(default="Aria", min_length=1, max_length=60)
    timezone: str = "Asia/Kolkata"
    auto_send_invites: bool = True
    response_days: int = Field(default=3, ge=1, le=30)
    voice: str = Field(default=DEFAULT_VOICE, min_length=1, max_length=100)
    calling_hours: CallingHours = Field(
        default_factory=lambda: CallingHours(start="10:00", end="19:00")
    )

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
    voice: str | None = Field(default=None, min_length=1, max_length=100)
    calling_hours: CallingHours | None = None

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
    voice: str
    calling_hours: CallingHours
    criteria_stale: bool
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


PHONE = re.compile(r"^\+[1-9]\d{7,14}$")  # E.164
SIP_ADDRESS = re.compile(r"^(sip:)?[^@\s:]+@[^@\s]+$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _blank_to_none(v):
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


class CandidateCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(max_length=320)
    phone: str | None = Field(default=None, description="international format, e.g. +919876543210")
    sip_address: str | None = Field(
        default=None, max_length=200, examples=["priya@sip.linphone.org"]
    )
    resume_text: str = Field(default="", max_length=20000)

    @field_validator("name", "resume_text", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if not EMAIL.match(v):
                raise ValueError(f"{v!r} is not an email address")
        return v

    @field_validator("phone", mode="before")
    @classmethod
    def _phone(cls, v):
        v = _blank_to_none(v)
        if isinstance(v, str):
            v = re.sub(r"[\s().-]", "", v)
            if not PHONE.match(v):
                raise ValueError("phone must be in international format, like +919876543210")
        return v

    @field_validator("sip_address", mode="before")
    @classmethod
    def _sip(cls, v):
        v = _blank_to_none(v)
        if isinstance(v, str) and not SIP_ADDRESS.match(v):
            raise ValueError("SIP address must look like user@host")
        return v

    @model_validator(mode="after")
    def _reachable(self) -> "CandidateCreate":
        if self.phone is None and self.sip_address is None:
            raise ValueError("give a phone number or a SIP address")
        return self


class CandidateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    job_id: uuid.UUID
    name: str
    email: str
    phone: str | None
    sip_address: str | None
    resume_text: str
    status: str
    created_at: datetime


class CallRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="call a candidate who is already screened, declined or withdrew",
    )


class CallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    candidate_id: uuid.UUID
    criteria_version_id: uuid.UUID
    room: str
    attempt: int
    status: str
    end_reason: str | None
    callback_at: datetime | None
    consent_given: bool | None
    started_at: datetime | None
    ended_at: datetime | None
    error: str | None
    transcript: list[dict]
    created_at: datetime


class RowError(BaseModel):
    row: int = Field(description="spreadsheet row; the header is row 1")
    errors: list[str]


class ImportResult(BaseModel):
    created: int
    errors: list[RowError]
