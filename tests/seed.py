"""Rows and objects for tests: an active job with approved criteria, a candidate, a call."""

from sqlalchemy.orm import Session

from voiceagent.criteria.schema import Criteria
from voiceagent.db.models import Call, Candidate, CriteriaVersion, Job
from voiceagent.settings import DEFAULT_VOICE

JD = (
    "Backend Engineer at Acme. Requirements: 3+ years of Python experience, "
    "on-site in Pune, 3 days a week. Budget: 25-35 LPA."
)
PYTHON = {
    "id": "python_years",
    "label": "Python experience",
    "kind": "number",
    "unit": "years",
    "question": "How many years have you worked with Python?",
    "hard": True,
    "rule": {"op": ">=", "value": 3},
    "source": "3+ years of Python experience",
}
ONSITE = {
    "id": "onsite_pune",
    "label": "Works from the Pune office",
    "kind": "boolean",
    "question": "Would working from our Pune office three days a week work for you?",
    "hard": True,
    "rule": {"op": "==", "value": True},
    "disclose": True,
    "source": "on-site in Pune, 3 days a week",
}
SALARY = {
    "id": "expected_salary",
    "label": "Expected salary",
    "kind": "number",
    "unit": "LPA",
    "question": "What salary are you expecting?",
    "weight": 1.0,
    "sensitive": True,
    "source": "Budget: 25-35 LPA",
}
REQUIREMENTS = [PYTHON, ONSITE, SALARY]
FACTS = {
    "summary": "Build payment APIs in Python.",
    "location": "Pune",
    "work_mode": "Hybrid, 3 days a week in the office",
    "interview_process": "Two technical rounds and a chat with the founder",
    "salary_band": None,
    "other": ["Health insurance from day one"],
}
CRITERIA = Criteria(requirements=REQUIREMENTS, job_facts=FACTS)


def sample_job(**kw) -> Job:
    """Every field a call reads, set explicitly (ORM defaults only apply on insert)."""
    fields = dict(
        title="Backend Engineer",
        company_name="Acme",
        description=JD,
        recruiter_notes="",
        agent_name="Aria",
        timezone="Asia/Kolkata",
        response_days=3,
        voice=DEFAULT_VOICE,
        status="active",
    )
    return Job(**(fields | kw))


def sample_candidate(**kw) -> Candidate:
    fields = dict(
        name="Priya Sharma",
        email="priya@example.com",
        phone="+919876543210",
        sip_address=None,
        resume_text="4 years at Infosys building Django services.",
        status="pending",
    )
    return Candidate(**(fields | kw))


def seed_job(
    session: Session, *, criteria_status: str = "approved", **kw
) -> tuple[Job, CriteriaVersion]:
    job = sample_job(**kw)
    session.add(job)
    session.flush()
    cv = CriteriaVersion(
        job_id=job.id,
        version=1,
        status=criteria_status,
        requirements=REQUIREMENTS,
        job_facts=FACTS,
        model="fake",
    )
    session.add(cv)
    session.commit()
    return job, cv


def seed_candidate(session: Session, job: Job, **kw) -> Candidate:
    candidate = sample_candidate(job_id=job.id, **kw)
    session.add(candidate)
    session.commit()
    return candidate


def seed_call(session: Session, candidate: Candidate, cv: CriteriaVersion, **kw) -> Call:
    call = Call(candidate_id=candidate.id, criteria_version_id=cv.id, **({"status": "queued"} | kw))
    session.add(call)
    session.commit()
    return call
