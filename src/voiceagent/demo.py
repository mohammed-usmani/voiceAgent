"""screening-demo-seed: a demo workspace for recordings and live demos. Two jobs from the
reference jobs in evals/jobs/ with approved criteria, twelve candidates across the
statuses, finished calls with transcripts, decisions with reasons and drafted messages.

It never dials or sends: nothing is queued, and every candidate has an unroutable SIP
address and an example.com email. Run from the repository root."""

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker as SessionMaker

from voiceagent.db.base import make_engine, make_sessionmaker
from voiceagent.db.models import Answer, Call, Candidate, CriteriaVersion, Decision, Job, Message
from voiceagent.evals.jobs import RefJob, load_jobs
from voiceagent.rules import apply_rule, decide
from voiceagent.settings import get_settings

DEMO = " (demo)"  # every seeded job's company name ends with this
RESPONSE_DAYS = 3


@dataclass(frozen=True)
class Said:
    """A candidate's answer to one requirement; `score` and `note` for weighted ones."""

    value: object
    quote: str
    score: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Person:
    name: str
    status: str  # the candidate's status once seeded
    resume: str
    answers: dict[str, Said] = field(default_factory=dict)
    end_reason: str | None = None  # how their conversation ended; None: no conversation
    unanswered: int = 0  # calls before it that nobody picked up
    decision: str | None = None  # the decision's status: pending_review, overridden, sent
    excluded: bool = False


SE = [
    Person(
        "Ananya Krishnan",
        "decided",
        "5 years of backend Python (Django, FastAPI) at a payments startup; PostgreSQL, Kafka.",
        {
            "python_years": Said(5, "About five years now, mostly Django and FastAPI services."),
            "hybrid_bengaluru": Said(
                True, "Yes, three days in Koramangala works, I live close by."
            ),
            "notice_days": Said(30, "My notice period is thirty days."),
            "sql_depth": Said(
                "composite index after reading the query plan",
                "Our settlement report took forty seconds. I read the query plan, added a "
                "composite index on merchant and date, and it dropped under a second.",
                0.9,
                "A concrete slow query, read from the plan, fixed with an index.",
            ),
            "api_design": Said(
                "idempotent refunds API used by other teams",
                "I designed the refunds API the checkout and support teams call. Every request "
                "takes an idempotency key, so a retry never refunds twice.",
                0.85,
                "Names the consumers and a real design decision (idempotency).",
            ),
            "expected_ctc": Said(26, "Around twenty-six lakhs.", 0.8, "Inside the 18-28 LPA band."),
        },
        end_reason="completed",
        decision="pending_review",
    ),
    Person(
        "Rahul Verma",
        "decided",
        "4 years of Python and Flask at an e-commerce company; reporting services.",
        {
            "python_years": Said(4, "Four years, building backend services in Python and Flask."),
            "hybrid_bengaluru": Said(True, "Yes, that's fine for me."),
            "notice_days": Said(90, "It's ninety days, and my company doesn't buy it out."),
            "sql_depth": Said(
                "window function instead of a self-join",
                "I rewrote a sales report to use a window function instead of a self-join.",
                0.6,
                "A real fix, but no word on how the problem was found.",
            ),
            "api_design": Said(
                "internal reporting API",
                "I built an internal API for the reporting team, with a v1 and a v2.",
                0.5,
                "Versioning mentioned; little about who used it or why.",
            ),
            "expected_ctc": Said(30, "Thirty lakhs.", 0.4, "Above the 18-28 LPA band."),
        },
        end_reason="completed",
        decision="pending_review",
    ),
    Person(
        "Farah Khan",
        "notified",
        "6 years of Python; owns payout APIs at a fintech; led a Postgres migration.",
        {
            "python_years": Said(6, "Six years of Python, the last three on payment APIs."),
            "hybrid_bengaluru": Said(True, "Yes, Koramangala three days a week is fine."),
            "notice_days": Said(45, "Forty-five days."),
            "sql_depth": Said(
                "partial index on unsettled rows",
                "A dashboard query scanned the whole ledger table, so I added a partial index "
                "on the unsettled rows.",
                0.8,
                "Concrete problem and fix.",
            ),
            "api_design": Said(
                "versioned payouts API used by four teams",
                "I own the payouts API four teams use. We version it and return clear error "
                "codes so callers know what to retry.",
                0.9,
                "Consumers, versioning and error handling.",
            ),
            "expected_ctc": Said(27, "Twenty-seven lakhs.", 0.75, "Inside the band."),
        },
        end_reason="completed",
        decision="sent",
    ),
    Person(
        "Dev Patel",
        "callback",
        "3 years of Python microservices at a logistics startup.",
        end_reason="callback_requested",
    ),
    Person("Meghna Rao", "pending", "7 years of Java, 1 year of Python.", excluded=True),
    Person("Arjun Iyer", "unreachable", "4 years of Go and Python.", unanswered=3),
]

CS = [
    Person(
        "Sana Sheikh",
        "decided",
        "B.A. graduate; 1 year in a domestic voice process for a bank.",
        {
            "graduate": Said(True, "Yes, I finished my B.A. last year."),
            "night_shifts": Said(None, "I'd have to check with my family about working nights."),
            "joining_days": Said(10, "I can join in ten days."),
            "frustrated_customer": Said(
                "listened, apologised, fixed the card block",
                "A customer was angry about a blocked card. I let him finish, apologised, and "
                "unblocked it while he was on the line.",
                0.7,
                "Listening, apology and a concrete fix.",
            ),
            "intl_voice_years": Said(
                0, "No international experience yet, only a domestic process.", 0.2, "None."
            ),
            "expected_salary": Said(3.5, "Around three and a half lakhs.", 0.8, "In the band."),
        },
        end_reason="completed",
        decision="pending_review",
    ),
    Person(
        "Vikram Singh",
        "decided",
        "B.Com graduate; 2 years of chat support for a US retailer.",
        {
            "graduate": Said(True, "Yes, B.Com from Osmania University."),
            "night_shifts": Said(True, "Yes, I already work US hours."),
            "joining_days": Said(15, "Fifteen days."),
            "frustrated_customer": Said(
                "stayed calm, escalated",
                "I stayed calm and passed it to my team lead.",
                0.3,
                "Escalated, but no listening or acknowledging described.",
            ),
            "intl_voice_years": Said(0, "None, I've only done chat support.", 0.0, "None."),
            "expected_salary": Said(
                4.2, "About four point two lakhs.", 0.6, "Upper end of the band."
            ),
        },
        end_reason="completed",
        decision="overridden",
    ),
    Person(
        "Pooja Nair",
        "declined_consent",
        "B.Sc graduate; fresher.",
        end_reason="declined_consent",
    ),
    Person(
        "Imran Ali",
        "withdrawn",
        "B.A. graduate; 3 years in an international voice process.",
        end_reason="not_interested",
    ),
    Person("Neha Gupta", "no_answer", "B.Com graduate; 1 year of telecalling.", unanswered=1),
    Person("Karthik Subramanian", "pending", "BBA graduate; 6 months of chat support."),
]

PEOPLE = {"software-engineer": SE, "customer-support": CS}

# What a short call sounds like after the greeting, by how it ended.
SHORT_CALLS = {
    "callback_requested": [
        ("candidate", "I'm driving right now. Could you call me tomorrow at five in the evening?"),
        ("agent", "Of course, {first}. I'll call you tomorrow at 5 PM. Drive safe!"),
    ],
    "declined_consent": [
        ("candidate", "Sorry, I'm not comfortable talking to an AI or being recorded."),
        ("agent", "I understand, {first}. The team will reach out by email instead. Thank you."),
    ],
    "not_interested": [
        ("candidate", "I've actually accepted another offer last week, so I'm no longer looking."),
        ("agent", "Congratulations, {first}! I'll let the team know. All the best."),
    ],
}
MESSAGES = {
    "invite": (
        "Next steps for your application to {company}",
        "Dear {first},\n\nThank you for speaking with us about the {title} role. The team "
        "would like to move forward with your application. The next step: {process}. We will "
        "be in touch shortly to schedule it.\n\nWarm regards,\nThe {company} hiring team",
    ),
    "rejection": (
        "Your application to {company}",
        "Dear {first},\n\nThank you for your time speaking with us about the {title} role. "
        "After careful consideration, the team has decided not to move forward with your "
        "application at this stage. We wish you the very best in your search.\n\nKind "
        "regards,\nThe {company} hiring team",
    ),
}
MESSAGE_KIND = {"suitable": "invite", "not_suitable": "rejection"}
OVERRIDE_NOTE = "Calm under pressure on the call; chat experience is enough for this batch."


def transcript(ref: RefJob, person: Person, start: datetime) -> list[dict]:
    first = person.name.split()[0]
    lines = [
        (
            "agent",
            f"Hi {first}, this is Aria, an AI assistant calling from {ref.company} about the "
            f"{ref.title} role you applied for. This call is recorded so the hiring team can "
            "review it. Is now a good time for a few quick questions?",
        )
    ]
    if person.end_reason in SHORT_CALLS:
        lines += [(role, text.format(first=first)) for role, text in SHORT_CALLS[person.end_reason]]
    else:
        lines.append(("candidate", "Yes, sure, go ahead."))
        for r in ref.criteria.requirements:
            if r.id in person.answers:
                lines += [("agent", r.question), ("candidate", person.answers[r.id].quote)]
        lines.append(
            (
                "agent",
                f"Thanks, {first}. The team will review and get back to you by email within "
                f"{RESPONSE_DAYS} days.",
            )
        )
    return [
        {"role": role, "text": text, "at": (start + timedelta(seconds=8 * i)).isoformat()}
        for i, (role, text) in enumerate(lines)
    ]


def assess(ref: RefJob, person: Person) -> dict:
    """The decision's fields, worked out from the answers with the real rules."""
    reasons, hard, failures, weighted, weights = [], [], [], 0.0, 0.0
    for r in ref.criteria.requirements:
        said = person.answers.get(r.id)
        if said is None:
            continue
        if r.hard:
            verdict = apply_rule(r, said.value)
            hard.append(verdict)
            if verdict == "fail":
                failures.append(r.id)
            note = f"{said.value} {r.unit or ''} vs rule {r.rule.op} {r.rule.value}: {verdict}"
            reasons.append(
                {
                    "requirement_id": r.id,
                    "verdict": verdict,
                    "score": None,
                    "quote": said.quote,
                    "confidence": 0.95 if said.value is not None else 0.4,
                    "note": " ".join(note.split()),
                }
            )
        else:
            weighted += r.weight * said.score
            weights += r.weight
            reasons.append(
                {
                    "requirement_id": r.id,
                    "verdict": "scored",
                    "score": said.score,
                    "quote": said.quote,
                    "confidence": 0.9,
                    "note": said.note,
                }
            )
    score = round(weighted / weights, 4) if weights else None
    threshold = ref.criteria.pass_threshold
    return {
        "result": decide(hard, score, threshold),
        "score": score,
        "hard_failures": failures,
        "reasons": reasons,
        "summary": f"Score {score:.2f} (pass mark {threshold:.2f})",
    }


def add_job(session: Session, ref: RefJob, now: datetime) -> tuple[Job, CriteriaVersion]:
    job = Job(
        title=ref.title,
        company_name=ref.company + DEMO,
        description=ref.description,
        recruiter_notes=ref.notes,
        auto_send_invites=False,  # every message waits for a recruiter
        response_days=RESPONSE_DAYS,
        status="active",
        created_at=now - timedelta(days=3),
    )
    session.add(job)
    session.flush()
    cv = CriteriaVersion(
        job_id=job.id,
        version=1,
        status="approved",
        requirements=[r.model_dump(mode="json") for r in ref.criteria.requirements],
        job_facts=ref.criteria.job_facts.model_dump(mode="json"),
        pass_threshold=ref.criteria.pass_threshold,
        warnings=[],
        model="demo",
        approved_at=now - timedelta(days=2),
    )
    session.add(cv)
    session.flush()
    return job, cv


def add_person(
    session: Session, ref: RefJob, job: Job, cv: CriteriaVersion, person: Person, start: datetime
) -> None:
    slug = person.name.split()[0].lower()
    candidate = Candidate(
        job_id=job.id,
        name=person.name,
        email=f"{slug}@example.com",
        sip_address=f"{slug}@demo.invalid",  # .invalid never resolves: nothing can ring
        resume_text=person.resume,
        status=person.status,
        excluded_at=start if person.excluded else None,
        created_at=start - timedelta(days=1),
    )
    session.add(candidate)
    session.flush()
    for n in range(person.unanswered):
        at = start + timedelta(hours=2 * n)
        session.add(
            Call(
                candidate_id=candidate.id,
                criteria_version_id=cv.id,
                room=f"screen-{uuid.uuid4().hex[:12]}",
                attempt=n + 1,
                status="no_answer",
                error="SIP 480: Temporarily Unavailable",
                prior_candidate_status="pending",
                created_at=at,
                ended_at=at + timedelta(seconds=40),
            )
        )
    if person.end_reason is None:
        return
    turns = transcript(ref, person, start)
    callback = person.end_reason == "callback_requested"
    ended = start + timedelta(seconds=8 * len(turns) + 5)
    call = Call(
        candidate_id=candidate.id,
        criteria_version_id=cv.id,
        room=f"screen-{uuid.uuid4().hex[:12]}",
        attempt=person.unanswered + 1,
        status="completed",
        end_reason=person.end_reason,
        callback_at=start + timedelta(days=1) if callback else None,
        consent_given={"completed": True, "declined_consent": False}.get(person.end_reason),
        started_at=start,
        ended_at=ended,
        transcript=turns,
        prior_candidate_status="pending",
        created_at=start - timedelta(seconds=20),
    )
    session.add(call)
    session.flush()
    for rid, said in person.answers.items():
        session.add(
            Answer(
                call_id=call.id,
                requirement_id=rid,
                value=said.value,
                quote=said.quote,
                confidence=0.95 if said.value is not None else 0.4,
            )
        )
    if person.decision is not None:
        add_decision(session, ref, candidate, call, person, ended)


def add_decision(
    session: Session, ref: RefJob, candidate: Candidate, call: Call, person: Person, at: datetime
) -> None:
    fields = assess(ref, person)
    overridden = person.decision == "overridden"
    decision = Decision(
        candidate_id=candidate.id,
        call_id=call.id,
        criteria_version_id=call.criteria_version_id,
        model="demo",
        status=person.decision,
        final_result="suitable" if overridden else None,
        reviewer_note=OVERRIDE_NOTE if overridden else None,
        agent_review={
            "agree": True,
            "notes": "The failed must-have is stated plainly in the candidate's own words.",
            "challenged": False,
            "model": "demo",
        }
        if fields["result"] == "not_suitable" and not overridden
        else None,
        created_at=at + timedelta(minutes=1),
        **fields,
    )
    session.add(decision)
    session.flush()
    kind = MESSAGE_KIND.get(decision.final_result or decision.result)
    if kind is None:
        return
    first = person.name.split()[0]
    process = ref.criteria.job_facts.interview_process or "an interview with the team"
    subject, body = (
        t.format(first=first, title=ref.title, company=ref.company, process=process)
        for t in MESSAGES[kind]
    )
    sent = person.decision == "sent"
    session.add(
        Message(
            decision_id=decision.id,
            kind=kind,
            to=candidate.email,
            subject=subject,
            body=body,
            status="sent" if sent else "draft",
            model="demo",
            sent_at=at + timedelta(hours=1) if sent else None,
        )
    )


def seed(session: Session, refs: list[RefJob], now: datetime) -> list[Job]:
    """Replace the demo jobs (and everything under them) with a fresh set. Not committed."""
    demo_jobs = select(Job.id).where(Job.company_name.endswith(DEMO))
    # Candidates first: their calls hold the criteria versions the jobs would cascade to.
    session.execute(delete(Candidate).where(Candidate.job_id.in_(demo_jobs)))
    session.execute(delete(Job).where(Job.id.in_(demo_jobs)))
    jobs = []
    for ref in refs:
        job, cv = add_job(session, ref, now)
        for i, person in enumerate(PEOPLE[ref.slug]):
            add_person(session, ref, job, cv, person, now - timedelta(hours=26 - 4 * i))
        jobs.append(job)
    return jobs


def main(argv: list[str] | None = None, sessionmaker: SessionMaker | None = None) -> int:
    parser = argparse.ArgumentParser(prog="screening-demo-seed", description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="seed even though the database has real jobs"
    )
    args = parser.parse_args(argv)
    sm = sessionmaker or make_sessionmaker(make_engine(get_settings().database_url))
    with sm() as session:
        real = session.scalar(
            select(func.count()).select_from(Job).where(~Job.company_name.endswith(DEMO))
        )
        if real and not args.force:
            print(
                f"This database has {real} real job(s); the demo is meant for an empty one. "
                "Pass --force to add it beside them.",
                file=sys.stderr,
            )
            return 2
        jobs = seed(session, load_jobs(list(PEOPLE)), datetime.now(UTC))
        session.commit()
    print(f"Seeded {len(jobs)} demo jobs: " + ", ".join(j.title for j in jobs))
    return 0
