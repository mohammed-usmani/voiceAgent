"""Campaigns: call a job's candidates in the background, within the job's calling hours,
retrying unanswered calls and honouring callbacks. `tick` runs from the worker loop."""

import uuid
from collections import defaultdict
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from voiceagent.call.store import queue_call
from voiceagent.db.models import (
    ACTIVE_CALL_STATUSES,
    MAX_UNANSWERED,
    UNANSWERED_CALL_STATUSES,
    Call,
    Campaign,
    Candidate,
    CriteriaVersion,
    Job,
)
from voiceagent.events import emit

# Unanswered calls (no answer, busy, voicemail, failed) are retried this long after the
# last one ended, up to MAX_UNANSWERED calls in all.
RETRY_GAP = timedelta(hours=2)
# A requested callback is honoured at the time given even outside the job's calling hours,
# but never at night: between these local times only.
CALLBACK_HOURS = (time(9), time(21))
# A candidate who keeps asking to be called back is called at most this many times for it.
MAX_CALLBACKS = 3
# Candidates a campaign may call: never called, not answered yet, or asked for a callback.
CAMPAIGN_STATUSES = ("pending", "no_answer", "callback")
ANY_TIME = datetime.min.replace(tzinfo=UTC)


def next_call_at(status: str, calls: list[Call]) -> datetime | None:
    """The earliest time a campaign may call this candidate again (calling hours aside), or
    None if it never will. `calls` are the candidate's calls, oldest first."""
    if status not in CAMPAIGN_STATUSES:
        return None
    missed = [c for c in calls if c.status in UNANSWERED_CALL_STATUSES]
    if len(missed) >= MAX_UNANSWERED:
        return None
    due = missed[-1].ended_at + RETRY_GAP if missed else ANY_TIME
    if status == "callback":
        requested = [c for c in calls if c.end_reason == "callback_requested"]
        if len(requested) > MAX_CALLBACKS:
            return None
        if requested and requested[-1].callback_at is not None:
            due = max(due, requested[-1].callback_at)
    return due


def within_hours(now: datetime, job: Job, *, callback: bool = False) -> bool:
    """Whether `now` is inside the job's calling hours in its timezone (or, for a requested
    callback, inside CALLBACK_HOURS). Converting the instant to local time keeps this right
    across daylight-saving changes."""
    local = now.astimezone(ZoneInfo(job.timezone)).time()
    start, end = (time.fromisoformat(job.calling_hours[k]) for k in ("start", "end"))
    if start <= local < end:
        return True
    return callback and CALLBACK_HOURS[0] <= local < CALLBACK_HOURS[1]


def calls_by_candidate(session: Session, candidate_ids: list[uuid.UUID]) -> dict:
    calls = defaultdict(list)
    stmt = select(Call).where(Call.candidate_id.in_(candidate_ids)).order_by(Call.attempt)
    for call in session.scalars(stmt):
        calls[call.candidate_id].append(call)
    return calls


def tick(session: Session, now: datetime) -> int:
    """Queue the calls that are due for every active campaign; returns how many.

    Each campaign is handled in its own transaction under its row lock, taken with SKIP
    LOCKED: with several workers, each campaign is ticked by one of them at a time.
    """
    queued = 0
    for job_id in session.scalars(select(Campaign.job_id).where(Campaign.status == "active")):
        campaign = session.scalars(
            select(Campaign)
            .where(Campaign.job_id == job_id, Campaign.status == "active")
            .with_for_update(skip_locked=True)
        ).first()
        if campaign is not None:
            queued += run_campaign(session, campaign, now)
        session.commit()
    return queued


def run_campaign(session: Session, campaign: Campaign, now: datetime) -> int:
    job = session.get(Job, campaign.job_id)
    criteria = session.scalars(
        select(CriteriaVersion).where(
            CriteriaVersion.job_id == job.id, CriteriaVersion.status == "approved"
        )
    ).first()
    if job.status != "active" or criteria is None:
        return 0  # e.g. criteria being regenerated: wait, don't end the campaign
    in_flight = session.scalar(
        select(func.count())
        .select_from(Call)
        .join(Candidate)
        .where(Candidate.job_id == job.id, Call.status.in_(ACTIVE_CALL_STATUSES))
    )
    waiting = list(
        session.scalars(
            select(Candidate)
            .where(Candidate.job_id == job.id, Candidate.status.in_(CAMPAIGN_STATUSES))
            .order_by(Candidate.created_at, Candidate.id)
        )
    )
    calls = calls_by_candidate(session, [c.id for c in waiting])
    due = [(when, c) for c in waiting if (when := next_call_at(c.status, calls[c.id])) is not None]
    if not due and not in_flight:
        campaign.status = "done"
        campaign.finished_at = now
        emit(session, "job", job.id, "campaign_done")
        return 0
    queued, slots = 0, campaign.max_concurrent - in_flight
    # Callbacks first (the candidate is expecting that call), then whoever waited longest.
    for when, candidate in sorted(due, key=lambda d: (d[1].status != "callback", d[0])):
        if queued >= slots:
            break
        if when <= now and within_hours(now, job, callback=candidate.status == "callback"):
            queued += call_if_due(session, job, criteria, candidate.id, now)
    return queued


def call_if_due(
    session: Session, job: Job, criteria: CriteriaVersion, candidate_id: uuid.UUID, now: datetime
) -> int:
    """Queue a call to one candidate if, under their row lock, it is still due and allowed
    now. Returns 1 if a call was queued."""
    candidate = session.get(Candidate, candidate_id, with_for_update=True, populate_existing=True)
    if candidate is None:  # deleted meanwhile
        return 0
    calls = calls_by_candidate(session, [candidate.id])[candidate.id]
    when = next_call_at(candidate.status, calls)
    callback = candidate.status == "callback"
    if when is None or when > now or not within_hours(now, job, callback=callback):
        return 0
    if any(c.status in ACTIVE_CALL_STATUSES for c in calls):
        return 0  # a manual call got there first
    reason = "callback" if callback else "retry" if calls else "first_call"
    queue_call(session, candidate, criteria.id, by="campaign", reason=reason)
    return 1
