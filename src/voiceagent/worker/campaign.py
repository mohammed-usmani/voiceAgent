"""Campaigns: call a job's candidates in the background, within the job's calling hours,
retrying unanswered calls and honouring callbacks. `tick` runs from the worker loop."""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from voiceagent.db.models import MAX_UNANSWERED, UNANSWERED_CALL_STATUSES, Call, Job

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
