from datetime import UTC, datetime, timedelta

import pytest

from tests.seed import sample_job
from voiceagent.db.models import Call
from voiceagent.worker.campaign import (
    ANY_TIME,
    MAX_CALLBACKS,
    RETRY_GAP,
    next_call_at,
    within_hours,
)

T = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)  # 11:30 in Kolkata, a Monday


def missed(ended_at, status="no_answer"):
    return Call(status=status, ended_at=ended_at)


def callback(at):
    return Call(status="completed", end_reason="callback_requested", callback_at=at, ended_at=T)


def test_a_new_candidate_can_be_called_any_time():
    assert next_call_at("pending", []) == ANY_TIME


@pytest.mark.parametrize("status", ["no_answer", "voicemail", "failed"])
def test_an_unanswered_call_is_retried_after_the_gap(status):
    assert next_call_at("no_answer", [missed(T, status)]) == T + RETRY_GAP


def test_a_failed_first_call_waits_too():
    # Review focus 3: a broken number or trunk is not redialled every minute; the
    # candidate went back to pending, but the failed call still counts.
    assert next_call_at("pending", [missed(T, "failed")]) == T + RETRY_GAP


def test_the_gap_counts_from_the_latest_unanswered_call():
    calls = [missed(T - timedelta(hours=5)), missed(T)]
    assert next_call_at("no_answer", calls) == T + RETRY_GAP


def test_three_unanswered_calls_end_the_retries():
    assert next_call_at("no_answer", [missed(T), missed(T), missed(T)]) is None


def test_a_callback_asked_for_after_three_missed_calls_is_still_made():
    # Unreachable, then a recruiter's manual call got through and they asked for a callback.
    at = T + timedelta(hours=7)
    calls = [missed(T - timedelta(days=1))] * 3 + [callback(at)]
    assert next_call_at("callback", calls) == at


@pytest.mark.parametrize(
    "status",
    ["calling", "screened", "decided", "notified", "declined_consent", "withdrawn", "unreachable"],
)
def test_other_statuses_are_never_called_by_a_campaign(status):
    assert next_call_at(status, []) is None


def test_a_callback_is_due_at_the_time_they_gave():
    at = T + timedelta(hours=7)
    assert next_call_at("callback", [callback(at)]) == at


def test_a_callback_after_a_missed_call_waits_for_both():
    # They asked for 12:00, the 12:00 call wasn't answered at 12:01: retry at 14:01.
    calls = [callback(T), missed(T + timedelta(minutes=1))]
    assert next_call_at("callback", calls) == T + timedelta(minutes=1) + RETRY_GAP


def test_endless_callback_requests_stop_being_honoured():
    calls = [callback(T + timedelta(days=i)) for i in range(MAX_CALLBACKS + 1)]
    assert next_call_at("callback", calls[:MAX_CALLBACKS]) is not None
    assert next_call_at("callback", calls) is None


def kolkata(hour, minute=0):
    return datetime(2026, 10, 12, hour, minute) - timedelta(hours=5, minutes=30)


@pytest.mark.parametrize(
    ("local", "inside"),
    [((9, 59), False), ((10, 0), True), ((18, 59), True), ((19, 0), False), ((23, 0), False)],
)
def test_calling_hours_are_the_jobs_local_hours(local, inside):
    job = sample_job(calling_hours={"start": "10:00", "end": "19:00"})
    assert within_hours(kolkata(*local).replace(tzinfo=UTC), job) is inside


@pytest.mark.parametrize(
    ("local", "inside"), [((8, 59), False), ((9, 0), True), ((20, 59), True), ((21, 0), False)]
)
def test_a_callback_may_fall_outside_calling_hours_but_never_at_night(local, inside):
    job = sample_job(calling_hours={"start": "10:00", "end": "19:00"})
    assert within_hours(kolkata(*local).replace(tzinfo=UTC), job, callback=True) is inside


def test_calling_hours_follow_daylight_saving():
    # Review focus 1: New York moves from EST (UTC-5) to EDT (UTC-4) on 8 March 2026.
    # 14:30 UTC is 09:30 the day before (too early) and 10:30 on the day (allowed).
    job = sample_job(timezone="America/New_York", calling_hours={"start": "10:00", "end": "19:00"})
    assert not within_hours(datetime(2026, 3, 7, 14, 30, tzinfo=UTC), job)
    assert within_hours(datetime(2026, 3, 8, 14, 30, tzinfo=UTC), job)
