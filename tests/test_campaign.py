from datetime import UTC, datetime, timedelta

from sqlalchemy import event, select

from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.call.store import queue_call
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Campaign, Candidate, Event, Task
from voiceagent.worker.campaign import RETRY_GAP, call_if_due, tick

IST = timedelta(hours=5, minutes=30)


def ist(day, hour, minute=0):
    """A moment given in Kolkata time (the seeded job's timezone), as UTC."""
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC) - IST


NOW = ist(12, 11, 30)  # Monday 11:30, inside the default 10:00-19:00


def campaign_job(session, max_concurrent=2, status="active", **job):
    job, cv = seed_job(session, **job)
    session.add(
        Campaign(job_id=job.id, started_at=NOW, max_concurrent=max_concurrent, status=status)
    )
    session.commit()
    return job, cv


def person(session, job, n, **kw):
    return seed_candidate(session, job, name=f"Candidate {n}", email=f"c{n}@example.com", **kw)


def queued_calls(session):
    return session.scalars(select(Call).where(Call.status == "queued")).all()


def queued_reasons(session):
    events = session.scalars(select(Event).where(Event.type == "queued").order_by(Event.id))
    return [(e.data["by"], e.data["reason"]) for e in events]


def test_a_campaign_calls_up_to_its_concurrency_cap(session):
    job, _ = campaign_job(session, max_concurrent=2)
    for n in range(3):
        person(session, job, n)
    assert tick(session, NOW) == 2
    assert len(queued_calls(session)) == 2
    assert [t.kind for t in session.scalars(select(Task))] == ["dial", "dial"]
    assert queued_reasons(session) == [("campaign", "first_call")] * 2
    assert tick(session, NOW) == 0  # both slots are still taken


def test_calls_already_in_progress_count_toward_the_cap(session):
    job, cv = campaign_job(session, max_concurrent=2)
    busy = person(session, job, 0, status="calling")
    seed_call(session, busy, cv, status="in_progress")  # e.g. a recruiter's manual call
    person(session, job, 1)
    person(session, job, 2)
    assert tick(session, NOW) == 1


def test_ticking_twice_never_calls_a_candidate_twice(session):
    job, _ = campaign_job(session, max_concurrent=5)
    person(session, job, 0)
    person(session, job, 1)
    assert tick(session, NOW) == 2
    assert tick(session, NOW) == 0
    assert len(session.scalars(select(Call)).all()) == 2


def test_nobody_is_called_outside_calling_hours(session):
    job, _ = campaign_job(session)
    person(session, job, 0)
    assert tick(session, ist(12, 9, 59)) == 0
    assert tick(session, ist(12, 19, 0)) == 0
    assert session.get(Campaign, job.id).status == "active"
    assert tick(session, ist(12, 10, 0)) == 1


def test_unanswered_candidates_are_retried_after_two_hours(session):
    job, cv = campaign_job(session)
    candidate = person(session, job, 0, status="no_answer")
    seed_call(session, candidate, cv, status="no_answer", ended_at=NOW - RETRY_GAP)
    assert tick(session, NOW - timedelta(minutes=1)) == 0
    assert tick(session, NOW) == 1
    assert queued_reasons(session) == [("campaign", "retry")]
    assert queued_calls(session)[0].attempt == 2


def test_a_failed_dial_is_not_redialled_every_minute(session):
    # Review focus 3: the trunk rejected the call and the candidate went back to pending.
    job, cv = campaign_job(session)
    candidate = person(session, job, 0)
    seed_call(session, candidate, cv, status="failed", ended_at=NOW - timedelta(minutes=10))
    assert tick(session, NOW) == 0


def test_a_callback_is_made_at_the_requested_time_even_after_hours(session):
    job, cv = campaign_job(session)
    candidate = person(session, job, 0, status="callback")
    at = ist(12, 19, 30)  # after the job's calling hours, before 21:00
    seed_call(
        session,
        candidate,
        cv,
        status="completed",
        end_reason="callback_requested",
        callback_at=at,
        ended_at=NOW,
    )
    assert tick(session, at - timedelta(minutes=1)) == 0
    assert tick(session, at) == 1
    assert queued_reasons(session) == [("campaign", "callback")]


def test_a_late_night_callback_waits_for_the_morning(session):
    # Review focus 4: "call me at 11 pm" is not honoured at 11 pm; it is made at 09:00.
    job, cv = campaign_job(session)
    candidate = person(session, job, 0, status="callback")
    at = ist(12, 23, 0)
    seed_call(
        session,
        candidate,
        cv,
        status="completed",
        end_reason="callback_requested",
        callback_at=at,
        ended_at=NOW,
    )
    assert tick(session, at) == 0
    assert tick(session, ist(13, 8, 59)) == 0
    assert tick(session, ist(13, 9, 0)) == 1


def test_a_due_callback_goes_before_new_candidates(session):
    job, cv = campaign_job(session, max_concurrent=1)
    person(session, job, 0)  # added first, never called
    candidate = person(session, job, 1, status="callback")
    seed_call(
        session,
        candidate,
        cv,
        status="completed",
        end_reason="callback_requested",
        callback_at=NOW,
        ended_at=NOW - timedelta(hours=1),
    )
    assert tick(session, NOW) == 1
    assert queued_calls(session)[0].candidate_id == candidate.id


def test_the_campaign_ends_when_nobody_can_be_called_again(session):
    job, cv = campaign_job(session)
    person(session, job, 0, status="screened")
    person(session, job, 1, status="unreachable")
    waiting = person(session, job, 2, status="no_answer")
    seed_call(session, waiting, cv, status="no_answer", ended_at=NOW)  # retry due later
    assert tick(session, NOW) == 0
    assert session.get(Campaign, job.id).status == "active"

    waiting.status = "screened"
    session.commit()
    assert tick(session, NOW) == 0
    campaign = session.get(Campaign, job.id)
    assert (campaign.status, campaign.finished_at) == ("done", NOW)
    assert session.scalars(select(Event.type).where(Event.entity == "job")).all() == [
        "campaign_done"
    ]


def test_a_campaign_with_a_call_in_flight_is_not_done(session):
    job, cv = campaign_job(session)
    talking = person(session, job, 0, status="calling")
    seed_call(session, talking, cv, status="in_progress")
    assert tick(session, NOW) == 0
    assert session.get(Campaign, job.id).status == "active"


def test_a_manual_call_queued_mid_tick_does_not_end_the_campaign(engine, session):
    # A recruiter calls the last waiting candidate while the tick is reading the job.
    job, cv = campaign_job(session)
    candidate = person(session, job, 0)
    called = []

    def manual_call(conn, cursor, statement, *args):
        if "FROM candidates" in statement and "ORDER BY" in statement and not called:
            called.append(True)
            with make_sessionmaker(engine)() as other:
                c = other.get(Candidate, candidate.id, with_for_update=True)
                queue_call(other, c, cv.id, by="recruiter")
                other.commit()

    event.listen(engine, "before_cursor_execute", manual_call)
    try:
        assert tick(session, NOW) == 0
    finally:
        event.remove(engine, "before_cursor_execute", manual_call)
    assert called
    assert session.get(Campaign, job.id).status == "active"


def test_a_paused_campaign_calls_nobody(session):
    job, _ = campaign_job(session, status="paused")
    person(session, job, 0)
    assert tick(session, NOW) == 0


def test_a_campaign_waits_while_its_job_is_not_active(session):
    job, _ = campaign_job(session, status="active", criteria_status="draft")
    person(session, job, 0)
    assert tick(session, NOW) == 0
    assert session.get(Campaign, job.id).status == "active"  # not done: it resumes later


def test_a_campaign_another_worker_is_ticking_is_skipped(engine, session):
    # Review focus 2: two workers tick at once; one takes the campaign, the other skips it
    # (SKIP LOCKED) instead of waiting or queueing the same candidates again.
    job, _ = campaign_job(session)
    person(session, job, 0)
    other = make_sessionmaker(engine)()
    try:
        other.scalars(select(Campaign).with_for_update()).all()  # the other worker's lock
        assert tick(session, NOW) == 0
    finally:
        other.rollback()
        other.close()
    assert tick(session, NOW) == 1


def test_a_campaign_skips_excluded_candidates(session):
    job, _ = campaign_job(session, max_concurrent=5)
    person(session, job, 0)
    person(session, job, 1, excluded_at=NOW - timedelta(hours=1))
    assert tick(session, NOW) == 1
    [call] = queued_calls(session)
    assert session.get(Candidate, call.candidate_id).name == "Candidate 0"


def test_a_candidate_excluded_after_the_tick_read_them_is_not_called(session):
    job, cv = campaign_job(session)
    candidate = person(session, job, 0, excluded_at=NOW)
    assert call_if_due(session, job, cv, candidate.id, NOW) == 0
    assert queued_calls(session) == []


def test_a_campaign_with_only_excluded_candidates_left_is_done(session):
    # Resume calling doesn't reopen it: the recruiter starts the campaign again.
    job, _ = campaign_job(session)
    person(session, job, 0, excluded_at=NOW)
    assert tick(session, NOW) == 0
    assert session.get(Campaign, job.id).status == "done"
