import re
import uuid
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import event, select, update

from tests.seed import seed_call, seed_candidate, seed_job
from voiceagent.call.store import (
    MAX_CALL_MINUTES,
    CallOutcome,
    close_stale_calls,
    fail_call,
    finish_call,
    load_call,
    queue_call,
    stop_call,
)
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import Call, Candidate, Event, Task

TURNS = [
    {"role": "agent", "text": "Hi Priya", "at": "2026-10-09T09:00:00+00:00"},
    {"role": "candidate", "text": "Hello", "at": "2026-10-09T09:00:03+00:00"},
]


def a_call(session, status="in_progress"):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="calling")
    return seed_call(session, candidate, cv, status=status)


def test_load_call_gives_the_prompt_its_inputs_and_marks_it_in_progress(engine, session):
    call = a_call(session, status="ringing")
    loaded = load_call(make_sessionmaker(engine), call.id)
    assert (loaded.job.title, loaded.candidate.name) == ("Backend Engineer", "Priya Sharma")
    assert [r.id for r in loaded.criteria.requirements] == [
        "python_years",
        "onsite_pune",
        "expected_salary",
    ]
    session.expire_all()
    c = session.get(Call, call.id)
    assert c.status == "in_progress" and c.started_at is not None
    assert [e.type for e in session.scalars(select(Event))] == ["in_progress"]


@pytest.mark.parametrize("status", ["completed", "failed", "no_answer"])
def test_load_call_refuses_a_finished_call(engine, session, status):
    call = a_call(session, status=status)
    with pytest.raises(LookupError, match=status):
        load_call(make_sessionmaker(engine), call.id)


def test_load_call_refuses_an_unknown_call(engine, session):
    with pytest.raises(LookupError):
        load_call(make_sessionmaker(engine), uuid.uuid4())


@pytest.mark.parametrize(
    ("reason", "consent", "candidate_status"),
    [
        ("completed", True, "screened"),
        ("declined_consent", False, "declined_consent"),
        ("not_interested", None, "withdrawn"),
    ],
)
def test_finish_call_saves_the_result_and_queues_postcall(
    engine, session, reason, consent, candidate_status
):
    call = a_call(session)
    finish_call(make_sessionmaker(engine), call.id, CallOutcome(reason), TURNS)
    session.expire_all()
    c = session.get(Call, call.id)
    assert (c.status, c.end_reason, c.consent_given, c.transcript) == (
        "completed",
        reason,
        consent,
        TURNS,
    )
    assert c.ended_at is not None
    assert session.get(Candidate, c.candidate_id).status == candidate_status
    task = session.scalars(select(Task)).one()
    assert (task.kind, task.payload) == ("postcall", {"call_id": str(call.id)})
    ended = [e for e in session.scalars(select(Event)) if e.type == "ended"]
    assert ended[0].data == {"end_reason": reason, "turns": 2}


def test_hang_up_without_end_call_saves_the_transcript(engine, session):
    # Review focus 3: the candidate hung up, end_call never ran. What was said is kept
    # and assessed (milestone 3 sends incomplete calls to review).
    call = a_call(session)
    finish_call(make_sessionmaker(engine), call.id, None, TURNS)
    session.expire_all()
    c = session.get(Call, call.id)
    assert (c.status, c.end_reason, c.transcript) == ("completed", "candidate_hung_up", TURNS)
    assert session.get(Candidate, c.candidate_id).status == "screened"
    assert session.scalars(select(Task)).one().kind == "postcall"


def test_callback_time_is_saved(engine, session):
    call = a_call(session)
    when = datetime(2026, 10, 10, 17, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    finish_call(make_sessionmaker(engine), call.id, CallOutcome("callback_requested", when), [])
    session.expire_all()
    c = session.get(Call, call.id)
    assert c.callback_at == when
    assert session.get(Candidate, c.candidate_id).status == "callback"


def test_finishing_twice_saves_once(engine, session):
    call = a_call(session)
    sm = make_sessionmaker(engine)
    finish_call(sm, call.id, CallOutcome("completed"), TURNS)
    finish_call(sm, call.id, None, [])
    session.expire_all()
    assert session.get(Call, call.id).transcript == TURNS
    assert len(session.scalars(select(Task)).all()) == 1


def test_finishing_a_deleted_call_is_a_no_op(engine, session):
    finish_call(make_sessionmaker(engine), uuid.uuid4(), None, TURNS)
    assert session.scalars(select(Task)).all() == []


def test_fail_call_lets_the_candidate_be_called_again(engine, session):
    call = a_call(session)
    fail_call(make_sessionmaker(engine), call.id, "LookupError('no criteria')")
    session.expire_all()
    c = session.get(Call, call.id)
    assert (c.status, c.end_reason, c.error) == ("failed", "error", "LookupError('no criteria')")
    assert session.get(Candidate, c.candidate_id).status == "pending"


def age(session, call, minutes):
    """Pretend nothing has touched the call for `minutes`."""
    session.execute(
        update(Call)
        .where(Call.id == call.id)
        .values(updated_at=datetime.now(UTC) - timedelta(minutes=minutes))
    )
    session.commit()


@pytest.mark.parametrize("status", ["queued", "ringing", "in_progress"])
def test_a_call_nobody_reported_back_is_closed_as_failed(session, status):
    # I1: the agent died (or never came) and the call would stay active forever.
    stale = a_call(session, status=status)
    age(session, stale, MAX_CALL_MINUTES + 1)
    assert close_stale_calls(session) == 1
    session.commit()
    session.expire_all()
    c = session.get(Call, stale.id)
    assert (c.status, c.end_reason) == ("failed", "error")
    assert "never reported" in c.error
    assert session.get(Candidate, c.candidate_id).status == "pending"


def test_a_live_call_is_left_alone(session):
    live = a_call(session)
    age(session, live, MAX_CALL_MINUTES - 1)
    assert close_stale_calls(session) == 0
    session.expire_all()
    assert session.get(Call, live.id).status == "in_progress"


def for_update_targets(engine, run) -> list[str]:
    """The tables each SELECT ... FOR UPDATE locks while `run` runs, in order."""
    seen = []

    def record(conn, cursor, statement, *args):
        if "FOR UPDATE" in statement:
            seen.append(re.search(r"FROM (\w+)", statement).group(1))

    event.listen(engine, "before_cursor_execute", record)
    try:
        run()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return seen


@pytest.mark.parametrize(
    "write",
    [
        lambda sm, call_id: finish_call(sm, call_id, CallOutcome("completed"), TURNS),
        lambda sm, call_id: fail_call(sm, call_id, "boom"),
        lambda sm, call_id: load_call(sm, call_id),
    ],
    ids=["finish_call", "fail_call", "load_call"],
)
def test_call_writes_lock_the_candidate_before_the_call(engine, session, write):
    # M9: DELETE /candidates/{id} locks the candidate, then its calls (cascade). Taking
    # the call first here would deadlock against it.
    call = a_call(session, status="ringing")
    sm = make_sessionmaker(engine)
    assert for_update_targets(engine, lambda: write(sm, call.id))[:2] == ["candidates", "calls"]


def test_an_agent_error_is_saved_as_a_failed_call_not_a_screening(engine, session):
    # I4: the session died (TTS quota, bad voice): the candidate wasn't screened.
    call = a_call(session)
    finish_call(make_sessionmaker(engine), call.id, None, TURNS, error="TTSError('quota')")
    session.expire_all()
    c = session.get(Call, call.id)
    assert (c.status, c.end_reason, c.error, c.transcript) == (
        "failed",
        "error",
        "TTSError('quota')",
        TURNS,
    )
    assert session.get(Candidate, c.candidate_id).status == "pending"
    assert session.scalars(select(Task)).all() == []  # nothing to assess


def unanswered(session, candidate, cv, n):
    for attempt in range(1, n + 1):
        seed_call(session, candidate, cv, attempt=attempt, status="no_answer")


def test_the_third_unanswered_call_makes_the_candidate_unreachable(session):
    # Review focus 3: someone who never picks up is called three times, then left alone.
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="calling")
    unanswered(session, candidate, cv, 2)
    call = seed_call(session, candidate, cv, attempt=3, status="ringing")
    stop_call(session, call, "voicemail", "answered by a machine")
    session.commit()
    assert session.get(Candidate, candidate.id).status == "unreachable"
    assert [(e.entity, e.type) for e in session.scalars(select(Event).order_by(Event.id))] == [
        ("candidate", "unreachable"),
        ("call", "voicemail"),
    ]


def test_fewer_unanswered_calls_leave_the_candidate_retryable(session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="calling")
    unanswered(session, candidate, cv, 1)
    call = seed_call(session, candidate, cv, attempt=2, status="ringing")
    stop_call(session, call, "voicemail", "answered by a machine")
    session.commit()
    assert (call.status, call.end_reason) == ("voicemail", "voicemail")
    assert session.get(Candidate, candidate.id).status == "no_answer"


def test_a_finished_candidate_stays_finished_however_many_calls_failed(session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="calling")
    unanswered(session, candidate, cv, 2)
    call = seed_call(
        session, candidate, cv, attempt=3, status="queued", prior_candidate_status="screened"
    )
    stop_call(session, call, "failed", "trunk down")
    session.commit()
    assert session.get(Candidate, candidate.id).status == "screened"


def test_queue_call_numbers_the_attempt_and_queues_the_dial(session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job, status="no_answer")
    unanswered(session, candidate, cv, 1)
    call = queue_call(session, candidate, cv.id, by="campaign", reason="retry")
    session.commit()
    assert (call.attempt, call.status, call.prior_candidate_status) == (2, "queued", "no_answer")
    assert candidate.status == "calling"
    task = session.scalars(select(Task)).one()
    assert (task.kind, task.payload) == ("dial", {"call_id": str(call.id)})
    event = session.scalars(select(Event)).one()
    assert event.data == {
        "candidate_id": str(candidate.id),
        "attempt": 2,
        "by": "campaign",
        "reason": "retry",
    }
