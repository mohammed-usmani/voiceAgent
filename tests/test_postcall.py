import pytest
from sqlalchemy import select, update

from tests.fakes import FakeModels
from tests.seed import TRANSCRIPT, seed_call, seed_decision, seed_finished_call, seed_message
from voiceagent.agents.assessment import Evidence, EvidenceSet, Review, SoftScore, SoftScores
from voiceagent.agents.messages import Flag, InviteCheck, Letter, RejectionCheck
from voiceagent.db.base import make_sessionmaker
from voiceagent.db.models import (
    Answer,
    Call,
    Candidate,
    CriteriaVersion,
    Decision,
    Event,
    Message,
    Task,
)
from voiceagent.llm import StructuredOutputError
from voiceagent.worker.main import run_once
from voiceagent.worker.queue import enqueue

PY = "I have been writing Python for about five years now."
PUNE = "Yes, Pune three days a week works for me."
SALARY = "I am expecting around 30 lakhs per annum."
TWO = "Honestly only about two years, mostly scripting."
TWO_CALL = [*TRANSCRIPT[:3], {"role": "candidate", "text": TWO}, *TRANSCRIPT[4:]]
INVITE = Letter(
    subject="Next steps at Acme",
    body="Hi Priya, we'd like to move forward. The team will be in touch to schedule.",
)
REJECTION = Letter(subject="Your application to Acme", body="Hi Priya, thank you for your time.")


def replies(python=5.0, salary_score=0.8, letter=INVITE, check=None):
    py = PY if python == 5.0 else TWO
    return {
        EvidenceSet: [
            EvidenceSet(
                answers=[
                    Evidence(requirement_id="python_years", value=python, quote=py, confidence=0.9),
                    Evidence(requirement_id="onsite_pune", value=True, quote=PUNE, confidence=0.9),
                    Evidence(
                        requirement_id="expected_salary", value=30, quote=SALARY, confidence=0.9
                    ),
                ]
            )
        ],
        SoftScores: [
            SoftScores(
                scores=[
                    SoftScore(requirement_id="expected_salary", score=salary_score, quote=SALARY)
                ]
            )
        ],
        Review: [Review(check="ok", problems=[])],
        Letter: [letter],
        InviteCheck: [check or InviteCheck()],
        RejectionCheck: [RejectionCheck()],
    }


def postcall(session, **kw):
    call = seed_finished_call(session, **kw)
    enqueue(session, "postcall", {"call_id": str(call.id)})
    session.commit()
    return call


def drain(engine, models) -> int:
    sm, n = make_sessionmaker(engine), 0
    while run_once(sm, models):
        n += 1
    return n


def events(session) -> list[tuple[str, str]]:
    return [(e.entity, e.type) for e in session.scalars(select(Event).order_by(Event.id))]


def test_a_suitable_candidate_is_invited_automatically(engine, session, mailer):
    call = postcall(session)
    assert drain(engine, FakeModels(replies())) == 3  # postcall, draft_message, send_message
    session.expire_all()
    d = session.scalars(select(Decision)).one()
    assert (d.result, d.score, d.status, d.call_id) == ("suitable", 0.8, "sent", call.id)
    assert d.model == "fake; reviewer fake"
    assert [s["step"] for s in d.trace][:2] == ["calibrate", "evidence"]
    answers = {a.requirement_id: a.value for a in session.scalars(select(Answer))}
    assert answers == {"python_years": 5.0, "onsite_pune": True, "expected_salary": 30.0}
    m = session.scalars(select(Message)).one()
    assert (m.kind, m.to, m.status, m.subject) == (
        "invite",
        "priya@example.com",
        "sent",
        "Next steps at Acme",
    )
    assert [e.subject for e in mailer.sent] == ["Next steps at Acme"]
    assert session.get(Candidate, call.candidate_id).status == "notified"
    assert events(session) == [
        ("decision", "created"),
        ("message", "drafted"),
        ("message", "queued"),
        ("message", "sent"),
    ]


def test_an_invite_the_reviewer_doubts_is_not_sent_unattended(engine, session, mailer):
    postcall(session)
    models = FakeModels(replies())
    models.replies[Review] = [Review(check="checked", problems=["the salary answer was hedged"])]
    drain(engine, models)
    session.expire_all()
    d = session.scalars(select(Decision)).one()
    assert (d.result, d.status, d.agent_review["challenged"]) == (
        "needs_review",
        "pending_review",
        "suitable",
    )
    assert mailer.sent == []


def test_invites_wait_for_the_recruiter_when_auto_send_is_off(engine, session, mailer):
    call = postcall(session, auto_send_invites=False)
    drain(engine, FakeModels(replies()))
    session.expire_all()
    d = session.scalars(select(Decision)).one()
    assert (d.result, d.status) == ("suitable", "pending_review")
    assert session.scalars(select(Message)).one().status == "draft"
    assert session.get(Candidate, call.candidate_id).status == "decided"
    assert mailer.sent == []


def test_an_invite_the_checker_never_passed_is_not_sent(engine, session, mailer):
    postcall(session)
    sentence = "The team will be in touch to schedule."
    flag = Flag(flagged=True, sentence=sentence, why="promises an interview date")
    drain(engine, FakeModels(replies(check=InviteCheck(invents_facts=flag))))
    session.expire_all()
    m = session.scalars(select(Message)).one()
    assert (m.status, m.issues) == (
        "draft",
        [f'invents facts: promises an interview date ("{sentence}")'],
    )
    assert session.scalars(select(Decision)).one().status == "pending_review"
    assert mailer.sent == []


def test_a_rejection_is_drafted_and_waits_for_approval(engine, session, mailer):
    postcall(session, transcript=TWO_CALL)
    drain(engine, FakeModels(replies(python=2.0, letter=REJECTION)))
    session.expire_all()
    d = session.scalars(select(Decision)).one()
    assert (d.result, d.hard_failures, d.status) == (
        "not_suitable",
        ["python_years"],
        "pending_review",
    )
    assert d.agent_review["agree"] is True
    m = session.scalars(select(Message)).one()
    assert (m.kind, m.status) == ("rejection", "draft")
    assert mailer.sent == []


def test_needs_review_gets_no_message_until_a_recruiter_decides(engine, session):
    postcall(session)
    drain(engine, FakeModels(replies(salary_score=0.55)))
    session.expire_all()
    assert session.scalars(select(Decision)).one().result == "needs_review"
    assert session.scalars(select(Message)).all() == []
    assert [t.kind for t in session.scalars(select(Task))] == ["postcall"]


def test_declined_consent_is_not_assessed(engine, session):
    call = postcall(session, end_reason="declined_consent")
    session.execute(
        update(Candidate).where(Candidate.id == call.candidate_id).values(status="declined_consent")
    )
    session.commit()
    models = FakeModels()  # no scripted replies: any model call would fail the task
    drain(engine, models)
    session.expire_all()
    assert session.scalars(select(Decision)).all() == []
    assert session.get(Candidate, call.candidate_id).status == "declined_consent"
    assert events(session) == [("call", "not_assessed")]
    assert models.prompts == []


def test_a_hang_up_mid_call_needs_review(engine, session):
    call = postcall(session, end_reason="candidate_hung_up", transcript=TWO_CALL)
    drain(engine, FakeModels(replies(python=2.0)))
    session.expire_all()
    d = session.scalars(select(Decision)).one()
    assert d.result == "needs_review" and "ended early" in d.summary
    assert session.get(Candidate, call.candidate_id).status == "decided"


@pytest.mark.parametrize("said", [[], TRANSCRIPT[:1]])
def test_a_hang_up_before_answering_the_consent_question_is_not_assessed(engine, session, said):
    call = postcall(session, end_reason="candidate_hung_up", transcript=said)
    models = FakeModels(replies())
    drain(engine, models)
    session.expire_all()
    assert models.prompts == []  # no model reads a call the candidate never agreed to
    d = session.scalars(select(Decision)).one()
    assert (d.result, d.reasons) == ("needs_review", [])
    assert "before answering" in d.summary
    assert session.scalars(select(Answer)).all() == []
    assert session.get(Candidate, call.candidate_id).status == "decided"


def test_rerunning_postcall_never_adds_a_second_decision(engine, session):
    # A retried or reclaimed task, or a duplicate enqueue: the call id keys the decision.
    call = postcall(session, auto_send_invites=False)
    enqueue(session, "postcall", {"call_id": str(call.id)})
    session.commit()
    models = FakeModels(replies())
    drain(engine, models)
    session.expire_all()
    assert len(session.scalars(select(Decision)).all()) == 1
    assert len(session.scalars(select(Answer)).all()) == 3
    assert len(session.scalars(select(Message)).all()) == 1
    assert [s for s, _, _ in models.prompts].count("EvidenceSet") == 1  # second run: no LLM


def test_rerunning_draft_message_never_adds_a_second_message(engine, session):
    postcall(session, auto_send_invites=False)
    drain(engine, FakeModels(replies()))
    decision = session.scalars(select(Decision)).one()
    enqueue(session, "draft_message", {"decision_id": str(decision.id)})
    session.commit()
    models = FakeModels(replies())
    drain(engine, models)
    session.expire_all()
    assert len(session.scalars(select(Message)).all()) == 1
    assert models.prompts == []


def test_an_assessment_that_keeps_failing_goes_to_a_recruiter(engine, session):
    call = postcall(session)
    task = session.scalars(select(Task)).one()
    task.attempts = task.max_attempts - 1
    session.commit()
    drain(engine, FakeModels({EvidenceSet: [StructuredOutputError("both models failed")]}))
    session.expire_all()
    d = session.scalars(select(Decision)).one()
    assert (d.result, d.status) == ("needs_review", "pending_review")
    assert "assessment failed" in d.summary
    assert session.get(Candidate, call.candidate_id).status == "decided"


class Hooked(FakeModels):
    def __init__(self, replies, hook):
        super().__init__(replies)
        self.hook = hook

    def structured(self, schema, system, user):
        self.hook(schema)
        return super().structured(schema, system, user)


def test_an_override_during_drafting_drops_the_stale_draft(engine, session):
    # The invite is being written when the recruiter overrides to not_suitable: the invite
    # must not land; the override's own draft_message writes the rejection.
    postcall(session, auto_send_invites=False)
    drain(engine, FakeModels(replies(salary_score=0.55)))  # needs_review: no message yet
    decision = session.scalars(select(Decision)).one()
    decision.final_result, decision.status = "suitable", "overridden"
    enqueue(session, "draft_message", {"decision_id": str(decision.id)})
    session.commit()

    def override_mid_draft(schema):
        if schema is InviteCheck:
            with make_sessionmaker(engine)() as s:
                s.execute(
                    update(Decision)
                    .where(Decision.id == decision.id)
                    .values(final_result="not_suitable")
                )
                s.commit()

    drain(engine, Hooked(replies(), override_mid_draft))
    session.expire_all()
    assert session.scalars(select(Message)).all() == []


def test_a_candidate_deleted_mid_assessment_leaves_nothing(engine, session):
    call = postcall(session)

    def delete_candidate(schema):
        if schema is SoftScores:
            with make_sessionmaker(engine)() as s:
                s.delete(s.get(Candidate, call.candidate_id))
                s.commit()

    drain(engine, Hooked(replies(), delete_candidate))
    session.expire_all()
    assert session.scalars(select(Call)).all() == []
    assert session.scalars(select(Decision)).all() == []
    assert session.scalars(select(Task)).one().status == "done"


@pytest.mark.parametrize("prior", ["sent", "sending"])
def test_a_candidate_already_emailed_is_not_auto_invited(engine, session, mailer, prior):
    call = postcall(session)
    old_call = seed_call(
        session,
        session.get(Candidate, call.candidate_id),
        session.get(CriteriaVersion, call.criteria_version_id),
        status="completed",
    )
    old = seed_decision(session, old_call, status="sent")
    seed_message(session, old, status=prior)
    drain(engine, FakeModels(replies()))
    session.expire_all()
    d = session.scalars(select(Decision).where(Decision.call_id == call.id)).one()
    assert (d.result, d.status) == ("suitable", "pending_review")
    assert "already emailed" in d.summary
    assert (
        session.scalars(select(Message).where(Message.decision_id == d.id)).one().status == "draft"
    )
    assert mailer.sent == []
