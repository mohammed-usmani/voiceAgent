"""After a call: `postcall` assesses it into a decision, and `draft_message` writes the
candidate's email for the decision's result. A `suitable` decision on a job with
auto_send_invites sends its invite straight away; everything else waits for a recruiter.

Both are idempotent: the call id keys the decision (one per call) and the decision id keys
the message (one per decision), so a retried or reclaimed task never adds a second one.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from voiceagent.agents.assessment import run_assessment
from voiceagent.agents.messages import run_message_agent
from voiceagent.call.store import lock_call
from voiceagent.criteria.schema import Requirement
from voiceagent.db.models import (
    Answer,
    Call,
    Candidate,
    CriteriaVersion,
    Decision,
    Job,
    Message,
    Task,
)
from voiceagent.events import emit
from voiceagent.knowledge.store import search_overrides
from voiceagent.llm import Models
from voiceagent.worker.queue import enqueue, heartbeat
from voiceagent.worker.send import queue_send

# Calls that end before any screening: nothing to assess. The candidate's status already
# says why (declined_consent, callback).
NOT_ASSESSED = ("declined_consent", "callback_requested")
MESSAGE_KIND = {"suitable": "invite", "not_suitable": "rejection"}
REVIEWABLE = ("pending_review", "overridden")


def _decided(session: Session, call: Call) -> bool:
    return session.scalar(select(Decision.id).where(Decision.call_id == call.id)) is not None


def _record(session: Session, call: Call, decision: Decision) -> None:
    """Add the decision; the candidate is decided; queue its message. Not committed."""
    session.add(decision)
    session.flush()
    candidate = session.get(Candidate, call.candidate_id)
    if candidate.status == "screened":  # a withdrawn candidate stays withdrawn
        candidate.status = "decided"
    emit(
        session,
        "decision",
        decision.id,
        "created",
        {"call_id": str(call.id), "candidate_id": str(candidate.id), "result": decision.result},
    )
    if decision.result in MESSAGE_KIND:
        enqueue(session, "draft_message", {"decision_id": str(decision.id)})


def postcall(session: Session, models: Models, task: Task) -> None:
    call_id = uuid.UUID(task.payload["call_id"])
    call = session.get(Call, call_id)
    if call is None or call.status != "completed" or _decided(session, call):
        return
    if call.end_reason in NOT_ASSESSED:
        emit(session, "call", call.id, "not_assessed", {"end_reason": call.end_reason})
        return
    candidate = session.get(Candidate, call.candidate_id)
    job = session.get(Job, candidate.job_id)
    cv = session.get(CriteriaVersion, call.criteria_version_id)
    requirements = [Requirement.model_validate(r) for r in cv.requirements]
    pass_threshold, title = cv.pass_threshold, job.title
    transcript, resume = list(call.transcript), candidate.resume_text
    end_reason = call.end_reason or "unknown"
    session.rollback()  # hold no transaction (or locks) through the slow agent run

    def overrides(r: Requirement) -> list[str]:
        try:
            return search_overrides(session, models, title, r)
        finally:
            session.rollback()

    a = run_assessment(
        models,
        requirements=requirements,
        pass_threshold=pass_threshold,
        transcript=transcript,
        resume=resume,
        end_reason=end_reason,
        overrides_lookup=overrides,
        heartbeat=lambda: heartbeat(session, task),
    )

    call = lock_call(session, call_id)  # the candidate, then the call: no deadlock with delete
    if call is None or _decided(session, call):  # deleted, or assessed by another run
        return
    session.add_all(
        Answer(
            call_id=call.id,
            requirement_id=e.requirement_id,
            value=e.value,
            declined=e.declined,
            quote=e.quote,
            confidence=e.confidence,
            inconsistency=e.inconsistency,
        )
        for e in a.answers
    )
    decision = Decision(
        candidate_id=call.candidate_id,
        call_id=call.id,
        criteria_version_id=call.criteria_version_id,
        result=a.result,
        score=a.score,
        hard_failures=a.hard_failures,
        reasons=a.reasons,
        summary=a.summary,
        model=f"{models.name}; reviewer {models.reviewer.name}",
        agent_review=a.agent_review,
        trace=a.trace,
    )
    _record(session, call, decision)


def postcall_failed(session: Session, payload: dict, error: str) -> None:
    """The assessment kept failing: a recruiter decides from the transcript instead."""
    call = lock_call(session, uuid.UUID(payload["call_id"]))
    if call is None or call.end_reason in NOT_ASSESSED or _decided(session, call):
        return
    decision = Decision(
        candidate_id=call.candidate_id,
        call_id=call.id,
        criteria_version_id=call.criteria_version_id,
        result="needs_review",
        score=None,
        hard_failures=[],
        reasons=[],
        summary=f"The assessment failed ({error[:300]}); read the transcript and decide.",
        model="",
        trace=[{"step": "error", "error": error[:2000]}],
    )
    _record(session, call, decision)


def draft_message(session: Session, models: Models, task: Task) -> None:
    decision_id = uuid.UUID(task.payload["decision_id"])
    decision = session.get(Decision, decision_id)
    if decision is None or decision.status not in REVIEWABLE:
        return
    kind = MESSAGE_KIND.get(decision.final_result or decision.result)
    existing = session.scalar(select(Message).where(Message.decision_id == decision.id))
    if kind is None or (existing is not None and existing.kind == kind):
        return
    candidate = session.get(Candidate, decision.candidate_id)
    job = session.get(Job, candidate.job_id)
    cv = session.get(CriteriaVersion, decision.criteria_version_id)
    inputs = dict(
        candidate_name=candidate.name,
        job_title=job.title,
        company_name=job.company_name,
        job_facts=cv.job_facts,
        requirement_labels=[r["label"] for r in cv.requirements],
    )
    session.rollback()

    d = run_message_agent(models, kind=kind, heartbeat=lambda: heartbeat(session, task), **inputs)

    decision = session.get(Decision, decision_id, with_for_update=True, populate_existing=True)
    if (
        decision is None
        or decision.status not in REVIEWABLE
        or MESSAGE_KIND.get(decision.final_result or decision.result) != kind
    ):
        return  # deleted, approved, or overridden meanwhile (that override queued its own draft)
    existing = session.scalar(select(Message).where(Message.decision_id == decision.id))
    if existing is not None:
        if existing.kind == kind:
            return
        session.delete(existing)
        session.flush()
    candidate = session.get(Candidate, decision.candidate_id)
    message = Message(
        decision_id=decision.id,
        kind=kind,
        to=candidate.email,
        subject=d.subject[:300],
        body=d.body,
        status="draft",
        issues=d.issues,
        model=models.name,
    )
    session.add(message)
    session.flush()
    emit(
        session,
        "message",
        message.id,
        "drafted",
        {"decision_id": str(decision.id), "kind": kind, "issues": d.issues},
    )
    job = session.get(Job, candidate.job_id, populate_existing=True)
    auto_send = (
        decision.status == "pending_review"
        and decision.final_result is None
        and decision.result == "suitable"
        and job.auto_send_invites
        and not d.issues
    )
    if auto_send:
        queue_send(session, decision, message)


def draft_message_failed(session: Session, payload: dict, error: str) -> None:
    emit(session, "decision", payload["decision_id"], "message_failed", {"error": error[:500]})
