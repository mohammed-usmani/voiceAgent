"""Recruiter review: decisions, their approval or override, and the outcome messages."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from voiceagent.api.deps import get_models, get_session
from voiceagent.api.schemas import (
    AnswerOut,
    ApproveRequest,
    DecisionOut,
    DecisionSummary,
    MessageEdit,
    MessageOut,
    Override,
)
from voiceagent.criteria.schema import Requirement
from voiceagent.db.models import Answer, Candidate, CriteriaVersion, Decision, Job, Message
from voiceagent.events import emit
from voiceagent.knowledge.store import index_override
from voiceagent.llm import Models
from voiceagent.worker.assess import MESSAGE_KIND, REVIEWABLE
from voiceagent.worker.queue import enqueue
from voiceagent.worker.send import queue_send

router = APIRouter()
DB = Annotated[Session, Depends(get_session)]


def get_decision(session: Session, decision_id: uuid.UUID, *, lock: bool = False) -> Decision:
    """`lock=True` serializes everything that changes a decision or its message (approve,
    override, and the worker's draft_message)."""
    decision = session.get(
        Decision, decision_id, with_for_update=lock or None, populate_existing=lock
    )
    if decision is None:
        raise HTTPException(status_code=404, detail="decision not found")
    return decision


def _message(session: Session, decision: Decision, *, lock: bool = False) -> Message | None:
    stmt = select(Message).where(Message.decision_id == decision.id)
    if lock:
        stmt = stmt.with_for_update()
    return session.scalars(stmt.execution_options(populate_existing=lock)).first()


def _summary(decision: Decision, candidate: Candidate) -> dict:
    return dict(
        id=decision.id,
        job_id=candidate.job_id,
        candidate_id=candidate.id,
        candidate_name=candidate.name,
        call_id=decision.call_id,
        result=decision.result,
        final_result=decision.final_result,
        score=decision.score,
        status=decision.status,
        summary=decision.summary,
        created_at=decision.created_at,
    )


def decision_out(session: Session, decision: Decision) -> DecisionOut:
    candidate = session.get(Candidate, decision.candidate_id)
    cv = session.get(CriteriaVersion, decision.criteria_version_id)
    order = {r["id"]: i for i, r in enumerate(cv.requirements)}
    answers = sorted(
        session.scalars(select(Answer).where(Answer.call_id == decision.call_id)),
        key=lambda a: order.get(a.requirement_id, len(order)),
    )
    message = _message(session, decision)
    return DecisionOut(
        **_summary(decision, candidate),
        criteria_version_id=decision.criteria_version_id,
        hard_failures=decision.hard_failures,
        reasons=decision.reasons,
        model=decision.model,
        reviewer_note=decision.reviewer_note,
        agent_review=decision.agent_review,
        trace=decision.trace,
        answers=[AnswerOut.model_validate(a) for a in answers],
        message=MessageOut.model_validate(message) if message else None,
    )


@router.get("/decisions", response_model=list[DecisionSummary])
def list_decisions(
    session: DB,
    job_id: uuid.UUID | None = None,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DecisionSummary]:
    """Oldest first: the review queue is `?status=pending_review`."""
    stmt = (
        select(Decision, Candidate)
        .join(Candidate, Decision.candidate_id == Candidate.id)
        .order_by(Decision.created_at, Decision.id)
        .limit(limit)
        .offset(offset)
    )
    if job_id is not None:
        stmt = stmt.where(Candidate.job_id == job_id)
    if status:
        stmt = stmt.where(Decision.status == status)
    return [DecisionSummary(**_summary(d, c)) for d, c in session.execute(stmt)]


@router.get("/decisions/{decision_id}", response_model=DecisionOut)
def read_decision(decision_id: uuid.UUID, session: DB) -> DecisionOut:
    return decision_out(session, get_decision(session, decision_id))


@router.post("/decisions/{decision_id}:approve", response_model=DecisionOut)
def approve_decision(
    decision_id: uuid.UUID, session: DB, body: ApproveRequest | None = None
) -> DecisionOut:
    """Send the drafted message for the decision's result (the recruiter's, if overridden).
    Also retries a message whose sending failed. A candidate who was already notified, or
    has a newer decision (a re-call), is only emailed again with `{"force": true}`: two
    different outcomes would contradict each other."""
    candidate = session.get(Candidate, get_decision(session, decision_id).candidate_id)
    session.refresh(candidate, with_for_update=True)  # the lock order: candidate, decision
    decision = get_decision(session, decision_id, lock=True)
    result = decision.final_result or decision.result
    if decision.status == "sent":
        raise HTTPException(status_code=409, detail="the message was already sent")
    if not (body and body.force):
        newer = session.scalar(
            select(Decision.id).where(
                Decision.candidate_id == candidate.id,
                Decision.id != decision.id,
                Decision.created_at > decision.created_at,
            )
        )
        if candidate.status == "notified" or newer is not None:
            why = "was already notified" if newer is None else "has a newer decision"
            raise HTTPException(
                status_code=409,
                detail=f'the candidate {why}; send {{"force": true}} to email them anyway',
            )
    if result not in MESSAGE_KIND:
        raise HTTPException(
            status_code=409, detail=f"the result is {result}: choose one with :override"
        )
    message = _message(session, decision, lock=True)
    if message is None or message.kind != MESSAGE_KIND[result]:
        raise HTTPException(status_code=409, detail="the message is still being drafted")
    if message.status not in ("draft", "failed"):
        raise HTTPException(status_code=409, detail=f"the message is already {message.status}")
    queue_send(session, decision, message)
    emit(session, "decision", decision.id, "approved", {"result": result})
    session.commit()
    return decision_out(session, decision)


@router.post("/decisions/{decision_id}:override", response_model=DecisionOut)
def override_decision(
    decision_id: uuid.UUID,
    body: Override,
    session: DB,
    models: Annotated[Models, Depends(get_models)],
) -> DecisionOut:
    """Replace the agent's result with the recruiter's, while it is in review or its send
    failed. The message is re-drafted for the new result (send it with :approve), and the
    override becomes a calibration example (only for `requirement_ids`, if given)."""
    decision = get_decision(session, decision_id, lock=True)
    message = _message(session, decision, lock=True)
    failed_send = decision.status == "approved" and message and message.status == "failed"
    if decision.status not in REVIEWABLE and not failed_send:
        raise HTTPException(status_code=409, detail=f"the decision is already {decision.status}")
    candidate = session.get(Candidate, decision.candidate_id)
    job = session.get(Job, candidate.job_id)
    cv = session.get(CriteriaVersion, decision.criteria_version_id)
    requirements = [Requirement.model_validate(r) for r in cv.requirements]
    weighted = {r.id for r in requirements if not r.hard}
    if body.requirement_ids is not None and (bad := set(body.requirement_ids) - weighted):
        raise HTTPException(
            status_code=422,
            detail=f"not weighted requirements of these criteria: {', '.join(sorted(bad))}",
        )
    decision.final_result = body.result
    decision.reviewer_note = body.note
    decision.status = "overridden"
    if message is not None and message.kind != MESSAGE_KIND[body.result]:
        session.delete(message)  # never send the old result's message by mistake
        message = None
    if message is None:
        enqueue(session, "draft_message", {"decision_id": str(decision.id)})
    try:
        index_override(
            session,
            models,
            job=job,
            decision=decision,
            requirements=requirements,
            final_result=body.result,
            note=body.note,
            requirement_ids=body.requirement_ids,
        )
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=502, detail=f"couldn't index the override: {e}") from e
    emit(
        session,
        "decision",
        decision.id,
        "overridden",
        {"from": decision.result, "to": body.result},
    )
    session.commit()
    return decision_out(session, decision)


def get_message(session: Session, message_id: uuid.UUID, *, lock: bool = False) -> Message:
    message = session.get(Message, message_id, with_for_update=lock or None, populate_existing=lock)
    if message is None:
        raise HTTPException(status_code=404, detail="message not found")
    return message


@router.get("/messages/{message_id}", response_model=MessageOut)
def read_message(message_id: uuid.UUID, session: DB) -> Message:
    return get_message(session, message_id)


@router.patch("/messages/{message_id}", response_model=MessageOut)
def edit_message(message_id: uuid.UUID, body: MessageEdit, session: DB) -> Message:
    """Edit the subject or body before sending (while draft, or failed)."""
    message = get_message(session, message_id)
    get_decision(session, message.decision_id, lock=True)  # the order approve locks in
    message = get_message(session, message_id, lock=True)
    if message.status not in ("draft", "failed"):
        raise HTTPException(status_code=409, detail=f"the message is already {message.status}")
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in changes.items():
        setattr(message, field, value)
    emit(session, "message", message.id, "edited", {"fields": sorted(changes)})
    session.commit()
    return message
