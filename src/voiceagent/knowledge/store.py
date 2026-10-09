"""Retrievable examples in pgvector: approved criteria of past jobs, which ground the
criteria agent, and recruiter overrides, which calibrate the assessment's soft scoring."""

import uuid

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from voiceagent.criteria.schema import Criteria, Requirement
from voiceagent.db.models import Decision, Job, Knowledge
from voiceagent.llm import Models


def add_knowledge(
    session: Session,
    models: Models,
    kind: str,
    entries: list[tuple[uuid.UUID | None, str, dict]],
) -> None:
    if not entries:
        return
    vectors = models.embed([text for _, text, _ in entries])
    for (job_id, text, data), vector in zip(entries, vectors, strict=True):
        session.add(Knowledge(kind=kind, job_id=job_id, text=text, data=data, embedding=vector))


def search_knowledge(
    session: Session,
    models: Models,
    kind: str,
    query: str,
    k: int = 8,
    exclude_job_id: uuid.UUID | None = None,
    min_similarity: float | None = None,
) -> list[Knowledge]:
    vector = models.embed_query(query)
    stmt = select(Knowledge).where(Knowledge.kind == kind)
    if min_similarity is not None:
        stmt = stmt.where(Knowledge.embedding.cosine_distance(vector) <= 1 - min_similarity)
    if exclude_job_id is not None:
        stmt = stmt.where(or_(Knowledge.job_id.is_(None), Knowledge.job_id != exclude_job_id))
    stmt = stmt.order_by(Knowledge.embedding.cosine_distance(vector)).limit(k)
    return list(session.scalars(stmt))


def index_criteria(session: Session, models: Models, job: Job, criteria: Criteria) -> None:
    """One entry per requirement, so retrieval finds individual checks of similar roles.
    Replaces the job's earlier entries, so only its approved criteria are retrievable."""
    session.execute(
        delete(Knowledge).where(Knowledge.kind == "criteria", Knowledge.job_id == job.id)
    )
    entries = [
        (
            job.id,
            f"{job.title}: {r.label}. {r.question}",
            {"job_title": job.title, **r.model_dump(mode="json")},
        )
        for r in criteria.requirements
    ]
    add_knowledge(session, models, "criteria", entries)


def _fmt(v) -> str:
    # Pydantic stores 30 as 30.0; show whole numbers without the ".0".
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _describe(data: dict) -> str:
    if data.get("hard") and data.get("rule"):
        rule = data["rule"]
        unit = f" {data['unit']}" if data.get("unit") else ""
        kind = f"must-have: {rule['op']} {_fmt(rule['value'])}{unit}"
    else:
        kind = "weighted"
    return f"{data['job_title']}: {data['label']} ({kind}) — asked as: {data['question']}"


# Below this cosine similarity a past role isn't comparable. Measured with bge-base on job
# descriptions vs. indexed requirements: the same or a related role scored 0.63-0.81
# (sales vs. business development 0.65); unrelated roles at most 0.58 (sales vs. backend).
MIN_SIMILARITY = 0.62


def similar_criteria(
    session: Session,
    models: Models,
    query: str,
    exclude_job_id: uuid.UUID | None = None,
    k: int = 8,
) -> list[str]:
    hits = search_knowledge(
        session,
        models,
        "criteria",
        query,
        k=k,
        exclude_job_id=exclude_job_id,
        min_similarity=MIN_SIMILARITY,
    )
    return [_describe(h.data) for h in hits]


def _requirement_text(job_title: str, r: Requirement) -> str:
    return f"{job_title}: {r.label}. {r.question}"


def index_override(
    session: Session,
    models: Models,
    *,
    job: Job,
    decision: Decision,
    requirements: list[Requirement],
    final_result: str,
    note: str,
    requirement_ids: list[str] | None = None,
) -> None:
    """Index a recruiter's override as calibration examples: one per soft requirement the
    agent scored on a quote, or only those in `requirement_ids` (the ones the override is
    about). Hard requirements are never calibrated (rules decide them), and the
    candidate's name is never stored, so examples can't carry it into scoring.
    Replaces the decision's earlier examples."""
    delete_overrides(session, decision_id=decision.id)
    soft = {
        r.id: r
        for r in requirements
        if not r.hard and (requirement_ids is None or r.id in requirement_ids)
    }
    entries = []
    for reason in decision.reasons:
        r = soft.get(reason["requirement_id"])
        if r is None or reason.get("score") is None or not reason.get("quote"):
            continue
        data = {
            "candidate_id": str(decision.candidate_id),
            "decision_id": str(decision.id),
            "job_title": job.title,
            "requirement_id": r.id,
            "label": r.label,
            "quote": reason["quote"],
            "agent_score": reason["score"],
            "agent_result": decision.result,
            "recruiter_result": final_result,
            "note": note,
        }
        entries.append((job.id, _requirement_text(job.title, r), data))
    add_knowledge(session, models, "override", entries)


def delete_overrides(
    session: Session,
    *,
    decision_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
) -> None:
    """Remove override examples of one decision, or of every decision about a candidate
    (their quotes leave with them when they are deleted)."""
    key, value = ("decision_id", decision_id) if decision_id else ("candidate_id", candidate_id)
    session.execute(
        delete(Knowledge).where(
            Knowledge.kind == "override", Knowledge.data[key].astext == str(value)
        )
    )


def _describe_override(d: dict) -> str:
    return (
        f'{d["job_title"]} — {d["label"]}: the candidate said "{d["quote"]}". The agent '
        f"scored it {d['agent_score']:.2f} and judged the candidate {d['agent_result']}; the "
        f"recruiter decided {d['recruiter_result']}: {d['note']}"
    )


def search_overrides(
    session: Session, models: Models, job_title: str, requirement: Requirement, k: int = 3
) -> list[tuple[str, str]]:
    """Recruiter overrides on this requirement of this job or a similar one, as (knowledge
    id, description for the soft scorer) calibration examples. Callers store the id, never
    the text: the text is another candidate's words, which must leave when they do."""
    hits = search_knowledge(
        session,
        models,
        "override",
        _requirement_text(job_title, requirement),
        k=k,
        min_similarity=MIN_SIMILARITY,
    )
    return [(str(h.id), _describe_override(h.data)) for h in hits]
