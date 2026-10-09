from sqlalchemy import select

from tests.fakes import FakeModels
from voiceagent.criteria.schema import Criteria, JobFacts, Requirement, Rule
from voiceagent.db.models import Job, Knowledge
from voiceagent.knowledge.store import (
    add_knowledge,
    index_criteria,
    search_knowledge,
    similar_criteria,
)


def make_job(session, title):
    job = Job(title=title, company_name="Acme", description="x" * 60)
    session.add(job)
    session.flush()
    return job


def criteria_with(req_id):
    return Criteria(
        job_facts=JobFacts(summary="APIs"),
        requirements=[
            Requirement(id=req_id, label=req_id, kind="boolean", question="?", weight=1, source="x")
        ],
    )


def test_search_ranks_by_similarity_and_filters_kind(session):
    models = FakeModels()
    add_knowledge(
        session,
        models,
        "criteria",
        [
            (None, "python backend api experience", {}),
            (None, "forklift warehouse safety certificate", {}),
        ],
    )
    add_knowledge(session, models, "override", [(None, "python backend api", {})])
    session.commit()
    hits = search_knowledge(session, models, "criteria", "python api backend", k=2)
    assert [h.text for h in hits] == [
        "python backend api experience",
        "forklift warehouse safety certificate",
    ]


def test_search_can_exclude_a_job(session):
    models = FakeModels()
    mine, other = make_job(session, "Mine"), make_job(session, "Other")
    add_knowledge(
        session,
        models,
        "criteria",
        [
            (mine.id, "python years", {}),
            (other.id, "python years", {}),
        ],
    )
    session.commit()
    hits = search_knowledge(session, models, "criteria", "python", exclude_job_id=mine.id)
    assert [h.job_id for h in hits] == [other.id]


def test_index_and_describe_similar_criteria(session):
    models = FakeModels()
    job = make_job(session, "Backend Engineer")
    criteria = Criteria(
        job_facts=JobFacts(summary="APIs"),
        requirements=[
            Requirement(
                id="notice_days",
                label="Notice period",
                kind="number",
                unit="days",
                question="What's your notice period?",
                hard=True,
                rule=Rule(op="<=", value=30),
                source="30 days notice",
            )
        ],
    )
    index_criteria(session, models, job, criteria)
    session.commit()
    lines = similar_criteria(session, models, "backend engineer notice period")
    assert lines == [
        "Backend Engineer: Notice period (must-have: <= 30 days) — asked as: "
        "What's your notice period?"
    ]


def test_reindexing_a_job_replaces_its_criteria(session):
    models = FakeModels()
    job, other = make_job(session, "Backend Engineer"), make_job(session, "Data Engineer")
    add_knowledge(session, models, "criteria", [(other.id, "other job", {})])
    add_knowledge(session, models, "override", [(job.id, "an override", {})])
    index_criteria(session, models, job, criteria_with("notice_days"))
    index_criteria(session, models, job, criteria_with("python_years"))
    session.commit()
    rows = {(k.kind, k.job_id, k.data.get("id")) for k in session.scalars(select(Knowledge))}
    assert rows == {
        ("criteria", other.id, None),
        ("override", job.id, None),
        ("criteria", job.id, "python_years"),
    }
