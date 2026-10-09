from sqlalchemy import select

from tests.fakes import FakeModels
from tests.seed import seed_call, seed_candidate, seed_decision, seed_job
from voiceagent.criteria.schema import Criteria, JobFacts, Requirement, Rule
from voiceagent.db.models import Job, Knowledge
from voiceagent.knowledge.store import (
    add_knowledge,
    delete_overrides,
    index_criteria,
    index_override,
    search_knowledge,
    search_overrides,
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


def test_search_embeds_the_query_as_a_query(session):
    models = FakeModels()
    add_knowledge(session, models, "criteria", [(None, "python backend api", {})])
    session.commit()

    def as_documents(texts):
        raise AssertionError("the query was embedded as a document")

    models.embed = as_documents
    assert [h.text for h in search_knowledge(session, models, "criteria", "python")] == [
        "python backend api"
    ]


def test_similar_criteria_ignores_unrelated_roles(session):
    # Retrieval used to return the nearest rows however far: a sales job was offered
    # "Python" and "PostgreSQL" checks from a backend role.
    models = FakeModels()
    job = make_job(session, "Backend Engineer")
    add_knowledge(
        session,
        models,
        "criteria",
        [
            (job.id, "python backend api experience", _data("Python experience")),
            (job.id, "forklift warehouse safety certificate", _data("Forklift licence")),
        ],
    )
    session.commit()
    lines = similar_criteria(session, models, "python backend api experience")
    assert [line.split(":")[1].split("(")[0].strip() for line in lines] == ["Python experience"]


def _data(label: str) -> dict:
    return {"job_title": "Backend Engineer", "label": label, "hard": False, "question": "?"}


SALARY = Requirement(
    id="expected_salary",
    label="Expected salary",
    kind="number",
    unit="LPA",
    question="What salary are you expecting?",
    weight=1.0,
    sensitive=True,
    source="Budget: 25-35 LPA",
)
PYTHON_HARD = Requirement(
    id="python_years",
    label="Python experience",
    kind="number",
    unit="years",
    question="How many years of Python?",
    hard=True,
    rule=Rule(op=">=", value=3),
    source="3+ years of Python",
)


def overridden_decision(session):
    job, cv = seed_job(session)
    candidate = seed_candidate(session, job)
    call = seed_call(session, candidate, cv, status="completed")
    decision = seed_decision(
        session,
        call,
        reasons=[
            {
                "requirement_id": "python_years",
                "verdict": "fail",
                "score": None,
                "quote": "two years of Python",
                "confidence": 0.9,
                "note": "2 >= 3 fails",
            },
            {
                "requirement_id": "expected_salary",
                "verdict": "scored",
                "score": 0.2,
                "quote": "around 36 lakhs",
                "confidence": 0.9,
                "note": "above the budget",
            },
        ],
    )
    return job, decision


def test_an_override_becomes_calibration_for_soft_requirements_only(session):
    models = FakeModels()
    job, decision = overridden_decision(session)
    index_override(
        session,
        models,
        job=job,
        decision=decision,
        requirements=[PYTHON_HARD, SALARY],
        final_result="suitable",
        note="36 is fine for a senior hire",
    )
    session.commit()
    rows = session.scalars(select(Knowledge).where(Knowledge.kind == "override")).all()
    assert [r.data["requirement_id"] for r in rows] == ["expected_salary"]
    assert "Priya" not in rows[0].text + str(rows[0].data)  # no names in calibration
    assert [text for _, text in search_overrides(session, models, "Backend Engineer", SALARY)] == [
        'Backend Engineer — Expected salary: the candidate said "around 36 lakhs". The agent '
        "scored it 0.20 and judged the candidate not_suitable; the recruiter decided "
        "suitable: 36 is fine for a senior hire"
    ]
    assert search_overrides(session, models, "Warehouse Supervisor", PYTHON_HARD) == []


def test_overriding_again_replaces_the_examples(session):
    models = FakeModels()
    job, decision = overridden_decision(session)
    for result in ("suitable", "not_suitable"):
        index_override(
            session,
            models,
            job=job,
            decision=decision,
            requirements=[SALARY],
            final_result=result,
            note=result,
        )
    session.commit()
    rows = session.scalars(select(Knowledge).where(Knowledge.kind == "override")).all()
    assert [r.data["recruiter_result"] for r in rows] == ["not_suitable"]


def test_a_candidates_overrides_can_be_deleted(session):
    models = FakeModels()
    job, decision = overridden_decision(session)
    index_override(
        session,
        models,
        job=job,
        decision=decision,
        requirements=[SALARY],
        final_result="suitable",
        note="ok",
    )
    add_knowledge(session, models, "override", [(job.id, "someone else", {"candidate_id": "x"})])
    delete_overrides(session, candidate_id=decision.candidate_id)
    session.commit()
    assert [k.text for k in session.scalars(select(Knowledge))] == ["someone else"]
