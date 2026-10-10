import json

from tests.fakes import FakeModels
from voiceagent.agents.criteria import (
    MAX_ROUNDS,
    Critique,
    Draft,
    Issue,
    run_criteria_agent,
)
from voiceagent.criteria.schema import MAX_ASKED, Criteria, JobFacts, Requirement, Rule

JD = (
    "Backend Engineer (Python), Pune. You will build APIs with FastAPI and PostgreSQL. "
    "Requirements: 3+ years of Python experience. Hybrid: 3 days a week in our Pune office. "
    "Notice period of 30 days or less preferred."
)
FACTS = JobFacts(summary="Build APIs with FastAPI.", location="Pune", work_mode="hybrid")

PYTHON = Requirement(
    id="python_years",
    label="Python experience",
    kind="number",
    unit="years",
    question="How many years have you worked with Python?",
    hard=True,
    rule=Rule(op=">=", value=3),
    source="3+ years of Python experience",
)
HYBRID = Requirement(
    id="hybrid_ok",
    label="Hybrid in Pune",
    kind="boolean",
    question="The role is three days a week in our Pune office. Would that work for you?",
    hard=True,
    rule=Rule(op="==", value=True),
    disclose=True,
    source="Hybrid: 3 days a week in our Pune office",
)
NOTICE = Requirement(
    id="notice_days",
    label="Notice period",
    kind="number",
    unit="days",
    question="What's your notice period?",
    weight=0.8,
    source="Notice period of 30 days or less preferred",
)
INVENTED = Requirement(
    id="age",
    label="Age",
    kind="number",
    unit="years",
    question="How old are you?",
    hard=True,
    rule=Rule(op="<=", value=30),
    source="Candidates under 30 preferred",
)


def draft(*reqs, suggestions=()):
    return Draft(
        criteria=Criteria(requirements=list(reqs), job_facts=FACTS), suggestions=list(suggestions)
    )


def run(models, similar=()):
    return run_criteria_agent(
        models,
        title="Backend Engineer",
        description=JD,
        notes="",
        similar_lookup=lambda q: list(similar),
    )


def test_sound_draft_passes_in_one_round_with_normalized_weights():
    models = FakeModels({Draft: [draft(PYTHON, HYBRID, NOTICE)], Critique: [Critique()]})
    result = run(models)
    assert result.rounds == 1
    assert [r.id for r in result.criteria.requirements] == [
        "python_years",
        "hybrid_ok",
        "notice_days",
    ]
    assert result.criteria.requirements[2].weight == 1.0
    assert result.warnings == []


def test_ungrounded_requirement_is_sent_back_and_fixed():
    models = FakeModels(
        {
            Draft: [draft(PYTHON, INVENTED), draft(PYTHON, HYBRID)],
            Critique: [Critique()],
        }
    )
    result = run(models)
    assert result.rounds == 2
    assert [r.id for r in result.criteria.requirements] == ["python_years", "hybrid_ok"]
    revision_prompt = [u for name, _, u in models.prompts if name == "Draft"][1]
    assert "Candidates under 30 preferred" in revision_prompt
    assert "ungrounded" in revision_prompt


def test_unresolved_illegal_requirement_is_dropped_after_max_rounds():
    illegal = Issue(
        requirement_id="age_limit", kind="illegal", detail="screens on age", fix="remove it"
    )
    age = INVENTED.model_copy(update={"id": "age_limit", "source": "3+ years of Python"})
    models = FakeModels({Draft: [draft(PYTHON, age)], Critique: [Critique(issues=[illegal])]})
    result = run(models)
    assert result.rounds == MAX_ROUNDS
    assert [r.id for r in result.criteria.requirements] == ["python_years"]
    assert any("age_limit" in w and "removed" in w for w in result.warnings)


def test_unresolved_soft_issue_becomes_a_warning():
    vague = Issue(
        requirement_id="notice_days", kind="units", detail="days or weeks?", fix="say days"
    )
    models = FakeModels({Draft: [draft(PYTHON, NOTICE)], Critique: [Critique(issues=[vague])]})
    result = run(models)
    assert [r.id for r in result.criteria.requirements] == ["python_years", "notice_days"]
    assert any("notice_days" in w and "days or weeks?" in w for w in result.warnings)


def test_retrieved_examples_reach_the_drafter_and_suggestions_become_warnings():
    models = FakeModels(
        {
            Draft: [draft(PYTHON, suggestions=["Work authorisation in India"])],
            Critique: [Critique()],
        }
    )
    result = run(models, similar=["Data Engineer: Notice period (weighted) — asked as: ..."])
    first_prompt = [u for name, _, u in models.prompts if name == "Draft"][0]
    assert "Data Engineer: Notice period" in first_prompt
    assert any("Work authorisation in India" in w for w in result.warnings)


def test_no_requirements_left_is_a_warning():
    illegal = Issue(requirement_id="age", kind="illegal", detail="age", fix="remove")
    models = FakeModels({Draft: [draft(INVENTED)], Critique: [Critique(issues=[illegal])]})
    result = run(models)
    assert result.criteria.requirements == []
    assert any("No checkable requirements" in w for w in result.warnings)


def test_heartbeat_runs_before_every_step():
    beats = []
    models = FakeModels({Draft: [draft(PYTHON)], Critique: [Critique()]})
    run_criteria_agent(
        models,
        title="Backend Engineer",
        description=JD,
        notes="",
        similar_lookup=lambda q: [],
        heartbeat=lambda: beats.append(len(models.prompts)),
    )
    assert beats == [0, 0, 1, 2]  # retrieve, draft, critique, finalize


def test_invalid_rule_is_tied_to_its_requirement_and_left_to_the_api():
    no_rule = PYTHON.model_copy(update={"rule": None})
    models = FakeModels({Draft: [draft(no_rule)], Critique: [Critique()]})
    result = run(models)
    revision_prompt = [u for name, _, u in models.prompts if name == "Draft"][1]
    assert "[invalid] python_years: a hard requirement needs a rule" in revision_prompt
    assert not any("needs a rule" in w for w in result.warnings)  # the API reports it


def test_repeated_issues_give_one_warning_and_one_removal():
    units = Issue(requirement_id="notice_days", kind="units", detail="days or weeks?")
    illegal = Issue(requirement_id="age", kind="illegal", detail="screens on age")
    models = FakeModels(
        {
            Draft: [draft(PYTHON, NOTICE, INVENTED)],  # INVENTED is also ungrounded
            Critique: [Critique(issues=[units, units.model_copy(), illegal])],
        }
    )
    result = run(models)
    assert sum("notice_days" in w for w in result.warnings) == 1
    assert sum(w.startswith("age: removed") for w in result.warnings) == 1


def test_critic_question_count_is_ignored():
    too_many = Issue(kind="too_many", detail="too many questions")
    models = FakeModels({Draft: [draft(PYTHON)], Critique: [Critique(issues=[too_many])]})
    result = run(models)
    assert result.rounds == 1
    assert result.warnings == []


def test_too_many_questions_are_sent_back_once_and_warned_once():
    many = [NOTICE.model_copy(update={"id": f"q{i}"}) for i in range(MAX_ASKED + 1)]
    models = FakeModels({Draft: [draft(*many)], Critique: [Critique()]})
    result = run(models)
    revision_prompt = [u for name, _, u in models.prompts if name == "Draft"][1]
    assert "[too_many]" in revision_prompt
    assert [w for w in result.warnings if "questions" in w] == [
        f"{MAX_ASKED + 1} questions; calls stay short with at most {MAX_ASKED}"
    ]


def test_failed_retrieval_does_not_stop_generation():
    def down(query):
        raise RuntimeError("embeddings down")

    models = FakeModels({Draft: [draft(PYTHON)], Critique: [Critique()]})
    result = run_criteria_agent(
        models, title="Backend Engineer", description=JD, notes="", similar_lookup=down
    )
    assert [r.id for r in result.criteria.requirements] == ["python_years"]


def test_prompts_read_experience_ranges_as_minimums():
    # "1-3 years" became a hard "between 1 and 3", which would reject a candidate with 5.
    from voiceagent.agents.criteria import CRITIC, DRAFTER

    assert "1-3 years" in DRAFTER and ">= 1" in DRAFTER
    assert "upper bound" in CRITIC


def steps_of(models, similar=()):
    steps = []
    run_criteria_agent(
        models,
        title="Backend Engineer",
        description=JD,
        notes="",
        similar_lookup=lambda q: list(similar),
        on_step=lambda *report: steps.append(report),
    )
    return steps


AGE_IS_ILLEGAL = Critique(issues=[Issue(requirement_id="age", kind="illegal", detail="age")])


def test_each_step_is_reported_with_counts_and_a_redraft_is_a_revise():
    models = FakeModels(
        {
            Draft: [draft(PYTHON, INVENTED), draft(PYTHON, HYBRID)],
            Critique: [AGE_IS_ILLEGAL, Critique()],
        }
    )
    steps = steps_of(models, similar=["Notice period"])
    assert steps == [
        ("retrieve", 0, {"similar": 1}),
        ("draft", 1, {"requirements": 2, "suggestions": 0}),
        ("critique", 1, {"issues": 2, "kinds": {"illegal": 1, "ungrounded": 1}}),
        ("revise", 2, {"requirements": 2, "suggestions": 0}),
        ("critique", 2, {"issues": 0, "kinds": {}}),
        ("finalize", 2, {"requirements": 2, "removed": 0, "warnings": []}),
    ]
    assert JD not in json.dumps(steps)  # counts and warnings, never the prompts


def test_finalize_reports_what_it_removed_and_why():
    models = FakeModels({Draft: [draft(PYTHON, INVENTED)], Critique: [AGE_IS_ILLEGAL]})
    steps = steps_of(models)
    last_two = [(step, rnd) for step, rnd, _ in steps][-2:]
    assert last_two == [("critique", MAX_ROUNDS), ("finalize", MAX_ROUNDS)]
    summary = steps[-1][2]
    assert (summary["requirements"], summary["removed"]) == (1, 1)
    assert summary["warnings"][0].startswith("age: removed")
