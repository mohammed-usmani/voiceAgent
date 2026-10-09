import pytest

from tests.fakes import FakeModels
from voiceagent.criteria.schema import Requirement, Rule
from voiceagent.evals.jobs import load_jobs
from voiceagent.evals.personas import (
    EXTRAS,
    PersonaDraft,
    SoftAnswer,
    expected_outcome,
    make_persona,
    persona_keys,
    renamed,
    rule_value,
)
from voiceagent.rules import apply_rule

JOBS = load_jobs()
SE = next(j for j in JOBS if j.slug == "software-engineer")


def draft(**answers) -> PersonaDraft:
    return PersonaDraft(
        resume="Backend engineer at a payments startup since 2021.",
        soft_answers=[SoftAnswer(requirement_id=k, value=v) for k, v in answers.items()],
    )


@pytest.mark.parametrize("job", JOBS, ids=lambda j: j.slug)
def test_rule_values_land_where_they_are_meant_to(job):
    for r in job.criteria.requirements:
        if r.hard:
            assert apply_rule(r, rule_value(r, "pass")) == "pass", r.id
            assert apply_rule(r, rule_value(r, "edge")) == "pass", r.id
            assert apply_rule(r, rule_value(r, "fail")) == "fail", r.id


def test_boundary_values():
    years = next(r for r in SE.criteria.requirements if r.id == "python_years")
    notice = next(r for r in SE.criteria.requirements if r.id == "notice_days")
    assert [rule_value(years, o) for o in ("pass", "edge", "fail")] == [5, 3, 2]
    assert [rule_value(notice, o) for o in ("pass", "edge", "fail")] == [30, 60, 120]


def test_a_rule_that_accepts_every_option_cannot_be_missed():
    r = Requirement(
        id="shift",
        label="Shift",
        kind="choice",
        options=["day", "night"],
        question="?",
        hard=True,
        rule=Rule(op="in", value=["day", "night"]),
        source="x",
    )
    assert rule_value(r, "fail") is None
    job = SE.model_copy(update={"criteria": SE.criteria.model_copy(update={"requirements": [r]})})
    assert not [k for k in persona_keys(job, 20) if k.startswith("miss")]


def test_expected_outcome_runs_the_rules_before_the_label():
    reqs = SE.criteria.requirements
    fit = {
        "python_years": 5,
        "hybrid_bengaluru": True,
        "notice_days": 30,
        "sql_depth": "Tuned a query.",
        "api_design": "Built the payouts API.",
        "expected_ctc": 20,
    }
    assert expected_outcome("clear_fit", reqs, fit, []) == ("completed", "suitable")
    assert expected_outcome("borderline", reqs, fit, []) == ("completed", "needs_review")
    assert expected_outcome("overstates", reqs, fit, []) == ("completed", "needs_review")
    short = fit | {"python_years": 2}
    assert expected_outcome("miss", reqs, short, []) == ("completed", "not_suitable")
    assert expected_outcome("borderline", reqs, short, []) == ("completed", "not_suitable")
    unsure = fit | {"notice_days": None}
    assert expected_outcome("evasive", reqs, unsure, []) == ("completed", "needs_review")
    # a declined sensitive answer never fails anything
    salary = fit | {"expected_ctc": None}
    assert expected_outcome("declines_salary", reqs, salary, ["expected_ctc"]) == (
        "completed",
        "suitable",
    )
    # missing soft evidence never makes anyone suitable
    no_api = fit | {"api_design": None}
    assert expected_outcome("clear_fit", reqs, no_api, []) == ("completed", "needs_review")
    assert expected_outcome("bad_time", reqs, fit, []) == ("callback_requested", None)
    assert expected_outcome("refuses_consent", reqs, fit, []) == ("declined_consent", None)


def test_a_small_run_covers_every_kind_across_the_six_jobs():
    keys = [persona_keys(job, 4, offset=i) for i, job in enumerate(JOBS)]
    for k in keys:
        assert k[0] == "clear_fit" and k[1].startswith("miss:") and k[2] == "borderline"
    assert {k[3] for k in keys} == set(EXTRAS)


def test_a_large_run_misses_each_hard_requirement_once():
    keys = persona_keys(SE, 50)
    hard = [r.id for r in SE.criteria.requirements if r.hard]
    assert sorted(k for k in keys if k.startswith("miss:")) == sorted(f"miss:{h}" for h in hard)
    assert len(keys) == len(set(keys))


def test_a_miss_persona_fails_exactly_its_requirement():
    models = FakeModels({PersonaDraft: [draft(sql_depth="I tuned a report query.")]})
    p = make_persona(models, SE, "miss:notice_days", "Vikram Nair")
    assert p.truth["notice_days"] == 120
    assert p.truth["python_years"] == 5 and p.truth["hybrid_bengaluru"] is True
    assert (p.expected_end, p.expected_decision) == ("completed", "not_suitable")
    assert p.truth["sql_depth"] == "I tuned a report query."
    assert p.truth["api_design"] is None  # the persona agent left it out: unanswered


def test_the_persona_agent_cannot_overwrite_a_hard_answer_or_invent_one():
    answers = dict(sql_depth="Indexed a slow join.", api_design="Built the payouts API.")
    models = FakeModels(
        {PersonaDraft: [draft(python_years=1, made_up="x", expected_ctc=24, **answers)]}
    )
    p = make_persona(models, SE, "clear_fit", "Ananya Rao")
    assert p.truth["python_years"] == 5
    assert "made_up" not in p.truth
    assert p.truth["expected_ctc"] == 24
    assert p.expected_decision == "suitable"
    _, system, user = models.prompts[0]
    assert "strong" in system and '"Backend Python experience": 5' in user


def test_a_soft_number_answer_must_parse_as_a_number():
    answers = dict(sql_depth="Indexed a slow join.", api_design="Built the payouts API.")
    good = FakeModels({PersonaDraft: [draft(expected_ctc="1,200", **answers)]})
    assert make_persona(good, SE, "clear_fit", "Ananya Rao").truth["expected_ctc"] == 1200
    bad = FakeModels({PersonaDraft: [draft(expected_ctc="about 24 lakh", **answers)]})
    assert make_persona(bad, SE, "clear_fit", "Ananya Rao").truth["expected_ctc"] is None


def test_declining_salary_hides_it_from_the_persona_agent():
    answers = dict(sql_depth="Indexed a slow join.", api_design="Built the payouts API.")
    models = FakeModels({PersonaDraft: [draft(expected_ctc=30, **answers)]})
    p = make_persona(models, SE, "declines_salary", "Sneha Kulkarni")
    assert p.declined == ["expected_ctc"] and p.truth["expected_ctc"] is None
    assert p.expected_decision == "suitable"
    assert '"Expected salary": "declines to say"' in models.prompts[0][2]


def test_overstating_puts_a_failing_value_on_the_resume_only():
    models = FakeModels({PersonaDraft: [draft()]})
    p = make_persona(models, SE, "overstates", "Rohan Mehta")
    assert p.truth["python_years"] == 5
    assert "shows only 2 years" in models.prompts[0][2]
    assert p.expected_decision == "needs_review"


def test_renaming_changes_only_the_name():
    models = FakeModels({PersonaDraft: [draft()]})
    p = make_persona(models, SE, "borderline", "Kavya Reddy")
    q = renamed(p, "Imran Qureshi")
    assert q.name == "Imran Qureshi" and q.key == "fairness:Imran Qureshi"
    assert q.model_dump(exclude={"name", "key"}) == p.model_dump(exclude={"name", "key"})
