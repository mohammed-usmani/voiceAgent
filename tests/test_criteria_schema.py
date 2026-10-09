import pytest
from pydantic import ValidationError

from voiceagent.criteria.schema import (
    Criteria,
    JobFacts,
    Requirement,
    Rule,
    check_criteria,
    normalize_weights,
)

FACTS = JobFacts(summary="Build APIs.")


def req(**kw) -> Requirement:
    base = dict(
        id="python_years",
        label="Python experience",
        kind="number",
        unit="years",
        question="How many years have you worked with Python?",
        source="3+ years of Python",
    )
    return Requirement(**(base | kw))


def crit(*reqs: Requirement) -> Criteria:
    return Criteria(requirements=list(reqs), job_facts=FACTS)


def test_valid_hard_number_rule():
    errors, _ = check_criteria(crit(req(hard=True, rule=Rule(op=">=", value=3))))
    assert errors == []


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (dict(hard=True), "needs a rule"),
        (dict(hard=True, rule=Rule(op="in", value=["a"])), "doesn't fit a number answer"),
        (dict(hard=True, rule=Rule(op=">=", value="three")), "doesn't fit"),
        (dict(hard=True, rule=Rule(op="between", value=[5, 2])), "doesn't fit"),
        (dict(kind="boolean", hard=True, rule=Rule(op="==", value=3)), "doesn't fit"),
        (dict(kind="choice", options=None), "needs options"),
        (dict(kind="text", hard=True, rule=Rule(op="==", value="x")), "doesn't fit a text answer"),
        (dict(kind="text", weight=0.5), "needs a rubric"),
    ],
)
def test_invalid_requirements(kw, message):
    errors, _ = check_criteria(crit(req(**kw)))
    assert any(message in e for e in errors), errors


def test_choice_rule_must_use_listed_options():
    ok = req(
        id="mode",
        kind="choice",
        options=["onsite", "hybrid"],
        hard=True,
        rule=Rule(op="in", value=["onsite", "hybrid"]),
    )
    bad = req(
        id="mode2",
        kind="choice",
        options=["onsite", "hybrid"],
        hard=True,
        rule=Rule(op="==", value="remote"),
    )
    assert check_criteria(crit(ok))[0] == []
    assert check_criteria(crit(bad))[0]


def test_duplicate_ids_are_errors():
    errors, _ = check_criteria(crit(req(weight=0.5), req(weight=0.5)))
    assert any("duplicate" in e for e in errors)


def test_warnings_for_zero_weight_and_too_many_questions():
    reqs = [req(id=f"q{i}", weight=0.1) for i in range(9)] + [req(id="zero", weight=0)]
    errors, warnings = check_criteria(crit(*reqs))
    assert errors == []
    assert any("weight 0" in w for w in warnings)
    assert any("10 questions" in w for w in warnings)


def test_normalize_weights_sums_soft_weights_to_one():
    c = normalize_weights(
        crit(
            req(id="a", weight=0.2),
            req(id="b", weight=0.6),
            req(id="h", hard=True, rule=Rule(op=">=", value=1), weight=0.9),
        )
    )
    soft = {r.id: r.weight for r in c.requirements if not r.hard}
    assert soft == {"a": 0.25, "b": 0.75}
    assert next(r for r in c.requirements if r.id == "h").weight == 0.9


def test_requirement_id_must_be_a_slug():
    with pytest.raises(ValidationError):
        req(id="Python Years")


def test_boolean_rule_value_stays_boolean():
    r = req(kind="boolean", hard=True, rule=Rule(op="==", value=True))
    assert r.rule.value is True
    assert check_criteria(crit(r))[0] == []
