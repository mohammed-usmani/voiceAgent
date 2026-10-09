import pytest

from voiceagent.criteria.schema import Requirement, Rule
from voiceagent.rules import apply_rule, decide


def req(kind, op, value, **kw) -> Requirement:
    return Requirement(
        id="r",
        label="r",
        kind=kind,
        question="?",
        hard=True,
        rule=Rule(op=op, value=value),
        source="x",
        **kw,
    )


@pytest.mark.parametrize(
    ("op", "target", "value", "verdict"),
    [
        (">=", 3, 3, "pass"),
        (">=", 3, 2.5, "fail"),
        (">", 3, 3, "fail"),
        ("<=", 30, 30, "pass"),
        ("<", 30, 30, "fail"),
        ("==", 5, 5.0, "pass"),
        ("between", [25, 35], 25, "pass"),
        ("between", [25, 35], 35, "pass"),
        ("between", [25, 35], 36, "fail"),
        (">=", 3, "4", "pass"),  # numeric strings are numbers
        (">=", 3, " 2.5 ", "fail"),
        (">=", 3, "1,200", "pass"),
        (">=", 3, "about four", "unknown"),
        (">=", 3, "3+", "unknown"),
        (">=", 3, True, "unknown"),  # a bool is never a number of years
        (">=", 3, [3], "unknown"),
    ],
)
def test_number_rules(op, target, value, verdict):
    assert apply_rule(req("number", op, target, unit="years"), value) == verdict


@pytest.mark.parametrize(
    ("target", "value", "verdict"),
    [
        (True, True, "pass"),
        (True, False, "fail"),
        (False, False, "pass"),
        (True, "yes", "pass"),
        (True, " No ", "fail"),
        (True, 1, "unknown"),  # a number is not a yes/no
        (True, "maybe", "unknown"),
    ],
)
def test_boolean_rules(target, value, verdict):
    assert apply_rule(req("boolean", "==", target), value) == verdict


@pytest.mark.parametrize(
    ("op", "target", "value", "verdict"),
    [
        ("==", "Immediate", "immediate", "pass"),
        ("==", "Immediate", " IMMEDIATE ", "pass"),
        ("==", "Immediate", "30 days", "fail"),
        ("in", ["Immediate", "15 days"], "15 Days", "pass"),
        ("in", ["Immediate", "15 days"], "60 days", "fail"),
        ("in", ["Immediate"], 15, "unknown"),
        ("==", ["Immediate"], "Immediate", "unknown"),  # malformed rule
    ],
)
def test_choice_rules_ignore_case(op, target, value, verdict):
    options = ["Immediate", "15 days", "30 days", "60 days"]
    assert apply_rule(req("choice", op, target, options=options), value) == verdict


def test_missing_or_declined_answers_are_unknown_never_fail():
    salary = req("number", "<=", 30, unit="LPA", sensitive=True)
    assert apply_rule(salary, None) == "unknown"
    assert apply_rule(salary, 50, declined=True) == "unknown"


def test_a_requirement_without_a_rule_or_a_text_answer_is_unknown():
    no_rule = req("number", ">=", 3).model_copy(update={"rule": None})
    assert apply_rule(no_rule, 5) == "unknown"
    text = req("number", ">=", 3).model_copy(update={"kind": "text"})
    assert apply_rule(text, "lots") == "unknown"


@pytest.mark.parametrize(
    ("hard", "score", "result"),
    [
        (["pass", "fail"], 0.95, "not_suitable"),  # any failed hard requirement
        (["fail", "unknown"], None, "not_suitable"),
        (["pass", "unknown"], 0.95, "needs_review"),  # an unknown hard requirement
        (["pass"], None, "suitable"),  # no soft score: hard rules decide
        ([], 0.6, "suitable"),  # at the threshold
        ([], 0.5, "needs_review"),  # exactly 0.1 below
        ([], 0.55, "needs_review"),
        ([], 0.4999, "not_suitable"),
        ([], 0.0, "not_suitable"),
    ],
)
def test_decide(hard, score, result):
    assert decide(hard, score, 0.6) == result


MODES = ["On-site", "Hybrid", "Remote"]


@pytest.mark.parametrize(
    ("value", "verdict"),
    [
        ("Onsite", "pass"),
        ("on site", "pass"),
        ("ON-SITE.", "pass"),
        ("  hybrid ", "pass"),
        ("remote", "fail"),
        ("Fully remote", "unknown"),  # not one of the options: a human reads it
        ("Office", "unknown"),
    ],
)
def test_a_choice_outside_the_options_is_unknown_never_fail(value, verdict):
    r = req("choice", "in", ["On-site", "Hybrid"], options=MODES)
    assert apply_rule(r, value) == verdict
