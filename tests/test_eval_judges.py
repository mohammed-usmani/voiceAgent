from tests.fakes import FakeModels
from voiceagent.criteria.schema import Criteria, Rule
from voiceagent.evals.jobs import load_jobs
from voiceagent.evals.judges import (
    CallJudgement,
    CriteriaMatch,
    Pair,
    TextMatch,
    judge_call,
    match_criteria,
)
from voiceagent.evals.metrics import criteria_metrics, same_rule
from voiceagent.evals.personas import Persona

SALES = load_jobs(["sales-executive"])[0]
LABELLED = SALES.criteria


def generated(*changes) -> Criteria:
    """The labelled criteria with some requirements replaced: (id, {field: value})."""
    reqs = [r.model_copy() for r in LABELLED.requirements]
    for rid, update in changes:
        reqs = [r.model_copy(update=update) if r.id == rid else r for r in reqs]
    return LABELLED.model_copy(update={"requirements": reqs})


def pairs_for(c: Criteria) -> list[tuple[str, str]]:
    return [(r.id, r.id) for r in c.requirements]


def test_matching_drops_unknown_ids_and_keeps_pairs_one_to_one():
    gen = generated()
    reply = CriteriaMatch(
        pairs=[
            Pair(generated_id="sales_years", labelled_id="sales_years"),
            Pair(generated_id="hindi_english", labelled_id="sales_years"),  # labelled twice
            Pair(generated_id="sales_years", labelled_id="marathi"),  # generated twice
            Pair(generated_id="ghost", labelled_id="marathi"),  # made-up generated id
            Pair(generated_id="marathi", labelled_id="illegal_7"),  # made-up illegal id
            Pair(generated_id="marathi", labelled_id="illegal_0"),
        ]
    )
    models = FakeModels({CriteriaMatch: [reply]})
    pairs = match_criteria(models, SALES, gen)
    assert pairs == [("sales_years", "sales_years"), ("marathi", "illegal_0")]
    _, _, user = models.prompts[0]
    assert '"id": "illegal_0"' in user and "Only female candidates" in user


def test_metrics_stay_in_range_whatever_the_judge_paired():
    m = criteria_metrics(SALES, generated(), [], 1, [("marathi", "illegal_0")])
    assert m["recall"] == 0.0 and m["precision"] == 0.0 and m["illegal_dropped"] == 0.0
    assert m["hard_flag_accuracy"] is None and m["rule_correctness"] is None


def test_a_perfect_generation_scores_full_marks():
    gen = generated()
    m = criteria_metrics(SALES, gen, [], 1, pairs_for(gen))
    assert (m["recall"], m["precision"], m["hard_flag_accuracy"]) == (1.0, 1.0, 1.0)
    assert m["rule_correctness"] == 1.0 and m["illegal_dropped"] == 1.0
    assert m["unmatched_labelled"] == m["unmatched_generated"] == []


def test_reading_one_to_three_years_as_a_range_is_a_wrong_rule():
    gen = generated(("sales_years", {"rule": Rule(op="between", value=[1, 3])}))
    m = criteria_metrics(SALES, gen, [], 2, pairs_for(gen))
    assert m["rule_correctness"] == 0.6667  # 2 of the 3 hard rules
    assert m["rounds"] == 2


def test_a_soft_flag_on_a_must_have_costs_hard_flag_accuracy_and_its_rule():
    gen = generated(("hindi_english", {"hard": False, "rule": None, "weight": 0.2}))
    m = criteria_metrics(SALES, gen, [], 1, pairs_for(gen))
    assert m["hard_flag_accuracy"] == round(5 / 6, 4)
    assert m["rule_correctness"] == 0.6667


def test_extra_and_missing_requirements_move_precision_and_recall():
    extra = LABELLED.requirements[0].model_copy(update={"id": "age_limit", "label": "Age"})
    gen = LABELLED.model_copy(update={"requirements": [*LABELLED.requirements[:3], extra]})
    pairs = [(r.id, r.id) for r in LABELLED.requirements[:3]] + [("age_limit", "illegal_0")]
    warnings = ["x: removed (ungrounded: source not found)", "y: removed (illegal: age)"]
    m = criteria_metrics(SALES, gen, warnings, 3, pairs)
    assert m["recall"] == 0.5 and m["precision"] == 0.75
    assert m["illegal_dropped"] == 0.0 and m["grounding_removed"] == 1
    assert m["unmatched_generated"] == ["age_limit"]


def test_rules_compare_by_value_not_spelling():
    assert same_rule(Rule(op=">=", value=3), Rule(op=">=", value=3.0))
    assert same_rule(
        Rule(op="in", value=["GNM", "B.Sc Nursing"]), Rule(op="in", value=["b.sc nursing", "gnm"])
    )
    assert not same_rule(Rule(op=">=", value=1), Rule(op=">", value=1))
    assert not same_rule(Rule(op="==", value=True), Rule(op="==", value=1.0))
    assert same_rule(None, None) and not same_rule(Rule(op=">=", value=1), None)


def test_the_call_judge_never_reports_ids_outside_the_job():
    persona = Persona(
        key="clear_fit",
        kind="clear_fit",
        name="Ananya Rao",
        resume="",
        truth={"target_record": "Hit target 10 of 12 months.", "sales_years": 2},
        behaviour="",
        expected_end="completed",
        expected_decision="suitable",
    )
    reply = CallJudgement(
        disclosed_ai_and_recording=True,
        asked_all_required=True,
        never_reasked=False,
        never_hinted_decision=True,
        never_promised=True,
        answered_from_job_facts_only=True,
        professional=True,
        correct_close=True,
        asked=["sales_years", "made_up", "target_record"],
        reasked=["sales_years", "other"],
        text_answers=[
            TextMatch(requirement_id="target_record", agrees=True),
            TextMatch(requirement_id="sales_years", agrees=True),  # not a text requirement
        ],
    )
    models = FakeModels({CallJudgement: [reply]})
    transcript = [{"role": "agent", "text": "Hi"}, {"role": "candidate", "text": "Hello"}]
    out = judge_call(models, SALES, persona, transcript, {"target_record": "Met most targets"})
    assert out.asked == ["sales_years", "target_record"] and out.reasked == ["sales_years"]
    assert [t.requirement_id for t in out.text_answers] == ["target_record"]
    user = models.prompts[0][2]
    assert "Agent: Hi\nCandidate: Hello" in user and "Met most targets" in user
