import pytest

from voiceagent.evals.jobs import load_jobs
from voiceagent.evals.judges import CHECKS
from voiceagent.evals.metrics import (
    calibration_metrics,
    capture_metrics,
    decision_metrics,
    fairness_metrics,
    judge_rates,
    message_metrics,
    reviewer_stats,
    summarize,
)

JOBS = {j.slug: j for j in load_jobs(["software-engineer"])}
TRUTH = {
    "python_years": 5,
    "hybrid_bengaluru": True,
    "notice_days": 30,
    "sql_depth": "Added an index on merchant_id.",
    "api_design": None,
    "expected_ctc": 24,
}


def answer(rid, value, **kw):
    return {"requirement_id": rid, "value": value, "declined": False, **kw}


def record(
    expected="suitable",
    result="suitable",
    *,
    key="clear_fit",
    end="completed",
    expected_end="completed",
    answers=None,
    judgement=None,
    review=None,
    score=0.8,
    name="Ananya Rao",
    declined=(),
    **kw,
):
    return {
        "job": "software-engineer",
        "persona": {
            "key": key,
            "kind": key.split(":")[0],
            "name": name,
            "truth": TRUTH,
            "declined": list(declined),
            "expected_end": expected_end,
            "expected_decision": expected,
        },
        "run": 0,
        "call": {"end_reason": end, "callback_at": None, "turns": 8, "transcript": []},
        "assessment": None
        if result is None
        else {
            "result": result,
            "score": score,
            "answers": answers or [],
            "agent_review": review,
        },
        "judgement": judgement,
        "message": None,
        "calibrated": None,
        "cost_usd": 0.02,
        "error": None,
    } | kw


def judged(asked=(), reasked=(), text=(), **checks):
    return {c: checks.get(c, True) for c in CHECKS} | {
        "asked": list(asked),
        "reasked": list(reasked),
        "text_answers": [{"requirement_id": r, "agrees": ok} for r, ok in text],
    }


def test_capture_compares_numbers_with_tolerance_and_text_by_the_judge():
    rec = record(
        answers=[
            answer("python_years", 5.4),  # within 10%
            answer("hybrid_bengaluru", "yes"),
            answer("notice_days", 45),  # wrong
            answer("sql_depth", "Indexed merchant_id."),
            answer("expected_ctc", "24"),
        ],
        judgement=judged(
            asked=["python_years", "hybrid_bengaluru", "notice_days", "sql_depth", "api_design"],
            reasked=["notice_days"],
            text=[("sql_depth", True)],
        ),
    )
    m = capture_metrics(JOBS, [rec])
    assert m["by_kind"] == {"boolean": 1.0, "number": round(2 / 3, 4), "text": 1.0}
    assert m["accuracy"] == 0.8  # api_design has no ground truth: not counted
    assert m["missed_per_call"] == 1 and m["reasked_per_call"] == 1


def test_a_declined_answer_counts_only_when_recorded_as_declined():
    good = record(declined=["expected_ctc"], answers=[answer("expected_ctc", None, declined=True)])
    bad = record(declined=["expected_ctc"], answers=[answer("expected_ctc", 24)])
    assert capture_metrics(JOBS, [good])["by_kind"]["declined"] == 1.0
    assert capture_metrics(JOBS, [bad])["by_kind"]["declined"] == 0.0


def test_capture_skips_calls_that_ended_early_or_failed():
    early = record(end="candidate_hung_up", answers=[answer("python_years", 1)])
    failed = record(error="boom", answers=[answer("python_years", 1)])
    consent = record(None, None, expected_end="declined_consent", end="declined_consent")
    assert capture_metrics(JOBS, [early, failed, consent])["accuracy"] is None


def test_false_rejections_are_counted_apart_from_accuracy():
    recs = [
        record("suitable", "suitable"),
        record("suitable", "not_suitable", key="evasive"),  # a false rejection
        record("needs_review", "needs_review", key="borderline"),
        record("not_suitable", "not_suitable", key="miss:python_years"),
        record("not_suitable", "needs_review", key="miss:notice_days"),
        record(
            None, None, key="bad_time", end="callback_requested", expected_end="callback_requested"
        ),
        record(None, None, key="refuses_consent", end="completed", expected_end="declined_consent"),
    ]
    m = decision_metrics(recs)
    assert m["n"] == 5 and m["accuracy"] == 0.6
    assert m["false_rejection_rate"] == round(1 / 3, 4)
    assert m["needs_review_rate"] == 0.4
    assert m["end_reason_accuracy"] == round(6 / 7, 4)
    assert m["consistency"] is None  # one run per persona


def test_a_call_cut_short_before_its_decision_counts_as_wrong():
    recs = [
        record("suitable", "suitable"),
        record("suitable", None, key="evasive", end="callback_requested"),
    ]
    m = decision_metrics(recs)
    assert m["n"] == 2 and m["accuracy"] == 0.5
    assert m["false_rejection_rate"] == 0.0  # over decided calls only


def test_consistency_across_repeated_runs():
    recs = [
        record(run=0),
        record(run=1),
        record(key="borderline", run=0),
        record(result="needs_review", key="borderline", run=1),
    ]
    assert decision_metrics(recs)["consistency"] == 0.5


def test_reviewer_stats_score_its_challenges_against_the_expected_decision():
    def review(verdict, agree):
        return {"agree": agree, "challenged": verdict}

    recs = [
        record("suitable", "needs_review", review=review("not_suitable", False)),  # caught
        record("not_suitable", "needs_review", review=review("not_suitable", False)),  # needless
        record("suitable", "not_suitable", review=review("not_suitable", True)),  # let through
        record("not_suitable", "not_suitable", review=review("not_suitable", True)),
        record("needs_review", "needs_review", review=review("suitable", False)),  # caught
        record("suitable", "suitable", review=review("suitable", True)),
        record("suitable", "suitable"),  # not reviewed
    ]
    assert reviewer_stats(recs) == {
        "rejections_challenged": 4,
        "wrong_rejections_caught": 1,
        "needless_rejections_escalated": 1,
        "wrong_rejections_let_through": 1,
        "invites_challenged": 2,
        "wrong_invites_caught": 1,
        "needless_invites_escalated": 0,
        "wrong_invites_let_through": 0,
    }


def test_judge_and_message_rates():
    recs = [record(judgement=judged()), record(judgement=judged(never_promised=False))]
    assert judge_rates(recs)["never_promised"] == 0.5
    assert judge_rates(recs)["professional"] == 1.0
    msg = {
        "kind": "rejection",
        "issues": ["too cold"],
        "rounds": 3,
        "judgement": {"professional": True, "no_reasons_or_scores": False, "no_promises": True},
    }
    invite = {
        "kind": "invite",
        "issues": [],
        "rounds": 1,
        "judgement": {"professional": True, "no_reasons_or_scores": False, "no_promises": True},
    }
    m = message_metrics([record(message=msg), record(message=invite)])
    assert m["rejection_gives_no_reasons"] == 0.0  # invites aren't held to it
    assert m["unresolved_checker_issues"] == 0.5 and m["rounds"] == 2


def test_calibration_compares_the_same_calls_before_and_after_overrides():
    recs = [
        record("suitable", "needs_review", calibrated={"result": "suitable", "score": 0.7}),
        record("needs_review", "needs_review", calibrated={"result": "suitable", "score": 0.7}),
        record("suitable", "suitable"),  # not re-assessed
    ]
    assert calibration_metrics(recs) == {"n": 2, "accuracy_before": 0.5, "accuracy_after": 0.5}


def fair(name, result, score, **kw):
    return record(result=result, score=score, key=f"fairness:{name}", name=name, **kw)


def test_fairness_needs_the_same_decision_and_close_scores():
    ok = [fair("A", "needs_review", 0.55), fair("B", "needs_review", 0.58)]
    assert fairness_metrics(ok)["passed"] is True
    spread = [fair("A", "needs_review", 0.52), fair("B", "needs_review", 0.58)]
    assert fairness_metrics(spread)["score_spread"] == 0.06
    assert fairness_metrics(spread)["passed"] is False
    split = [fair("A", "needs_review", 0.55), fair("B", "suitable", 0.56)]
    assert fairness_metrics(split)["same_decision"] is False


def test_fairness_fails_when_one_name_errored_or_scored_and_another_did_not():
    errored = [fair("A", "suitable", 0.7), fair("B", "suitable", 0.7, error="timeout")]
    assert fairness_metrics(errored)["passed"] is False
    unscored = [fair("A", "suitable", None), fair("B", "suitable", 0.7)]
    assert fairness_metrics(unscored)["passed"] is False
    assert fairness_metrics([record()]) is None


def test_summary_is_flat_and_counts_errors():
    recs = [record(), record(error="boom", assessment=None)]
    rows = [{"recall": 1.0, "precision": 0.5, "illegal_dropped": None, "rounds": 2}]
    s = summarize(rows, recs, JOBS, usd=0.123456)
    assert s["criteria.recall"] == 1.0 and s["criteria.illegal_dropped"] is None
    assert s["calls.n"] == 1 and s["calls.errors"] == 1
    assert s["cost.usd"] == pytest.approx(0.1235) and s["cost.per_call_usd"] == 0.02
    assert all(isinstance(v, int | float) or v is None for v in s.values())
