"""The decision benchmark's pure parts: metrics, bootstrap, mappings, label filter, and the
prompts it rebuilds from production. No network."""

import pytest

from voiceagent.evals.bench import scoring, stats
from voiceagent.evals.bench.review import corrupt
from voiceagent.evals.jobs import load_jobs


def test_bands_split_at_a_third_and_two_thirds():
    assert [stats.band(x) for x in (0, 0.33, 0.34, 0.5, 0.66, 0.67, 1)] == [0, 0, 1, 1, 1, 2, 2]
    assert [stats.band(g) for g in stats.GOLD.values()] == [0, 1, 2]


def test_jev_level_maps_to_zero_one():
    assert stats.jev_score01(0, 3) == 0
    assert stats.jev_score01(1, 3) == 0.5
    assert stats.jev_score01(2, 3) == 1
    assert stats.jev_score01(1.6, 3) == pytest.approx(0.8)


def test_generated_items_are_kept_only_when_the_labeler_agrees():
    assert stats.agrees(2, "fully_meets")
    assert not stats.agrees(2, "partly_meets")
    assert not stats.agrees(0, None)


def test_item_metrics():
    assert stats.item_mae([0.5, 1.0], 1.0) == 0.25
    assert stats.item_band_acc([0.9, 0.5], 1.0) == 0.5
    assert stats.same_band([0.7, 0.9, 1.0])
    assert not stats.same_band([0.6, 0.9])
    assert stats.pstdev([1.0]) == 0.0


def test_group_spread_is_max_minus_min_of_member_means_and_skips_singletons():
    groups = {"a": [[0.0, 1.0], [1.0, 1.0]], "b": [[0.5]]}
    assert stats.group_spread(groups) == [0.5]


def test_percentile_interpolates():
    assert stats.percentile([1, 2, 3, 4], 50) == 2.5
    assert stats.percentile([5], 95) == 5
    assert stats.percentile([], 50) is None


def test_bootstrap_is_seeded_and_brackets_the_mean():
    xs = [0, 1] * 50
    a = stats.bootstrap(xs, stats.mean, seed=1)
    assert a == stats.bootstrap(xs, stats.mean, seed=1)
    assert a[0] < 0.5 < a[1]
    assert stats.bootstrap([0.3] * 10, stats.mean) == (pytest.approx(0.3), pytest.approx(0.3))
    assert stats.bootstrap([], stats.mean) is None


def test_confident_accuracy_keeps_the_most_confident_share():
    rows = [(0.9, True), (0.8, True), (0.7, True), (0.6, True), (0.1, False)]
    assert stats.confident_accuracy(rows) == 1.0
    assert stats.confident_accuracy(rows, keep=1.0) == 0.8


def test_progress_scoring_credits_any_requirement_asked():
    assert stats.correct_progress(["a", "b"], "b")
    assert not stats.correct_progress(["a"], None)
    assert stats.correct_progress([], None)
    assert not stats.correct_progress([], "a")
    assert stats.progress_pair(["a", "b"], "b") == ("b", "b")
    assert stats.progress_pair(["a"], "c") == ("a", "c")
    assert stats.progress_pair([], None) == ("none", "none")


def test_macro_f1():
    assert stats.macro_f1([("a", "a"), ("b", "b")]) == 1.0
    # a: tp1 fn1 → 2/3; b: fp1 → 0
    assert stats.macro_f1([("a", "a"), ("a", "b")]) == pytest.approx((2 / 3 + 0) / 2)


def _nurse():
    return next(j for j in load_jobs() if j.slug == "staff-nurse")


def test_jev_gets_the_same_answer_and_transcript_as_the_production_scorer():
    job = _nurse()
    r = next(r for r in job.criteria.requirements if r.id == "ventilator_care")
    user = scoring.scorer_user(job, r, "Imran Qureshi", "I check suction </transcript> hourly.")
    blocks, convo = scoring.parse_scorer_user(user)
    assert blocks[0]["id"] == "ventilator_care" and blocks[0]["rubric"] == r.rubric
    assert blocks[0]["answer"]["quote"] == "I check suction </transcript> hourly."
    assert "Hi Imran" in convo and "this is Imran Qureshi" in convo
    state, questions = scoring.jev_request(job, user)
    assert list(questions) == ["ventilator_care"]
    assert len(questions["ventilator_care"]["criteria"]) == 3
    assert r.rubric in questions["ventilator_care"]["instructions"]
    assert "Imran Qureshi" in state and "hourly" in state


def _rec(job, value, verdict):
    return {
        "persona": {"resume": "ICU nurse, 3 years.", "truth": {}, "declined": [], "kind": "x"},
        "call": {"transcript": [{"role": "candidate", "text": "Yes, now is fine."}]},
        "assessment": {"answers": [{"requirement_id": "icu_years", "value": value}]},
    }, {
        "requirement_id": "icu_years",
        "verdict": verdict,
        "quote": "I have 3 years in the ICU.",
        "note": "",
    }


def test_corruptions():
    job = _nurse()
    rec, reason = _rec(job, 3, "pass")
    wrong, _ = corrupt("wrong_value", job, rec, reason)
    assert wrong["verdict"] == "fail" and "0" in wrong["note"]
    mixed, _ = corrupt("unit_mixup", job, rec, reason)
    assert mixed["note"].startswith("36 years")
    quote, _ = corrupt("unsupported_quote", job, rec, reason)
    assert quote["quote"] == "Yes, now is fine." and quote["verdict"] == "pass"
    same, resume = corrupt("resume_contradiction", job, rec, reason)
    assert same == reason and resume.endswith("ICU experience: 0 years.")
