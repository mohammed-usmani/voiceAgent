import pytest

from tests.fakes import FakeModels
from tests.seed import CRITERIA, TRANSCRIPT
from voiceagent.agents.assessment import (
    Evidence,
    EvidenceSet,
    Review,
    SoftScore,
    SoftScores,
    run_assessment,
)

REQS = CRITERIA.requirements  # python_years (hard >= 3), onsite_pune (hard == true), salary
PY_QUOTE = "I have been writing Python for about five years now."
PUNE_QUOTE = "Yes, Pune three days a week works for me."
SALARY_QUOTE = "I am expecting around 30 lakhs per annum."


def evidence(python=5.0, py_quote=PY_QUOTE, py_conf=0.9, salary=30.0, **salary_kw):
    salary_kw = {"quote": SALARY_QUOTE, "confidence": 0.9} | salary_kw
    return EvidenceSet(
        answers=[
            Evidence(
                requirement_id="python_years", value=python, quote=py_quote, confidence=py_conf
            ),
            Evidence(requirement_id="onsite_pune", value=True, quote=PUNE_QUOTE, confidence=0.95),
            Evidence(
                requirement_id="expected_salary",
                value=salary,
                **salary_kw,
            ),
        ]
    )


def scores(salary=0.8, quote=SALARY_QUOTE):
    return SoftScores(
        scores=[SoftScore(requirement_id="expected_salary", score=salary, quote=quote, note="ok")]
    )


def assess(models, *, transcript=TRANSCRIPT, end_reason="completed", lookup=None, **kw):
    return run_assessment(
        models,
        requirements=REQS,
        pass_threshold=0.6,
        transcript=transcript,
        resume="4 years at Infosys building Django services.",
        end_reason=end_reason,
        overrides_lookup=lookup or (lambda r: []),
        **kw,
    )


def asked(models) -> list[str]:
    return [schema for schema, _, _ in models.prompts]


def test_clear_fit_is_suitable_without_a_review():
    models = FakeModels({EvidenceSet: [evidence()], SoftScores: [scores(0.8)]})
    a = assess(models)
    assert (a.result, a.score, a.hard_failures, a.agent_review) == ("suitable", 0.8, [], None)
    assert [(r["requirement_id"], r["verdict"]) for r in a.reasons] == [
        ("python_years", "pass"),
        ("onsite_pune", "pass"),
        ("expected_salary", "scored"),
    ]
    assert asked(models) == ["EvidenceSet", "SoftScores"]  # the reviewer wasn't needed
    assert [t["step"] for t in a.trace] == ["calibrate", "evidence", "rules", "score", "verdict"]
    assert [e.value for e in a.answers] == [5.0, True, 30.0]


def test_failed_must_have_is_challenged_by_the_reviewer_model():
    reviewer = FakeModels({Review: [Review(agree=True, notes="2 years is clear")]})
    reviewer.name = "reviewer"
    models = FakeModels(
        {EvidenceSet: [evidence(python=2.0)], SoftScores: [scores(0.9)]}, reviewer=reviewer
    )
    a = assess(models)
    assert (a.result, a.hard_failures) == ("not_suitable", ["python_years"])
    assert a.agent_review == {
        "agree": True,
        "notes": "2 years is clear",
        "challenged": "not_suitable",
        "model": "reviewer",
    }
    assert asked(reviewer) == ["Review"] and "Review" not in asked(models)
    assert "Must-have not met: Python experience" in a.summary


def test_a_reviewer_who_disagrees_sends_it_to_a_human_never_to_suitable():
    models = FakeModels(
        {
            EvidenceSet: [evidence(python=2.0)],
            SoftScores: [scores(0.9)],
            Review: [Review(agree=False, notes="he said 'two' but later 'five'")],
        }
    )
    a = assess(models)
    assert a.result == "needs_review"
    assert a.agent_review["agree"] is False and a.agent_review["challenged"] == "not_suitable"
    assert "The reviewer disagreed: he said 'two' but later 'five'" in a.summary


def test_a_quote_that_is_not_the_candidates_words_decides_nothing():
    # The model quoted the agent's own question as the answer: the must-have is unknown.
    agent_words = "How many years have you worked with Python?"
    models = FakeModels(
        {
            EvidenceSet: [evidence(python=2.0, py_quote=agent_words)],
            SoftScores: [scores(0.9)],
            Review: [Review(agree=True, notes="")],
        }
    )
    a = assess(models)
    py = a.reasons[0]
    assert (py["verdict"], py["confidence"]) == ("unknown", 0.3)
    assert "quote not found" in a.answers[0].note
    assert a.result == "needs_review" and a.hard_failures == []


def test_a_one_word_answer_is_evidence():
    transcript = [
        {"role": "agent", "text": "How many years have you worked with Python?"},
        {"role": "candidate", "text": "Five."},
        {"role": "agent", "text": "Would working from our Pune office three days a week work?"},
        {"role": "candidate", "text": "Yes."},
        {"role": "agent", "text": "What salary are you expecting?"},
        {"role": "candidate", "text": "I am expecting around 30 lakhs per annum."},
    ]
    ev = evidence(py_quote="Five")
    ev.answers[1].quote = "Yes"
    models = FakeModels({EvidenceSet: [ev], SoftScores: [scores(0.8)]})
    assert assess(models, transcript=transcript).result == "suitable"


def test_low_confidence_must_have_is_unknown():
    models = FakeModels(
        {
            EvidenceSet: [evidence(python=5.0, py_conf=0.4)],
            SoftScores: [scores(0.9)],
            Review: [Review(agree=True, notes="")],
        }
    )
    a = assess(models)
    assert a.reasons[0]["verdict"] == "unknown"
    assert a.result == "needs_review"
    assert "Must-have unconfirmed: Python experience" in a.summary


def test_declined_salary_never_counts_against_the_candidate():
    models = FakeModels(
        {EvidenceSet: [evidence(salary=None, declined=True, quote="I'd rather not say")]}
    )
    transcript = [*TRANSCRIPT[:6], {"role": "candidate", "text": "I'd rather not say."}]
    a = assess(models, transcript=transcript)
    assert a.reasons[2]["verdict"] == "declined"
    assert (a.result, a.score) == ("suitable", None)  # the must-haves decide
    assert "SoftScores" not in asked(models)


def test_score_just_under_the_pass_mark_needs_review():
    models = FakeModels(
        {
            EvidenceSet: [evidence()],
            SoftScores: [scores(0.55)],
            Review: [Review(agree=True, notes="")],
        }
    )
    assert assess(models).result == "needs_review"


def test_low_score_is_not_suitable():
    models = FakeModels(
        {
            EvidenceSet: [evidence()],
            SoftScores: [scores(0.2)],
            Review: [Review(agree=True, notes="")],
        }
    )
    a = assess(models)
    assert (a.result, a.score) == ("not_suitable", 0.2)
    assert a.agent_review["agree"] is True


@pytest.mark.parametrize("end_reason", ["candidate_hung_up", "not_interested"])
def test_an_unfinished_call_is_never_an_automatic_rejection(end_reason):
    models = FakeModels({EvidenceSet: [evidence(python=1.0)], SoftScores: [scores(0.1)]})
    a = assess(models, end_reason=end_reason)
    assert a.result == "needs_review"
    assert a.hard_failures == ["python_years"]  # the evidence is still shown
    assert a.summary.startswith(f"The call ended early ({end_reason})")
    assert "Review" not in asked(models)


def test_a_requirement_the_model_skipped_is_unknown():
    ev = evidence()
    ev.answers = ev.answers[1:]  # nothing for python_years
    models = FakeModels(
        {EvidenceSet: [ev], SoftScores: [scores(0.8)], Review: [Review(agree=True, notes="")]}
    )
    a = assess(models)
    assert (a.reasons[0]["verdict"], a.reasons[0]["confidence"]) == ("unknown", 0.0)
    assert a.result == "needs_review"


def test_overrides_calibrate_soft_requirements_only():
    looked_up = []

    def lookup(r):
        looked_up.append(r.id)
        return ["Backend Engineer — Expected salary: recruiter decided suitable at 36 LPA"]

    models = FakeModels({EvidenceSet: [evidence()], SoftScores: [scores(0.8)]})
    assess(models, lookup=lookup)
    assert looked_up == ["expected_salary"]
    scorer_prompt = models.prompts[1][2]
    assert "recruiter decided suitable at 36 LPA" in scorer_prompt
    assert "36 LPA" not in models.prompts[0][2]  # never shown to the evidence step


def test_a_failed_override_lookup_still_assesses():
    def broken(r):
        raise RuntimeError("pgvector down")

    models = FakeModels({EvidenceSet: [evidence()], SoftScores: [scores(0.8)]})
    assert assess(models, lookup=broken).result == "suitable"


def test_a_soft_score_quoting_words_never_said_is_challenged():
    models = FakeModels(
        {
            EvidenceSet: [evidence()],
            SoftScores: [scores(0.9, quote="I want 20 lakhs")],
            Review: [Review(agree=False, notes="salary quote isn't in the call")],
        }
    )
    a = assess(models)
    assert a.reasons[2]["confidence"] == 0.3
    assert a.result == "needs_review"


def test_inconsistency_with_the_resume_is_kept():
    ev = evidence()
    ev.answers[0].inconsistency = "said 5 years; résumé shows 4"
    models = FakeModels({EvidenceSet: [ev], SoftScores: [scores(0.8)]})
    assert assess(models).answers[0].inconsistency == "said 5 years; résumé shows 4"


def test_heartbeat_before_every_step():
    beats = []
    models = FakeModels({EvidenceSet: [evidence()], SoftScores: [scores(0.8)]})
    assess(models, heartbeat=lambda: beats.append(1))
    assert len(beats) == 6  # calibrate, evidence, rules, score, verdict, finalize
