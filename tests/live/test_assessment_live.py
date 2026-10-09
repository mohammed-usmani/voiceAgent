"""Runs the assessment and message agents against the real models on canned calls. Costs a
handful of LLM calls: RUN_LIVE=1 uv run pytest tests/live/test_assessment_live.py -v -s"""

import os

import pytest

from tests.seed import CRITERIA, FACTS, TRANSCRIPT
from voiceagent.agents.assessment import run_assessment
from voiceagent.agents.messages import Letter, letter_issues, run_message_agent
from voiceagent.llm import default_models
from voiceagent.settings import get_settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1"),
]
LABELS = [r.label for r in CRITERIA.requirements]


@pytest.fixture(scope="module")
def models():
    return default_models(get_settings())


def assess(models, transcript):
    return run_assessment(
        models,
        requirements=CRITERIA.requirements,
        pass_threshold=0.6,
        transcript=transcript,
        resume="4 years at Infosys building Django services.",
        end_reason="completed",
        overrides_lookup=lambda r: [],
    )


def test_a_clear_fit_passes_every_must_have(models):
    a = assess(models, TRANSCRIPT)
    print(a.result, a.score, a.summary, *a.reasons, sep="\n")
    values = {e.requirement_id: e.value for e in a.answers}
    assert values["python_years"] == 5 and values["onsite_pune"] is True
    assert values["expected_salary"] == 30
    assert [r["verdict"] for r in a.reasons[:2]] == ["pass", "pass"]
    assert a.result in ("suitable", "needs_review")  # the soft score is the model's call


def test_a_missed_must_have_is_never_suitable_and_is_reviewed(models):
    short = [
        *TRANSCRIPT[:3],
        {"role": "candidate", "text": "Honestly only about two years, mostly scripting."},
        *TRANSCRIPT[4:],
    ]
    a = assess(models, short)
    print(a.result, a.summary, a.agent_review, sep="\n")
    assert a.reasons[0]["verdict"] == "fail"
    assert a.result in ("not_suitable", "needs_review")
    assert a.agent_review is not None  # the reviewer model challenged it


@pytest.mark.parametrize("kind", ["invite", "rejection"])
def test_messages_pass_their_own_checks(models, kind):
    d = run_message_agent(
        models,
        kind=kind,
        candidate_name="Priya Sharma",
        job_title="Backend Engineer",
        company_name="Acme",
        job_facts=FACTS,
        requirement_labels=LABELS,
    )
    print(d.subject, d.body, d.issues, f"rounds={d.rounds}", sep="\n")
    assert letter_issues(kind, Letter(subject=d.subject, body=d.body), labels=LABELS) == []
    assert "Priya" in d.body
