import pytest

from tests.fakes import FakeModels
from tests.seed import FACTS
from voiceagent.agents.messages import (
    MAX_ROUNDS,
    Check,
    Letter,
    letter_issues,
    run_message_agent,
)

INVITE = Letter(
    subject="Next steps for the Backend Engineer role at Acme",
    body="Hi Priya,\n\nThanks for speaking with us. We'd like to move forward: next come two "
    "technical rounds and a chat with the founder. The team will be in touch to schedule."
    "\n\nThe Acme hiring team",
)
REJECTION = Letter(
    subject="Your application to Acme",
    body="Hi Priya,\n\nThank you for your time on the call. We won't be moving forward with "
    "your application for this role, but we wish you every success.\n\nThe Acme hiring team",
)
LABELS = ["Python experience", "Works from the Pune office", "Expected salary"]


def draft(models, kind="invite"):
    return run_message_agent(
        models,
        kind=kind,
        candidate_name="Priya Sharma",
        job_title="Backend Engineer",
        company_name="Acme",
        job_facts=FACTS,
        requirement_labels=LABELS,
    )


def test_a_clean_invite_passes_in_one_round():
    models = FakeModels({Letter: [INVITE], Check: [Check()]})
    d = draft(models)
    assert (d.kind, d.subject, d.issues, d.rounds) == ("invite", INVITE.subject, [], 1)
    writer_prompt = models.prompts[0]
    assert "The Acme hiring team" in writer_prompt[1]
    assert "Priya" in writer_prompt[2] and "Sharma" not in writer_prompt[2]
    assert "Two technical rounds and a chat with the founder" in writer_prompt[2]


def test_a_rejection_that_gives_the_reason_is_rewritten():
    leaky = Letter(
        subject="Your application",
        body="Hi Priya, your Python experience did not meet our bar. The Acme hiring team",
    )
    models = FakeModels({Letter: [leaky, REJECTION], Check: [Check()]})
    d = draft(models, "rejection")
    assert (d.body, d.issues, d.rounds) == (REJECTION.body, [], 2)
    rewrite_prompt = models.prompts[2][2]  # write, check, write
    assert 'reveals the assessment: "python experience"' in rewrite_prompt
    assert 'reveals the assessment: "did not meet"' in rewrite_prompt


def test_the_checkers_objections_go_back_to_the_writer():
    models = FakeModels({Letter: [INVITE, INVITE], Check: [Check(issues=["sounds cold"]), Check()]})
    d = draft(models)
    assert (d.issues, d.rounds) == ([], 2)
    assert "- sounds cold" in models.prompts[2][2]


def test_a_draft_that_never_passes_keeps_its_issues_after_three_rounds():
    models = FakeModels({Letter: [INVITE], Check: [Check(issues=["promises a start date"])]})
    d = draft(models)
    assert d.rounds == MAX_ROUNDS
    assert d.issues == ["promises a start date"]
    assert d.body == INVITE.body  # still a draft for the recruiter to fix


@pytest.mark.parametrize(
    ("kind", "body", "issue"),
    [
        ("rejection", "Hi [Name], thanks. The Acme hiring team", "placeholder left in: [Name]"),
        ("invite", "We guarantee you an interview.", 'promises something: "guarantee"'),
        ("rejection", "You scored 0.45 overall.", 'reveals the assessment: "score"'),
        ("rejection", "You reached 45% of the bar.", 'reveals a score: "45%"'),
        ("rejection", "You were rated 4/10.", 'reveals a score: "4/10"'),
        (
            "rejection",
            "Your expected salary was too high.",
            'reveals the assessment: "expected salary"',
        ),
    ],
)
def test_deterministic_checks(kind, body, issue):
    assert issue in letter_issues(kind, Letter(subject="Acme", body=body), labels=LABELS)


def test_an_invite_may_talk_about_next_steps_and_numbers():
    body = "Next come 2 technical rounds; the team will reply within 3 days."
    assert letter_issues("invite", Letter(subject="Next steps", body=body), labels=LABELS) == []


def test_clean_rejection_has_no_issues():
    assert letter_issues("rejection", REJECTION, labels=LABELS) == []
