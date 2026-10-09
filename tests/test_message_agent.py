import pytest

from tests.fakes import FakeModels
from tests.seed import FACTS
from voiceagent.agents.messages import (
    MAX_ROUNDS,
    Flag,
    InviteCheck,
    Letter,
    RejectionCheck,
    letter_issues,
    run_message_agent,
)

INVITE = Letter(
    subject="Next steps for the Backend Engineer role at Acme",
    body="Hi Priya,\n\nThanks for speaking with us. We'd like to move forward: next come two "
    "technical rounds and a chat with the founder. The team will be in touch to schedule."
    "\n\nThe Acme hiring team",
)
GENERIC_INVITE = Letter(
    subject="Next steps at Acme",
    body="Hi Priya,\n\nThanks for speaking with us. We'd like to move forward, and the team "
    "will be in touch to schedule the next steps.\n\nThe Acme hiring team",
)
REJECTION = Letter(
    subject="Your application to Acme",
    body="Hi Priya,\n\nThank you for your time on the call. We won't be moving forward with "
    "your application for this role, but we wish you every success.\n\nThe Acme hiring team",
)
LABELS = ["Python experience", "Works from the Pune office", "Expected salary"]


def draft(models, kind="invite", facts=FACTS, name="Priya Sharma"):
    return run_message_agent(
        models,
        kind=kind,
        candidate_name=name,
        job_title="Backend Engineer",
        company_name="Acme",
        job_facts=facts,
        requirement_labels=LABELS,
    )


def issues(kind, body, *, labels=LABELS, subject="Acme", **kw):
    return letter_issues(
        kind, Letter(subject=subject, body=body), labels=labels, first_name="Priya", **kw
    )


def test_a_clean_invite_passes_in_one_round():
    models = FakeModels({Letter: [INVITE], InviteCheck: [InviteCheck()]})
    d = draft(models)
    assert (d.kind, d.subject, d.issues, d.rounds) == ("invite", INVITE.subject, [], 1)
    writer_prompt = models.prompts[0]
    assert "The Acme hiring team" in writer_prompt[1]
    assert "Priya" in writer_prompt[2] and "Sharma" not in writer_prompt[2]
    assert "Two technical rounds and a chat with the founder" in writer_prompt[2]


def test_a_clean_rejection_with_no_flags_has_no_issues():
    models = FakeModels({Letter: [REJECTION], RejectionCheck: [RejectionCheck()]})
    d = draft(models, "rejection")
    assert (d.issues, d.rounds) == ([], 1)
    checker_prompt = models.prompts[1][1]
    assert "next step" not in checker_prompt.lower()  # never asked of a rejection


def test_the_model_judges_only_tone_and_invented_facts():
    # Leaks, promises, names and next steps are code's job: the checker can't report them.
    assert set(InviteCheck.model_fields) == {"cold", "invents_facts"}
    assert set(RejectionCheck.model_fields) == {"discouraging", "invents_facts"}


def test_a_flag_on_a_sentence_that_is_not_in_the_letter_is_dropped():
    imagined = RejectionCheck(
        discouraging=Flag(flagged=True, sentence="You were not a good fit for us.", why="cold")
    )
    models = FakeModels({Letter: [REJECTION], RejectionCheck: [imagined]})
    assert draft(models, "rejection").issues == []


def test_an_invite_without_an_interview_process_can_pass():
    facts = FACTS | {"interview_process": None}
    models = FakeModels({Letter: [GENERIC_INVITE], InviteCheck: [InviteCheck()]})
    d = draft(models, facts=facts)
    assert (d.issues, d.rounds) == ([], 1)
    assert "in touch to schedule the next steps" in models.prompts[0][1]  # the writer's rule


def test_a_rejection_that_gives_the_reason_is_rewritten():
    leaky = Letter(
        subject="Your application",
        body="Hi Priya, your Python experience did not meet our bar. The Acme hiring team",
    )
    models = FakeModels({Letter: [leaky, REJECTION], RejectionCheck: [RejectionCheck()]})
    d = draft(models, "rejection")
    assert (d.body, d.issues, d.rounds) == (REJECTION.body, [], 2)
    rewrite_prompt = models.prompts[2][2]  # write, check, write
    assert 'reveals the assessment: "python experience"' in rewrite_prompt
    assert 'reveals the assessment: "did not meet"' in rewrite_prompt


def test_the_checkers_objections_go_back_to_the_writer():
    sentence = "Thanks for speaking with us."
    cold = InviteCheck(cold=Flag(flagged=True, sentence=sentence, why="sounds curt"))
    models = FakeModels({Letter: [INVITE, INVITE], InviteCheck: [cold, InviteCheck()]})
    d = draft(models)
    assert (d.issues, d.rounds) == ([], 2)
    assert f'- cold: sounds curt ("{sentence}")' in models.prompts[2][2]


def test_a_draft_that_never_passes_keeps_its_issues_after_three_rounds():
    sentence = "The team will be in touch to schedule."
    flag = Flag(flagged=True, sentence=sentence, why="names no date")
    models = FakeModels({Letter: [INVITE], InviteCheck: [InviteCheck(invents_facts=flag)]})
    d = draft(models)
    assert d.rounds == MAX_ROUNDS
    assert d.issues == [f'invents facts: names no date ("{sentence}")']
    assert d.body == INVITE.body  # still a draft for the recruiter to fix


def test_a_subject_with_a_line_break_is_flattened():
    broken = GENERIC_INVITE.model_copy(update={"subject": "Next steps\r\nBcc: x@example.com"})
    models = FakeModels({Letter: [broken], InviteCheck: [InviteCheck()]})
    assert draft(models).subject == "Next steps Bcc: x@example.com"


@pytest.mark.parametrize(
    ("kind", "body", "issue"),
    [
        ("rejection", "Hi [Name], thanks. The Acme hiring team", "placeholder left in: [Name]"),
        ("invite", "Hi Priya, we guarantee you an interview.", 'promises something: "guarantee"'),
        ("rejection", "Hi Priya, we promise to keep your CV.", 'promises something: "promise"'),
        ("rejection", "Hi Priya, you scored 0.45 overall.", 'reveals the assessment: "score"'),
        ("rejection", "Hi Priya, you reached 45% of the bar.", 'reveals a score: "45%"'),
        ("rejection", "Hi Priya, you were rated 4/10.", 'reveals a score: "4/10"'),
        (
            "rejection",
            "Hi Priya, your expected salary was too high.",
            'reveals the assessment: "expected salary"',
        ),
        ("rejection", "Hi Anita, thank you for your time.", "isn't addressed to Priya"),
        ("invite", "Hi Priya, we'd like to move forward.", "doesn't say what happens next"),
    ],
)
def test_deterministic_checks(kind, body, issue):
    assert any(i.startswith(issue) for i in issues(kind, body))


def test_an_invite_may_talk_about_next_steps_and_numbers():
    body = "Hi Priya, next come 2 technical rounds; the team will be in touch within 3 days."
    assert issues("invite", body, subject="Next steps") == []


def test_a_label_is_matched_as_a_whole_phrase():
    assert issues("rejection", "Hi Priya, we wish you good luck.", labels=["Go"]) == []
    assert issues("rejection", "Hi Priya, your Go skills fell short.", labels=["Go"]) != []


def test_a_label_that_names_the_role_is_not_a_leak():
    body = "Hi Priya, thank you for applying for the Python Developer role at Acme."
    assert issues("rejection", body, labels=["Python"], job_title="Python Developer") == []


def test_clean_rejection_has_no_issues():
    assert letter_issues("rejection", REJECTION, labels=LABELS, first_name="Priya") == []
