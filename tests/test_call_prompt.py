from datetime import datetime

from tests.seed import CRITERIA, sample_candidate, sample_job
from voiceagent.call.prompt import RESUME_LIMIT, build_instructions, closing, greeting

NOW = datetime(2026, 10, 9, 14, 30)


def prompt(**candidate) -> str:
    return build_instructions(sample_job(), sample_candidate(**candidate), CRITERIA, now=NOW)


def test_greeting_says_who_why_ai_and_recorded():
    text = greeting(sample_job(), sample_candidate())
    for part in ("Hi Priya", "Aria", "AI assistant", "Acme", "Backend Engineer", "recorded"):
        assert part in text
    assert text.endswith("?")


def test_questions_are_asked_in_criteria_order():
    text = prompt()
    positions = [text.index(r.question) for r in CRITERIA.requirements]
    assert positions == sorted(positions)
    assert "never re-ask" in text


def test_disclosed_and_sensitive_requirements_get_their_instructions():
    numbered = [line for line in prompt().splitlines() if line[:3] in ("1. ", "2. ", "3. ")]
    python, onsite, salary = numbered
    assert "requires" in onsite and "on-site in Pune, 3 days a week" in onsite
    assert "prefer not to say" in salary
    assert "requires" not in python and "prefer not to say" not in python


def test_vague_answers_get_one_follow_up():
    assert "one follow-up" in prompt()


def test_resume_is_included_to_confirm_not_reask():
    text = prompt()
    assert "4 years at Infosys building Django services." in text
    assert "confirm" in text
    assert "<resume>" not in prompt(resume_text="  ")


def test_long_resume_is_cut():
    text = prompt(resume_text="x" * (RESUME_LIMIT + 500))
    assert "x" * RESUME_LIMIT + " [...]" in text and "x" * (RESUME_LIMIT + 1) not in text


def test_job_facts_are_the_only_answers():
    text = prompt()
    for fact in ("Pune", "Two technical rounds", "Health insurance from day one"):
        assert fact in text
    assert "Salary band" not in text  # not given, so not offered
    assert "pass the question to the team" in text


def test_conduct_rules():
    text = prompt()
    assert "Never say or hint whether they are suitable" in text
    assert "never promise an interview, an offer or a salary" in text
    assert "Always reply in English" in text and "word for word" in text


def test_every_end_reason_is_explained():
    text = prompt()
    for reason in ("declined_consent", "callback_requested", "not_interested", "completed"):
        assert f"reason {reason}" in text
    assert "Friday 09 October 2026, 14:30" in text and "2026-10-10T17:00" in text


def test_neutral_close_promises_only_the_follow_up_time():
    job = sample_job(response_days=5)
    text = build_instructions(job, sample_candidate(), CRITERIA, now=NOW)
    close = closing(job, sample_candidate())
    assert close == (
        "Thanks, Priya. The team will review and get back to you by email within 5 days."
    )
    assert close in text
