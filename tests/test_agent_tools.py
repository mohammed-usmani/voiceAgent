from voiceagent.agents.tools import verify_quote

JD = """Requirements:
- 3+ years of Python experience.
- Hybrid: 3 days a week in our Pune office."""


def test_exact_and_case_or_whitespace_differences_match():
    assert verify_quote("3+ years of Python experience", JD)
    assert verify_quote("HYBRID:   3 days a week in our  Pune office", JD)


def test_small_drift_matches():
    assert verify_quote("3+ years of python experiences", JD)


def test_invented_quote_does_not_match():
    assert not verify_quote("Must be under 30 years old", JD)
    assert not verify_quote("5+ years of Java", JD)


def test_empty_quote_does_not_match():
    assert not verify_quote("   ", JD)
