import pytest

from voiceagent.agents.tools import verify_answer, verify_quote

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


# Two wordings of one job description, so a quote check that only samples some
# alignments of the text misses the match in at least one of them.
REQS = (
    "Requirements: at least 3 years of Python experience, solid SQL, and AWS. "
    "Salary up to 18 LPA. Notice period of 30 days or less."
)
JDS = [
    "We are hiring a Backend Engineer to manage our UPI payments platform. " + REQS,
    "We are hiring a Backend Engineer in Pune to manage our payments platform. " + REQS,
]


@pytest.mark.parametrize("jd", JDS, ids=["upi", "pune"])
def test_trailing_punctuation_matches(jd):
    assert verify_quote("at least 3 years of Python experience.", jd)
    assert verify_quote('"Salary up to 18 LPA."', jd)


@pytest.mark.parametrize("jd", JDS, ids=["upi", "pune"])
def test_one_dropped_word_matches(jd):
    assert verify_quote("at least 3 years Python experience", jd)


@pytest.mark.parametrize("jd", JDS, ids=["upi", "pune"])
def test_changed_number_does_not_match(jd):
    assert not verify_quote("at least 5 years of Python experience", jd)
    assert not verify_quote("Salary up to 28 LPA", jd)
    assert not verify_quote("Notice period of 60 days or less", jd)


@pytest.mark.parametrize("jd", JDS, ids=["upi", "pune"])
def test_short_or_partial_word_quotes_do_not_ground(jd):
    assert not verify_quote("age", jd)  # inside "manage"
    assert not verify_quote("Python", jd)
    assert not verify_quote("solid SQL", jd)


CALL = [
    {"role": "agent", "text": "Would working from our Pune office three days a week work?"},
    {"role": "candidate", "text": "Yes."},
    {"role": "agent", "text": "How many years of Python?"},
    {"role": "candidate", "text": "Five years, mostly Django and FastAPI services."},
]


def test_a_short_spoken_answer_is_a_quote():
    assert verify_answer("Yes", CALL)
    assert verify_answer("five years", CALL)
    assert verify_answer("Five years, mostly Django and FastAPI service", CALL)  # small drift


def test_the_agents_words_are_never_the_candidates_answer():
    assert not verify_answer("Pune office three days a week", CALL)
    assert not verify_answer("How many years of Python?", CALL)


def test_a_short_quote_must_match_exactly():
    assert not verify_answer("No", CALL)
    assert not verify_answer("six years", CALL)
    assert not verify_answer("", CALL)


def test_job_description_quotes_still_need_three_words():
    assert not verify_quote("Python", "Python")


def test_a_spelled_out_number_is_protected_like_a_digit():
    said = "I have two years of Python, mostly scripting."
    assert not verify_quote("I have five years of Python, mostly scripting.", said)
    assert verify_quote("I have 2 years of Python, mostly scripting.", said)  # same number


def test_a_quote_spanning_two_answers_is_not_the_candidates_words():
    call = [
        {"role": "candidate", "text": "I would love to work here."},
        {"role": "agent", "text": "And how many years of Python?"},
        {"role": "candidate", "text": "Ten years, give or take."},
    ]
    assert verify_answer("Ten years, give or take.", call)
    assert not verify_answer("work here. Ten years, give or take.", call)


def test_numbers_inside_hyphenated_words_and_a_unit_count():
    from voiceagent.agents.tools import numbers_in

    assert numbers_in("I have a 60-day notice period") == [60]
    assert numbers_in("about a year in telesales, and an eight-month stint") == [1, 8]
    assert numbers_in("a lot of a things") == []
