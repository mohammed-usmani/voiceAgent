"""Deterministic tools the agents use to check their own output."""

import re
from difflib import SequenceMatcher

MIN_QUOTE_WORDS = 3  # shorter quotes ("age", "Python") match almost any job description
_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
_TEENS = "fifteen sixteen seventeen eighteen nineteen twenty"
NUMBER_WORDS = {w: str(i) for i, w in enumerate(f"{_UNITS} {_TEENS}".split())} | {
    w: str(10 * i)
    for i, w in enumerate("thirty forty fifty sixty seventy eighty ninety".split(), 3)
}
_NUMBER = re.compile(r"\d+(\.\d+)?")


def _words(s: str) -> list[str]:
    """Lowercased words with punctuation stripped from their edges ("3+," → "3"), and
    spelled-out numbers as digits ("five" → "5"), so a number is protected either way."""
    words = (re.sub(r"^\W+|\W+$", "", t) for t in s.lower().split())
    return [NUMBER_WORDS.get(w, w) for w in words if w]


_ONE = re.compile(r"\ban? (year|month|week|day|lakh)\b")


def numbers_in(text: str) -> list[float]:
    """The numbers in `text`, spelled out or not ("five years, 18 months" → [5, 18]), also
    inside hyphenated words ("a 60-day notice" → [60]) and as "a" or "an" before a unit
    ("about a year" → [1])."""
    text = _ONE.sub(r"1 \1", text.lower().replace("-", " "))
    return [float(w) for w in _words(text) if _NUMBER.fullmatch(w)]


def verify_quote(
    quote: str, text: str, threshold: float = 0.9, *, min_words: int = MIN_QUOTE_WORDS
) -> bool:
    """Whether `quote` appears in `text`, allowing case, punctuation, spacing and small
    wording drift (a word added, dropped or changed) but never a different number.

    Grounds every generated requirement in the job description, so the criteria agent
    can't invent requirements the recruiter never wrote.
    """
    q, t = _words(quote), _words(text)
    if len(q) < max(min_words, 1):
        return False
    numbers = [w for w in q if any(c.isdigit() for c in w)]
    matcher = SequenceMatcher(None, b=" ".join(q))
    for size in (len(q), len(q) - 1, len(q) + 1):
        for i in range(len(t) - size + 1):
            window = t[i : i + size]
            if window == q:
                return True
            matcher.set_seq1(" ".join(window))
            if (
                matcher.quick_ratio() >= threshold
                and matcher.ratio() >= threshold
                and all(n in window for n in numbers)
            ):
                return True
    return False


def verify_answer(quote: str, transcript: list[dict]) -> bool:
    """Whether `quote` is the candidate's own words in one of their answers — never the
    agent's, and never stitched across two answers. Back-to-back candidate turns are one
    answer (a pause makes turn detection split it). Spoken answers are short ("Yes.",
    "Five years."), so one word is enough; a quote under three words gets no drift
    allowance, so it must match exactly."""
    return any(verify_quote(quote, answer, min_words=1) for answer in _answers(transcript))


def _answers(transcript: list[dict]) -> list[str]:
    """The candidate's answers: runs of their turns with no agent turn between, joined."""
    answers: list[str] = []
    run: list[str] = []
    for t in [*transcript, {"role": "agent"}]:
        if t.get("role") == "candidate":
            run.append(t.get("text", ""))
        elif run:
            answers.append(" ".join(run))
            run = []
    return answers


def untrusted(tag: str, text: str) -> str:
    """`text` from the candidate, fenced in <tag> tags for a prompt. Its own tags are removed,
    so it can't close the block and continue as instructions."""
    clean = re.sub(rf"<\s*/?\s*{tag}\s*>", "", text or "", flags=re.I)
    return f"<{tag}>\n{clean}\n</{tag}>"
