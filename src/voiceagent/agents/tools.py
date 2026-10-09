"""Deterministic tools the agents use to check their own output."""

import re
from difflib import SequenceMatcher

MIN_QUOTE_WORDS = 3  # shorter quotes ("age", "Python") match almost any job description


def _words(s: str) -> list[str]:
    """Lowercased words with punctuation stripped from their edges ("3+," → "3")."""
    return [w for w in (re.sub(r"^\W+|\W+$", "", t) for t in s.lower().split()) if w]


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
    """Whether `quote` is the candidate's own words in a call transcript — never the
    agent's. Spoken answers are short ("Yes.", "Five years."), so one word is enough; a
    quote under three words gets no drift allowance, so it must match exactly."""
    said = "\n".join(t.get("text", "") for t in transcript if t.get("role") == "candidate")
    return verify_quote(quote, said, min_words=1)


def untrusted(tag: str, text: str) -> str:
    """`text` from the candidate, fenced in <tag> tags for a prompt. Its own tags are removed,
    so it can't close the block and continue as instructions."""
    clean = re.sub(rf"<\s*/?\s*{tag}\s*>", "", text or "", flags=re.I)
    return f"<{tag}>\n{clean}\n</{tag}>"
