"""Deterministic tools the agents use to check their own output."""

import re
from difflib import SequenceMatcher


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s%+.-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def verify_quote(quote: str, text: str, threshold: float = 0.9) -> bool:
    """Whether `quote` appears in `text`, allowing case, spacing and small wording drift.

    Grounds every generated requirement in the job description, so the criteria agent
    can't invent requirements the recruiter never wrote.
    """
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    n = len(q)
    step = max(1, n // 4)
    for i in range(0, max(1, len(t) - n + 1), step):
        if SequenceMatcher(None, q, t[i : i + n]).ratio() >= threshold:
            return True
    return False
