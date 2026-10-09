"""Deterministic rules for hard requirements, and the thresholds that turn verdicts and a
score into a decision. No model is involved: every knockout is auditable."""

import re
from typing import Literal

from voiceagent.criteria.schema import Requirement

Verdict = Literal["pass", "fail", "unknown"]
Result = Literal["suitable", "not_suitable", "needs_review"]
# A score this far below the pass threshold (or closer) goes to a human.
BORDERLINE = 0.1
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_YES = {"yes", "true", "y"}
_NO = {"no", "false", "n"}


def _as_number(value) -> float | None:
    if isinstance(value, bool):  # True is an int in Python, never a number of years
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        return float(text) if _NUMBER.match(text) else None
    return None


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        return True if text in _YES else False if text in _NO else None
    return None


def _norm(value) -> str | None:
    """Case, spacing, hyphens and punctuation don't matter: "On-site" is "onsite"."""
    return re.sub(r"[\W_]+", "", value.casefold()) if isinstance(value, str) else None


def apply_rule(req: Requirement, value, *, declined: bool = False) -> Verdict:
    """Evaluate a hard requirement's rule on an extracted answer, in the requirement's unit.

    Anything that can't be read — no answer, a declined answer, a value of the wrong type,
    a choice that isn't one of the options, a malformed rule — is "unknown", never "fail":
    only a human rejects on a guess.
    """
    rule = req.rule
    if declined or value is None or rule is None:
        return "unknown"
    op, target = rule.op, rule.value
    if req.kind == "number":
        n = _as_number(value)
        if n is None:
            return "unknown"
        if op == "between":
            if not isinstance(target, list) or len(target) != 2:
                return "unknown"
            lo, hi = (_as_number(t) for t in target)
            if lo is None or hi is None:
                return "unknown"
            ok = lo <= n <= hi
        else:
            t = _as_number(target)
            if t is None:
                return "unknown"
            ok = {
                ">=": n >= t,
                "<=": n <= t,
                ">": n > t,
                "<": n < t,
                "==": abs(n - t) < 1e-9,
            }.get(op)
            if ok is None:
                return "unknown"
    elif req.kind == "boolean":
        b, t = _as_bool(value), _as_bool(target)
        if b is None or t is None or op != "==":
            return "unknown"
        ok = b == t
    elif req.kind == "choice":
        v = _norm(value)
        # Not one of the options (a misspelt or paraphrased answer): only a human can tell.
        if v is None or v not in {_norm(o) for o in req.options or []}:
            return "unknown"
        if op == "==" and isinstance(target, str):
            ok = v == _norm(target)
        elif op == "in" and isinstance(target, list):
            ok = v in {_norm(t) for t in target}
        else:
            return "unknown"
    else:  # text answers are scored against a rubric, never by a rule
        return "unknown"
    return "pass" if ok else "fail"


def decide(hard: list[Verdict], score: float | None, pass_threshold: float) -> Result:
    """Any failed hard requirement → not_suitable; any unknown one → needs_review; then the
    weighted score of the soft requirements: at or above the threshold → suitable, within
    BORDERLINE below it → needs_review, lower → not_suitable. No soft score (no weighted
    requirements, or none answered) leaves the hard requirements to decide."""
    if "fail" in hard:
        return "not_suitable"
    if "unknown" in hard:
        return "needs_review"
    if score is None:
        return "suitable"
    s = round(score, 6)  # 0.6 - 0.1 is 0.49999999999999994 in floating point
    if s >= round(pass_threshold, 6):
        return "suitable"
    if s >= round(pass_threshold - BORDERLINE, 6):
        return "needs_review"
    return "not_suitable"
