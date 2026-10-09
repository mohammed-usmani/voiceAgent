"""What a screening call checks: requirements with typed answers and rules.

Hard requirements are knockouts decided later by a deterministic rule on the extracted
answer; the rest carry a weight and are scored against their rubric.
"""

from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["number", "boolean", "choice", "text"]
Op = Literal[">=", "<=", ">", "<", "==", "in", "between"]
MAX_ASKED = 8  # more questions than this makes the call drag

OPS_BY_KIND: dict[str, set[str]] = {
    "number": {">=", "<=", ">", "<", "==", "between"},
    "boolean": {"=="},
    "choice": {"==", "in"},
    "text": set(),  # judged against a rubric, never by a rule
}


class Rule(BaseModel):
    op: Op
    value: bool | float | str | list[float] | list[str]


class Requirement(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$", description="snake_case id")
    label: str
    kind: Kind
    unit: str | None = Field(default=None, description="for numbers: years, days, LPA, ...")
    options: list[str] | None = Field(default=None, description="for choice answers")
    question: str = Field(description="how the agent asks it, one natural spoken sentence")
    hard: bool = Field(default=False, description="a knockout if not met")
    rule: Rule | None = Field(default=None, description="required when hard")
    weight: float = Field(default=0.0, ge=0, le=1, description="importance when not hard")
    rubric: str | None = Field(default=None, description="what a good answer contains")
    sensitive: bool = Field(default=False, description="candidate may decline to answer")
    disclose: bool = Field(default=False, description="state the requirement before asking")
    source: str = Field(description="exact words from the job description or notes")


class JobFacts(BaseModel):
    summary: str
    location: str | None = None
    work_mode: str | None = None
    interview_process: str | None = None
    salary_band: str | None = None
    other: list[str] = Field(default_factory=list)


class Criteria(BaseModel):
    requirements: list[Requirement]
    job_facts: JobFacts
    pass_threshold: float = Field(default=0.6, ge=0, le=1)


def _is_number(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _rule_fits(r: Requirement) -> bool:
    op, v = r.rule.op, r.rule.value
    if r.kind == "number":
        if op == "between":
            return isinstance(v, list) and len(v) == 2 and all(map(_is_number, v)) and v[0] <= v[1]
        return _is_number(v)
    if r.kind == "boolean":
        return isinstance(v, bool)
    values = v if isinstance(v, list) else [v]  # choice
    allowed = set(r.options or [])
    return (op == "in") == isinstance(v, list) and all(
        isinstance(x, str) and x in allowed for x in values
    )


def requirement_errors(c: Criteria) -> list[tuple[str, str]]:
    """(requirement id, problem) for every rule that blocks approval."""
    errors: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in c.requirements:
        if r.id in seen:
            errors.append((r.id, "duplicate id"))
        seen.add(r.id)
        if r.kind == "choice" and not r.options:
            errors.append((r.id, "a choice answer needs options"))
        if r.hard:
            if r.rule is None:
                errors.append((r.id, "a hard requirement needs a rule"))
            elif r.rule.op not in OPS_BY_KIND[r.kind]:
                errors.append((r.id, f"'{r.rule.op}' doesn't fit a {r.kind} answer"))
            elif not _rule_fits(r):
                errors.append(
                    (
                        r.id,
                        f"rule value {r.rule.value!r} doesn't fit '{r.rule.op}' "
                        f"on a {r.kind} answer",
                    )
                )
        elif r.kind == "text" and not r.rubric:
            errors.append((r.id, "a text requirement needs a rubric"))
    return errors


def check_criteria(c: Criteria) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block approval; warnings are shown to the recruiter."""
    errors = [f"{rid}: {problem}" for rid, problem in requirement_errors(c)]
    warnings = [
        f"{r.id}: weight 0, so it won't affect the score"
        for r in c.requirements
        if not r.hard and r.weight == 0
    ]
    if len(c.requirements) > MAX_ASKED:
        warnings.append(
            f"{len(c.requirements)} questions; calls stay short with at most {MAX_ASKED}"
        )
    return errors, warnings


def normalize_weights(c: Criteria) -> Criteria:
    """Scale the weights of non-hard requirements to sum to 1."""
    total = sum(r.weight for r in c.requirements if not r.hard)
    if total <= 0:
        return c
    reqs = [
        r if r.hard else r.model_copy(update={"weight": round(r.weight / total, 4)})
        for r in c.requirements
    ]
    return c.model_copy(update={"requirements": reqs})
