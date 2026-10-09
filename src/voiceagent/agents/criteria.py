"""Criteria agent: job description → screening criteria a recruiter can approve.

    retrieve → draft → critique ─┬─→ finalize
                 ↑                │
                 └── revise ──────┘   (at most MAX_ROUNDS drafts)

`critique` combines deterministic checks (every requirement quotes the job description;
rules fit their answer type) with an LLM critic (legality, phone-checkability,
contradictions, units). `finalize` drops anything still illegal or ungrounded and turns
other unresolved issues into warnings for the recruiter.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from voiceagent.agents.tools import verify_quote
from voiceagent.criteria.schema import MAX_ASKED, Criteria, check_criteria, normalize_weights
from voiceagent.llm import Models

MAX_ROUNDS = 3
PROTECTED = (
    "age, gender, religion, caste, marital status, pregnancy, disability, health, "
    "nationality or appearance"
)


class Draft(BaseModel):
    criteria: Criteria
    suggestions: list[str] = Field(
        default_factory=list,
        description="checks similar roles used that this job description doesn't mention; "
        "never added as requirements",
    )


IssueKind = Literal[
    "illegal",
    "not_phone_checkable",
    "contradiction",
    "units",
    "too_many",
    "ungrounded",
    "invalid",
    "other",
]


class Issue(BaseModel):
    requirement_id: str | None = None
    kind: IssueKind
    detail: str
    fix: str = ""


class Critique(BaseModel):
    issues: list[Issue] = Field(default_factory=list)


@dataclass
class CriteriaResult:
    criteria: Criteria
    warnings: list[str]
    rounds: int


class State(TypedDict, total=False):
    title: str
    description: str
    notes: str
    similar: list[str]
    draft: Draft
    issues: list[Issue]
    round: int
    result: CriteriaResult


DRAFTER = f"""You design phone-screening criteria for a recruiter.

From the job description and the recruiter's notes, list what a short screening call should
check: at most {MAX_ASKED} requirements, the most decisive first. For each requirement:
- id: snake_case; label: a few words
- kind: number, boolean, choice or text; unit for numbers (years, days, LPA, ...); options
  for choice
- question: one natural spoken question
- hard=true only for must-haves the job description or notes make non-negotiable, with a
  rule: number → >=, <=, >, <, == or between [min, max]; boolean → == true/false;
  choice → == one option, or in a list of options. Text answers are never hard.
- otherwise hard=false and a weight between 0 and 1 for how much it matters
- rubric for text answers: what a good answer contains
- sensitive=true for salary and other personal topics the candidate may decline
- disclose=true when the agent should state the requirement before asking (location,
  on-site days, shifts, travel)
- source: the exact words from the job description or notes it comes from

Recruiter notes override the job description. Never screen on {PROTECTED}.
Also extract job facts the agent may tell candidates: a two-sentence summary, location,
work mode, interview process and salary band (only if stated).
If similar past roles checked things this job description doesn't mention, list them under
suggestions instead of adding them as requirements."""

CRITIC = f"""You review phone-screening criteria before a recruiter sees them. Report only real
problems, each tied to a requirement id when it applies:
- illegal: screens on {PROTECTED}, or a proxy for one
- not_phone_checkable: a short spoken answer can't show it (e.g. "strong communication
  skills"); suggest an askable question instead
- contradiction: conflicts with the job description, the notes or another requirement
- units: wrong or missing unit, or a threshold in a different unit than the question asks
- too_many: more than {MAX_ASKED} requirements
- other: a hard rule stricter or looser than the text says, a leading question, or a
  must-have the job description states that is missing
Return no issues if the criteria are sound."""


def _job_block(state: State) -> str:
    notes = state.get("notes") or "(none)"
    return (
        f"Job title: {state['title']}\n\nJob description:\n{state['description']}\n\n"
        f"Recruiter notes:\n{notes}"
    )


def _issue_lines(issues: list[Issue]) -> str:
    return "\n".join(
        f"- [{i.kind}] {i.requirement_id or 'overall'}: {i.detail}"
        + (f" (fix: {i.fix})" if i.fix else "")
        for i in issues
    )


def run_criteria_agent(
    models: Models,
    *,
    title: str,
    description: str,
    notes: str,
    similar_lookup: Callable[[str], list[str]],
    heartbeat: Callable[[], None] = lambda: None,
) -> CriteriaResult:
    """`heartbeat` runs before every step, so a worker can keep its task's lease alive."""

    def retrieve(state: State) -> State:
        query = f"{state['title']}\n{state['description'][:1500]}"
        return {"similar": similar_lookup(query), "round": 0, "issues": []}

    def draft(state: State) -> State:
        parts = [_job_block(state)]
        if state["similar"]:
            parts.append(
                "Checks used for similar past roles (for suggestions only):\n"
                + "\n".join(f"- {s}" for s in state["similar"])
            )
        if state.get("draft") is not None:
            parts.append(
                "Your previous draft:\n"
                + state["draft"].model_dump_json(indent=1)
                + "\n\nFix these problems and return the full corrected criteria:\n"
                + _issue_lines(state["issues"])
            )
        d = models.structured(Draft, DRAFTER, "\n\n".join(parts))
        return {"draft": d, "round": state["round"] + 1}

    def critique(state: State) -> State:
        d = state["draft"]
        source = f"{state['description']}\n{state.get('notes', '')}"
        issues = [
            Issue(
                requirement_id=r.id,
                kind="ungrounded",
                detail=f"source not found in the job description or notes: {r.source!r}",
                fix="quote the exact words it comes from, or remove the requirement",
            )
            for r in d.criteria.requirements
            if not verify_quote(r.source, source)
        ]
        errors, _ = check_criteria(d.criteria)
        issues += [
            Issue(kind="invalid", detail=e, fix="make the rule fit the answer type") for e in errors
        ]
        user = (
            _job_block(state)
            + "\n\nCriteria to review:\n"
            + json.dumps(d.criteria.model_dump(mode="json"), indent=1)
        )
        issues += models.structured(Critique, CRITIC, user).issues
        return {"issues": issues}

    def route(state: State) -> str:
        return "finalize" if not state["issues"] or state["round"] >= MAX_ROUNDS else "draft"

    def finalize(state: State) -> State:
        d = state["draft"]
        blocking = {
            i.requirement_id
            for i in state["issues"]
            if i.kind in ("illegal", "ungrounded") and i.requirement_id
        }
        kept = [r for r in d.criteria.requirements if r.id not in blocking]
        criteria = normalize_weights(d.criteria.model_copy(update={"requirements": kept}))
        warnings = [
            f"{rid}: removed ({i.kind}: {i.detail})"
            for i in state["issues"]
            if (rid := i.requirement_id) in blocking
        ]
        warnings += [
            f"{i.requirement_id or 'overall'}: {i.detail}"
            for i in state["issues"]
            if i.requirement_id not in blocking
        ]
        warnings += check_criteria(criteria)[1]
        warnings += [f"Similar roles also screened for: {s}" for s in d.suggestions]
        if not kept:
            warnings.append("No checkable requirements found; add them before approving.")
        return {"result": CriteriaResult(criteria, warnings, state["round"])}

    def beating(step: Callable[[State], State]) -> Callable[[State], State]:
        def run(state: State) -> State:
            heartbeat()
            return step(state)

        return run

    graph = StateGraph(State)
    for step in (retrieve, draft, critique, finalize):
        graph.add_node(step.__name__, beating(step))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges("critique", route, {"draft": "draft", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    out = graph.compile().invoke({"title": title, "description": description, "notes": notes})
    return out["result"]
