"""Message agent: a decision → the email the candidate receives.

    write → check ─┬─→ finalize
      ↑             │
      └─────────────┘   (while the checker objects, at most MAX_ROUNDS drafts)

`check` combines deterministic checks (no placeholders, no promises, and for rejections no
reasons or scores) with an LLM checker (tone, promises, invented facts). A draft still
failing after MAX_ROUNDS keeps its issues: it is never sent without a recruiter.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from voiceagent.llm import Models

MAX_ROUNDS = 3
Kind = Literal["invite", "rejection"]
# A rejection gives no reasons: none of these words, and no requirement label.
LEAKS = (
    "score",
    "criteria",
    "criterion",
    "requirement",
    "rubric",
    "threshold",
    "not suitable",
    "unsuitable",
    "did not meet",
    "didn't meet",
    "does not meet",
    "doesn't meet",
)
PROMISES = (
    "guarantee",
    "promise",
    "job offer",
    "offer letter",
    "you will be hired",
    "you're hired",
    "you are hired",
    "we will hire",
)
PLACEHOLDER = re.compile(r"\[[^\]]*\]|\{[^}]*\}|<[^>]*>")
# "0.45", "45%", "4/10", "4 out of 10": a score in disguise
SCORE_LIKE = re.compile(r"\b0\.\d+\b|\d+\s*%|\b\d+\s*/\s*10\b|\b\d+\s+out\s+of\s+\d+\b")


class Letter(BaseModel):
    subject: str
    body: str = Field(description="plain text, signed by the hiring team")


class Check(BaseModel):
    issues: list[str] = Field(default_factory=list)


@dataclass
class MessageDraft:
    kind: Kind
    subject: str
    body: str
    issues: list[str]  # left after the last round: a recruiter must look before it's sent
    rounds: int


class State(TypedDict, total=False):
    letter: Letter
    issues: list[str]
    round: int
    draft: MessageDraft


WRITER = """You write the email a job candidate receives after an AI phone screening, on behalf
of the hiring team. Plain text, under 150 words, warm and professional, addressed to the
candidate by first name and signed "The {company} hiring team".
- invite: thank them for the call, say the team would like to move forward, and describe the
  next steps using only the job facts given. Never promise an offer, a salary or a date;
  say the team will be in touch to schedule.
- rejection: thank them warmly for their time, say the team won't be moving forward with
  their application for this role, and wish them well. Give no reasons, scores or criteria,
  and nothing that discourages them from applying again in future.
Never invent facts, names, links or dates. No placeholders such as [Name]."""

CHECKER = """You check an email to a job candidate before it is sent. Report each real problem:
- it reveals reasons, scores, criteria or how the candidate's answers were judged
- it promises anything: an offer, a salary, a start date, an interview date or an outcome
- it sounds discouraging, cold or condescending, or blames the candidate
- it states facts that are not in the job facts given
- (invite only) it doesn't explain the next steps
Return no issues if it is ready to send."""


def letter_issues(kind: Kind, letter: Letter, *, labels: list[str]) -> list[str]:
    """Problems code can find without a model."""
    text = f"{letter.subject}\n{letter.body}"
    low = text.lower()
    issues = []
    if not letter.subject.strip() or not letter.body.strip():
        issues.append("the subject and body must not be empty")
    if m := PLACEHOLDER.search(text):
        issues.append(f"placeholder left in: {m.group()}")
    issues += [f'promises something: "{p}"' for p in PROMISES if p in low]
    if kind == "rejection":
        leaks = [w for w in (*LEAKS, *(label.lower() for label in labels)) if w and w in low]
        issues += [f'reveals the assessment: "{w}"' for w in leaks]
        if m := SCORE_LIKE.search(text):
            issues.append(f'reveals a score: "{m.group()}"')
    return issues


def run_message_agent(
    models: Models,
    *,
    kind: Kind,
    candidate_name: str,
    job_title: str,
    company_name: str,
    job_facts: dict,
    requirement_labels: list[str],
    heartbeat: Callable[[], None] = lambda: None,
) -> MessageDraft:
    first_name = candidate_name.split()[0] if candidate_name.split() else candidate_name
    context = (
        f"Email: {kind}\nCandidate's first name: {first_name}\nRole: {job_title}\n"
        f"Company: {company_name}\nJob facts:\n{json.dumps(job_facts, indent=1)}"
    )
    writer = WRITER.format(company=company_name)

    def write(state: State) -> State:
        user = context
        if state.get("letter") is not None:
            user += (
                "\n\nYour previous draft:\n"
                + state["letter"].model_dump_json(indent=1)
                + "\n\nFix these problems and return the whole email:\n"
                + "\n".join(f"- {i}" for i in state["issues"])
            )
        letter = models.structured(Letter, writer, user)
        return {"letter": letter, "round": state.get("round", 0) + 1}

    def check(state: State) -> State:
        letter = state["letter"]
        issues = letter_issues(kind, letter, labels=requirement_labels)
        user = f"{context}\n\nEmail to check:\nSubject: {letter.subject}\n\n{letter.body}"
        issues += models.structured(Check, CHECKER, user).issues
        return {"issues": issues}

    def route(state: State) -> str:
        return "finalize" if not state["issues"] or state["round"] >= MAX_ROUNDS else "write"

    def finalize(state: State) -> State:
        letter = state["letter"]
        return {
            "draft": MessageDraft(
                kind=kind,
                subject=letter.subject.strip(),
                body=letter.body.strip(),
                issues=state["issues"],
                rounds=state["round"],
            )
        }

    def beating(step: Callable[[State], State]) -> Callable[[State], State]:
        def run(state: State) -> State:
            heartbeat()
            return step(state)

        return run

    graph = StateGraph(State)
    for step in (write, check, finalize):
        graph.add_node(step.__name__, beating(step))
    graph.add_edge(START, "write")
    graph.add_edge("write", "check")
    graph.add_conditional_edges("check", route, {"write": "write", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile().invoke({"issues": []})["draft"]
