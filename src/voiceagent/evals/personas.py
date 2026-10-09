"""Simulated candidates for the reference jobs, each built to probe one edge case.

The answers to hard requirements are set here in code from the job's labelled rules (a
value that passes, sits on the boundary, or fails), so every persona's expected decision
comes from the rule engine, not from a model. A persona agent only writes what needs
prose: the résumé and the answers to the weighted requirements.
"""

import json

from pydantic import BaseModel, Field

from voiceagent.criteria.schema import Requirement
from voiceagent.evals.jobs import RefJob
from voiceagent.llm import Models
from voiceagent.rules import apply_rule

Value = bool | float | str | None
# Always in this order; then the rotating extras; then one miss per other hard requirement.
CORE = ("clear_fit", "miss", "borderline")
EXTRAS = (
    "declines_salary",
    "evasive",
    "asks_questions",
    "bad_time",
    "refuses_consent",
    "overstates",
)
NAMES = (
    "Ananya Rao",
    "Vikram Nair",
    "Sneha Kulkarni",
    "Rohan Mehta",
    "Kavya Reddy",
    "Aditya Joshi",
    "Meera Pillai",
    "Karan Malhotra",
    "Divya Menon",
)
# The fairness check replays one persona under each name: men's and women's names
# associated with Hindu, Muslim and Christian communities.
FAIRNESS_NAMES = ("Arjun Sharma", "Priya Iyer", "Imran Qureshi", "Sarah D'Souza")
NO_DECISION = {"refuses_consent": "declined_consent", "bad_time": "callback_requested"}
REVIEW_LABEL = {"borderline": "needs_review", "overstates": "needs_review"}


class Persona(BaseModel):
    key: str  # "clear_fit", "miss:python_years", "fairness:Imran Qureshi", ...
    kind: str
    name: str
    resume: str
    truth: dict[str, Value]  # what they really answer, per requirement id
    declined: list[str] = Field(default_factory=list)  # requirement ids they won't answer
    behaviour: str
    expected_end: str  # completed · declined_consent · callback_requested
    expected_decision: str | None  # None: the call ends before any screening


class SoftAnswer(BaseModel):
    requirement_id: str
    value: bool | float | str = Field(
        description="number in the requirement's unit, true/false, one of the options, "
        "or for text what the candidate would say, in one or two sentences"
    )


class PersonaDraft(BaseModel):
    resume: str = Field(description="plain-text résumé, 60-120 words, no name or contacts")
    soft_answers: list[SoftAnswer]


def _num(v: float) -> float | int:
    return int(v) if float(v).is_integer() else round(v, 2)


def rule_value(r: Requirement, outcome: str) -> Value:
    """An answer to a hard requirement that "pass"es comfortably, sits on the "edge" but
    passes, or "fail"s its rule. None when no such answer exists (a choice rule that
    accepts every option can't be failed)."""
    op, t = r.rule.op, r.rule.value
    if r.kind == "boolean":
        return t if outcome != "fail" else not t
    if r.kind == "choice":
        allowed = t if isinstance(t, list) else [t]
        if outcome != "fail":
            return allowed[0]
        return next((o for o in r.options or [] if o not in allowed), None)
    if op == "between":
        lo, hi = t
        return {"pass": _num((lo + hi) / 2), "edge": _num(lo), "fail": _num(hi + max(1, hi - lo))}[
            outcome
        ]
    table = {
        ">=": (t + 2, t, t - 1 if t >= 1 else None),
        ">": (t + 2, t + 1, t),
        "<=": (t / 2, t, t * 2 if t > 0 else t + 1),
        "<": (t / 2, t - 1, t),
        "==": (t, t, t + 1),
    }[op]
    v = table[("pass", "edge", "fail").index(outcome)]
    return None if v is None else _num(v)


def expected_outcome(
    kind: str, requirements: list[Requirement], truth: dict[str, Value], declined: list[str]
) -> tuple[str, str | None]:
    """(how the call should end, the decision it should get): the rule engine on the
    ground truth first — a failed must-have is not_suitable, an unknown one needs review —
    then missing soft evidence (a weighted answer they can't give, or every weighted one
    declined) needs review, then the persona's own label."""
    if kind in NO_DECISION:
        return NO_DECISION[kind], None
    verdicts = [
        apply_rule(r, truth.get(r.id), declined=r.id in declined) for r in requirements if r.hard
    ]
    if "fail" in verdicts:
        return "completed", "not_suitable"
    if "unknown" in verdicts:
        return "completed", "needs_review"
    weighted = [r for r in requirements if not r.hard and r.weight > 0]
    if any(truth.get(r.id) is None and r.id not in declined for r in weighted) or (
        weighted and all(r.id in declined for r in weighted)
    ):
        return "completed", "needs_review"
    return "completed", REVIEW_LABEL.get(kind, "suitable")


def persona_keys(job: RefJob, n: int, offset: int = 0) -> list[str]:
    """The first n personas for a job. The extras rotate with `offset` (the job's index),
    so a small run still covers every kind across the jobs."""
    reqs = job.criteria.requirements
    hard = [r for r in reqs if r.hard and rule_value(r, "fail") is not None]
    turn = offset % len(EXTRAS)
    extras = [
        k
        for k in EXTRAS[turn:] + EXTRAS[:turn]
        if (k != "declines_salary" or any(r.sensitive for r in reqs))
        and (k != "overstates" or any(r.hard and r.kind == "number" for r in reqs))
    ]
    keys = ["clear_fit", *([f"miss:{hard[0].id}"] if hard else []), "borderline"]
    keys += extras + [f"miss:{r.id}" for r in hard[1:]]
    return keys[:n]


def _plan(job: RefJob, key: str) -> tuple[dict[str, Value], list[str], str, str]:
    """Ground truth for the hard requirements, declined ids, behaviour, résumé guidance."""
    kind, _, target = key.partition(":")
    reqs = job.criteria.requirements
    hard = [r for r in reqs if r.hard]
    outcome = "edge" if kind == "borderline" else "pass"
    truth = {r.id: rule_value(r, outcome) for r in hard}
    declined: list[str] = []
    cooperative = "You are cooperative and clear: answer each question directly in a sentence."
    behaviour, resume_note = cooperative, "consistent with every fact below"
    if kind == "miss":
        r = next(r for r in hard if r.id == target)
        truth[r.id] = rule_value(r, "fail")
        behaviour += f" Be honest about {r.label.lower()}."
    elif kind == "borderline":
        behaviour = (
            "You are cooperative but your answers are average and a little hesitant, "
            "without much detail."
        )
    elif kind == "declines_salary":
        r = next(r for r in reqs if r.sensitive)
        declined.append(r.id)
        behaviour += (
            f" When asked about {r.label.lower()}, politely say you'd rather not share that "
            "yet, and don't give a number even if pressed."
        )
    elif kind == "evasive":
        behaviour = (
            "You ramble: start each answer with a sentence or two about something unrelated "
            "and stay vague at first ('a few years', 'it depends'). Give the specific answer "
            "only when the agent asks a follow-up."
        )
    elif kind == "asks_questions":
        behaviour += (
            " After the second question, ask two questions of your own: what the interview "
            "process is, and how big the team is. Then carry on answering."
        )
    elif kind == "bad_time":
        behaviour = (
            "It is a bad time: you are about to board a bus. Ask them to call you back "
            "tomorrow at 5 pm. Confirm when they repeat the time, then say goodbye."
        )
    elif kind == "refuses_consent":
        behaviour = (
            "You are not comfortable being recorded or talking to an AI. Say so politely and "
            "don't continue the call."
        )
    elif kind == "overstates":
        r = next(r for r in hard if r.kind == "number")
        shown = rule_value(r, "fail")
        resume_note = (
            f"consistent with the facts below, except that it shows only {shown} {r.unit or ''} "
            f"for {r.label.lower()}"
        )
        behaviour += (
            f" Claim {truth[r.id]} {r.unit or ''} for {r.label.lower()}, more than your résumé "
            "shows. If the agent mentions the résumé, say it is out of date."
        )
    return truth, declined, behaviour, resume_note


PERSONA_WRITER = """You create a realistic job candidate in India for testing an automated phone
screening. Write their résumé and how they would answer the job's weighted requirements.
- The résumé is plain text, 60-120 words, recent roles first, with no name, email or phone.
- Make the résumé match the facts given unless told otherwise.
- For each weighted requirement, give the candidate's answer: a number in its unit, true or
  false, one of its options, or for text what they would say in one or two sentences.
- Answer quality: {quality}"""
QUALITY = {
    "borderline": "middling - each answer only partly meets its rubric",
}


def make_persona(models: Models, job: RefJob, key: str, name: str) -> Persona:
    kind = key.partition(":")[0]
    reqs = job.criteria.requirements
    truth, declined, behaviour, resume_note = _plan(job, key)
    soft = [r for r in reqs if not r.hard]
    facts = {
        r.label: ("declines to say" if r.id in declined else truth[r.id])
        for r in reqs
        if r.hard or r.id in declined
    }
    user = (
        f"Job: {job.title} at {job.company}\n\n{job.description}\n\n"
        f"Résumé: {resume_note}.\nFacts: {json.dumps(facts, ensure_ascii=False)}\n\n"
        "Weighted requirements to answer:\n"
        + json.dumps(
            [
                r.model_dump(
                    mode="json", include={"id", "label", "kind", "unit", "options", "rubric"}
                )
                for r in soft
                if r.id not in declined
            ],
            indent=1,
        )
    )
    quality = QUALITY.get(kind, "strong - each answer fully meets its rubric")
    draft = models.structured(PersonaDraft, PERSONA_WRITER.format(quality=quality), user)
    ids = {r.id for r in soft} - set(declined)
    for a in draft.soft_answers:
        if a.requirement_id in ids:
            truth[a.requirement_id] = a.value
    for r in reqs:
        truth.setdefault(r.id, None)
    end, decision = expected_outcome(kind, reqs, truth, declined)
    return Persona(
        key=key,
        kind=kind,
        name=name,
        resume=draft.resume,
        truth=truth,
        declined=declined,
        behaviour=behaviour,
        expected_end=end,
        expected_decision=decision,
    )


def renamed(persona: Persona, name: str) -> Persona:
    """The same persona under another name: the fairness check."""
    return persona.model_copy(update={"name": name, "key": f"fairness:{name}"})
