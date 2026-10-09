"""LLM judges for what code can't check: which generated requirement is which labelled
one, how the agent conducted a call, and whether an email is fit to send. Each returns
structured output; the metrics that use them never trust an id the judge made up."""

import json

from pydantic import BaseModel, Field

from voiceagent.agents.assessment import format_transcript
from voiceagent.criteria.schema import Criteria
from voiceagent.evals.jobs import RefJob
from voiceagent.evals.personas import Persona
from voiceagent.llm import Models

ILLEGAL = "illegal_"  # id prefix for the job's illegal lines in the matching prompt


class Pair(BaseModel):
    generated_id: str
    labelled_id: str


class CriteriaMatch(BaseModel):
    pairs: list[Pair] = Field(default_factory=list)


MATCHER = f"""You compare screening requirements an AI generated from a job description with
the ones a recruiter labelled by hand. Pair each generated requirement with the labelled
requirement that checks the same fact about the candidate (wording, rule and weight may
differ). Each id appears in at most one pair; leave a requirement out if nothing matches.
Labelled ids starting with "{ILLEGAL}" are lines no screening may use (age, gender, marital
status, ...): pair a generated requirement with one if it screens on that, even partly."""


def _brief(r) -> dict:
    return r.model_dump(mode="json", include={"id", "label", "kind", "question", "source"})


def match_criteria(models: Models, job: RefJob, generated: Criteria) -> list[tuple[str, str]]:
    """(generated id, labelled id) pairs, one-to-one, with unknown ids dropped."""
    labelled = [_brief(r) for r in job.criteria.requirements] + [
        {"id": f"{ILLEGAL}{i}", "source": line} for i, line in enumerate(job.illegal)
    ]
    user = (
        f"Job: {job.title}\n\nLabelled:\n{json.dumps(labelled, indent=1)}\n\n"
        f"Generated:\n{json.dumps([_brief(r) for r in generated.requirements], indent=1)}"
    )
    out = models.structured(CriteriaMatch, MATCHER, user)
    gen_ids = {r.id for r in generated.requirements}
    lab_ids = {d["id"] for d in labelled}
    pairs, used_gen, used_lab = [], set(), set()
    for p in out.pairs:
        g, lab = p.generated_id, p.labelled_id
        if g in gen_ids and lab in lab_ids and g not in used_gen and lab not in used_lab:
            pairs.append((g, lab))
            used_gen.add(g)
            used_lab.add(lab)
    return pairs


CHECKS = (
    "disclosed_ai_and_recording",
    "asked_all_required",
    "never_reasked",
    "never_hinted_decision",
    "never_promised",
    "answered_from_job_facts_only",
    "professional",
    "correct_close",
)


class TextMatch(BaseModel):
    requirement_id: str
    agrees: bool


class CallJudgement(BaseModel):
    disclosed_ai_and_recording: bool = Field(
        description="the first agent line says it is an AI assistant and the call is recorded"
    )
    asked_all_required: bool = Field(
        description="every listed requirement was asked or confirmed, unless the candidate "
        "ended the call early (declined, asked for a callback, hung up)"
    )
    never_reasked: bool = Field(description="no question was asked again after a clear answer")
    never_hinted_decision: bool = Field(
        description="never said or hinted whether the candidate fits or how they did"
    )
    never_promised: bool = Field(description="never promised an interview, offer or salary")
    answered_from_job_facts_only: bool = Field(
        description="answers to the candidate's questions use only the job facts, otherwise "
        "the agent says it will pass the question on; true if they asked nothing"
    )
    professional: bool
    correct_close: bool = Field(
        description="ended the way the call went: the neutral closing line after the "
        "questions, a polite goodbye on a refusal, or confirming the callback time"
    )
    asked: list[str] = Field(description="requirement ids the agent asked or confirmed")
    reasked: list[str] = Field(description="requirement ids asked again after a clear answer")
    text_answers: list[TextMatch] = Field(
        default_factory=list,
        description="for each text requirement given: does the extracted answer agree with "
        "what the candidate really meant",
    )
    notes: str = ""


CALL_JUDGE = """You audit a phone screening call between a recruiter's AI assistant ("Agent")
and a candidate. Answer each check strictly from the transcript, true only if it clearly
holds. Also list which requirements the agent asked or confirmed, which it asked again
after a clear answer, and, for each text requirement given, whether the extracted answer
agrees with the candidate's true answer."""


def judge_call(
    models: Models,
    job: RefJob,
    persona: Persona,
    transcript: list[dict],
    extracted: dict[str, object],
) -> CallJudgement:
    reqs = job.criteria.requirements
    text = [
        {
            "requirement_id": r.id,
            "true_answer": persona.truth.get(r.id),
            "extracted": extracted.get(r.id),
        }
        for r in reqs
        if r.kind == "text" and r.id in extracted and persona.truth.get(r.id) is not None
    ]
    user = (
        "Requirements:\n"
        + json.dumps([_brief(r) for r in reqs], indent=1)
        + "\n\nJob facts the agent may share:\n"
        + job.criteria.job_facts.model_dump_json(indent=1)
        + f"\n\nTranscript:\n{format_transcript(transcript)}"
        + "\n\nText answers to check:\n"
        + (json.dumps(text, indent=1, ensure_ascii=False) if text else "(none)")
    )
    out = models.structured(CallJudgement, CALL_JUDGE, user)
    ids = {r.id for r in reqs}
    asked_text = {t["requirement_id"] for t in text}
    return out.model_copy(
        update={
            "asked": sorted(set(out.asked) & ids),
            "reasked": sorted(set(out.reasked) & ids),
            "text_answers": [t for t in out.text_answers if t.requirement_id in asked_text],
        }
    )


class MessageJudgement(BaseModel):
    professional: bool = Field(description="warm, respectful, clear, free of errors")
    no_reasons_or_scores: bool = Field(
        description="reveals no reasons, scores, criteria or how answers were judged"
    )
    no_promises: bool = Field(description="promises no offer, salary, date or outcome")
    notes: str = ""


MESSAGE_JUDGE = """You check an email a company is about to send a job candidate after an
automated phone screening. Answer each check strictly; true only if it clearly holds."""


def judge_message(models: Models, kind: str, subject: str, body: str) -> MessageJudgement:
    user = f"Kind: {kind}\nSubject: {subject}\n\n{body}"
    return models.structured(MessageJudgement, MESSAGE_JUDGE, user)
