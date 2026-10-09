"""Stand-ins for every model, so `screening-eval --offline` runs the whole pipeline (the
real agent, assessment, message agent, metrics and report) with no API calls: a smoke
test for CI, not a measurement.

The voice model asks the job's questions in order; the candidate answers each from the
persona's ground truth; the backend models answer by schema name, so they don't depend on
the agents' internal classes.
"""

import json
from datetime import datetime, timedelta

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from voiceagent.evals.jobs import RefJob
from voiceagent.evals.personas import Persona

OFFLINE = "offline"
CLOSING = "Thanks. The team will review and get back to you by email within 3 days."
REFUSE = "No, I'm not okay with being recorded by an AI."
BUSY = "Sorry, I'm busy right now. Can you call me back tomorrow at 5 pm?"


def answer_line(persona: Persona, requirement_id: str, kind: str, unit: str | None) -> str:
    """What the offline candidate says for a requirement; also the evidence's quote."""
    if requirement_id in persona.declined:
        return "I'd rather not say for now."
    value = persona.truth.get(requirement_id)
    if value is None:
        return "I'm not sure, sorry."
    if isinstance(value, bool):
        return "Yes, that works for me." if value else "No, that doesn't work for me."
    if kind == "number":
        return f"{value} {unit or ''}".strip() + "."
    return f"{value}".rstrip(".") + "."


class OfflineSimulator:
    def reply(self, persona: Persona, job: RefJob, said: list[dict]) -> str | None:
        if not any(t["role"] == "candidate" for t in said):
            return {"refuses_consent": REFUSE, "bad_time": BUSY}.get(persona.kind, "Yes, sure.")
        last = next((t["text"] for t in reversed(said) if t["role"] == "agent"), "")
        for r in job.criteria.requirements:
            if r.question in last:
                return answer_line(persona, r.id, r.kind, r.unit)
        return None  # nothing asked: hang up


class OfflineAgentLLM(llm.LLM):
    """Asks each question in order after the candidate's first reply, then closes."""

    def __init__(self, job: RefJob, now: datetime) -> None:
        super().__init__()
        self._questions = [r.question for r in job.criteria.requirements]
        self._now = now

    @property
    def model(self) -> str:
        return OFFLINE

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **_kw):
        said = [i for i in chat_ctx.items if i.type == "message" and i.role == "user"]
        last = said[-1].text_content if said else ""
        if last == REFUSE:
            reply = ["Understood. The team will follow up by email.", _end_call("declined_consent")]
        elif last == BUSY:
            when = (self._now + timedelta(days=1)).replace(hour=17, minute=0, second=0)
            reply = [
                "Sure, I'll call you tomorrow at 5 pm.",
                _end_call("callback_requested", when.strftime("%Y-%m-%dT%H:%M")),
            ]
        elif len(said) <= len(self._questions):
            reply = self._questions[len(said) - 1]
        else:
            reply = [CLOSING, _end_call("completed")]
        return _Stream(self, reply, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def _end_call(reason: str, callback_at: str | None = None) -> llm.FunctionToolCall:
    args = {"reason": reason} | ({"callback_at": callback_at} if callback_at else {})
    return llm.FunctionToolCall(
        name="end_call", arguments=json.dumps(args), call_id=f"end_{reason}"
    )


class _Stream(llm.LLMStream):
    def __init__(self, owner: llm.LLM, reply, **kwargs) -> None:
        super().__init__(owner, **kwargs)
        self._reply = reply

    async def _run(self) -> None:
        for i, part in enumerate(self._reply if isinstance(self._reply, list) else [self._reply]):
            if isinstance(part, str):
                delta = llm.ChoiceDelta(role="assistant", content=part)
            else:
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[part])
            self._event_ch.send_nowait(llm.ChatChunk(id=f"offline_{i}", delta=delta))


class OfflineModels:
    """Backend models for one job (and persona): every requirement extracted exactly as
    the persona said it, soft answers scored 1.0 (0.55 for a borderline persona, unless a
    recruiter correction came with it: the corrections all call such answers suitable),
    every check passed."""

    name = OFFLINE

    def __init__(self, job: RefJob, persona: Persona | None = None) -> None:
        self.job, self.persona = job, persona
        self.reviewer = self

    def structured(self, schema, system: str, user: str):
        reqs = self.job.criteria.requirements
        p = self.persona
        name = schema.__name__
        if name == "Draft":
            data = {"criteria": self.job.criteria.model_dump(mode="json")}
        elif name == "CriteriaMatch":
            data = {"pairs": [{"generated_id": r.id, "labelled_id": r.id} for r in reqs]}
        elif name == "PersonaDraft":
            middling = "middling" in system
            guidance = user.split("Résumé: ", 1)[1].split("\n", 1)[0]
            data = {
                "resume": f"{self.job.title} candidate with relevant work; {guidance}",
                "soft_answers": [
                    {"requirement_id": r.id, "value": _soft_value(r, middling)}
                    for r in reqs
                    if not r.hard
                ],
            }
        elif name == "EvidenceSet":
            data = {"answers": [_evidence(p, r, user) for r in reqs]}
        elif name == "SoftScores":
            blocks = json.loads(user.split("\n", 1)[1].split("\n\nTranscript:")[0])
            corrected = {b["id"] for b in blocks if b.get("recruiter_corrections")}
            data = {
                "scores": [
                    {
                        "requirement_id": r.id,
                        "score": 0.55 if p.kind == "borderline" and r.id not in corrected else 1.0,
                        "quote": answer_line(p, r.id, r.kind, r.unit),
                        "note": "offline",
                    }
                    for r in reqs
                    if not r.hard
                ]
            }
        elif name == "Review":
            data = {"checks": []}
        elif name == "Letter":
            first = p.name.split()[0] if p else "there"
            news = (
                "We won't be moving forward with your application this time."
                if "Email: rejection" in user
                else "The team will be in touch to schedule the next steps."
            )
            data = {
                "subject": f"Your application for {self.job.title}",
                "body": f"Hello {first},\n\nThank you for your time on the call. {news}\n\n"
                f"The {self.job.company} hiring team",
            }
        elif name == "CallJudgement":
            data = {
                "disclosed_ai_and_recording": True,
                "asked_all_required": True,
                "never_reasked": True,
                "never_hinted_decision": True,
                "never_promised": True,
                "answered_from_job_facts_only": True,
                "professional": True,
                "correct_close": True,
                "asked": [r.id for r in reqs if r.question in user],
                "reasked": [],
                "text_answers": [
                    {"requirement_id": r.id, "agrees": True}
                    for r in reqs
                    if r.kind == "text" and f'"requirement_id": "{r.id}"' in user
                ],
            }
        elif name == "MessageJudgement":
            data = {"professional": True, "no_reasons_or_scores": True, "no_promises": True}
        else:  # Critique, Check, ...: an empty report means "no issues"
            data = {}
        return schema.model_validate(data)

    def embed(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


def _soft_value(r, middling: bool):
    if r.kind == "number":
        return 2 if middling else 10
    if r.kind == "boolean":
        return not middling
    if r.kind == "choice":
        return (r.options or ["Other"])[0]
    return "Somewhat, now and then." if middling else (r.rubric or r.label)


def _evidence(p: Persona, r, user: str) -> dict:
    quote = answer_line(p, r.id, r.kind, r.unit)
    if f"Candidate: {quote}" not in user:
        return {"requirement_id": r.id, "value": None, "quote": "", "confidence": 0.0}
    declined = r.id in p.declined
    return {
        "requirement_id": r.id,
        "value": None if declined else p.truth.get(r.id),
        "quote": quote,
        "confidence": 0.9,
        "declined": declined,
    }
