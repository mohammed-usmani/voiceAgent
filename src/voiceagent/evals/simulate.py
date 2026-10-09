"""A screening call in text mode: the real agent (prompt builder, disclosure, end_call,
llm_node) through LiveKit's AgentSession, against a model playing the persona."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from livekit.agents import AgentSession, llm

from voiceagent.call.prompt import build_instructions, greeting
from voiceagent.call.screening import ScreeningAgent, transcript
from voiceagent.db.models import Candidate, Job
from voiceagent.evals.cost import Meter
from voiceagent.evals.jobs import RefJob
from voiceagent.evals.personas import Persona

# Candidate turns before the call is cut off: a full screening takes about 10.
MAX_TURNS = 25
END = "[END]"


class Simulator(Protocol):
    def reply(self, persona: Persona, job: RefJob, said: list[dict]) -> str | None:
        """What the candidate says next, "" if they said nothing, or None to hang up."""


@dataclass
class SimCall:
    end_reason: str
    callback_at: str | None
    transcript: list[dict]  # as the agent stores it on the call
    candidate_turns: int
    # why the harness ended a call the agent didn't: turn_limit, simulator_end, simulator_empty
    stopped_by: str | None = None


CANDIDATE = """You are playing a job candidate on a phone screening call, to test a recruiter's
AI assistant. Stay in character and never say you are simulated or an AI.
You are {name}, and you applied for the {title} role at {company}.
Your résumé:
{resume}
How you behave: {behaviour}
Your true answers, to give in your own words when asked (never read this list out):
{answers}
Reply with only what you say next: one to three short spoken sentences, no stage
directions. Once the assistant has said goodbye, reply with exactly {end}."""


def candidate_prompt(persona: Persona, job: RefJob) -> str:
    labels = {r.id: r.label for r in job.criteria.requirements}
    answers = "\n".join(
        f"- {labels[rid]}: "
        + ("you'd rather not say" if rid in persona.declined else json.dumps(v))
        for rid, v in persona.truth.items()
        if rid in labels and (v is not None or rid in persona.declined)
    )
    return CANDIDATE.format(
        name=persona.name,
        title=job.title,
        company=job.company,
        resume=persona.resume,
        behaviour=persona.behaviour,
        answers=answers or "- (none)",
        end=END,
    )


class LLMSimulator:
    """The candidate, played by a chat model (LangChain), at temperature 0."""

    def __init__(self, chat) -> None:
        self._chat = chat

    def reply(self, persona: Persona, job: RefJob, said: list[dict]) -> str | None:
        messages = [("system", candidate_prompt(persona, job))] + [
            ("human" if t["role"] == "agent" else "ai", t["text"]) for t in said
        ]
        text = str(self._chat.invoke(messages).content).strip().strip('"')
        return None if END in text else text


def eval_job_row(job: RefJob) -> Job:
    """The job as the agent's prompt builder reads it (never saved)."""
    return Job(
        title=job.title,
        company_name=job.company,
        description=job.description,
        recruiter_notes=job.notes,
        agent_name="Aria",
        timezone="Asia/Kolkata",
        response_days=3,
    )


async def simulate_call(
    job: RefJob,
    persona: Persona,
    *,
    agent_llm: llm.LLM,
    backup_llm: llm.LLM,
    simulator: Simulator,
    now: datetime,
    meter: Meter | None = None,
    max_turns: int = MAX_TURNS,
) -> SimCall:
    """`now` is the call's local time (aware, in the job's timezone)."""
    row = eval_job_row(job)
    candidate = Candidate(
        name=persona.name, email="candidate@example.com", resume_text=persona.resume
    )
    agent = ScreeningAgent(
        instructions=build_instructions(row, candidate, job.criteria, now=now),
        greeting=greeting(row, candidate),
        timezone=row.timezone,
        backup=backup_llm,
        clock=lambda: now,
    )
    turns, stopped_by = 0, None
    async with AgentSession(llm=agent_llm) as session:
        await (await session.start(agent, capture_run=True))
        while agent.outcome is None and turns < max_turns:
            if meter is not None:
                meter.check()
            said = transcript(session.history)
            line = await asyncio.to_thread(simulator.reply, persona, job, said)
            if not line:
                stopped_by = "simulator_empty" if line == "" else "simulator_end"
                break
            turns += 1
            await session.run(user_input=line)
        if agent.outcome is None and stopped_by is None:
            stopped_by = "turn_limit"
        said = transcript(session.history)
    outcome = agent.outcome
    return SimCall(
        end_reason=outcome.reason if outcome else "candidate_hung_up",
        callback_at=outcome.callback_at.isoformat() if outcome and outcome.callback_at else None,
        transcript=said,
        candidate_turns=turns,
        stopped_by=stopped_by,
    )
