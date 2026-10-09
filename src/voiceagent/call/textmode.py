"""Run the real screening agent as text through LiveKit's AgentSession.run(): no audio,
no room; the same prompt, greeting, end_call and llm_node as a phone call.

The tests drive it with a scripted model; the evaluation suite (milestone 4) will drive
it with a simulated candidate.
"""

from dataclasses import dataclass

from livekit.agents import AgentSession, llm
from livekit.agents.voice.run_result import RunEvent

from voiceagent.call.screening import ScreeningAgent, transcript
from voiceagent.call.store import CallOutcome


@dataclass
class TextCall:
    outcome: CallOutcome | None
    transcript: list[dict]
    events: list[RunEvent]


async def run_text_call(
    agent: ScreeningAgent, model: llm.LLM, candidate_says: list[str]
) -> TextCall:
    """Start the call (the agent greets), then send the candidate's lines one at a time
    until they run out or the agent ends the call."""
    async with AgentSession(llm=model) as session:
        greeting = await session.start(agent, capture_run=True)
        events = list((await greeting).events)
        for line in candidate_says:
            if agent.outcome is not None:
                break
            events += (await session.run(user_input=line)).events
        return TextCall(agent.outcome, transcript(session.history), events)
