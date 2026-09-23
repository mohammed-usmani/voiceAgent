"""End-of-turn detection backed by Jev (Typesafe AI).

Silence alone can't tell "I need help with my..." (mid-thought pause) from
"My laptop won't boot." (finished turn). Instead of a fixed silence timeout,
each candidate turn is sent to Jev with the recent dialogue, and Jev classifies
it as a real turn, a mid-sentence pause, or a backchannel ("yeah", "uh-huh").
That classification is mapped to the end-of-turn probability LiveKit uses to
choose between its short and long endpointing delays.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from livekit.agents import llm

logger = logging.getLogger(__name__)

TurnAction = Literal["respond", "wait_pause", "backchannel"]

JEV_BASE_URL = "https://api.typesafe.ai"
JEV_MODEL = "jev-latest"
REQUEST_TIMEOUT_S = 1.5
HISTORY_WINDOW = 4  # prior messages sent as context; kept small for latency
RESPOND_THRESHOLD = 0.70  # min Jev completeness to treat a "respond" as a finished turn
HOLD_PROBABILITY = 0.2  # returned for pauses/backchannels; must stay below unlikely_threshold

QUESTIONS = {
    "is_turn_complete": {
        "type": "noul",
        "instructions": "Determine whether the speaker has concluded an active turn and expects an agent response.",
        "criteria": {
            "true": "The user asked a question, made a substantive statement, or gave an instruction requiring an agent response.",
            "false": "The user is merely acknowledging (e.g., 'right', 'yeah', 'uh-huh') while the agent was speaking, or is pausing mid-thought/filler.",
        },
    },
    "turn_action": {
        "type": "choice",
        "instructions": "Which conversational action best fits the state?",
        "criteria": {
            "respond": "The speaker completed their thought or question. The assistant must speak now.",
            "wait_pause": "The speaker paused mid-sentence, used a filler word, or is formulating a continuation.",
            "backchannel": "The speaker gave a short confirmation/listening cue while the agent was speaking, not a turn-take.",
        },
    },
}


@dataclass(frozen=True)
class TurnDecision:
    action: TurnAction
    completeness: float

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "TurnDecision":
        answers = data.get("answers", {})
        return cls(
            action=answers.get("turn_action", {}).get("choice", "respond"),
            completeness=float(answers.get("is_turn_complete", {}).get("noul", 1.0)),
        )

    def end_of_turn_probability(self) -> float:
        if self.action == "respond" and self.completeness >= RESPOND_THRESHOLD:
            return self.completeness
        # Pause or backchannel: stay under unlikely_threshold so LiveKit waits up to max_delay.
        return min(self.completeness, HOLD_PROBABILITY)


# Fail open: if Jev is unreachable the agent still answers, just without smart endpointing.
FALLBACK = TurnDecision(action="respond", completeness=1.0)


class JevClient:
    def __init__(self, api_key: str | None = None) -> None:
        self._http = httpx.AsyncClient(
            base_url=JEV_BASE_URL,
            headers={"Authorization": f"Bearer {api_key or os.environ['TYPESAFE_API_KEY']}"},
            timeout=REQUEST_TIMEOUT_S,
        )

    async def evaluate_turn(
        self,
        transcript: str,
        history: list[dict[str, str]],
        silence_ms: int,
        assistant_was_speaking: bool = False,
    ) -> TurnDecision:
        state = {
            "dialogue_history": history,
            "recent_transcript": transcript,
            "silence_duration_ms": silence_ms,
            "assistant_was_speaking": assistant_was_speaking,
        }
        try:
            resp = await self._http.post(
                "/v1/systemone",
                json={"model": JEV_MODEL, "state": state, "questions": QUESTIONS},
            )
            resp.raise_for_status()
            return TurnDecision.from_response(resp.json())
        except (httpx.HTTPError, ValueError) as e:  # ValueError: malformed JSON body
            logger.warning("Jev request failed, falling back to respond: %r", e)
            return FALLBACK

    async def aclose(self) -> None:
        await self._http.aclose()


class JevTurnDetector:
    """Implements LiveKit's turn-detector protocol (livekit.agents.voice.turn._TurnDetector)."""

    def __init__(
        self,
        client: JevClient,
        *,
        silence_ms: int,
        unlikely_threshold: float = 0.5,
    ) -> None:
        self._client = client
        self._silence_ms = silence_ms
        self._unlikely_threshold = unlikely_threshold

    @property
    def model(self) -> str:
        return JEV_MODEL

    @property
    def provider(self) -> str:
        return "typesafe"

    async def supports_language(self, language: object) -> bool:
        return True

    async def unlikely_threshold(self, language: object) -> float:
        return self._unlikely_threshold

    async def predict_end_of_turn(
        self, chat_ctx: llm.ChatContext, *, timeout: float | None = None
    ) -> float:
        messages = chat_ctx.messages()
        if not messages:
            return 1.0

        *earlier, current = messages
        transcript = current.text_content or ""
        history = [
            {"speaker": m.role, "text": m.text_content or ""} for m in earlier[-HISTORY_WINDOW:]
        ]

        decision = await self._client.evaluate_turn(transcript, history, self._silence_ms)
        prob = decision.end_of_turn_probability()
        logger.info(
            "turn %r -> action=%s completeness=%.2f p_eot=%.2f",
            transcript,
            decision.action,
            decision.completeness,
            prob,
        )
        return prob
