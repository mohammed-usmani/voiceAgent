"""LiveKit's turn detector, plus one rule: a bare yes or no to a question is a finished turn.

The audio model judges a turn by how it sounds. On phone audio a flat "No." often sounds
unfinished, so the agent waited the full endpointing delay (about 2 s) before replying.
The words settle it: a lone "No." right after the agent asked a question is an answer.

The rule only ever raises the audio model's end-of-turn probability, never lowers it.
The audio prediction usually lands before the caller's transcript, so when the audio
model is unsure and no transcript is in yet, the rule waits up to FINAL_TRANSCRIPT_WAIT_S
for it. That costs nothing: LiveKit's end-of-turn wait counts from when speech stopped,
and an unsure turn waits max_delay (2 s) anyway.
"""

import asyncio
import dataclasses
import logging
import re
from collections.abc import Callable

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, AgentSession, APIConnectOptions, inference
from livekit.agents.voice.turn import TurnDetectionEvent

logger = logging.getLogger(__name__)

# Answers that mean yes or no on their own. "okay", "right" and "so" are left out:
# callers often say them before continuing ("Okay... so my current role is...").
_SHORT_ANSWER = re.compile(
    r"(yes|yeah|yep|yup|no|nope|nah|sure|correct|absolutely|definitely|not really|of course)"
    r"( (yes|yeah|no|sure))*"
)
DONE_PROBABILITY = 0.95
# Deepgram's final transcript landed 0.3-0.6 s after the caller stopped speaking.
FINAL_TRANSCRIPT_WAIT_S = 0.6


def is_short_answer(caller_text: str, agent_text: str) -> bool:
    """A bare yes/no-style answer, given right after the agent asked a question."""
    words = re.sub(r"[^\w\s']", " ", caller_text.lower()).split()
    asked = agent_text.rstrip().endswith("?")
    return asked and bool(words) and _SHORT_ANSWER.fullmatch(" ".join(words)) is not None


class CallerText:
    """The caller's final transcripts since their last committed turn."""

    def __init__(self) -> None:
        self._finals: list[str] = []
        self._arrived = asyncio.Event()

    def add_final(self, text: str) -> None:
        self._finals.append(text)
        self._arrived.set()

    async def wait_for_final(self, timeout: float) -> None:
        if self._finals:
            return
        self._arrived.clear()
        try:
            await asyncio.wait_for(self._arrived.wait(), timeout)
        except TimeoutError:
            pass

    def reset(self) -> None:
        self._finals = []

    def text(self) -> str:
        return " ".join(self._finals).strip()


class ShortAnswerStream:
    """Wraps LiveKit's turn-detector stream; implements livekit's _StreamingTurnDetectorStream."""

    def __init__(self, inner, caller: CallerText, last_agent_text: Callable[[], str]) -> None:
        self._inner = inner
        self._caller = caller
        self._last_agent_text = last_agent_text
        self._task: asyncio.Task | None = None
        self._out: asyncio.Future | None = None

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def is_fallback(self) -> bool:
        return self._inner.is_fallback

    @property
    def prediction_timeout(self) -> float:
        return self._inner.prediction_timeout + FINAL_TRANSCRIPT_WAIT_S

    async def unlikely_threshold(self, language) -> float | None:
        return await self._inner.unlikely_threshold(language)

    async def backchannel_threshold(self, language) -> float | None:
        return await self._inner.backchannel_threshold(language)

    async def supports_language(self, language) -> bool:
        return await self._inner.supports_language(language)

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        audio = self._inner.predict()
        out: asyncio.Future[TurnDetectionEvent] = asyncio.get_running_loop().create_future()
        self._out = out
        self._task = asyncio.create_task(self._decide(audio, out))
        return out

    async def _decide(self, audio_fut, out: asyncio.Future) -> None:
        try:
            event = await asyncio.shield(audio_fut)
        except asyncio.CancelledError:
            out.cancel()
            return
        except Exception as e:
            if not out.done():
                out.set_exception(e)
            return

        p = event.end_of_turn_probability
        if p < DONE_PROBABILITY:
            await self._caller.wait_for_final(FINAL_TRANSCRIPT_WAIT_S)
        text = self._caller.text()
        if p < DONE_PROBABILITY and is_short_answer(text, self._last_agent_text()):
            logger.info("short answer %r: end of turn %.2f -> %.2f", text, p, DONE_PROBABILITY)
            event = dataclasses.replace(event, end_of_turn_probability=DONE_PROBABILITY)
        if not out.done():
            out.set_result(event)

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        if self._out is not None and not self._out.done():
            self._out.cancel()  # LiveKit may check this right away
        if self._task is not None:
            self._task.cancel()
        self._inner.cancel_inference(timed_out=timed_out)

    def flush(self, reason: str | None = None) -> None:
        self._inner.flush(reason)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._inner.push_audio(frame)

    def end_input(self) -> None:
        self._inner.end_input()

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
        await self._inner.aclose()


class ShortAnswerTurnDetector(inference.TurnDetector):
    """LiveKit's audio turn detector (same model and settings), plus the short-answer rule."""

    def __init__(self, caller: CallerText, last_agent_text: Callable[[], str]) -> None:
        super().__init__()
        self._caller = caller
        self._last_agent_text = last_agent_text

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ShortAnswerStream:
        inner = super().stream(conn_options=conn_options)
        return ShortAnswerStream(inner, self._caller, self._last_agent_text)


def last_agent_text(session: AgentSession) -> str:
    for item in reversed(session.history.messages()):
        if item.role == "assistant":
            return item.text_content or ""
    return ""


def watch_caller(session: AgentSession, caller: CallerText) -> None:
    @session.on("user_input_transcribed")
    def _transcribed(ev) -> None:
        if ev.is_final:
            caller.add_final(ev.transcript)

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        if getattr(ev.item, "role", None) == "user":  # turn committed; start fresh
            caller.reset()
