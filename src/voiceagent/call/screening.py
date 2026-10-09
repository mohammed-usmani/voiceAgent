"""The screening call's agent: says the disclosure without the LLM, then holds the
conversation its prompt describes, with one tool, end_call."""

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, ToolError, function_tool, llm

from voiceagent.call.store import CallOutcome
from voiceagent.replies import ReliableAgent

EndReason = Literal["completed", "declined_consent", "callback_requested", "not_interested"]


def call_id_from(metadata: str) -> uuid.UUID | None:
    """The screening call a dispatch is for (outbound.dial sends {"call_id": ...}).

    No metadata means a demo call. Metadata that doesn't name a call raises, so a broken
    dispatch never reaches a real candidate as the demo recruiter.
    """
    if not metadata:
        return None
    return uuid.UUID(json.loads(metadata)["call_id"])


def transcript(history: llm.ChatContext) -> list[dict]:
    """What was said, in order, as stored on the call."""
    return [
        {
            "role": "agent" if item.role == "assistant" else "candidate",
            "text": item.text_content,
            "at": datetime.fromtimestamp(item.created_at, UTC).isoformat(),
        }
        for item in history.items
        if item.type == "message" and item.role in ("assistant", "user") and item.text_content
    ]


class ScreeningAgent(ReliableAgent):
    def __init__(
        self,
        *,
        instructions: str,
        greeting: str,
        timezone: str,
        hang_up: Callable[[], object] | None = None,
        backup: llm.LLM | None = None,
    ) -> None:
        super().__init__(instructions=instructions, backup=backup)
        self.greeting = greeting
        self.outcome: CallOutcome | None = None
        self.error: str | None = None  # why the session failed, if it did (agent.py)
        self._tz = ZoneInfo(timezone)
        self._hang_up = hang_up

    async def on_enter(self) -> None:
        # Fixed text, so no LLM: the AI and recording disclosure is said word for word on
        # every call, and the first words come without model latency. Uninterruptible: a
        # "Hello?" over it must not cut the disclosure short, since the prompt tells the
        # model it was said in full.
        self.session.say(self.greeting, allow_interruptions=False)

    @function_tool
    async def end_call(
        self, ctx: RunContext, reason: EndReason, callback_at: str | None = None
    ) -> None:
        """End the call. Use it in the same reply as your last sentence, never before
        saying goodbye.

        Args:
            reason: completed: all questions are done and you said the closing line.
                declined_consent: they object to talking to an AI or to the recording.
                callback_requested: it's a bad time and they gave a time to call back.
                not_interested: they no longer want the role.
            callback_at: Only with callback_requested: the local date and time to call
                back, like 2026-10-10T17:00.
        """
        when = None
        if reason == "callback_requested":
            try:
                when = datetime.fromisoformat(callback_at or "")
            except ValueError:
                raise ToolError(
                    "callback_at must be the local date and time they gave, like 2026-10-10T17:00"
                ) from None
            if when.tzinfo is None:
                when = when.replace(tzinfo=self._tz)
        if self.outcome is None:  # the first decision stands
            self.outcome = CallOutcome(reason, when)
        if self._hang_up is not None:
            # Hang up once this reply (the goodbye) has been spoken, as LiveKit's own
            # EndCallTool does. Returning None asks the model for no further reply.
            hang_up = self._hang_up
            ctx.speech_handle.add_done_callback(lambda _handle: hang_up())
