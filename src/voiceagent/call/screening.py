"""The screening call's agent: says the disclosure without the LLM, then holds the
conversation its prompt describes, with one tool, end_call."""

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, ToolError, function_tool, llm

from voiceagent.call.store import CallOutcome
from voiceagent.replies import ReliableAgent

# A callback further out than this is more likely a misheard date than a plan.
MAX_CALLBACK_DAYS = 60

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
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        detect_machine: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        super().__init__(instructions=instructions, backup=backup)
        self.greeting = greeting
        self.outcome: CallOutcome | None = None
        self.error: str | None = None  # why the session failed, if it did (agent.py)
        self._tz = ZoneInfo(timezone)
        self._hang_up = hang_up
        self._clock = clock
        # Listens to the line before the greeting: True when a machine answered (agent.py
        # runs LiveKit's answering-machine detection, and drops any reply to a "Hello?"
        # heard meanwhile). None: always greet (text mode).
        self._detect_machine = detect_machine

    async def on_enter(self) -> None:
        if self._detect_machine is not None:
            if await self._detect_machine():
                # Voicemail or an automated menu: leave no message, just hang up; a
                # campaign tries again later.
                self.outcome = CallOutcome("voicemail")
                if self._hang_up is not None:
                    self._hang_up()
                return
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
        when = self._callback_time(callback_at) if reason == "callback_requested" else None
        if self.outcome is None:  # the first decision stands
            self.outcome = CallOutcome(reason, when)
        if self._hang_up is not None:
            # Hang up once this reply (the goodbye) has been spoken, as LiveKit's own
            # EndCallTool does. Returning None asks the model for no further reply.
            hang_up = self._hang_up
            ctx.speech_handle.add_done_callback(lambda _handle: hang_up())

    def _callback_time(self, callback_at: str | None) -> datetime:
        """The callback time the model gave, or a ToolError it can correct and retry."""
        text = (callback_at or "").strip()
        try:
            date.fromisoformat(text)
        except ValueError:
            pass
        else:  # a date alone would be midnight
            raise ToolError(
                "callback_at needs a time of day too: ask them what time suits them on that day"
            )
        try:
            when = datetime.fromisoformat(text)
        except ValueError:
            raise ToolError(
                "callback_at must be the local date and time they gave, like 2026-10-10T17:00"
            ) from None
        if when.tzinfo is None:
            when = when.replace(tzinfo=self._tz)
        now = self._clock()
        if when <= now:
            raise ToolError("that callback time has already passed: confirm the date and time")
        if when > now + timedelta(days=MAX_CALLBACK_DAYS):
            raise ToolError(
                f"a callback must be within {MAX_CALLBACK_DAYS} days: confirm the date with them"
            )
        return when
