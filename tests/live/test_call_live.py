"""A short text-mode screening call against the real voice model (DeepSeek-V4.1-Flash on
Together, backup MiniMax-M3). A few cheap LLM calls:
RUN_LIVE=1 uv run pytest tests/live/test_call_live.py -v -s"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from dotenv import load_dotenv

from tests.seed import CRITERIA, sample_candidate, sample_job
from voiceagent.call.prompt import build_instructions, greeting
from voiceagent.call.screening import ScreeningAgent
from voiceagent.call.textmode import run_text_call
from voiceagent.replies import main_llm

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1"),
]
load_dotenv()


def agent() -> ScreeningAgent:
    job, candidate = sample_job(), sample_candidate()
    return ScreeningAgent(
        instructions=build_instructions(
            job, candidate, CRITERIA, now=datetime.now(ZoneInfo(job.timezone))
        ),
        greeting=greeting(job, candidate),
        timezone=job.timezone,
    )


async def test_declining_the_recording_ends_the_call():
    result = await run_text_call(
        agent(),
        main_llm(),
        ["No. I'm not okay with an AI recording me, please stop.", "No, goodbye."],
    )
    print(*result.transcript, sep="\n")
    assert result.outcome is not None and result.outcome.reason == "declined_consent"


async def test_bad_time_books_a_callback():
    result = await run_text_call(
        agent(),
        main_llm(),
        [
            "I'm driving right now, can you call me later?",
            "Tomorrow at 5 pm works.",
            "Yes, that's right. Bye.",
        ],
    )
    print(*result.transcript, sep="\n")
    assert result.outcome is not None and result.outcome.reason == "callback_requested"
    assert result.outcome.callback_at.hour == 17
