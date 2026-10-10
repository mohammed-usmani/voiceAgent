import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from livekit.agents import AgentSession, llm

from tests.fakes import ScriptedLLM, tool_call
from tests.seed import CRITERIA, sample_candidate, sample_job
from voiceagent.call.prompt import build_instructions, closing, greeting
from voiceagent.call.screening import ScreeningAgent, call_id_from, transcript
from voiceagent.call.store import CallOutcome
from voiceagent.call.textmode import run_text_call

JOB, CANDIDATE = sample_job(), sample_candidate()
NOW = datetime(2026, 10, 9, 14, 30, tzinfo=ZoneInfo("Asia/Kolkata"))


def make_agent(model: llm.LLM, **kw) -> ScreeningAgent:
    return ScreeningAgent(
        instructions=build_instructions(JOB, CANDIDATE, CRITERIA, now=NOW),
        greeting=greeting(JOB, CANDIDATE),
        timezone=JOB.timezone,
        backup=kw.pop("backup", model),
        clock=lambda: NOW,
        **kw,
    )


def texts(result) -> list[str]:
    return [t["text"] for t in result.transcript]


def test_end_call_is_the_only_tool():
    assert [t.id for t in make_agent(ScriptedLLM([])).tools] == ["end_call"]


async def test_greeting_is_spoken_without_the_llm():
    model = ScriptedLLM([])
    result = await run_text_call(make_agent(model), model, [])
    assert texts(result) == [greeting(JOB, CANDIDATE)]
    assert model.requests == []


async def test_the_disclosure_cannot_be_interrupted(monkeypatch):
    # I3: a "Hello?" over the greeting must not cut off the AI and recording disclosure.
    said = []
    say = AgentSession.say

    def record(self, text, **kw):
        handle = say(self, text, **kw)
        said.append((text, handle.allow_interruptions))
        return handle

    monkeypatch.setattr(AgentSession, "say", record)
    model = ScriptedLLM([])
    await run_text_call(make_agent(model), model, [])
    assert said == [(greeting(JOB, CANDIDATE), False)]


async def test_a_machine_hears_nothing_and_the_call_ends():
    # Review focus 5: voicemail answers. No greeting, nothing generated, so no message is
    # left on it; the call ends as voicemail.
    async def machine() -> bool:
        return True

    model = ScriptedLLM(["This must never be said."])
    hung_up = []
    agent = make_agent(model, detect_machine=machine, hang_up=lambda **kw: hung_up.append(kw))
    result = await run_text_call(agent, model, [])
    assert result.outcome == CallOutcome("voicemail")
    assert result.transcript == []
    assert model.requests == []
    assert hung_up == [{"drain": False}]  # nothing still queued plays into the voicemail


async def test_a_call_nobody_answered_says_nothing_and_leaves_no_outcome():
    async def rang_out() -> None:
        return None

    model = ScriptedLLM(["This must never be said."])
    hung_up = []
    agent = make_agent(model, detect_machine=rang_out, hang_up=lambda **kw: hung_up.append(kw))
    result = await run_text_call(agent, model, [])
    assert (result.outcome, result.transcript, model.requests) == (None, [], [])
    assert hung_up == [{"drain": False}]


async def test_a_person_is_greeted_once_detection_says_so():
    async def person() -> bool:
        return False

    model = ScriptedLLM([])
    result = await run_text_call(make_agent(model, detect_machine=person), model, [])
    assert texts(result) == [greeting(JOB, CANDIDATE)]
    assert result.outcome is None


async def test_declined_consent_ends_the_call_after_the_goodbye():
    model = ScriptedLLM(
        [
            [
                "No problem. The team will follow up by email. Goodbye.",
                tool_call("end_call", reason="declined_consent"),
            ]
        ]
    )
    hung_up = []
    agent = make_agent(model, hang_up=lambda: hung_up.append(True))
    result = await run_text_call(agent, model, ["No, I don't want to talk to a bot.", "Hello?"])
    assert result.outcome == CallOutcome("declined_consent")
    assert hung_up == [True]
    assert [t["role"] for t in result.transcript] == ["agent", "candidate", "agent"]
    assert "reason declined_consent" in model.requests[0].items[0].text_content


async def test_completed_call_ends_with_the_neutral_close():
    close = closing(JOB, CANDIDATE)
    model = ScriptedLLM(
        [
            "Great. How many years have you worked with Python?",
            [close, tool_call("end_call", reason="completed")],
        ]
    )
    result = await run_text_call(
        make_agent(model), model, ["Yes, now is fine.", "About five years."]
    )
    assert result.outcome == CallOutcome("completed")
    assert texts(result)[-1] == close


async def test_callback_time_must_be_a_real_time():
    model = ScriptedLLM(
        [
            ["Sure.", tool_call("end_call", reason="callback_requested", callback_at="tomorrow")],
            [
                "I'll call you tomorrow at five. Bye!",
                tool_call("end_call", reason="callback_requested", callback_at="2026-10-10T17:00"),
            ],
        ]
    )
    result = await run_text_call(make_agent(model), model, ["Busy now. Call me tomorrow at 5."])
    tz = ZoneInfo("Asia/Kolkata")
    assert result.outcome == CallOutcome(
        "callback_requested", datetime(2026, 10, 10, 17, 0, tzinfo=tz)
    )
    errors = [
        e.item.output for e in result.events if e.type == "function_call_output" and e.item.is_error
    ]
    assert len(errors) == 1 and "2026-10-10T17:00" in errors[0]


async def test_unknown_reason_is_rejected_and_a_bare_tool_call_skips_the_backup():
    model = ScriptedLLM([tool_call("end_call", reason="hired"), "Sorry, where were we?"])
    backup = ScriptedLLM([])
    result = await run_text_call(make_agent(model, backup=backup), model, ["Bye."])
    assert result.outcome is None
    assert backup.requests == []
    assert texts(result)[-1] == "Sorry, where were we?"


@pytest.mark.parametrize(
    ("callback_at", "complaint"),
    [
        ("2026-10-09T10:00", "already passed"),  # this morning
        ("2027-01-15T10:00", "60 days"),
        ("2026-10-10", "time of day"),
    ],
)
async def test_callback_time_must_be_soon_and_in_the_future(callback_at, complaint):
    # M7: the model retries with a sane time instead of saving a past or far-off one.
    model = ScriptedLLM(
        [
            ["Sure.", tool_call("end_call", reason="callback_requested", callback_at=callback_at)],
            ["Okay, bye!", tool_call("end_call", reason="callback_requested", callback_at=GOOD)],
        ]
    )
    result = await run_text_call(make_agent(model), model, ["Call me later."])
    assert result.outcome.callback_at == datetime.fromisoformat(GOOD).replace(
        tzinfo=ZoneInfo("Asia/Kolkata")
    )
    errors = [
        e.item.output for e in result.events if e.type == "function_call_output" and e.item.is_error
    ]
    assert len(errors) == 1 and complaint in errors[0]


GOOD = "2026-10-10T17:00"


def test_transcript_keeps_spoken_messages_only():
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content="instructions")
    ctx.add_message(role="assistant", content="Hi Priya")
    ctx.add_message(role="user", content="Hello")
    ctx.items.append(llm.FunctionCall(name="end_call", arguments="{}", call_id="c1"))
    turns = transcript(ctx)
    assert [(t["role"], t["text"]) for t in turns] == [
        ("agent", "Hi Priya"),
        ("candidate", "Hello"),
    ]
    assert datetime.fromisoformat(turns[0]["at"]).tzinfo is not None


def test_call_id_from_metadata():
    call_id = uuid.uuid4()
    assert call_id_from(f'{{"call_id": "{call_id}"}}') == call_id
    assert call_id_from("") is None


@pytest.mark.parametrize("metadata", ["{}", "not json", '{"call_id": "nope"}', "[]"])
def test_metadata_that_names_no_call_is_an_error_not_a_demo_call(metadata):
    with pytest.raises((ValueError, KeyError, TypeError)):
        call_id_from(metadata)
