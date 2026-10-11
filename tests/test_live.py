"""The live view's data: which approved question the agent is asking (plain word overlap),
and turn timings without words."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.fakes import FakeJev
from voiceagent.call.live import (
    NONE,
    PROGRESS_TOPIC,
    TURNS_TOPIC,
    Progress,
    RoomPublisher,
    publish_progress,
    turn_record,
)
from voiceagent.evals.jobs import load_jobs
from voiceagent.jev import Jev, JevError

SUPPORT = load_jobs(["customer-support"])[0].criteria.requirements
# The agent's turns on a simulated customer-support screening (evaluation run
# 20261009T223414Z, persona Ananya Rao), each with the question it asks. The agent words
# questions around the résumé, as on real calls.
CALL = [
    (
        "Hi Ananya, this is Aria, an AI assistant calling from Brightline Support Services "
        "about the Customer Support Associate, International Voice role you applied for. This "
        "call is recorded so the hiring team can review it. Is now a good time for a few "
        "quick questions?",
        None,
    ),
    (
        "Great, let's get started. Your résumé mentions you're a B.Com graduate — is that correct?",
        "graduate",
    ),
    (
        "Got it. Just to confirm — the role requires US night shifts, 6:30 pm to 3:30 am IST. "
        "Would that work for you?",
        "night_shifts",
    ),
    ("Thanks. If you got an offer, how many days would you need before joining?", "joining_days"),
    (
        "Tell me about a time you calmed down an upset customer or friend. What did you do?",
        "frustrated_customer",
    ),
    (
        "Your résumé mentions 1.5 years in an international voice process for a US telecom "
        "client — is that right?",
        "intl_voice_years",
    ),
    ("Got it. What annual salary are you expecting, in lakhs?", "expected_salary"),
    (
        "Thanks, Ananya. The team will review and get back to you by email within 3 days.",
        None,
    ),
]


def test_each_agent_turn_ticks_the_question_it_asks():
    progress = Progress(SUPPORT)
    assert [progress.agent_said(text)["asking"] for text, _ in CALL] == [q for _, q in CALL]
    assert progress.agent_said("Bye!")["asked"] == [
        "graduate",
        "night_shifts",
        "joining_days",
        "frustrated_customer",
        "intl_voice_years",
        "expected_salary",
    ]


def test_a_repeated_question_is_listed_once_in_the_order_asked():
    progress = Progress(SUPPORT)
    progress.agent_said("What annual salary are you expecting, in lakhs?")
    progress.agent_said("Sorry, I meant: what annual salary are you expecting, in lakhs?")
    assert progress.agent_said("Have you completed your graduation?") == {
        "asking": "graduate",
        "asked": ["expected_salary", "graduate"],
    }


def test_answering_the_candidates_questions_asks_nothing():
    progress = Progress(SUPPORT)
    for text in (
        "Thanks, Vikram. Do you have any questions about the role?",
        "I don't have specifics on training, but I'll pass that question to the team to "
        "follow up with you. Any other questions?",
    ):
        assert progress.agent_said(text) == {"asking": None, "asked": []}


def test_a_question_in_unusual_words_is_missed():
    # The heuristic's known limit: the UI shows the checklist as a guide, not a record.
    assert Progress(SUPPORT).agent_said("Your résumé lists a B.Com. Correct?")["asking"] is None


def test_turn_records_carry_timings_never_words():
    record = {
        "i": 3,
        "kind": "overlap",
        "t": 12.4,
        "transcript": "my salary is 30 lakhs",
        "agent_before": "What salary are you expecting?",
        "outcome": "interrupt",
        "eot_delay_ms": None,
        "reply_delay_ms": None,
        "paused_then_resumed": False,
        "suspect": None,
    }
    assert turn_record(record) == {
        "i": 3,
        "kind": "overlap",
        "t": 12.4,
        "outcome": "interrupt",
        "eot_delay_ms": None,
        "reply_delay_ms": None,
        "paused_then_resumed": False,
        "suspect": None,
    }


class Participant:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.sent: list = []

    async def publish_data(self, payload, *, reliable=True, topic=""):
        if self.error is not None:
            raise self.error
        self.sent.append((topic, json.loads(payload), reliable))


def room(error: Exception | None = None):
    return SimpleNamespace(local_participant=Participant(error))


async def test_publishes_json_reliably_on_the_topic():
    r = room()
    RoomPublisher(r).publish(PROGRESS_TOPIC, {"asking": "graduate", "asked": ["graduate"]})
    await asyncio.sleep(0)
    assert r.local_participant.sent == [
        ("screening.progress", {"asking": "graduate", "asked": ["graduate"]}, True)
    ]


async def test_a_failed_publish_is_logged_not_raised(caplog):
    # Review focus 4: the room may be gone at hang-up; the call must not notice.
    RoomPublisher(room(ConnectionError("room closed"))).publish(TURNS_TOPIC, {"i": 0})
    await asyncio.sleep(0)
    assert "couldn't publish on screening.turns" in caplog.text


def test_publishing_outside_the_event_loop_is_logged_not_raised(caplog):
    RoomPublisher(room()).publish(TURNS_TOPIC, {"i": 0})
    assert "couldn't publish on screening.turns" in caplog.text


class FakeSession:
    def __init__(self) -> None:
        self.handlers: dict = {}

    def on(self, event):
        return lambda f: self.handlers.setdefault(event, f)


def item(role: str, text: str):
    return SimpleNamespace(item=SimpleNamespace(role=role, text_content=text))


async def test_progress_is_published_after_each_agent_turn_only():
    session, r = FakeSession(), room()
    publish_progress(session, RoomPublisher(r), SUPPORT)
    added = session.handlers["conversation_item_added"]
    added(item("user", "Have you completed your graduation?"))
    added(item("assistant", "Have you completed your graduation?"))
    await asyncio.sleep(0)
    assert r.local_participant.sent == [
        (PROGRESS_TOPIC, {"asking": "graduate", "asked": ["graduate"]}, True)
    ]


# --- Jev (the decider) ---------------------------------------------------------------------


def choosing(*choices):
    """A FakeJev answering each progress request with the next choice, or raising it."""
    queue = list(choices)

    def answer(state, questions):
        c = queue.pop(0)
        if isinstance(c, Exception):
            raise c
        return {"q": {"choice": c, "confidence": 0.9}}

    return FakeJev(answer)


async def test_jev_picks_the_question_a_turn_asks():
    decider = choosing("graduate", NONE)
    progress = Progress(SUPPORT, decider)
    assert await progress.agent_said_async("Your résumé lists a B.Com. Correct?") == {
        "asking": "graduate",
        "asked": ["graduate"],
    }
    assert await progress.agent_said_async("Thanks!") == {"asking": None, "asked": ["graduate"]}
    state, questions = decider.asked[0]
    assert state == "Agent turn: Your résumé lists a B.Com. Correct?"
    q = questions["q"]
    assert q["type"] == "choice" and list(q["criteria"])[-1] == NONE
    assert set(q["criteria"]) == {r.id for r in SUPPORT} | {NONE}
    graduate = next(r for r in SUPPORT if r.id == "graduate")
    assert q["criteria"]["graduate"] == f"{graduate.label}: {graduate.question}"


async def test_word_overlap_answers_when_jev_fails_or_is_slow():
    progress = Progress(SUPPORT, choosing(JevError("timed out"), "not_a_requirement"))
    text = "Have you completed your graduation?"
    assert (await progress.agent_said_async(text))["asking"] == "graduate"
    assert (await progress.agent_said_async("Thanks!"))["asking"] is None


async def test_the_client_gives_up_after_its_timeout(monkeypatch):
    async def slow(*a, **kw):
        await asyncio.sleep(1)

    monkeypatch.setattr("httpx.AsyncClient.post", slow)
    with pytest.raises(JevError):
        await Jev("k").ask_async("Agent turn: hi", {"q": {}}, timeout_s=0.01)


async def test_with_jev_progress_is_published_in_the_background_in_turn_order():
    session, r = FakeSession(), room()
    publish_progress(session, RoomPublisher(r), SUPPORT, choosing("graduate", JevError("503")))
    added = session.handlers["conversation_item_added"]
    added(item("assistant", "Your résumé lists a B.Com. Correct?"))
    added(item("assistant", "What annual salary are you expecting, in lakhs?"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert [m for _, m, _ in r.local_participant.sent] == [
        {"asking": "graduate", "asked": ["graduate"]},
        {"asking": "expected_salary", "asked": ["graduate", "expected_salary"]},
    ]
