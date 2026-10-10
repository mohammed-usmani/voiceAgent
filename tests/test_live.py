"""The live view's data: which approved question the agent is asking (plain word overlap),
and turn timings without words."""

import asyncio
import json
from types import SimpleNamespace

from voiceagent.call.live import (
    PROGRESS_TOPIC,
    TURNS_TOPIC,
    Progress,
    RoomPublisher,
    publish_progress,
    turn_record,
)
from voiceagent.evals.jobs import load_jobs

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
