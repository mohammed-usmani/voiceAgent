"""agent.entrypoint with fakes for the job, the session and the database: which agent a
dispatch gets, what is said, and what is saved when the call ends."""

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from livekit.agents import AMDCategory, AMDPredictionEvent, llm
from livekit.agents.voice.events import CloseReason

from tests.seed import CRITERIA, sample_candidate, sample_job
from voiceagent import agent as agent_module
from voiceagent import replies
from voiceagent.call.prompt import APOLOGY
from voiceagent.call.screening import ScreeningAgent
from voiceagent.call.store import CallOutcome, LoadedCall

CALL_ID = uuid.uuid4()


class FakeSession:
    def __init__(self, log: list, start_error: Exception | None = None) -> None:
        self.log = log
        self.start_error = start_error
        self.handlers: dict = {}
        self.history = llm.ChatContext()
        self.agent = None

    def on(self, event, callback=None):
        if callback is None:  # used as a decorator
            return lambda f: self.handlers.setdefault(event, f)
        self.handlers[event] = callback

    async def start(self, agent, room, room_options):
        self.agent = agent
        self.log.append(("start", type(agent).__name__))
        if self.start_error:
            raise self.start_error

    def say(self, text, *, allow_interruptions=True):
        self.log.append(("say", text, allow_interruptions))

        async def spoken():
            return None

        return spoken()

    def shutdown(self):
        self.log.append(("shutdown",))

    def close(self, error=None):
        reason = CloseReason.ERROR if error else CloseReason.PARTICIPANT_DISCONNECTED
        self.handlers["close"](SimpleNamespace(error=error, reason=reason))


class FakeParticipant:
    def __init__(self) -> None:
        self.published: list = []

    async def publish_data(self, payload, *, reliable=True, topic=""):
        self.published.append((topic, json.loads(payload)))


class FakeCtx:
    def __init__(self, metadata: str) -> None:
        self.job = SimpleNamespace(metadata=metadata)
        self.room = SimpleNamespace(name="screen-abc", local_participant=FakeParticipant())
        self.callbacks = []

    async def connect(self):
        pass

    def add_shutdown_callback(self, callback):
        self.callbacks.append(callback)

    async def shut_down(self):
        # As JobContext does: a callback may take the shutdown reason, or nothing.
        for callback in self.callbacks:
            await (callback("job shutdown") if callback.__code__.co_argcount else callback())


class FakeTracker:
    opened: list = []

    @classmethod
    def open(cls, room, *, redact=False, on_record=None):
        cls.opened.append((room, redact))
        cls.on_record = on_record
        return cls()

    def close(self, t):
        pass


@pytest.fixture
def fakes(monkeypatch):
    """Patch everything the entrypoint touches; returns (log, settings) to tweak."""
    log: list = []
    state = SimpleNamespace(load_error=None, fail_error=None, start_error=None, session=None)

    def make_session(**_kw):
        state.session = FakeSession(log, state.start_error)
        return state.session

    def load_call(sm, call_id):
        log.append(("load_call", call_id))
        if state.load_error:
            raise state.load_error
        return LoadedCall(job=sample_job(), candidate=sample_candidate(), criteria=CRITERIA)

    def fail_call(sm, call_id, error):
        log.append(("fail_call", call_id, error))
        if state.fail_error:
            raise state.fail_error

    def finish_call(sm, call_id, outcome, turns, error=None):
        log.append(("finish_call", call_id, outcome, error))

    FakeTracker.opened = []
    m = agent_module
    for name, value in {
        "AgentSession": make_session,
        "stt": lambda: None,
        "tts": lambda voice=None: None,
        "main_llm": lambda: None,
        "ShortAnswerTurnDetector": lambda *a: None,
        "turn_handling": lambda detector=None: {},
        "room_options": lambda: None,
        "watch_caller": lambda *a: None,
        "log_reply_stages": lambda *a: None,
        "watch": lambda *a: None,
        "DecisionTracker": FakeTracker,
        "default_sessionmaker": lambda: None,
        "load_call": load_call,
        "fail_call": fail_call,
        "finish_call": finish_call,
    }.items():
        monkeypatch.setattr(m, name, value)
    monkeypatch.setattr(replies, "backup_llm", lambda: None)
    monkeypatch.setattr(m, "CLOSE_WAIT_S", 0.01)
    return log, state


def metadata() -> str:
    return json.dumps({"call_id": str(CALL_ID)})


async def test_a_screening_dispatch_runs_the_screening_agent(fakes):
    log, state = fakes
    ctx = FakeCtx(metadata())
    await agent_module.entrypoint(ctx)
    assert log[:2] == [("load_call", CALL_ID), ("start", "ScreeningAgent")]
    assert FakeTracker.opened == [("screen-abc", True)]  # M3: no candidate speech in logs
    state.session.close()
    await ctx.shut_down()
    assert log[-1] == ("finish_call", CALL_ID, None, None)  # hung up: no outcome, no error


async def test_a_dispatch_without_metadata_is_the_demo(fakes):
    log, _ = fakes
    await agent_module.entrypoint(FakeCtx(""))
    assert log == [("start", "ReliableAgent"), ("say", agent_module.GREETING, True)]
    assert FakeTracker.opened == [("screen-abc", False)]


async def test_a_call_that_cannot_be_loaded_is_apologised_for_first(fakes):
    # I2: Postgres is down: the apology is spoken (uninterruptibly) before the failed
    # database write, and the write's error doesn't stop the hang-up.
    log, state = fakes
    state.load_error = state.fail_error = ConnectionError("postgres is down")
    await agent_module.entrypoint(FakeCtx(metadata()))
    assert [entry[0] for entry in log] == ["load_call", "start", "say", "fail_call", "shutdown"]
    assert log[2] == ("say", APOLOGY, False)


async def test_broken_metadata_still_gets_the_apology(fakes):
    log, _ = fakes
    await agent_module.entrypoint(FakeCtx('{"nope": 1}'))
    assert [entry[0] for entry in log] == ["start", "say", "shutdown"]


async def test_a_session_that_closes_on_an_error_is_saved_as_a_failed_call(fakes):
    # I4: an exhausted TTS quota is not the candidate hanging up.
    log, state = fakes
    ctx = FakeCtx(metadata())
    await agent_module.entrypoint(ctx)
    state.session.close(error=RuntimeError("TTS quota exhausted"))
    await ctx.shut_down()
    name, call_id, outcome, error = log[-1]
    assert (name, call_id, outcome) == ("finish_call", CALL_ID, None)
    assert "TTS quota exhausted" in error


async def test_a_session_that_fails_to_start_is_saved_as_a_failed_call(fakes):
    log, state = fakes
    state.start_error = RuntimeError("bad voice id")
    ctx = FakeCtx(metadata())
    with pytest.raises(RuntimeError):
        await agent_module.entrypoint(ctx)
    await ctx.shut_down()
    assert log[-1][0] == "finish_call" and "bad voice id" in log[-1][3]


async def test_an_error_after_end_call_keeps_the_agents_outcome(fakes):
    log, state = fakes
    ctx = FakeCtx(metadata())
    await agent_module.entrypoint(ctx)
    agent: ScreeningAgent = state.session.agent
    agent.outcome = CallOutcome("completed")
    state.session.close(error=RuntimeError("TTS dropped during the goodbye"))
    await ctx.shut_down()
    assert log[-1] == ("finish_call", CALL_ID, agent.outcome, None)


async def test_saving_retries_a_database_blip(fakes, monkeypatch):
    log, state = fakes
    attempts = []

    def flaky_finish(sm, call_id, outcome, turns, error=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("server closed the connection")

    monkeypatch.setattr(agent_module, "finish_call", flaky_finish)
    monkeypatch.setattr(agent_module, "SAVE_RETRY_S", 0)
    ctx = FakeCtx(metadata())
    await agent_module.entrypoint(ctx)
    state.session.close()
    await ctx.shut_down()
    assert len(attempts) == 2


async def test_a_screening_agent_listens_for_a_machine_first(fakes):
    log, state = fakes
    await agent_module.entrypoint(FakeCtx(metadata()))
    assert state.session.agent._detect_machine is not None


class FakeAMD:
    """Stands in for livekit.agents.AMD: returns `category`, or raises `error`."""

    category = AMDCategory.HUMAN
    error: Exception | None = None
    kwargs: dict = {}

    def __init__(self, session, **kwargs):
        FakeAMD.kwargs = kwargs
        self.session = session

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc):
        self.session.log.append("amd closed")

    async def execute(self):
        return AMDPredictionEvent(
            speech_duration=1.0, category=self.category, reason="test", transcript="", delay=0.0
        )


class AMDSession:
    def __init__(self) -> None:
        self.log = []

    async def interrupt(self, *, force=False):
        self.log.append("interrupt" if force else "interrupt (not forced)")


@pytest.mark.parametrize(
    ("category", "machine"),
    [
        (AMDCategory.MACHINE_VM, True),
        (AMDCategory.MACHINE_UNAVAILABLE, True),
        (AMDCategory.MACHINE_IVR, True),
        (AMDCategory.HUMAN, False),
        (AMDCategory.UNCERTAIN, False),  # unsure: greet, never hang up on a person
    ],
)
async def test_answering_machine_verdicts(monkeypatch, category, machine):
    monkeypatch.setattr(FakeAMD, "category", category)
    monkeypatch.setattr(agent_module, "AMD", FakeAMD)
    session = AMDSession()
    assert await agent_module.answered_by_machine(session) is machine
    assert FakeAMD.kwargs["ivr_detection"] is False
    assert (FakeAMD.kwargs["llm"], FakeAMD.kwargs["stt"]) == (None, None)
    assert FakeAMD.kwargs["wait_until_finished"] is False  # the timeout is a hard cap
    # For a person, a reply to a "Hello?" held back while listening is dropped before
    # detection lets the agent speak; a machine was already interrupted by AMD itself.
    assert session.log == (["amd closed"] if machine else ["interrupt", "amd closed"])


async def test_a_failing_detector_greets_as_if_a_person_answered(monkeypatch):
    monkeypatch.setattr(FakeAMD, "error", ValueError("AMD classifier could not be resolved"))
    monkeypatch.setattr(agent_module, "AMD", FakeAMD)
    assert await agent_module.answered_by_machine(AMDSession()) is False


async def test_nobody_answering_is_not_taken_for_a_person(monkeypatch):
    # The agent waits in the room while the phone rings; if it rings out, AMD settles as
    # "participant_missing", which must not greet an empty line.
    class RangOut(FakeAMD):
        async def execute(self):
            return AMDPredictionEvent(
                speech_duration=0.0,
                category=AMDCategory.UNCERTAIN,
                reason="participant_missing",
                transcript="",
                delay=0.0,
            )

    monkeypatch.setattr(agent_module, "AMD", RangOut)
    session = AMDSession()
    assert await agent_module.answered_by_machine(session) is None
    assert session.log == ["amd closed"]


async def test_a_screening_call_publishes_question_progress_and_turn_timings(fakes):
    _, state = fakes
    ctx = FakeCtx(metadata())
    await agent_module.entrypoint(ctx)
    asked = SimpleNamespace(role="assistant", text_content=CRITERIA.requirements[0].question)
    state.session.handlers["conversation_item_added"](SimpleNamespace(item=asked))
    FakeTracker.on_record(
        {"i": 0, "kind": "pause", "t": 1.2, "transcript": "five years", "outcome": "respond"}
    )
    await asyncio.sleep(0)
    assert ctx.room.local_participant.published == [
        ("screening.progress", {"asking": "python_years", "asked": ["python_years"]}),
        ("screening.turns", {"i": 0, "kind": "pause", "t": 1.2, "outcome": "respond"}),
    ]
