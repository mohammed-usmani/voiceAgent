from livekit.agents import inference

from voiceagent.agent import turn_handling


def test_turn_handling_is_livekit_recommended_setup():
    # docs.livekit.io/agents/logic/turns/tuning: default detector, adaptive interruption,
    # preemptive generation left on. Only endpointing is tuned.
    cfg = turn_handling()
    assert isinstance(cfg["turn_detection"], inference.TurnDetector)
    assert cfg["interruption"] == {"mode": "adaptive", "min_duration": 0.5, "min_words": 0}
    assert cfg["endpointing"] == {"mode": "dynamic", "max_delay": 2.0}
    assert set(cfg) == {"turn_detection", "endpointing", "interruption"}


def test_speech_models_accept_their_settings(monkeypatch):
    # Plugins validate options at construction; a bad value must fail here, not on a call.
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    from voiceagent.agent import stt, tts

    assert tts().model == "sonic-3.6"
    # "hi" gave English words Hindi pronunciation ("do" came out wrong).
    assert tts()._opts.language.language == "en"
    assert stt().model == "nova-3"
    # "multi" guessed among ten languages on phone audio ("hello" came back as Spanish).
    assert stt()._opts.language == "en-IN"


def test_prompt_handles_repeats_and_answered_questions():
    from voiceagent.agent import INSTRUCTIONS

    assert "word for word" in INSTRUCTIONS
    assert "never re-ask" in INSTRUCTIONS
    assert "Always reply in English" in INSTRUCTIONS


def test_llm_does_not_think_before_speaking(monkeypatch):
    # Thinking tokens arrive before the first spoken word; with it on, replies start ~1 s later.
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-long-enough-for-hs256-signing")
    from voiceagent.agent import backup_llm, main_llm

    assert main_llm().model == "google/gemini-2.5-flash"
    assert main_llm()._opts.extra_kwargs == {"reasoning_effort": "none"}
    assert backup_llm()._opts.reasoning_effort == "low"


def test_back_to_back_caller_messages_are_merged():
    from livekit.agents import llm

    from voiceagent.agent import clean_history

    ctx = llm.ChatContext()
    ctx.add_message(role="assistant", content="What does your role involve?")
    ctx.add_message(role="user", content="I build agents.")
    ctx.add_message(role="user", content="Mostly voice ones.")
    merged = clean_history(ctx).items
    assert [m.role for m in merged] == ["assistant", "user"]
    assert merged[1].text_content == "I build agents.\nMostly voice ones."


def test_barely_heard_agent_fragments_are_dropped():
    # The model copied a cut-off "Thanks for" back ("Thanks for.Thanks for.").
    from livekit.agents import llm

    from voiceagent.agent import clean_history

    ctx = llm.ChatContext()
    ctx.add_message(role="assistant", content="Thanks for", interrupted=True)
    ctx.add_message(role="assistant", content="Are you open to relocating?", interrupted=True)
    ctx.add_message(role="assistant", content="Got it.")
    assert [m.text_content for m in clean_history(ctx).items] == [
        "Are you open to relocating?",
        "Got it.",
    ]


def test_provider_metadata_is_dropped_from_history():
    # A Gemini reply carried it; Groq rejected every later request because of it.
    from livekit.agents import llm

    from voiceagent.agent import clean_history

    ctx = llm.ChatContext()
    ctx.add_message(role="assistant", content="How many years?", extra={"google": {"x": 1}})
    assert clean_history(ctx).items[0].extra == {}


async def test_failed_fast_reply_falls_back_to_backup(monkeypatch):
    from livekit.agents import Agent, llm

    from voiceagent import agent as agent_mod

    async def failing_llm_node(*_args):
        raise RuntimeError("400 from Groq")
        yield

    class Backup:
        def chat(self, **_kw):
            class Stream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_):
                    return False

                def __aiter__(self):
                    async def gen():
                        yield "Sorry, could you say that again?"

                    return gen()

            return Stream()

    monkeypatch.setattr(agent_mod, "backup_llm", lambda: Backup())
    monkeypatch.setattr(Agent.default, "llm_node", failing_llm_node)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Hello?")
    out = [c async for c in agent_mod.Aria().llm_node(ctx, [], None)]
    assert out == ["Sorry, could you say that again?"]


async def test_empty_fast_reply_falls_back_to_backup(monkeypatch):
    from livekit.agents import Agent, llm

    from voiceagent import agent as agent_mod

    async def empty_llm_node(*_args):
        return
        yield

    class Backup:
        def chat(self, **_kw):
            class Stream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_):
                    return False

                def __aiter__(self):
                    async def gen():
                        yield "Sorry, could you say that again?"

                    return gen()

            return Stream()

    monkeypatch.setattr(agent_mod, "backup_llm", lambda: Backup())
    monkeypatch.setattr(Agent.default, "llm_node", empty_llm_node)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Hello?")
    out = [c async for c in agent_mod.Aria().llm_node(ctx, [], None)]
    assert out == ["Sorry, could you say that again?"]
