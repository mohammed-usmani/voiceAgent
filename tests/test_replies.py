def test_llm_does_not_think_before_speaking(monkeypatch):
    # Thinking tokens arrive before the first spoken word; with it on, replies start ~1 s later.
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-long-enough-for-hs256-signing")
    from voiceagent.replies import backup_llm, main_llm

    assert main_llm().model == "google/gemini-2.5-flash"
    assert main_llm()._opts.extra_kwargs == {"reasoning_effort": "none"}
    assert backup_llm()._opts.reasoning_effort == "low"


def test_back_to_back_caller_messages_are_merged():
    from livekit.agents import llm

    from voiceagent.replies import clean_history

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

    from voiceagent.replies import clean_history

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

    from voiceagent.replies import clean_history

    ctx = llm.ChatContext()
    ctx.add_message(role="assistant", content="How many years?", extra={"google": {"x": 1}})
    assert clean_history(ctx).items[0].extra == {}


class FakeBackup:
    def chat(self, **_kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def __aiter__(self):
        yield "Sorry, could you say that again?"


async def reply_when_main_llm(monkeypatch, main_llm_node) -> list:
    from livekit.agents import Agent, llm

    from voiceagent import replies

    monkeypatch.setattr(replies, "backup_llm", FakeBackup)
    monkeypatch.setattr(Agent.default, "llm_node", main_llm_node)
    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Hello?")
    agent = replies.ReliableAgent(instructions="test")
    return [chunk async for chunk in agent.llm_node(ctx, [], None)]


async def test_failed_main_reply_falls_back_to_backup(monkeypatch):
    async def failing(*_args):
        raise RuntimeError("400 from the provider")
        yield

    assert await reply_when_main_llm(monkeypatch, failing) == ["Sorry, could you say that again?"]


async def test_empty_main_reply_falls_back_to_backup(monkeypatch):
    async def empty(*_args):
        return
        yield

    assert await reply_when_main_llm(monkeypatch, empty) == ["Sorry, could you say that again?"]


async def test_good_main_reply_is_used_as_is(monkeypatch):
    async def good(*_args):
        yield "Great, how many years of experience do you have?"

    assert await reply_when_main_llm(monkeypatch, good) == [
        "Great, how many years of experience do you have?"
    ]
