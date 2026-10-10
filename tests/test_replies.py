def test_voice_models_run_on_together_without_thinking(monkeypatch):
    # Reasoning tokens arrive before the first spoken word. With reasoning off,
    # DeepSeek-V4.1-Flash on Together answered the screening prompt in 0.60 s median.
    monkeypatch.setenv("TOGETHER_API_KEY", "test")
    from voiceagent.replies import backup_llm, main_llm

    main, backup = main_llm(), backup_llm()
    assert (main.model, backup.model) == ("deepseek-ai/DeepSeek-V4.1-Flash", "MiniMaxAI/MiniMax-M3")
    for m in (main, backup):
        assert m._opts.extra_body == {"reasoning": {"enabled": False}}
        assert str(m._client.base_url).startswith("https://api.together.xyz/v1")


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


async def test_reply_that_is_only_a_tool_call_is_not_empty(monkeypatch):
    # end_call with no words is a complete reply: asking the backup would answer twice.
    from livekit.agents import llm

    end = llm.FunctionToolCall(name="end_call", arguments='{"reason": "completed"}', call_id="c1")
    chunk = llm.ChatChunk(id="x", delta=llm.ChoiceDelta(role="assistant", tool_calls=[end]))

    async def tool_only(*_args):
        yield chunk

    assert await reply_when_main_llm(monkeypatch, tool_only) == [chunk]


def test_backup_model_can_be_given(monkeypatch):
    # The text-mode harness passes its own model; building the default needs an API key.
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from voiceagent.replies import ReliableAgent

    backup = FakeBackup()
    assert ReliableAgent(instructions="test", backup=backup)._backup is backup
