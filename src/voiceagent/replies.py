"""How the agent's replies are produced: the models, the history they see, a fallback
for bad replies, and one log line per reply stage."""

import logging

from livekit.agents import Agent, AgentSession, inference, llm
from livekit.plugins import groq

logger = logging.getLogger("voiceagent")


def main_llm() -> inference.LLM:
    # Replayed on a real call's history, first spoken token from India: Gemini 2.5 Flash
    # ~1.1 s (thinks by default), with thinking off ~0.5 s. Qwen3.8-27B on Groq was
    # ~0.16 s but its replies came back empty or cut off mid-sentence (19/20 on a long
    # answer).
    return inference.LLM(model="google/gemini-2.5-flash", extra_kwargs={"reasoning_effort": "none"})


def backup_llm() -> groq.LLM:
    # A different provider, so one provider's outage doesn't silence the agent.
    return groq.LLM(model="openai/gpt-oss-20b", reasoning_effort="low")


def clean_history(chat_ctx: llm.ChatContext) -> llm.ChatContext:
    """Make the history safe to send to either model.

    - Back-to-back caller messages are joined into one. A long answer with pauses
      arrives as several user messages in a row, and models returned empty replies far
      more often on that shape.
    - Agent replies cut off after a few words are dropped. The caller barely heard
      them, and the model copied them ("Thanks for." -> "Thanks for.Thanks for.").
    - Provider metadata is dropped. Groq rejects any request containing it (400
      "extra_content is unsupported"); a Gemini reply in the history once broke every
      later turn.
    """
    items: list = []
    for item in chat_ctx.items:
        if (
            item.type == "message"
            and item.role == "assistant"
            and item.interrupted
            and len((item.text_content or "").split()) <= 3
        ):
            continue
        if item.type == "message" and item.extra:
            item = item.model_copy(update={"extra": {}})
        prev = items[-1] if items else None
        if (
            prev is not None
            and item.type == "message"
            and prev.type == "message"
            and item.role == prev.role == "user"
        ):
            items[-1] = prev.model_copy(update={"content": prev.content + item.content})
        else:
            items.append(item)
    return llm.ChatContext(items)


def _has_text(chunk) -> bool:
    if isinstance(chunk, str):
        return bool(chunk.strip())
    delta = getattr(chunk, "delta", None)
    return bool(delta and delta.content and delta.content.strip())


class ReliableAgent(Agent):
    """An agent whose reply never comes back empty.

    An empty reply (returned without any error) once left the agent silent for the rest
    of a call, so an empty or failed reply is retried once on the backup model.
    """

    def __init__(self, *, instructions: str) -> None:
        super().__init__(instructions=instructions)
        self._backup = backup_llm()

    async def llm_node(self, chat_ctx, tools, model_settings):
        chat_ctx = clean_history(chat_ctx)
        spoke = False
        try:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                spoke = spoke or _has_text(chunk)
                yield chunk
        except Exception:
            if spoke:
                raise
            logger.exception("main LLM failed; answering with the backup")
        else:
            if spoke:
                return
            logger.warning("main LLM replied with nothing; answering with the backup")
        async with self._backup.chat(chat_ctx=chat_ctx, tools=tools) as stream:
            async for chunk in stream:
                yield chunk


def log_reply_stages(session: AgentSession) -> None:
    """One log line per reply stage, so a silent stall shows which stage it stopped at."""

    @session.on("speech_created")
    def _speech(ev) -> None:
        logger.info("reply created (source=%s)", ev.source)

    @session.on("agent_state_changed")
    def _state(ev) -> None:
        logger.info("agent %s -> %s", ev.old_state, ev.new_state)

    @session.on("metrics_collected")
    def _metrics(ev) -> None:
        m = ev.metrics
        if m.type in ("llm_metrics", "tts_metrics"):
            first = m.ttft if m.type == "llm_metrics" else m.ttfb
            logger.info("%s first=%.0f ms cancelled=%s", m.type, first * 1000, m.cancelled)

    @session.on("error")
    def _error(ev) -> None:
        logger.error("session error from %s: %s", type(ev.source).__name__, ev.error)
