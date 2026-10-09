"""How the agent's replies are produced: the models, the history they see, a fallback
for bad replies, and one log line per reply stage."""

import logging
import os

from livekit.agents import Agent, AgentSession, llm
from livekit.plugins import openai

logger = logging.getLogger("voiceagent")


TOGETHER_BASE_URL = "https://api.together.xyz/v1"
# Together's switch for reasoning models: thinking tokens arrive before the first spoken
# word, so with reasoning on every reply starts later.
NO_REASONING = {"reasoning": {"enabled": False}}


def _together(model: str) -> openai.LLM:
    return openai.LLM(
        model=model,
        base_url=TOGETHER_BASE_URL,
        api_key=os.environ.get("TOGETHER_API_KEY", ""),
        extra_body=NO_REASONING,
    )


def main_llm() -> openai.LLM:
    # Streamed on the real screening prompt from India, reasoning off (9 Together models,
    # 5 runs each): Qwen3.5-9B was fastest and steadiest, 0.64 s median to the first token,
    # 0.74 s worst, no empty replies. (Earlier: Gemini 2.5 Flash ~0.5 s with thinking off;
    # Qwen3.8-27B on Groq returned empty or cut-off replies.)
    return _together("Qwen/Qwen3.5-9B")


def backup_llm() -> openai.LLM:
    # A different model family, so a bad reply from one isn't repeated by the other:
    # MiniMax-M3 answered in 0.77 s median with no empty replies.
    return _together("MiniMaxAI/MiniMax-M3")


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
