import logging
import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, inference, llm, room_io
from livekit.plugins import cartesia, deepgram, groq, noise_cancellation, silero

from voiceagent.decision_log import DecisionTracker, watch
from voiceagent.outbound import AGENT_NAME
from voiceagent.short_answers import (
    CallerText,
    ShortAnswerTurnDetector,
    last_agent_text,
    watch_caller,
)

load_dotenv()

logger = logging.getLogger("voiceagent")

INSTRUCTIONS = (
    "You are Aria from the Nimbus Labs hiring team, calling someone who applied for the "
    "AI Engineer role in Bangalore, to ask a few pre-screening questions before the "
    "interview. "
    "CRITICAL INSTRUCTION: Reply in exactly ONE or TWO short, natural sentences. "
    "Never exceed 25 words total. Never offer lists or paragraphs. "
    "Ask one question at a time, in this order, skipping anything already answered: "
    "what they work on in their current role; how many years of experience they have; "
    "whether they are open to relocating to Bangalore; whether they are serving a notice "
    "period and when they could join; their current and expected salary; whether they "
    "have any questions. "
    "Briefly acknowledge each answer before the next question. If they say it is not a "
    "good time, ask when to call back and end politely. When done, thank them and say "
    "the team will email next steps. "
    "Always reply in English. "
    "Build on what the caller has already told you and never re-ask a question they "
    "answered. If the caller says 'sorry?' or 'what?' or asks you to repeat, repeat your "
    "last message word for word. If their message looks cut off or garbled, ask them to "
    "finish or repeat it instead of changing the topic."
)
# Spoken as-is, without the LLM: it's fixed text, so there's nothing to generate.
GREETING = (
    "Hi, this is Aria from the Nimbus Labs hiring team, calling about your application "
    "for the AI Engineer role. Is now a good time for a quick five-minute chat?"
)

# Default Silero settings. Loaded once per process so incoming calls don't block on
# ONNX initialization.
_vad = silero.VAD.load()

# One pre-warmed idle process so calls attach without a cold start.
server = AgentServer(num_idle_processes=1)


def turn_handling(detector: inference.TurnDetector | None = None) -> dict:
    """LiveKit's recommended turn-taking setup, with endpointing tuned.

    See docs.livekit.io/agents/logic/turns/tuning: audio turn detector at its default
    version, adaptive interruption, default endpointing and preemptive generation.
    Preemptive generation starts the LLM and TTS while the turn is still being
    decided, so their latency overlaps the endpointing delay instead of adding to it.

    One change from the defaults, endpointing. When the detector is unsure it waits
    max_delay; at the default 2.5 s one-word answers stalled ("No." took 3.6 s), at
    1.3 s callers were cut off while pausing to think in long answers. 2.0 s splits
    the difference, and "dynamic" learns each caller's mid-sentence pauses and raises
    the shorter wait used when the detector is confident.

    The call passes a ShortAnswerTurnDetector: the same audio model, plus a rule that a
    bare yes or no to a question ends the turn (see short_answers.py).
    """
    return {
        "turn_detection": detector or inference.TurnDetector(),
        "endpointing": {"mode": "dynamic", "max_delay": 2.0},
        "interruption": {"mode": "adaptive", "min_duration": 0.5, "min_words": 0},
    }


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


class Aria(Agent):
    """An empty reply (returned without any error) once left the agent silent for the
    rest of a call, so an empty or failed reply is retried once on the backup model."""

    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)
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
            logger.exception("fast LLM failed; answering with the backup")
        else:
            if spoke:
                return
            logger.warning("fast LLM replied with nothing; answering with the backup")
        async with self._backup.chat(chat_ctx=chat_ctx, tools=tools) as stream:
            async for chunk in stream:
                yield chunk


def stt() -> deepgram.STT:
    # Indian English. "multi" chooses among ten languages and on phone audio it
    # misheard English ("on its own" -> "on YouTube") and drifted into Spanish.
    return deepgram.STT(model="nova-3", language="en-IN")


def tts() -> cartesia.TTS:
    return cartesia.TTS(
        model="sonic-3.6",
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        # "hi" gave English words Hindi pronunciation ("do" came out wrong).
        language="en",
        speed=0.9,  # 1.0 is natural pacing, which sounded rushed on phone calls
        sample_rate=24000,
    )


def log_pipeline(session: AgentSession) -> None:
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


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    caller = CallerText()
    detector = ShortAnswerTurnDetector(caller, lambda: last_agent_text(session))
    session = AgentSession(
        stt=stt(),
        llm=main_llm(),
        tts=tts(),
        vad=_vad,
        turn_handling=turn_handling(detector),
    )
    watch_caller(session, caller)
    agent = Aria()
    log_pipeline(session)

    tracker = DecisionTracker.open(ctx.room.name)
    watch(session, tracker)

    async def close_log() -> None:
        tracker.close(time.time())

    ctx.add_shutdown_callback(close_log)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # Calls arrive over SIP: LiveKit recommends the telephony-tuned BVC model.
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony()
            ),
        ),
    )
    await session.say(GREETING)


def main() -> None:
    agents.cli.run_app(server)
