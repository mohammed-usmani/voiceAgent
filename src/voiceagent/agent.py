import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, inference, room_io
from livekit.plugins import cartesia, deepgram, groq, noise_cancellation, silero

from voiceagent.decision_log import DecisionTracker, watch

load_dotenv()

INSTRUCTIONS = (
    "You are an expert phone support technician. "
    "CRITICAL INSTRUCTION: Reply in exactly ONE or TWO short, punchy sentences. "
    "Never exceed 25 words total. Never offer multi-step lists or paragraphs. "
    "Ask one clarifying question at a time so the caller can respond naturally. "
    "Always reply in English. "
    "Build on what the caller has already told you and never re-ask a question they "
    "answered. If the caller says 'sorry?' or 'what?' or asks you to repeat, repeat your "
    "last message word for word. If their message looks cut off or garbled, ask them to "
    "finish or repeat it instead of changing the topic."
)
# Spoken as-is, without the LLM: it's fixed text, and Qwen on Groq rejects a request
# that has no user message yet.
GREETING = "Hello! What issue can I help fix on your computer today?"

# Default Silero settings. Loaded once per process so incoming calls don't block on
# ONNX initialization.
_vad = silero.VAD.load()

# One pre-warmed idle process so calls attach without a cold start.
server = AgentServer(num_idle_processes=1)


def turn_handling() -> dict:
    """LiveKit's recommended turn-taking setup, unmodified.

    See docs.livekit.io/agents/logic/turns/tuning: audio turn detector at its default
    version, adaptive interruption, default endpointing and preemptive generation.
    Preemptive generation starts the LLM and TTS while the turn is still being
    decided, so their latency overlaps the endpointing delay instead of adding to it.

    One change from the defaults: when the detector is unsure, it waited up to 2.5 s,
    and on phone audio it is often unsure about one-word answers ("No." stalled 3.6 s).
    1.3 s caps that stall; a few more mid-sentence cut-offs are the trade-off.
    """
    return {
        "turn_detection": inference.TurnDetector(),
        "endpointing": {"max_delay": 1.3},
        "interruption": {"mode": "adaptive", "min_duration": 0.5, "min_words": 0},
    }


def llm() -> groq.LLM:
    # Measured from India, first spoken token: Gemini 2.5 Flash ~1.1 s (thinks by
    # default), with thinking off ~0.5 s; Qwen3.8-27B on Groq with thinking off ~0.11 s.
    return groq.LLM(model="qwen/qwen3.8-27b", reasoning_effort="none")


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


@server.rtc_session(agent_name="clinic-agent")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=stt(),
        llm=llm(),
        tts=tts(),
        vad=_vad,
        turn_handling=turn_handling(),
    )

    tracker = DecisionTracker.open(ctx.room.name)
    watch(session, tracker)

    async def close_log() -> None:
        tracker.close(time.time())

    ctx.add_shutdown_callback(close_log)

    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
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
