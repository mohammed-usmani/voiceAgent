import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, inference, room_io
from livekit.plugins import cartesia, deepgram, noise_cancellation, silero

from voiceagent.decision_log import DecisionTracker, watch

load_dotenv()

INSTRUCTIONS = (
    "You are an expert phone support technician. "
    "CRITICAL INSTRUCTION: Reply in exactly ONE or TWO short, punchy sentences. "
    "Never exceed 25 words total. Never offer multi-step lists or paragraphs. "
    "Ask one clarifying question at a time so the caller can respond naturally."
)
GREETING = "Say: 'Hello! What issue can I help fix on your computer today?'"

# Default Silero settings. Loaded once per process so incoming calls don't block on
# ONNX initialization.
_vad = silero.VAD.load()

# One pre-warmed idle process so calls attach without a cold start.
server = AgentServer(num_idle_processes=1)


def turn_handling() -> dict:
    """LiveKit's recommended turn-taking setup, minus preemptive generation.

    See docs.livekit.io/agents/logic/turns/tuning: audio turn detector at its default
    version, adaptive interruption, default endpointing. Preemptive generation is off
    so the LLM never drafts replies to a caller who is still mid-sentence.
    """
    return {
        "turn_detection": inference.TurnDetector(),
        "interruption": {"mode": "adaptive", "min_duration": 0.5, "min_words": 0},
        "preemptive_generation": {"enabled": False},
    }


@server.rtc_session(agent_name="clinic-agent")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm="google/gemini-2.5-flash",
        tts=cartesia.TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            sample_rate=24000,
        ),
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
    await session.generate_reply(instructions=GREETING)


def main() -> None:
    agents.cli.run_app(server)
