from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, room_io
from livekit.plugins import cartesia, deepgram, silero

from voiceagent.jev_detector import JevClient, JevTurnDetector

load_dotenv()

INSTRUCTIONS = (
    "You are an expert phone support technician. "
    "CRITICAL INSTRUCTION: Reply in exactly ONE or TWO short, punchy sentences. "
    "Never exceed 25 words total. Never offer multi-step lists or paragraphs. "
    "Ask one clarifying question at a time so the caller can respond naturally."
)
GREETING = "Say: 'Hello! What issue can I help fix on your computer today?'"

VAD_MIN_SILENCE_S = 0.20

# Loaded once per process so incoming calls don't block on ONNX initialization.
_vad = silero.VAD.load(min_speech_duration=0.1, min_silence_duration=VAD_MIN_SILENCE_S)

# One pre-warmed idle process so calls attach without a cold start.
server = AgentServer(num_idle_processes=1)


@server.rtc_session(agent_name="clinic-agent")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    jev = JevClient()
    ctx.add_shutdown_callback(jev.aclose)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm="google/gemini-2.5-flash",
        tts=cartesia.TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            sample_rate=24000,
        ),
        vad=_vad,
        turn_handling={
            "turn_detection": JevTurnDetector(
                jev, silence_ms=int(VAD_MIN_SILENCE_S * 1000), unlikely_threshold=0.5
            ),
            "interruption": {"enabled": True},
            # Respond after 350ms on a complete thought; hold up to 2s on pauses and fillers.
            "endpointing": {"min_delay": 0.35, "max_delay": 2.0},
            "preemptive_generation": {"enabled": False},
        },
    )

    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=None),
        ),
    )
    await session.generate_reply(instructions=GREETING)


def main() -> None:
    agents.cli.run_app(server)
