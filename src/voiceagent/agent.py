import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, AgentSession, JobContext, inference, room_io
from livekit.plugins import cartesia, deepgram, noise_cancellation, silero

from voiceagent.decision_log import DecisionTracker, watch
from voiceagent.outbound import AGENT_NAME
from voiceagent.replies import ReliableAgent, log_reply_stages, main_llm
from voiceagent.short_answers import (
    CallerText,
    ShortAnswerTurnDetector,
    last_agent_text,
    watch_caller,
)

load_dotenv()

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
    version, adaptive interruption and preemptive generation. Preemptive generation
    starts the LLM and TTS while the turn is still being decided, so their latency
    overlaps the endpointing delay instead of adding to it.

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
    log_reply_stages(session)

    tracker = DecisionTracker.open(ctx.room.name)
    watch(session, tracker)

    async def close_log() -> None:
        tracker.close(time.time())

    ctx.add_shutdown_callback(close_log)

    await session.start(
        agent=ReliableAgent(instructions=INSTRUCTIONS),
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
