import asyncio
import contextlib
import logging
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AMD, Agent, AgentServer, AgentSession, JobContext, inference, room_io
from livekit.agents.voice.events import CloseEvent, CloseReason
from livekit.plugins import cartesia, deepgram, noise_cancellation, silero

from voiceagent.call.live import TURNS_TOPIC, RoomPublisher, publish_progress, turn_record
from voiceagent.call.prompt import APOLOGY, build_instructions, greeting
from voiceagent.call.screening import ScreeningAgent, call_id_from, transcript
from voiceagent.call.store import (
    LoadedCall,
    default_sessionmaker,
    fail_call,
    finish_call,
    load_call,
)
from voiceagent.decision_log import DecisionTracker, watch
from voiceagent.jev import default_jev
from voiceagent.outbound import AGENT_NAME
from voiceagent.replies import ReliableAgent, log_reply_stages, main_llm
from voiceagent.settings import DEFAULT_VOICE, get_settings
from voiceagent.short_answers import (
    CallerText,
    ShortAnswerTurnDetector,
    last_agent_text,
    watch_caller,
)

load_dotenv()
logger = logging.getLogger("voiceagent")

# The demo recruiter, used when a call arrives without a screening call id.
INSTRUCTIONS = (
    "You are Aria, a recruiter at Nimbus Labs, on a phone call with an applicant who was "
    "shortlisted for the AI Engineer role in Bangalore. The purpose of the call is to "
    "confirm a few details and schedule their final interview. "
    "CRITICAL INSTRUCTION: Reply in exactly ONE or TWO short, natural sentences. "
    "Never exceed 25 words total. Never offer lists or paragraphs. "
    "Ask one question at a time, in this order, skipping anything already answered: "
    "what they work on in their current role; their total years of experience; their "
    "notice period or earliest joining date; their current and expected salary (if they "
    "prefer not to say, reply that it's fine and the team will discuss it later); whether "
    "they are comfortable with the hybrid setup, three days a week in the Bangalore "
    "office; which final interview slot works for them, Thursday at 11 AM or Friday at "
    "3 PM; whether the email on their application is still the best one for the invite; "
    "whether they have any questions. "
    "If they ask about the final interview: it is a one-hour technical round with the "
    "engineering lead, covering system design and a project they have built. "
    "Briefly acknowledge each answer before the next question. Never judge or comment on "
    "whether they meet the requirements, and never promise an offer or a salary: you only "
    "collect details, and the team decides after the interview. If they say it is not a "
    "good time, ask when to call back and end politely. When done, confirm the slot they "
    "picked, say a calendar invite is on its way, and thank them. "
    "Always reply in English. "
    "Build on what the caller has already told you and never re-ask a question they "
    "answered. If the caller says 'sorry?' or 'what?' or asks you to repeat, repeat your "
    "last message word for word. If their message looks cut off or garbled, ask them to "
    "finish or repeat it instead of changing the topic."
)
# Spoken as-is, without the LLM: it's fixed text, so there's nothing to generate.
GREETING = (
    "Hi, this is Aria from Nimbus Labs. You've been shortlisted for the AI Engineer role, "
    "and I'm calling to confirm a few details before we schedule your final interview. "
    "Is now a good time?"
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


def tts(voice: str = DEFAULT_VOICE) -> cartesia.TTS:
    return cartesia.TTS(
        model="sonic-3.6",
        voice=voice,
        # "hi" gave English words Hindi pronunciation ("do" came out wrong).
        language="en",
        speed=0.9,  # 1.0 is natural pacing, which sounded rushed on phone calls
        sample_rate=24000,
    )


def room_options() -> room_io.RoomOptions:
    return room_io.RoomOptions(
        # Calls arrive over SIP: LiveKit recommends the telephony-tuned BVC model.
        audio_input=room_io.AudioInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
        # Closing the session (end_call) deletes the room, which hangs up the SIP leg.
        delete_room_on_close=True,
    )


# Answering-machine detection. The agent joins a screening call while it rings and
# detection starts at pickup, so a "Hello?" settles it as a person in about 0.5 s; a person
# who waits silently for us is greeted after this much silence ("unsure"), rather than
# LiveKit's 10 s default (3 s felt long on a real call). TIMEOUT caps the wait for a
# verdict (wait_until_finished=False: even while the line is still talking).
AMD_NO_SPEECH_S = 2.0
AMD_TIMEOUT_S = 8.0


async def answered_by_machine(session: AgentSession) -> bool | None:
    """LiveKit's answering-machine detection on what the line says after pickup: True for
    voicemail or an automated menu, None when nobody ever answered. Unsure, or detection
    failing, counts as a person, so a candidate is never hung up on for it."""
    try:
        async with AMD(
            session,
            llm=None,  # the session's own LLM and STT: no extra provider
            stt=None,
            ivr_detection=False,  # never navigate a phone menu: hang up on it
            suppress_compatibility_warning=True,
            detection_options={"no_speech_threshold": AMD_NO_SPEECH_S, "timeout": AMD_TIMEOUT_S},
            wait_until_finished=False,
        ) as detector:
            result = await detector.execute()
            if result.reason == "participant_missing":
                logger.info("answering-machine detection: nobody answered")
                return None
            if not result.is_machine:
                # execute() lets the agent speak again: drop a reply to a "Hello?" heard
                # while listening before it plays, so nothing comes before the disclosure.
                await session.interrupt(force=True)
    except Exception:
        logger.exception("answering-machine detection failed; greeting as if a person answered")
        return False
    logger.info("answering-machine detection: %s (%s)", result.category.value, result.reason)
    return result.is_machine


# How long the shutdown save waits for the session's own close (so the last reply is in
# the history), and the pause between attempts to save through a database blip.
CLOSE_WAIT_S = 10.0
SAVE_ATTEMPTS = 3
SAVE_RETRY_S = 2.0
# The apology is spoken before anything touches the database, but never holds the line
# open longer than this.
APOLOGY_TIMEOUT_S = 15.0


def save_on_shutdown(
    ctx: JobContext, session: AgentSession, agent: ScreeningAgent, call_id: uuid.UUID
) -> None:
    """Save the transcript and end reason when the call ends, however it ends."""
    closed = asyncio.Event()

    @session.on("close")
    def _closed(ev: CloseEvent) -> None:
        if (ev.error is not None or ev.reason == CloseReason.ERROR) and agent.error is None:
            agent.error = f"the session closed on an error: {ev.error!r}"
        closed.set()

    async def save(_reason: str) -> None:
        # Shutdown callbacks run alongside the session's own close; wait for it so the
        # last reply is in the history.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(closed.wait(), timeout=CLOSE_WAIT_S)
        turns = transcript(session.history)
        # An error after end_call (e.g. during the goodbye) doesn't undo the outcome.
        error = agent.error if agent.outcome is None else None
        for attempt in range(1, SAVE_ATTEMPTS + 1):
            try:
                await asyncio.to_thread(
                    finish_call, default_sessionmaker(), call_id, agent.outcome, turns, error
                )
                return
            except Exception:
                # The worker's stale-call sweep closes the call if every attempt fails.
                logger.exception("couldn't save call %s (attempt %s)", call_id, attempt)
                if attempt < SAVE_ATTEMPTS:
                    await asyncio.sleep(SAVE_RETRY_S * attempt)

    ctx.add_shutdown_callback(save)


async def apologise_and_hang_up(
    session: AgentSession, call_id: uuid.UUID | None, error: str
) -> None:
    """The call couldn't be loaded, most likely because Postgres is down: speak the
    apology first, then try to record the failure, then hang up."""
    try:
        await asyncio.wait_for(
            session.say(APOLOGY, allow_interruptions=False), timeout=APOLOGY_TIMEOUT_S
        )
    except Exception:
        logger.exception("couldn't speak the apology")
    if call_id is not None:
        try:
            await asyncio.to_thread(fail_call, default_sessionmaker(), call_id, error)
        except Exception:
            # The worker's stale-call sweep closes the call later.
            logger.exception("couldn't mark call %s failed", call_id)
    session.shutdown()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # A screening dispatch carries {"call_id": ...} (outbound.dial). A call without
    # metadata (someone phoning in, `voiceagent console`) is the Nimbus Labs demo.
    screening = bool(ctx.job.metadata)
    call_id: uuid.UUID | None = None
    loaded: LoadedCall | None = None
    load_error = ""
    if screening:
        try:
            call_id = call_id_from(ctx.job.metadata)
            loaded = await asyncio.to_thread(load_call, default_sessionmaker(), call_id)
        except Exception as e:
            logger.exception("couldn't load the screening call (metadata %r)", ctx.job.metadata)
            load_error = repr(e)

    caller = CallerText()
    detector = ShortAnswerTurnDetector(caller, lambda: last_agent_text(session))
    session = AgentSession(
        stt=stt(),
        llm=main_llm(),
        tts=tts(loaded.job.voice if loaded else DEFAULT_VOICE),
        vad=_vad,
        turn_handling=turn_handling(detector),
    )
    watch_caller(session, caller)
    log_reply_stages(session)

    # Screening calls log turn timings only: what the candidate says stays in the call's
    # transcript, which deleting the candidate purges. The live view (a hidden viewer in the
    # room) gets the same timings, never words, plus question progress on screening calls.
    publisher = RoomPublisher(ctx.room)
    tracker = DecisionTracker.open(
        ctx.room.name,
        redact=screening,
        on_record=lambda record: publisher.publish(TURNS_TOPIC, turn_record(record)),
    )
    watch(session, tracker)

    async def close_log() -> None:
        tracker.close(time.time())

    ctx.add_shutdown_callback(close_log)

    if loaded is not None:
        job = loaded.job
        decider = default_jev(get_settings().typesafe_api_key)
        publish_progress(session, publisher, loaded.criteria.requirements, decider)
        agent = ScreeningAgent(
            instructions=build_instructions(
                job, loaded.candidate, loaded.criteria, now=datetime.now(ZoneInfo(job.timezone))
            ),
            greeting=greeting(job, loaded.candidate),
            timezone=job.timezone,
            hang_up=session.shutdown,
            detect_machine=lambda: answered_by_machine(session),
        )
        save_on_shutdown(ctx, session, agent, call_id)
        try:
            await session.start(agent=agent, room=ctx.room, room_options=room_options())
        except Exception as e:
            agent.error = f"the session failed to start: {e!r}"  # saved as a failed call
            raise
    elif screening:
        await session.start(
            agent=Agent(instructions="You can't help on this call. Say goodbye politely."),
            room=ctx.room,
            room_options=room_options(),
        )
        await apologise_and_hang_up(session, call_id, load_error)
    else:
        await session.start(
            agent=ReliableAgent(instructions=INSTRUCTIONS),
            room=ctx.room,
            room_options=room_options(),
        )
        await session.say(GREETING)


def main() -> None:
    agents.cli.run_app(server)
