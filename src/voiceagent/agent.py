import logging
from pathlib import Path
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    room_io,
)
from livekit.plugins import silero, deepgram, cartesia
from voiceagent.jev_detector import JevTurnDetector

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an expert phone support technician. "
                "CRITICAL INSTRUCTION: Reply in exactly ONE or TWO short, punchy sentences. "
                "Never exceed 25 words total. Never offer multi-step lists or paragraphs. "
                "Ask one clarifying question at a time so the caller can respond naturally."
            ),
        )


# Pre-load Silero VAD once at module/process start so incoming calls don't block on ONNX initialization
_vad = silero.VAD.load(
    min_speech_duration=0.1,
    min_silence_duration=0.20,
)

# Keep 1 idle process pre-warmed so incoming calls attach instantly without cold-start race conditions
server = AgentServer(
    num_idle_processes=1,
)

class JevLiveKitTurnDetector:
    """
    LiveKit-compatible Turn Detector that delegates end-of-turn evaluation
    to Jev (Typesafe AI API). Implements the LiveKit _TurnDetector protocol.
    """

    def __init__(
        self, detector: JevTurnDetector | None = None, unlikely_threshold: float = 0.5
    ) -> None:
        self._detector = detector or JevTurnDetector()
        self._unlikely_threshold = unlikely_threshold

    @property
    def model(self) -> str:
        return "jev"

    @property
    def provider(self) -> str:
        return "typesafe"

    async def supports_language(self, language: object) -> bool:
        return True

    async def unlikely_threshold(self, language: object) -> float:
        return self._unlikely_threshold

    async def predict_end_of_turn(
        self, chat_ctx: agents.llm.ChatContext, *, timeout: float | None = None
    ) -> float:
        """
        Evaluates current utterance and conversation history using Jev.
        Returns end-of-turn probability between 0.0 and 1.0.
        If probability < unlikely_threshold, LiveKit waits up to max_delay.
        """
        messages = list(chat_ctx.messages())
        if not messages:
            return 1.0

        current_message = messages[-1]
        recent_transcript = current_message.text_content or ""

        # Format sliding window dialogue history
        dialogue_history = []
        for m in messages[:-1][-4:]:
            dialogue_history.append({"speaker": m.role, "text": m.text_content or ""})

        try:
            decision = await self._detector.evaluate_turn(
                recent_transcript=recent_transcript,
                dialogue_history=dialogue_history,
                silence_ms=200,
                assistant_was_speaking=False,
            )

            action = decision.get("action", "respond")
            is_complete = float(decision.get("is_complete", 1.0))

            # Case 1: Active user speaking completed
            if action == "respond" and is_complete >= 0.70:
                prob = is_complete
            else:
                # Case 2: Passive backchannel or mid-sentence pause
                # Return low probability (< unlikely_threshold of 0.5) so LiveKit extends delay to max_delay
                prob = min(is_complete, 0.2)

            logging.info(
                "Jev Turn: '%s' -> action=%s, complete=%.2f => prob=%.2f",
                recent_transcript,
                action,
                is_complete,
                prob,
            )
            return prob
        except Exception:
            logging.exception("Error evaluating turn with Jev detector")
            return 1.0


@server.rtc_session(agent_name="clinic-agent")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en",
        ),
        llm="google/gemini-2.5-flash",
        tts=cartesia.TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            sample_rate=24000,
        ),
        vad=_vad,
        turn_handling={
            # Hook the custom Jev detector here
            "turn_detection": JevLiveKitTurnDetector(unlikely_threshold=0.5),
            "interruption": {
                "enabled": True,
            },
            "endpointing": {
                # Fast response (350ms) on complete thoughts; generous wait (2.0s) on fillers/pauses
                "min_delay": 0.35,
                "max_delay": 2.0,
            },
            "preemptive_generation": {
                "enabled": False,
            },
        },
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=None,
            ),
        ),
    )

    await session.generate_reply(
        instructions="Say: 'Hello! What issue can I help fix on your computer today?'"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)