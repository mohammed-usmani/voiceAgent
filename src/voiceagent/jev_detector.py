import logging
import os
import httpx

logger = logging.getLogger("voiceagent.jev")

JEV_API_KEY = os.getenv("TYPESAFE_API_KEY")
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

QUESTIONS_SCHEMA = {
    "is_turn_complete": {
        "type": "noul",
        "instructions": "Determine whether the speaker has concluded an active turn and expects an agent response.",
        "criteria": {
            "true": "The user asked a question, made a substantive statement, or gave an instruction requiring an agent response.",
            "false": "The user is merely acknowledging (e.g., 'right', 'yeah', 'uh-huh') while the agent was speaking, or is pausing mid-thought/filler.",
        },
    },
    "turn_action": {
        "type": "choice",
        "instructions": "Which conversational action best fits the state?",
        "criteria": {
            "respond": "The speaker completed their thought or question. The assistant must speak now.",
            "wait_pause": "The speaker paused mid-sentence, used a filler word, or is formulating a continuation.",
            "backchannel": "The speaker gave a short confirmation/listening cue while the agent was speaking, not a turn-take.",
        },
    },
}


class JevTurnDetector:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("TYPESAFE_API_KEY") or JEV_API_KEY or ""
        self.client = httpx.AsyncClient(
            base_url="https://api.typesafe.ai",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=1.5,  # 1.5s timeout for network roundtrips
        )

    async def evaluate_turn(
        self,
        recent_transcript: str,
        dialogue_history: list,
        silence_ms: int,
        assistant_was_speaking: bool = False,
    ) -> dict:
        state = {
            "dialogue_history": dialogue_history[-4:],  # Keep sliding window small for latency
            "recent_transcript": recent_transcript,
            "silence_duration_ms": silence_ms,
            "assistant_was_speaking": assistant_was_speaking,
        }

        try:
            resp = await self.client.post(
                "/v1/systemone",
                json={
                    "model": "jev-latest",
                    "state": state,
                    "questions": QUESTIONS_SCHEMA,
                },
            )
            if resp.status_code != 200:
                logger.error("Jev API error %s: %s", resp.status_code, resp.text)
                return {"action": "respond", "confidence": 1.0, "is_complete": 1.0}

            data = resp.json()
            answers = data.get("answers", {})
            turn_action = answers.get("turn_action", {})
            is_turn_complete = answers.get("is_turn_complete", {})

            return {
                "action": turn_action.get("choice", "respond"),
                "confidence": turn_action.get("confidence", 1.0),
                "is_complete": float(is_turn_complete.get("noul", 1.0)),
            }
        except Exception as e:
            logger.warning("Jev request failed (%s), falling back to respond: %s", type(e).__name__, e)
            return {"action": "respond", "confidence": 1.0, "is_complete": 1.0}