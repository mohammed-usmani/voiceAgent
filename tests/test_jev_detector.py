import json

import httpx
import pytest
from livekit.agents import llm

from voiceagent.jev_detector import (
    FALLBACK,
    HOLD_PROBABILITY,
    JevClient,
    JevTurnDetector,
    TurnDecision,
)


def jev_response(action: str, completeness: float) -> dict:
    return {
        "answers": {
            "turn_action": {"choice": action, "confidence": 0.9},
            "is_turn_complete": {"noul": completeness},
        }
    }


@pytest.mark.parametrize(
    ("action", "completeness", "expected"),
    [
        ("respond", 0.95, 0.95),  # finished turn: pass Jev's confidence through
        ("respond", 0.50, HOLD_PROBABILITY),  # "respond" but low completeness: hold
        ("wait_pause", 0.90, HOLD_PROBABILITY),  # mid-sentence pause: hold
        ("backchannel", 0.10, 0.10),  # never raise a low score
    ],
)
def test_end_of_turn_probability(action, completeness, expected):
    assert TurnDecision(action, completeness).end_of_turn_probability() == expected


def test_from_response_defaults_to_respond():
    assert TurnDecision.from_response({}) == FALLBACK


def client_returning(handler) -> JevClient:
    client = JevClient(api_key="test")
    client._http = httpx.AsyncClient(
        base_url="https://jev.test", transport=httpx.MockTransport(handler)
    )
    return client


def timeout(req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("slow")


@pytest.mark.parametrize(
    "handler",
    [
        lambda req: httpx.Response(500),
        lambda req: httpx.Response(200, content=b"not json"),
        timeout,
    ],
)
async def test_client_fails_open(handler):
    assert await client_returning(handler).evaluate_turn("hi", [], 200) == FALLBACK


async def test_detector_sends_windowed_history():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content)["state"])
        return httpx.Response(200, json=jev_response("wait_pause", 0.3))

    ctx = llm.ChatContext()
    for i in range(6):
        ctx.add_message(role="user" if i % 2 else "assistant", content=f"m{i}")
    ctx.add_message(role="user", content="my laptop is, um")

    detector = JevTurnDetector(client_returning(handler), silence_ms=200)
    assert await detector.predict_end_of_turn(ctx) == HOLD_PROBABILITY
    assert seen["recent_transcript"] == "my laptop is, um"
    assert [m["text"] for m in seen["dialogue_history"]] == ["m2", "m3", "m4", "m5"]
    assert seen["silence_duration_ms"] == 200
