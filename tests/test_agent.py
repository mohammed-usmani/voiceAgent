from livekit.agents import inference

from voiceagent.agent import turn_handling


def test_turn_handling_is_livekit_recommended_setup():
    # docs.livekit.io/agents/logic/turns/tuning: default detector, adaptive interruption,
    # endpointing and preemptive generation left to LiveKit's defaults (preemptive on).
    cfg = turn_handling()
    assert isinstance(cfg["turn_detection"], inference.TurnDetector)
    assert cfg["interruption"] == {"mode": "adaptive", "min_duration": 0.5, "min_words": 0}
    assert set(cfg) == {"turn_detection", "interruption"}


def test_speech_models_accept_their_settings(monkeypatch):
    # Plugins validate options at construction; a bad value must fail here, not on a call.
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    from voiceagent.agent import stt, tts

    assert tts().model == "sonic-3.6"
    assert stt().model == "nova-3"
