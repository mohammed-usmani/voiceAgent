from livekit.agents import inference

from voiceagent.agent import turn_handling


def test_turn_handling_is_livekit_recommended_setup():
    # docs.livekit.io/agents/logic/turns/tuning: default detector, adaptive interruption,
    # preemptive generation left on. Only endpointing is tuned.
    cfg = turn_handling()
    assert isinstance(cfg["turn_detection"], inference.TurnDetector)
    assert cfg["interruption"] == {"mode": "adaptive", "min_duration": 0.5, "min_words": 0}
    assert cfg["endpointing"] == {"mode": "dynamic", "max_delay": 2.0}
    assert set(cfg) == {"turn_detection", "endpointing", "interruption"}


def test_speech_models_accept_their_settings(monkeypatch):
    # Plugins validate options at construction; a bad value must fail here, not on a call.
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    from voiceagent.agent import stt, tts

    assert tts().model == "sonic-3.6"
    # "hi" gave English words Hindi pronunciation ("do" came out wrong).
    assert tts()._opts.language.language == "en"
    assert stt().model == "nova-3"
    # "multi" guessed among ten languages on phone audio ("hello" came back as Spanish).
    assert stt()._opts.language == "en-IN"


def test_prompt_handles_repeats_and_answered_questions():
    from voiceagent.agent import INSTRUCTIONS

    assert "word for word" in INSTRUCTIONS
    assert "never re-ask" in INSTRUCTIONS
    assert "Always reply in English" in INSTRUCTIONS


def test_tts_speaks_in_the_jobs_voice(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    from voiceagent.agent import tts
    from voiceagent.settings import DEFAULT_VOICE

    assert tts()._opts.voice == DEFAULT_VOICE
    assert tts("voice-123")._opts.voice == "voice-123"


def test_closing_the_session_hangs_up_the_phone():
    # Deleting the room drops the SIP leg, so end_call (session.shutdown) really hangs up.
    from voiceagent.agent import room_options

    assert room_options().delete_room_on_close is True
