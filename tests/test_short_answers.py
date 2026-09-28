import asyncio

import pytest
from livekit.agents.voice.turn import TurnDetectionEvent

from voiceagent.short_answers import CallerText, ShortAnswerStream, is_short_answer

QUESTION = "Great, thanks. Are you open to relocating to Bangalore?"


@pytest.mark.parametrize(
    "text", ["No.", "Yes.", "yeah", "Nope!", "No, no.", "Not really.", "Sure."]
)
def test_bare_yes_or_no_to_a_question_ends_the_turn(text):
    assert is_short_answer(text, QUESTION)


@pytest.mark.parametrize(
    "text",
    [
        "Okay.",  # often followed by the actual answer
        "No, but I could relocate next year",
        "I mostly work on voice agents.",
        "",
    ],
)
def test_anything_else_is_left_to_the_audio_model(text):
    assert not is_short_answer(text, QUESTION)


def test_only_after_a_question():
    assert not is_short_answer("No.", "Thanks, the team will email next steps.")


class FakeAudioStream:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict(self):
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(
            TurnDetectionEvent(type="eot", end_of_turn_probability=self.p, last_speaking_time=0.0)
        )
        return fut


async def decide(p_audio: float, caller_says: str) -> float:
    caller = CallerText()
    caller.add_final(caller_says)
    stream = ShortAnswerStream(FakeAudioStream(p_audio), caller, lambda: QUESTION)
    return (await stream.predict()).end_of_turn_probability


async def test_raises_an_unsure_audio_score_for_a_short_answer():
    assert await decide(0.2, "No.") == 0.95


async def test_never_lowers_the_audio_score():
    assert await decide(0.99, "No.") == 0.99
    assert await decide(0.9, "I mostly work on") == 0.9


async def test_leaves_long_answers_to_the_audio_model():
    assert await decide(0.2, "I mostly work on voice agents") == 0.2


async def test_waits_for_a_transcript_that_lands_after_the_audio_prediction():
    caller = CallerText()
    stream = ShortAnswerStream(FakeAudioStream(0.3), caller, lambda: QUESTION)
    fut = stream.predict()
    await asyncio.sleep(0.1)  # audio prediction is done; transcript not yet in
    caller.add_final("Yes.")
    assert (await fut).end_of_turn_probability == 0.95


async def test_gives_up_waiting_and_keeps_the_audio_score():
    caller = CallerText()
    stream = ShortAnswerStream(FakeAudioStream(0.3), caller, lambda: QUESTION)
    assert (await stream.predict()).end_of_turn_probability == 0.3
