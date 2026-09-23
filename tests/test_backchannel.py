import pytest

from voiceagent.backchannel import is_backchannel, words


def test_words_strips_punctuation_and_case():
    assert words("Yeah, OK... uh-huh!") == ["yeah", "ok", "uh-huh"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yeah", True),
        ("Mm-hmm.", True),
        ("got it", True),
        ("yeah okay", True),
        ("", False),
        ("yeah but that's wrong", False),
        ("wait", False),
    ],
)
def test_is_backchannel(text, expected):
    assert is_backchannel(text) is expected
