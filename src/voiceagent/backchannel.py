"""Short listener cues that should not be treated as taking the turn."""

import re

BACKCHANNELS = frozenset(
    {
        "yeah",
        "yes",
        "yep",
        "uh-huh",
        "mm-hmm",
        "mhm",
        "okay",
        "ok",
        "right",
        "sure",
        "got it",
        "i see",
    }
)
_SINGLE_WORDS = frozenset(b for b in BACKCHANNELS if " " not in b)


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s'-]", " ", text.lower()).split()


def is_backchannel(text: str) -> bool:
    tokens = words(text)
    if not tokens:
        return False
    return " ".join(tokens) in BACKCHANNELS or all(t in _SINGLE_WORDS for t in tokens)
