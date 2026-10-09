"""Deterministic stand-ins for the model clients."""

import hashlib
import math
import re

from voiceagent.settings import EMBEDDING_DIM


def hash_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Bag-of-words vector: texts sharing words land close together."""
    v = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeModels:
    name = "fake"

    def __init__(
        self, replies: dict[type, list] | None = None, embed_error: Exception | None = None
    ):
        self.replies = {k: list(v) for k, v in (replies or {}).items()}
        self.embed_error = embed_error
        self.prompts: list[tuple[str, str, str]] = []

    def structured(self, schema, system, user):
        self.prompts.append((schema.__name__, system, user))
        queue = self.replies.get(schema)
        if not queue:
            raise AssertionError(f"no scripted reply for {schema.__name__}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def embed(self, texts):
        if self.embed_error:
            raise self.embed_error
        return [hash_embed(t) for t in texts]
