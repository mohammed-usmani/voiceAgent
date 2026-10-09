"""Deterministic stand-ins for the model clients."""

import hashlib
import itertools
import json
import math
import re

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from voiceagent.outcomes.email import Email
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
        self,
        replies: dict[type, list] | None = None,
        embed_error: Exception | None = None,
        reviewer: "FakeModels | None" = None,
    ):
        self.replies = {k: list(v) for k, v in (replies or {}).items()}
        self.embed_error = embed_error
        self.prompts: list[tuple[str, str, str]] = []
        self.reviewer = reviewer if reviewer is not None else self

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

    def embed_query(self, text):
        if self.embed_error:
            raise self.embed_error
        return hash_embed(text)


_call_ids = itertools.count()


def tool_call(name: str, **arguments) -> llm.FunctionToolCall:
    return llm.FunctionToolCall(
        name=name, arguments=json.dumps(arguments), call_id=f"call_{next(_call_ids)}"
    )


class ScriptedLLM(llm.LLM):
    """A model that replies from a script, one entry per request: text, a `tool_call(...)`,
    or a list of both (a sentence, then a tool call). Out of script, it replies with
    nothing. Every request's chat context is kept in `requests`."""

    def __init__(self, script: list) -> None:
        super().__init__()
        self.script = list(script)
        self.requests: list[llm.ChatContext] = []

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **_kw):
        self.requests.append(chat_ctx.copy())
        reply = self.script.pop(0) if self.script else ""
        return _ScriptedStream(
            self, reply, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _ScriptedStream(llm.LLMStream):
    def __init__(self, owner: ScriptedLLM, reply, **kwargs) -> None:
        super().__init__(owner, **kwargs)
        self._reply = reply

    async def _run(self) -> None:
        parts = self._reply if isinstance(self._reply, list) else [self._reply]
        for i, part in enumerate(parts):
            if isinstance(part, str):
                delta = llm.ChoiceDelta(role="assistant", content=part)
            else:
                delta = llm.ChoiceDelta(role="assistant", tool_calls=[part])
            self._event_ch.send_nowait(llm.ChatChunk(id=f"chunk_{i}", delta=delta))


class FakeMailer:
    """Records what would have been emailed; `error` makes every send fail."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.sent: list[Email] = []

    def send(self, email: Email) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append(email)
