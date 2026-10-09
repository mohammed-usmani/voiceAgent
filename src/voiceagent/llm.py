"""Model clients for the backend agents: structured output with a fallback provider,
and embeddings."""

import logging
import os
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel

from voiceagent.settings import Settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MODEL_TIMEOUT_S = 60
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
# Waits before the 2nd and 3rd attempts: back-to-back retries hit the same "high demand"
# 503 from the provider.
BACKOFF_S = (2, 4)


class Models(Protocol):
    name: str

    def structured(self, schema: type[T], system: str, user: str) -> T: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class StructuredOutputError(RuntimeError):
    pass


class FallbackModels:
    """Main model, retried once, then the backup model on a different provider."""

    def __init__(self, main, backup, embedder, *, name: str) -> None:
        self._attempts = (("main", main), ("main", main), ("backup", backup))
        self._embedder = embedder
        self.name = name

    def structured(self, schema: type[T], system: str, user: str) -> T:
        messages = [("system", system), ("human", user)]
        reasons: list[str] = []
        for attempt, (label, chat) in enumerate(self._attempts):
            if attempt:
                time.sleep(BACKOFF_S[attempt - 1])
            try:
                out = chat.with_structured_output(schema).invoke(messages)
            except Exception as e:  # provider errors, schema validation errors
                logger.warning("%s model failed on %s: %r", label, schema.__name__, e)
                reasons.append(f"{label}: {e!r}")
                continue
            if isinstance(out, schema):
                return out
            reasons.append(f"{label}: returned {type(out).__name__}")
        raise StructuredOutputError(f"{schema.__name__}: " + "; ".join(reasons))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embeds a search query (Gemini's RETRIEVAL_QUERY task, vs. documents)."""
        return self._embedder.embed_query(text)


class LocalEmbedder:
    """BAAI/bge-base-en-v1.5 on the CPU via fastembed: 768 dims, no API calls."""

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5") -> None:
        from fastembed import TextEmbedding

        cache = os.path.expanduser("~/.cache/fastembed")  # the default lives in /tmp
        self._model = TextEmbedding(model, cache_dir=cache)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge expects a retrieval instruction on queries; query_embed adds it.
        return next(iter(self._model.query_embed([text]))).tolist()


def _together_chat(model: str, api_key: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=TOGETHER_BASE_URL,
        temperature=0,
        max_retries=0,  # FallbackModels owns the retry policy
        timeout=MODEL_TIMEOUT_S,
        # Reasoning stays on for these offline agents and is billed as output; the
        # benchmark ran with this cap and no reply was cut off.
        max_tokens=16000,
    )


class _JsonSchemaChat:
    """Structured output through Together's JSON-schema mode, which every candidate model
    followed in the benchmark (function calling broke on the large criteria schema)."""

    def __init__(self, chat) -> None:
        self._chat = chat

    def __getattr__(self, name):
        return getattr(self._chat, name)

    def with_structured_output(self, schema):
        return self._chat.with_structured_output(schema, method="json_schema")


def default_models(settings: Settings) -> FallbackModels:
    # Worst case per structured() call: 3 attempts x MODEL_TIMEOUT_S plus the backoff.
    main = _JsonSchemaChat(_together_chat(settings.main_model, settings.together_api_key))
    backup = _JsonSchemaChat(_together_chat(settings.backup_model, settings.together_api_key))
    return FallbackModels(
        main,
        backup,
        LocalEmbedder(settings.embedding_model),
        name=f"{settings.main_model} (backup {settings.backup_model})",
    )
