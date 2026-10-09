"""Model clients for the backend agents: structured output with a fallback provider,
and embeddings."""

import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel

from voiceagent.settings import EMBEDDING_DIM, Settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MODEL_TIMEOUT_S = 60


class Models(Protocol):
    name: str

    def structured(self, schema: type[T], system: str, user: str) -> T: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


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
        for label, chat in self._attempts:
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


def default_models(settings: Settings) -> FallbackModels:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_groq import ChatGroq

    # FallbackModels owns the retry policy, so the clients don't retry on their own.
    # Worst case per structured() call: 3 attempts x MODEL_TIMEOUT_S.
    main = ChatGoogleGenerativeAI(
        model=settings.main_model,
        api_key=settings.gemini_api_key,
        temperature=0,
        max_retries=0,
        timeout=MODEL_TIMEOUT_S,
    )
    backup = ChatGroq(
        model=settings.backup_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=0,
        timeout=MODEL_TIMEOUT_S,
    )
    embedder = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.gemini_api_key,
        output_dimensionality=EMBEDDING_DIM,
    )
    return FallbackModels(
        main, backup, embedder, name=f"{settings.main_model} (backup {settings.backup_model})"
    )
