import pytest
from pydantic import BaseModel

from voiceagent.llm import FallbackModels, StructuredOutputError, default_models
from voiceagent.settings import Settings


class Out(BaseModel):
    title: str


class FakeChat:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        return [-float(len(text))]


def models(main, backup):
    return FallbackModels(main, backup, FakeEmbedder(), name="test")


def test_main_model_answers():
    main, backup = FakeChat(Out(title="Engineer")), FakeChat()
    assert models(main, backup).structured(Out, "sys", "user") == Out(title="Engineer")
    assert (main.calls, backup.calls) == (1, 0)


def test_main_retried_once_then_backup():
    main = FakeChat(RuntimeError("500"), None)  # an error, then an unparsed reply
    backup = FakeChat(Out(title="Engineer"))
    assert models(main, backup).structured(Out, "sys", "user").title == "Engineer"
    assert (main.calls, backup.calls) == (2, 1)


def test_all_failures_raise_with_every_reason():
    main = FakeChat(RuntimeError("quota"), RuntimeError("quota"))
    backup = FakeChat(RuntimeError("down"))
    with pytest.raises(StructuredOutputError) as e:
        models(main, backup).structured(Out, "sys", "user")
    assert "quota" in str(e.value) and "down" in str(e.value)


def test_embed_delegates():
    assert models(FakeChat(), FakeChat()).embed(["ab", "c"]) == [[2.0], [1.0]]


def test_default_clients_leave_retries_to_fallback_models_and_time_out():
    settings = Settings(database_url="", api_key="", gemini_api_key="g", groq_api_key="q")
    m = default_models(settings)
    (_, main), _, (_, backup) = m._attempts
    assert (main.max_retries, main.timeout) == (0, 60)
    assert (backup.max_retries, backup.request_timeout) == (0, 60)


def test_embed_query_uses_the_query_embedding():
    assert models(FakeChat(), FakeChat()).embed_query("abc") == [-3.0]
