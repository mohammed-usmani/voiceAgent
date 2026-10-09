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


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr("voiceagent.llm.time.sleep", lambda s: None)


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


def test_default_clients_run_on_together_and_leave_retries_to_fallback_models():
    settings = Settings(database_url="", api_key="", together_api_key="t")
    m = default_models(settings)
    (_, main), _, (_, backup) = m._attempts
    assert (main.model_name, backup.model_name) == (
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "MiniMaxAI/MiniMax-M3",
    )
    for chat in (main, backup):
        assert chat.openai_api_base == "https://api.together.xyz/v1"
        assert (chat.max_retries, chat.request_timeout) == (0, 60)


def test_retries_wait_between_attempts(monkeypatch):
    # Two back-to-back calls during a provider's "high demand" 503 both failed.
    waits = []
    monkeypatch.setattr("voiceagent.llm.time.sleep", waits.append)
    main = FakeChat(RuntimeError("503"), RuntimeError("503"))
    backup = FakeChat(Out(title="Engineer"))
    models(main, backup).structured(Out, "sys", "user")
    assert waits == [2, 4]


def test_no_wait_after_success(monkeypatch):
    waits = []
    monkeypatch.setattr("voiceagent.llm.time.sleep", waits.append)
    models(FakeChat(Out(title="x")), FakeChat()).structured(Out, "sys", "user")
    assert waits == []


def test_embed_query_uses_the_query_embedding():
    assert models(FakeChat(), FakeChat()).embed_query("abc") == [-3.0]


def test_local_embedder_makes_768_dim_query_and_document_vectors():
    from voiceagent.llm import LocalEmbedder

    e = LocalEmbedder()
    docs = e.embed_documents(["Python backend experience", "Forklift licence"])
    query = e.embed_query("years of Python")
    assert [len(v) for v in docs] == [768, 768] and len(query) == 768
    similarity = [sum(a * b for a, b in zip(query, d, strict=True)) for d in docs]
    assert similarity[0] > similarity[1]
