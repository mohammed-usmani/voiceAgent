"""The local embedding model (downloads ~400 MB on first use). Its onnxruntime thread pool
aborts the interpreter at exit in a few percent of full-suite runs (exit 134), so it stays
out of the default run: RUN_LIVE=1 uv run pytest tests/live/test_embedder_live.py"""

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1"),
]


def test_local_embedder_makes_768_dim_query_and_document_vectors():
    from voiceagent.llm import LocalEmbedder

    e = LocalEmbedder()
    docs = e.embed_documents(["Python backend experience", "Forklift licence"])
    query = e.embed_query("years of Python")
    assert [len(v) for v in docs] == [768, 768] and len(query) == 768
    similarity = [sum(a * b for a, b in zip(query, d, strict=True)) for d in docs]
    assert similarity[0] > similarity[1]
