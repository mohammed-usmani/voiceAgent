import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from tests.fakes import ScriptedLLM
from voiceagent.evals.cost import PRICES, UNKNOWN_PRICE, UNREPORTED, BudgetExceeded, Meter, price

FLASH = "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_prices_come_from_the_table_and_unknown_models_cost_the_most():
    assert price(FLASH) == (0.14, 0.28)
    assert price("DeepSeek-V4-Flash-0731") == (0.14, 0.28)  # provider dropped the org
    assert price("some/new-model") == UNKNOWN_PRICE == max(PRICES.values())


def test_cost_is_tokens_times_price():
    m = Meter()
    m.add(FLASH, 1_000_000, 500_000)
    m.add("Qwen/Qwen3.5-9B", 2_000_000, 0)
    assert m.usd == pytest.approx(0.14 + 0.14 + 0.34)


def test_the_budget_stops_the_run_only_once_exceeded():
    m = Meter(budget_usd=0.14)
    m.add(FLASH, 1_000_000, 0)
    m.check()  # exactly at the budget: carry on
    m.add(FLASH, 10, 0)
    with pytest.raises(BudgetExceeded, match=r"of a \$0.14 budget"):
        m.check()


def test_no_budget_never_stops():
    m = Meter(budget_usd=None)
    m.add(FLASH, 10**9, 10**9)
    m.check()


async def test_langchain_usage_is_counted_inside_threads():
    reply = AIMessage(
        "hi",
        usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
        response_metadata={"model_name": FLASH},
    )
    chat = FakeMessagesListChatModel(responses=[reply, reply])
    m = Meter()
    with m.tracking():
        chat.invoke("hello")
        await asyncio.to_thread(chat.invoke, "hello again")
        assert m.tokens() == {FLASH: (2000, 400)}  # live, for the budget check
    assert m.tokens() == {FLASH: (2000, 400)}  # and kept after the context ends
    chat.invoke("outside")
    assert m.tokens() == {FLASH: (2000, 400)}


def test_livekit_models_are_counted_from_their_metrics():
    model = ScriptedLLM([])
    m = Meter()
    m.watch(model)
    model.emit("metrics_collected", SimpleNamespace(prompt_tokens=300, completion_tokens=20))
    model.emit("metrics_collected", SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    assert m.tokens() == {model.model: (300 + UNREPORTED[0], 20 + UNREPORTED[1])}


def test_langchain_usage_is_counted_inside_langgraph_nodes():
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    reply = AIMessage(
        "hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": FLASH},
    )
    chat = FakeMessagesListChatModel(responses=[reply])

    class S(TypedDict, total=False):
        x: str

    graph = StateGraph(S)
    graph.add_node("ask", lambda s: {"x": chat.invoke("hello").content})
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    m = Meter()
    with m.tracking():
        graph.compile().invoke({})
    assert m.tokens() == {FLASH: (10, 5)}
