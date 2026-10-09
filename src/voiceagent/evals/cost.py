"""What an evaluation run spends, from the token usage the providers report, and the
budget that stops it."""

from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback
from livekit.agents import llm

# USD per million tokens (input, output): Together AI list prices, 9 Oct 2026
# (https://www.together.ai/pricing). Update them here when they change.
PRICES: dict[str, tuple[float, float]] = {
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1.32, 3.96),
    "MiniMaxAI/MiniMax-M3": (0.30, 1.20),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
    "zai-org/GLM-5.3": (1.40, 4.40),
    "offline": (0.0, 0.0),
}
# A model missing from the table is charged at the dearest price, so the budget errs high.
UNKNOWN_PRICE = max(PRICES.values())
# A voice-model request whose provider reported no usage is counted as this many tokens
# (input, output), so the budget never treats it as free.
UNREPORTED = (4000, 500)


class BudgetExceeded(RuntimeError):
    pass


def price(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    low = model.lower()
    for name, p in PRICES.items():  # providers sometimes drop or add the org prefix
        if name.lower().endswith(low) or low.endswith(name.lower()):
            return p
    return UNKNOWN_PRICE


class Meter:
    """Tokens per model and their cost. LangChain calls (the backend agents, judges and
    simulated candidate) are counted inside `tracking()`; LiveKit models through
    `watch()`."""

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = budget_usd
        self._counted: dict[str, list[int]] = {}
        self._langchain: UsageMetadataCallbackHandler | None = None

    @contextmanager
    def tracking(self) -> Iterator["Meter"]:
        """Counts every LangChain model call made in this context, including in threads
        and LangGraph nodes started from it."""
        with get_usage_metadata_callback() as cb:
            self._langchain = cb
            try:
                yield self
            finally:
                self._absorb(cb)

    def _absorb(self, cb: UsageMetadataCallbackHandler) -> None:
        for model, u in cb.usage_metadata.items():
            self.add(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
        self._langchain = None

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        t = self._counted.setdefault(model, [0, 0])
        t[0] += input_tokens
        t[1] += output_tokens

    def watch(self, model: llm.LLM) -> llm.LLM:
        """Count a LiveKit model's usage from its metrics events."""
        name = model.model

        def on_metrics(m) -> None:
            reported = (m.prompt_tokens, m.completion_tokens)
            self.add(name, *(reported if any(reported) else UNREPORTED))

        model.on("metrics_collected", on_metrics)
        return model

    def tokens(self) -> dict[str, tuple[int, int]]:
        out = {m: (t[0], t[1]) for m, t in self._counted.items()}
        if self._langchain is not None:
            for m, u in self._langchain.usage_metadata.items():
                i, o = out.get(m, (0, 0))
                out[m] = (i + u.get("input_tokens", 0), o + u.get("output_tokens", 0))
        return out

    @property
    def usd(self) -> float:
        total = 0.0
        for model, (i, o) in self.tokens().items():
            p_in, p_out = price(model)
            total += (i * p_in + o * p_out) / 1_000_000
        return round(total, 6)

    def check(self) -> None:
        """Raise BudgetExceeded once the run has spent more than its budget."""
        if self.budget_usd is not None and self.usd > self.budget_usd:
            raise BudgetExceeded(f"spent ${self.usd:.4f} of a ${self.budget_usd:.2f} budget")
