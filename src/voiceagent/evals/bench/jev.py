"""The benchmark's Jev: production's client (voiceagent.jev) with the bench's patience
(6 attempts, backoff up to 30 s), its in-flight limit and its metering."""

import os
import threading
import time

from voiceagent import jev
from voiceagent.evals.bench.common import JEV_WORKERS
from voiceagent.evals.cost import Meter
from voiceagent.jev import JevError

__all__ = ["Jev", "JevError"]
PRICE_KEY = "typesafe/jev-latest"  # in voiceagent.evals.cost.PRICES
ATTEMPTS = 6


class Jev(jev.Jev):
    def __init__(self, meter: Meter) -> None:
        from voiceagent.settings import get_settings

        get_settings()  # loads .env
        super().__init__(
            os.environ.get("TYPESAFE_API_KEY", ""),
            attempts=ATTEMPTS,
            timeout_s=60,
            budget_s=float("inf"),
            backoff_s=2,
            backoff_cap_s=30,
        )
        self._slots = threading.BoundedSemaphore(JEV_WORKERS)
        self._lock = threading.Lock()
        self.meter = meter
        self.versions: set[str] = set()
        self.retries = 0

    def _post(self, body: dict):
        with self._slots:
            return super()._post(body)

    def ask(self, state: str, questions: dict) -> tuple[dict, float, int]:
        """(answers by question id, seconds including any retries, attempts)."""
        start = time.perf_counter()
        answers, data, attempt = self._ask(state, questions)
        u = data.get("usage") or {}
        with self._lock:
            self.meter.add(PRICE_KEY, u.get("input_tokens", 0), u.get("output_tokens", 0))
            self.versions.add(data.get("model", "?"))
            self.retries += attempt - 1
        return answers, time.perf_counter() - start, attempt
