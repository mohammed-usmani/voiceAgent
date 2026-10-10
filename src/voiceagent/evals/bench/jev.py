"""TypeSafe's Jev decision model (early access): POST /v1/systemone with a state and
named questions (score, choice or noul), each answered independently."""

import os
import random
import threading
import time

import httpx

from voiceagent.evals.bench.common import JEV_WORKERS
from voiceagent.evals.cost import Meter

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
PRICE_KEY = "typesafe/jev-latest"  # in voiceagent.evals.cost.PRICES
ATTEMPTS = 6


class JevError(RuntimeError):
    pass


class Jev:
    def __init__(self, meter: Meter) -> None:
        from voiceagent.settings import get_settings

        get_settings()  # loads .env
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key:
            raise JevError("TYPESAFE_API_KEY is not set")
        self._client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})
        self._slots = threading.BoundedSemaphore(JEV_WORKERS)
        self._lock = threading.Lock()
        self.meter = meter
        self.versions: set[str] = set()
        self.retries = 0

    def ask(self, state: str, questions: dict) -> tuple[dict, float, int]:
        """(answers by question id, seconds including any retries, attempts). Retries 429s,
        5xx and network errors with exponential backoff."""
        body = {"state": state, "model": MODEL, "questions": questions}
        start = time.perf_counter()
        last = ""
        for attempt in range(1, ATTEMPTS + 1):
            with self._slots:
                try:
                    r = self._client.post(URL, json=body)
                except httpx.HTTPError as e:
                    r, last = None, repr(e)
            if r is not None and r.status_code == 200:
                data = r.json()
                u = data.get("usage") or {}
                with self._lock:
                    self.meter.add(PRICE_KEY, u.get("input_tokens", 0), u.get("output_tokens", 0))
                    self.versions.add(data.get("model", "?"))
                    self.retries += attempt - 1
                answers = data.get("answers") or {}
                missing = set(questions) - set(answers)
                if missing:
                    raise JevError(f"no answer for {sorted(missing)}")
                return answers, time.perf_counter() - start, attempt
            if r is not None and r.status_code != 429 and r.status_code < 500:
                raise JevError(f"HTTP {r.status_code}: {r.text[:300]}")
            if r is not None:
                last = f"HTTP {r.status_code}"
            time.sleep(min(2**attempt, 30) * (0.5 + random.random()))
        raise JevError(f"gave up after {ATTEMPTS} attempts: {last}")
