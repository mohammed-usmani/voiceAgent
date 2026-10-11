"""TypeSafe's Jev decision model (early access): POST /v1/systemone with a state and named
questions (score, choice or noul), each answered independently.

Production uses it for three decisions the decision benchmark measured (evals/bench):
rubric scoring and reviewer checks in the assessment, and question progress during a call.
Each caller falls back to its previous LLM or matcher when Jev fails."""

import asyncio
import random
import time

import httpx

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class JevError(RuntimeError):
    pass


class Jev:
    """`ask` retries 429s, 5xx and network errors with short exponential backoff, at most
    `attempts` times and within `budget_s`; anything else raises JevError. `ask_async` is
    for the call: one attempt, a short timeout, no retries."""

    def __init__(
        self,
        api_key: str,
        *,
        attempts: int = 3,
        timeout_s: float = 4.0,
        budget_s: float = 10.0,
        backoff_s: float = 0.5,
        backoff_cap_s: float = 2.0,
    ) -> None:
        if not api_key:
            raise JevError("TYPESAFE_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout_s, headers=self._headers)
        self.attempts, self.budget_s = attempts, budget_s
        self.backoff_s, self.backoff_cap_s = backoff_s, backoff_cap_s
        self.version = MODEL  # the model version of the last answer, e.g. "jev-1.13"

    def ask(self, state: str, questions: dict) -> dict:
        """Answers by question id."""
        return self._ask(state, questions)[0]

    def _post(self, body: dict) -> httpx.Response:
        return self._client.post(URL, json=body)

    def _ask(self, state: str, questions: dict) -> tuple[dict, dict, int]:
        """(answers, the whole response, attempts made)."""
        body = {"state": state, "model": MODEL, "questions": questions}
        start, last = time.monotonic(), ""
        for attempt in range(1, self.attempts + 1):
            try:
                r = self._post(body)
            except httpx.HTTPError as e:
                r, last = None, repr(e)
            if r is not None and r.status_code == 200:
                data = r.json()
                return self._answers(data, questions), data, attempt
            if r is not None and r.status_code != 429 and r.status_code < 500:
                raise JevError(f"HTTP {r.status_code}: {r.text[:300]}")
            if r is not None:
                last = f"HTTP {r.status_code}"
            wait = min(self.backoff_s * 2 ** (attempt - 1), self.backoff_cap_s)
            wait *= 0.5 + random.random()
            if attempt == self.attempts or time.monotonic() - start + wait > self.budget_s:
                break
            time.sleep(wait)
        raise JevError(f"gave up after {attempt} attempts: {last}")

    async def ask_async(self, state: str, questions: dict, timeout_s: float = 2.0) -> dict:
        # ponytail: a new connection per request (~a TLS handshake each); keep one
        # AsyncClient per call if that latency shows up on the live view.
        body = {"state": state, "model": MODEL, "questions": questions}
        try:
            async with asyncio.timeout(timeout_s):
                async with httpx.AsyncClient(timeout=timeout_s, headers=self._headers) as c:
                    r = await c.post(URL, json=body)
        except (httpx.HTTPError, TimeoutError) as e:
            raise JevError(repr(e)) from e
        if r.status_code != 200:
            raise JevError(f"HTTP {r.status_code}: {r.text[:300]}")
        return self._answers(r.json(), questions)

    def _answers(self, data: dict, questions: dict) -> dict:
        answers = data.get("answers") or {}
        missing = set(questions) - set(answers)
        if missing:
            raise JevError(f"no answer for {sorted(missing)}")
        self.version = data.get("model") or MODEL
        return answers


def default_jev(api_key: str) -> Jev | None:
    """The production client, or None without a key (the LLM and matcher paths)."""
    return Jev(api_key) if api_key else None
