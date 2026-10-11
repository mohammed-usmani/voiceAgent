"""The production Jev client: retries, limits and errors. No network."""

import httpx
import pytest

from voiceagent import jev
from voiceagent.jev import Jev, JevError, default_jev

Q = {"q": {"type": "noul", "instructions": "x"}}
OK = {"answers": {"q": {"noul": 0.9}}, "model": "jev-1.13", "usage": {"input_tokens": 9}}


@pytest.fixture
def client(monkeypatch):
    waits = []
    monkeypatch.setattr(jev.time, "sleep", waits.append)

    def make(*replies, **kw) -> tuple[Jev, list]:
        return _client(replies, kw), waits

    return make


def _client(replies, kw) -> Jev:
    """A Jev whose posts return `replies` in turn (a status code, a body dict, or an
    exception); the waits it would sleep are collected."""
    j, queue = Jev("k", **kw), list(replies)

    def post(body):
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, int):
            return httpx.Response(r, text="busy")
        return httpx.Response(200, json=r)

    j._post = post
    return j


def test_retries_429_5xx_and_network_errors_then_answers(client):
    j, waits = client(429, httpx.ConnectError("down"), OK)
    assert j.ask("s", Q) == {"q": {"noul": 0.9}}
    assert j.version == "jev-1.13"
    assert len(waits) == 2 and all(w <= 2.0 * 1.5 for w in waits)


def test_gives_up_after_three_attempts(client):
    j, waits = client(503, 503, 503)
    with pytest.raises(JevError, match="gave up after 3 attempts: HTTP 503"):
        j.ask("s", Q)
    assert len(waits) == 2


def test_stops_within_its_time_budget(client):
    j, waits = client(503, 503, 503, budget_s=0.1)
    with pytest.raises(JevError, match="gave up after 1 attempts"):
        j.ask("s", Q)
    assert waits == []


def test_a_client_error_is_not_retried(client):
    j, waits = client(400, OK)
    with pytest.raises(JevError, match="HTTP 400"):
        j.ask("s", Q)
    assert waits == []


def test_a_missing_answer_is_an_error(client):
    j, _ = client({"answers": {}})
    with pytest.raises(JevError, match="no answer for"):
        j.ask("s", Q)


def test_no_key_no_jev():
    assert default_jev("") is None
    assert isinstance(default_jev("k"), Jev)
