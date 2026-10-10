"""GET /evals/runs: evaluation runs in evals/runs/, read-only."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeModels
from voiceagent.api.app import create_app
from voiceagent.db.base import make_sessionmaker

KEY = {"X-API-Key": "test-key"}
SUMMARY = {
    "decision.accuracy": 0.85,
    "decision.false_rejection_rate": 0.05,
    "criteria.recall": 0.84,
    "cost.per_call_usd": 0.011,
    "calls.n": 28,
}


def write_run(root, name, *, offline=False, usd=0.31):
    d = root / name
    d.mkdir()
    results = {
        "name": name,
        "started_at": "2026-10-09T22:34:14+00:00",
        "finished_at": "2026-10-09T22:51:02+00:00",
        "config": {"offline": offline, "jobs": ["staff-nurse"], "personas": 4, "runs": 1},
        "aborted": None,
        "summary": SUMMARY,
        "cost": {"usd": usd, "tokens": {}},
    }
    (d / "results.json").write_text(json.dumps(results))


def app_client(engine, evals_dir):
    app = create_app(
        sessionmaker=make_sessionmaker(engine),
        models=FakeModels(),
        api_key="test-key",
        evals_dir=evals_dir,
    )
    return TestClient(app)


@pytest.fixture
def client(engine, tmp_path):
    write_run(tmp_path, "20261009T192610Z")
    write_run(tmp_path, "20261009T223414Z", offline=True, usd=0.0)
    (tmp_path / "20261010T000000Z").mkdir()  # a run still going: no results.json yet
    broken = tmp_path / "20261010T010000Z"
    broken.mkdir()
    (broken / "results.json").write_text("{not json")
    (tmp_path / "notes.txt").write_text("not a run")
    return app_client(engine, tmp_path)


def test_runs_are_listed_newest_first_with_their_headline(client):
    r = client.get("/evals/runs", headers=KEY)
    assert r.status_code == 200
    runs = r.json()
    assert [run["name"] for run in runs] == ["20261009T223414Z", "20261009T192610Z"]
    assert [(run["offline"], run["cost_usd"]) for run in runs] == [(True, 0.0), (False, 0.31)]
    assert runs[1]["headline"] == {
        "decision.accuracy": 0.85,
        "decision.false_rejection_rate": 0.05,
        "criteria.recall": 0.84,
        "cost.per_call_usd": 0.011,  # the overview's cost-per-call tile
    }
    assert runs[1]["aborted"] is None


def test_a_run_is_served_whole(client):
    r = client.get("/evals/runs/20261009T192610Z", headers=KEY)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["summary"]["calls.n"] == 28


@pytest.mark.parametrize(
    "name",
    ["20261001T000000Z", "20261010T000000Z", "notes.txt", "..", "..%2F..%2Fpyproject.toml"],
)
def test_only_listed_runs_can_be_read(client, name):
    assert client.get(f"/evals/runs/{name}", headers=KEY).status_code == 404


def test_no_runs_directory_lists_nothing(engine, tmp_path):
    assert app_client(engine, tmp_path / "missing").get("/evals/runs", headers=KEY).json() == []


def test_runs_need_the_api_key(client):
    assert client.get("/evals/runs").status_code == 401
