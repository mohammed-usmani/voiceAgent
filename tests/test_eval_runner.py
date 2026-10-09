import json

import pytest

from voiceagent.evals import cli
from voiceagent.evals.cost import Meter
from voiceagent.evals.jobs import load_jobs
from voiceagent.evals.offline import OfflineModels, OfflineSimulator
from voiceagent.evals.runner import EvalConfig, offline_backend, overrides_lookup, run_eval

FLASH = "deepseek-ai/DeepSeek-V4-Flash-0731"


def config(tmp_path, **kw) -> EvalConfig:
    return EvalConfig(**({"jobs": ["software-engineer"], "personas": 4, "out": tmp_path} | kw))


async def run(cfg, meter=None):
    meter = meter or Meter(cfg.budget_usd)
    with meter.tracking():
        return await run_eval(cfg, offline_backend(), meter)


async def test_an_offline_run_goes_end_to_end_and_writes_the_report(tmp_path):
    r = await run(config(tmp_path, jobs=["sales-executive", "staff-nurse"], personas=9))
    s = r["summary"]
    assert r["aborted"] is None and s["calls.errors"] == 0
    assert s["criteria.recall"] == 1.0 and s["criteria.illegal_dropped"] == 1.0
    assert s["decision.end_reason_accuracy"] == 1.0
    assert s["decision.false_rejection_rate"] == 0.0
    assert s["capture.accuracy"] == 1.0
    assert s["reviewer.invites_challenged"] > 0  # invites go out unattended: reviewed
    assert r["fairness"]["passed"] is True and len(r["fairness"]["results"]) == 4
    kinds = {c["persona"]["kind"] for c in r["calls"]}
    assert {"clear_fit", "miss", "borderline", "bad_time", "refuses_consent"} <= kinds
    calibrated = [c["persona"]["key"] for c in r["calls"] if c["calibrated"]]
    assert sorted(calibrated) == ["borderline", "borderline", "clear_fit", "clear_fit"]
    consent = next(c for c in r["calls"] if c["persona"]["kind"] == "refuses_consent")
    assert consent["assessment"] is None and consent["message"] is None  # never assessed
    run_dir = tmp_path / r["name"]
    assert json.loads((run_dir / "results.json").read_text())["summary"] == s
    assert "## Fairness" in (run_dir / "report.md").read_text()


async def test_the_next_run_compares_with_the_previous_one(tmp_path):
    first = await run(config(tmp_path))
    (tmp_path / first["name"]).rename(tmp_path / "20000101T000000Z")
    second = await run(config(tmp_path))
    assert second["comparison"]["previous"] == "20000101T000000Z"
    row = next(x for x in second["comparison"]["rows"] if x["metric"] == "decision.accuracy")
    assert row["delta"] == 0.0
    third = await run(config(tmp_path, compare=False))
    assert third["comparison"] == {"previous": None, "rows": []}


async def test_the_budget_stops_the_run_cleanly_with_a_partial_report(tmp_path, monkeypatch):
    # Every backend model call costs $0.14 (a million Flash input tokens).
    meter = Meter(budget_usd=0.5)
    real = OfflineModels.structured

    def costly(self, schema, system, user):
        meter.add(FLASH, 1_000_000, 0)
        return real(self, schema, system, user)

    monkeypatch.setattr(OfflineModels, "structured", costly)
    r = await run(config(tmp_path, budget_usd=0.5), meter)
    assert r["aborted"].startswith("spent $0.5600 of a $0.50 budget")
    assert r["summary"]["cost.usd"] == pytest.approx(0.56)
    assert r["criteria"] and r["calls"]  # what ran before the stop is kept
    assert r["calls"][-1]["error"].startswith("stopped: spent")
    report = (tmp_path / r["name"] / "report.md").read_text()
    assert "**Stopped early:**" in report


async def test_one_persona_failing_never_stops_the_run(tmp_path, monkeypatch):
    real = OfflineSimulator.reply

    def flaky(self, persona, job, said):
        if persona.kind == "declines_salary":
            raise TimeoutError("simulator timed out")
        return real(self, persona, job, said)

    monkeypatch.setattr(OfflineSimulator, "reply", flaky)
    r = await run(config(tmp_path))
    errors = [c for c in r["calls"] if c["error"]]
    assert [c["persona"]["key"] for c in errors] == ["declines_salary"]
    assert "TimeoutError: simulator timed out" in errors[0]["error"]
    assert r["aborted"] is None and r["summary"]["calls.errors"] == 1
    assert r["summary"]["decision.n"] == 3  # the other personas were still judged
    assert r["fairness"]["passed"] is True


async def test_a_failing_criteria_run_is_recorded_and_the_calls_still_run(tmp_path, monkeypatch):
    real = OfflineModels.structured

    def broken(self, schema, system, user):
        if schema.__name__ == "Draft":
            raise RuntimeError("provider down")
        return real(self, schema, system, user)

    monkeypatch.setattr(OfflineModels, "structured", broken)
    r = await run(config(tmp_path))
    assert r["criteria"] == [{"job": "software-engineer", "error": "RuntimeError: provider down"}]
    assert r["summary"]["criteria.recall"] is None and r["summary"]["calls.n"] > 0


def test_the_cli_runs_offline_and_exits_3_when_the_budget_stops_it(tmp_path, monkeypatch, capsys):
    assert (
        cli.main(["--offline", "--jobs", "data-analyst", "--personas", "3", "--out", str(tmp_path)])
        == 0
    )
    assert "Wrote" in capsys.readouterr().out
    monkeypatch.setattr("voiceagent.evals.cli.Meter", lambda budget: Meter(-1.0))
    assert (
        cli.main(["--offline", "--jobs", "data-analyst", "--out", str(tmp_path)]) == cli.BUDGET_EXIT
    )


@pytest.mark.parametrize(
    "argv",
    [["--jobs", "nope", "--offline"], ["--personas", "0", "--offline"], ["--budget-usd", "0"]],
)
def test_the_cli_rejects_bad_arguments(argv, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([*argv, "--out", str(tmp_path)])
    assert e.value.code == 2


def test_a_real_run_needs_an_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "voiceagent.settings.get_settings", lambda: type("S", (), {"together_api_key": ""})()
    )
    with pytest.raises(SystemExit):
        cli.main(["--out", str(tmp_path)])
    assert "TOGETHER_API_KEY is not set" in capsys.readouterr().err


def test_overrides_reach_only_their_own_requirement_as_id_and_text():
    job = load_jobs(["data-analyst"])[0]
    reqs = {r.id: r for r in job.criteria.requirements}
    lookup = overrides_lookup(job)
    [(oid, text)] = lookup(reqs["ab_testing"])
    assert oid == "data-analyst:override:0"
    assert text.startswith("Data Analyst — A/B test analysis: the candidate said \"I'd check")
    assert "the recruiter decided suitable" in text
    assert lookup(reqs["pandas"]) == []


async def test_a_hang_up_before_consent_is_sent_to_review_without_assessing(tmp_path, monkeypatch):
    real = OfflineSimulator.reply

    def silent(self, persona, job, said):
        return None if persona.kind == "clear_fit" else real(self, persona, job, said)

    monkeypatch.setattr(OfflineSimulator, "reply", silent)
    r = await run(config(tmp_path, personas=1))
    [rec] = [c for c in r["calls"] if c["persona"]["kind"] == "clear_fit"]
    assert rec["call"]["end_reason"] == "candidate_hung_up" and rec["call"]["turns"] == 0
    assert rec["assessment"]["result"] == "needs_review" and rec["assessment"]["answers"] == []
    assert rec["message"] is None and rec["judgement"] is not None


async def test_the_fairness_replays_are_scored_with_the_recruiter_overrides(tmp_path, monkeypatch):
    real = OfflineModels.structured
    corrected: dict[str, bool] = {}

    def spy(self, schema, system, user):
        if schema.__name__ == "SoftScores":
            key = self.persona.key
            corrected[key] = corrected.get(key, False) or "recruiter_corrections" in user
        return real(self, schema, system, user)

    monkeypatch.setattr(OfflineModels, "structured", spy)
    r = await run(config(tmp_path, personas=3))
    fairness = [c["persona"]["key"] for c in r["calls"] if c["persona"]["key"].startswith("fair")]
    assert len(fairness) == 4 and all(corrected[k] for k in fairness)
