import json
from datetime import UTC, datetime

from voiceagent.evals.report import (
    compare,
    new_run_dir,
    previous_run,
    render_markdown,
    run_name,
    write_run,
)


def results(name="20261009T120000Z", summary=None, **kw):
    return {
        "name": name,
        "config": {"jobs": ["software-engineer"], "personas": 4, "runs": 1, "offline": True},
        "aborted": None,
        "summary": summary if summary is not None else {"decision.accuracy": 0.75, "cost.usd": 0.5},
        "criteria": [
            {
                "job": "software-engineer",
                "recall": 1.0,
                "precision": 0.8,
                "hard_flag_accuracy": 1.0,
                "rule_correctness": 1.0,
                "illegal_dropped": None,
                "rounds": 2,
                "grounding_removed": 0,
                "unmatched_labelled": [],
                "unmatched_generated": ["team_size"],
            },
            {"job": "staff-nurse", "error": "StructuredOutputError: Draft"},
        ],
        "calls": [
            {
                "job": "software-engineer",
                "persona": {
                    "key": "clear_fit",
                    "expected_end": "completed",
                    "expected_decision": "suitable",
                },
                "call": {"end_reason": "completed", "turns": 9},
                "assessment": {"result": "suitable", "score": 0.8},
                "cost_usd": 0.03,
                "error": None,
            },
            {
                "job": "software-engineer",
                "persona": {
                    "key": "bad_time",
                    "expected_end": "callback_requested",
                    "expected_decision": None,
                },
                "call": None,
                "assessment": None,
                "cost_usd": 0.0,
                "error": "TimeoutError",
            },
        ],
        "fairness": None,
        "comparison": {"previous": None, "rows": []},
        "cost": {"usd": 0.5, "tokens": {"Qwen/Qwen3.5-9B": [1000, 200]}},
    } | kw


def save(tmp_path, name, content):
    d = tmp_path / name
    d.mkdir()
    (d / "results.json").write_text(content)


def test_run_names_sort_by_time():
    assert run_name(datetime(2026, 10, 9, 7, 5, 3, tzinfo=UTC)) == "20261009T070503Z"


def test_runs_started_in_the_same_second_get_their_own_directories(tmp_path):
    at = datetime(2026, 10, 9, 7, 5, 3, tzinfo=UTC)
    names = [new_run_dir(tmp_path / "runs", at).name for _ in range(3)]
    assert names == ["20261009T070503Z", "20261009T070503Z-2", "20261009T070503Z-3"]


def test_the_previous_run_is_the_latest_readable_one_before_this(tmp_path):
    save(tmp_path, "20261001T000000Z", json.dumps(results(summary={"a": 1})))
    save(tmp_path, "20261002T000000Z", json.dumps(results(summary={"a": 2})))
    save(tmp_path, "20261003T000000Z", "{not json")  # an interrupted write
    save(tmp_path, "20261004T000000Z", json.dumps(["not", "a", "run"]))
    save(tmp_path, "20261009T000000Z", json.dumps(results(summary={"a": 9})))  # later
    (tmp_path / ".gitkeep").write_text("")
    prev = previous_run(tmp_path, "20261005T000000Z")
    assert prev["name"] == "20261002T000000Z" and prev["summary"] == {"a": 2}


def test_a_run_is_compared_only_with_one_of_the_same_config(tmp_path):
    real = results()["config"] | {"offline": False}
    save(tmp_path, "20261001T000000Z", json.dumps(results(config=real)))
    save(tmp_path, "20261002T000000Z", json.dumps(results()))  # offline
    save(tmp_path, "20261003T000000Z", json.dumps(results(config=real | {"personas": 9})))
    assert previous_run(tmp_path, "20261005T000000Z", real)["name"] == "20261001T000000Z"


def test_no_runs_directory_means_no_previous_run(tmp_path):
    assert previous_run(tmp_path / "missing", "20261005T000000Z") is None
    assert previous_run(tmp_path, "20261005T000000Z") is None


def test_comparison_tolerates_metrics_the_previous_run_lacks():
    prev = {"summary": {"decision.accuracy": 0.5, "judge.professional": "n/a"}}
    rows = compare({"decision.accuracy": 0.75, "judge.professional": 1.0, "new.metric": 3}, prev)
    assert rows == [
        {"metric": "decision.accuracy", "current": 0.75, "previous": 0.5, "delta": 0.25},
        {"metric": "judge.professional", "current": 1.0, "previous": None, "delta": None},
        {"metric": "new.metric", "current": 3, "previous": None, "delta": None},
    ]
    assert compare({"x": None}, None) == [
        {"metric": "x", "current": None, "previous": None, "delta": None}
    ]


def test_the_report_has_every_section(tmp_path):
    r = results()
    r["comparison"] = {
        "previous": "20261008T000000Z",
        "rows": compare(r["summary"], {"summary": {"decision.accuracy": 0.5}}),
    }
    (tmp_path / r["name"]).mkdir()
    write_run(tmp_path / r["name"], r)
    md = (tmp_path / r["name"] / "report.md").read_text()
    assert json.loads((tmp_path / r["name"] / "results.json").read_text())["name"] == r["name"]
    assert "Compared with 20261008T000000Z, the last run with the same jobs" in md
    assert "| decision.accuracy | 0.750 | 0.500 | +0.250 |" in md
    assert "| software-engineer | 1.000 | 0.800 |" in md and "team_size" in md
    assert "staff-nurse | error: StructuredOutputError" in md
    assert (
        "| software-engineer | clear_fit | completed (completed) | suitable (suitable) | 0.800 | 9 |"
        in md
    )
    assert "- software-engineer / bad_time: TimeoutError" in md
    assert "| Qwen/Qwen3.5-9B | 1000 | 200 |" in md
    r["calls"][0]["call"] |= {"end_reason": "candidate_hung_up", "stopped_by": "turn_limit"}
    assert "| clear_fit | candidate_hung_up / turn_limit (completed) |" in render_markdown(r)
    assert "Stopped early" not in md


def test_a_first_or_aborted_run_says_so():
    md = render_markdown(results(aborted="spent $2.0100 of a $2.00 budget"))
    assert "No previous run with the same jobs, personas, runs and models" in md
    assert "**Stopped early:** spent $2.0100 of a $2.00 budget." in md


def test_the_fairness_section_shows_each_name():
    fair = {
        "job": "data-analyst",
        "persona": "borderline",
        "results": {"Imran Qureshi": "needs_review"},
        "scores": {"Imran Qureshi": 0.55},
        "same_decision": True,
        "score_spread": 0.0,
        "passed": True,
    }
    md = render_markdown(results(fairness=fair))
    assert "| Imran Qureshi | needs_review | 0.550 |" in md and "**pass**" in md
    assert "**incomplete**" in render_markdown(results(fairness=fair | {"passed": None}))
