"""An evaluation run on disk: evals/runs/<UTC timestamp>/results.json + report.md, with
the summary compared against the previous run."""

import json
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("evals/runs")
# A run is compared only with one of the same shape: offline numbers say nothing of a real run.
SAME = ("offline", "jobs", "personas", "runs")
FAIRNESS = {True: "pass", False: "FAIL", None: "incomplete"}


def run_name(at: datetime) -> str:
    return at.strftime("%Y%m%dT%H%M%SZ")


def previous_run(out: Path, current: str, config: dict | None = None) -> dict | None:
    """The latest run before `current` whose results.json can be read and, given `config`,
    that ran with the same SAME settings."""
    if not out.is_dir():
        return None
    for d in sorted((p for p in out.iterdir() if p.is_dir()), reverse=True):
        if d.name >= current:
            continue
        try:
            results = json.loads((d / "results.json").read_text())
        except (OSError, ValueError):
            continue  # an aborted write or a hand-edited file: look further back
        if not isinstance(results, dict) or not isinstance(results.get("summary"), dict):
            continue
        theirs = results.get("config") or {}
        if config is None or all(theirs.get(k) == config.get(k) for k in SAME):
            return results | {"name": d.name}
    return None


def _number(v) -> float | None:
    return v if isinstance(v, int | float) and not isinstance(v, bool) else None


def compare(summary: dict, previous: dict | None) -> list[dict]:
    """One row per metric of this run; metrics the previous run lacks have no change."""
    before = (previous or {}).get("summary", {})
    rows = []
    for metric, value in summary.items():
        prev = _number(before.get(metric))
        now = _number(value)
        delta = round(now - prev, 4) if now is not None and prev is not None else None
        rows.append({"metric": metric, "current": value, "previous": prev, "delta": delta})
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _ended(call: dict | None) -> str:
    """How the call ended, and why the harness stopped it if it did."""
    if not call:
        return "—"
    return " / ".join(filter(None, [call["end_reason"], call.get("stopped_by")]))


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return lines + [""]


def render_markdown(results: dict) -> str:
    cfg = results["config"]
    out = [f"# Screening evaluation {results['name']}", ""]
    if results.get("aborted"):
        out += [f"**Stopped early:** {results['aborted']}. The numbers cover what ran.", ""]
    out += [
        f"Jobs: {', '.join(cfg['jobs'])} · personas per job: {cfg['personas']} · runs: "
        f"{cfg['runs']} · {'offline (fakes)' if cfg['offline'] else 'real models'} · "
        f"cost ${results['cost']['usd']:.4f}",
        "",
        "## Summary",
        "",
    ]
    cmp = results["comparison"]
    if cmp["previous"]:
        out += [
            f"Compared with {cmp['previous']}, the last run with the same jobs, personas, "
            "runs and models (real or offline).",
            "",
        ]
        out += _table(
            ["Metric", "This run", "Previous", "Change"],
            [
                [
                    r["metric"],
                    r["current"],
                    r["previous"],
                    None if r["delta"] is None else f"{r['delta']:+.3f}",
                ]
                for r in cmp["rows"]
            ],
        )
    else:
        out += [
            "No previous run with the same jobs, personas, runs and models to compare with.",
            "",
        ]
        out += _table(["Metric", "This run"], [[k, v] for k, v in results["summary"].items()])
    out += ["## Criteria generation", ""]
    out += _table(
        [
            "Job",
            "Recall",
            "Precision",
            "Hard flags",
            "Rules",
            "Illegal dropped",
            "Rounds",
            "Ungrounded removed",
            "Missed (labelled)",
            "Extra (generated)",
        ],
        [
            [
                r["job"],
                r["recall"],
                r["precision"],
                r["hard_flag_accuracy"],
                r["rule_correctness"],
                r["illegal_dropped"],
                r["rounds"],
                r["grounding_removed"],
                ", ".join(r["unmatched_labelled"]) or None,
                ", ".join(r["unmatched_generated"]) or None,
            ]
            if "error" not in r
            else [r["job"], f"error: {r['error']}", None, None, None, None, None, None, None, None]
            for r in results["criteria"]
        ],
    )
    out += ["## Calls", ""]
    out += _table(
        ["Job", "Persona", "Ended (expected)", "Decision (expected)", "Score", "Turns", "Cost $"],
        [
            [
                c["job"],
                c["persona"]["key"],
                f"{_ended(c.get('call'))} ({c['persona']['expected_end']})",
                f"{(c.get('assessment') or {}).get('result', '—')} "
                f"({c['persona']['expected_decision'] or 'none'})",
                (c.get("assessment") or {}).get("score"),
                (c.get("call") or {}).get("turns"),
                c.get("cost_usd"),
            ]
            for c in results["calls"]
        ],
    )
    fair = results.get("fairness")
    if fair:
        out += ["## Fairness", "", f"{fair['persona']} persona of {fair['job']}, renamed:", ""]
        out += _table(
            ["Name", "Decision", "Score"],
            [[n, d, fair["scores"].get(n)] for n, d in fair["results"].items()],
        )
        out += [
            f"Same decision: {_fmt(fair['same_decision'])} · score spread "
            f"{_fmt(fair['score_spread'])} (limit 0.05) · **{FAIRNESS[fair['passed']]}**",
            "",
        ]
    errors = [c for c in results["calls"] if c.get("error")]
    if errors:
        out += (
            ["## Errors", ""]
            + [f"- {c['job']} / {c['persona']['key']}: {c['error']}" for c in errors]
            + [""]
        )
    out += ["## Cost by model", ""]
    out += _table(
        ["Model", "Input tokens", "Output tokens"],
        [[m, i, o] for m, (i, o) in sorted(results["cost"]["tokens"].items())],
    )
    return "\n".join(out)


def write_run(directory: Path, results: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False, default=str)
    )
    (directory / "report.md").write_text(render_markdown(results))
