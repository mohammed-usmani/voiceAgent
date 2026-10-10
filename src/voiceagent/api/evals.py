"""Evaluation runs (screening-eval writes evals/runs/<name>/results.json), read-only."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from voiceagent.api.schemas import EvalRunSummary

router = APIRouter()
HEADLINE = (
    "decision.accuracy",
    "decision.false_rejection_rate",
    "criteria.recall",
    "cost.per_call_usd",
)


def run_files(directory: Path) -> dict[str, Path]:
    """Run name -> its results.json, for every run directory that has one."""
    if not directory.is_dir():
        return {}
    return {
        d.name: d / "results.json"
        for d in directory.iterdir()
        if d.is_dir() and (d / "results.json").is_file()
    }


@router.get("/evals/runs", response_model=list[EvalRunSummary])
def list_eval_runs(request: Request) -> list[EvalRunSummary]:
    """Every evaluation run, newest first; offline runs (fake models) are flagged.

    ponytail: reads every results.json on each request; fine for tens of runs."""
    runs = []
    for name, path in sorted(run_files(request.app.state.evals_dir).items(), reverse=True):
        try:
            r = json.loads(path.read_text())
            runs.append(
                EvalRunSummary(
                    name=name,
                    started_at=r["started_at"],
                    finished_at=r.get("finished_at"),
                    offline=r["config"]["offline"],
                    aborted=r.get("aborted"),
                    headline={k: r["summary"].get(k) for k in HEADLINE},
                    cost_usd=r["cost"]["usd"],
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue  # being written, or hand-edited: leave it out
    return runs


@router.get("/evals/runs/{name}")
def read_eval_run(name: str, request: Request) -> FileResponse:
    """A run's full results.json. Only names in the listing are served."""
    path = run_files(request.app.state.evals_dir).get(name)
    if path is None:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    return FileResponse(path, media_type="application/json")
