"""screening-eval: run the offline evaluation suite and write evals/runs/<timestamp>/."""

import argparse
import asyncio
import logging
from pathlib import Path

from voiceagent.evals.cost import Meter
from voiceagent.evals.report import RUNS_DIR
from voiceagent.evals.runner import EvalConfig, offline_backend, real_backend, run_eval

BUDGET_EXIT = 3  # the run stopped at its budget; the report covers what ran


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="screening-eval",
        description="Evaluate criteria generation, calls, decisions and messages on the "
        "reference jobs in evals/jobs/. Run from the repository root.",
    )
    p.add_argument("--jobs", default="all", help="comma-separated job slugs, or 'all'")
    p.add_argument("--personas", type=int, default=4, help="simulated candidates per job")
    p.add_argument("--runs", type=int, default=1, help="calls per persona (for variance)")
    p.add_argument("--out", type=Path, default=RUNS_DIR, help="where runs are written")
    p.add_argument("--no-compare", action="store_true", help="skip the previous-run comparison")
    p.add_argument(
        "--budget-usd", type=float, default=2.0, help="stop once the run has spent this much"
    )
    p.add_argument("--offline", action="store_true", help="fake every model (smoke test)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.personas < 1 or args.runs < 1:
        parser.error("--personas and --runs must be at least 1")
    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    jobs = None if args.jobs == "all" else [s.strip() for s in args.jobs.split(",") if s.strip()]
    cfg = EvalConfig(
        jobs=jobs,
        personas=args.personas,
        runs=args.runs,
        out=args.out,
        compare=not args.no_compare,
        budget_usd=args.budget_usd,
        offline=args.offline,
    )
    logging.basicConfig(level=logging.WARNING)
    meter = Meter(cfg.budget_usd)
    if args.offline:
        backend = offline_backend()
    else:
        from voiceagent.settings import get_settings

        if not get_settings().together_api_key:
            parser.error("TOGETHER_API_KEY is not set (or use --offline)")
        backend = real_backend(meter)
    try:
        with meter.tracking():
            results = asyncio.run(run_eval(cfg, backend, meter))
    except ValueError as e:  # an unknown job slug or an invalid job file
        parser.error(str(e))
    s = results["summary"]
    print(f"Wrote {cfg.out / results['name']}/report.md")
    print(
        f"decision accuracy {s['decision.accuracy']} · false rejections "
        f"{s['decision.false_rejection_rate']} · criteria recall {s['criteria.recall']} · "
        f"cost ${results['cost']['usd']:.4f}"
    )
    if results["aborted"]:
        print(f"Stopped early: {results['aborted']}")
        return BUDGET_EXIT
    return 0
