"""decision-bench: Jev vs the production LLMs on three decisions of the screening backend
(evals/bench/README.md). Run from the repository root."""

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from voiceagent.evals.bench import progress, review, scoring
from voiceagent.evals.bench.common import RUNS, Jsonl, read_jsonl, record_spend, spent
from voiceagent.evals.cost import BudgetExceeded, Meter

BUDGET_EXIT = 3
TASKS = ("scoring", "progress", "review")


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, int | float) for x in v):
        return f"[{v[0]:.3f}, {v[1]:.3f}]"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table(rows: list[tuple[str, ...]], head: tuple[str, ...]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += [
        "| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows
    ]
    return out


def task_scoring(meter, out_dir, limit, build_only) -> dict:
    items = scoring.build(meter, limit)
    gen = [i for i in items if i["kind"] == "gen"]
    counts = {
        "generated": len(gen),
        "generated_kept": sum(i["kept"] for i in gen),
        "generated_dropped": Counter(i["drop"] for i in gen if not i["kept"]),
        "generated_kept_by_level": Counter(i["intended"] for i in gen if i["kept"]),
        "real": sum(i["kind"] == "real" for i in items),
        "real_by_label": Counter(i.get("label") for i in items if i["kind"] == "real"),
        "stress": Counter(i["kind"] for i in items if i["kind"] not in ("gen", "real")),
        "stress_kept": Counter(
            i["kind"] for i in items if i["kind"] not in ("gen", "real") and i["kept"]
        ),
    }
    if build_only:
        return {"dataset": counts}
    costs = scoring.run(meter, items, out_dir, limit)
    llm_rows = Jsonl(out_dir / "scoring_llm.jsonl").values()
    jev_rows = Jsonl(out_dir / "scoring_jev.jsonl").values()
    m = {
        "dataset": counts,
        "llm": scoring.contender_metrics(items, llm_rows, costs.get("llm", {})),
        "jev": scoring.contender_metrics(items, jev_rows, costs.get("jev", {})),
        "paired": scoring.paired(items, llm_rows, jev_rows),
        "jev_versions": costs.get("jev_versions"),
        "jev_retries": costs.get("jev_retries"),
    }
    m["verdict"], m["why"] = scoring.verdict(m)
    e2e_costs = scoring.end_to_end(meter, out_dir, limit)
    m["e2e"] = scoring.e2e_metrics(Jsonl(out_dir / "e2e.jsonl").values())
    m["e2e_costs"] = e2e_costs
    return m


def task_progress(meter, out_dir, limit, build_only) -> dict:
    items = progress.build(meter, limit)
    counts = {
        "turns": len(items),
        "labelled": sum(i["gold"] is not None for i in items),
        "by_label": progress.by_label_counts(items),
        "invalid_ids_dropped": sum(len(i["invalid"]) for i in items),
    }
    if build_only:
        return {"dataset": counts}
    costs = progress.run(meter, items, out_dir, limit)
    m = progress.metrics(items, Jsonl(out_dir / "progress.jsonl").values(), costs)
    m = {"dataset": counts, **m, "jev_retries": costs.get("jev_retries")}
    m["verdict"], m["why"] = progress.verdict({k: m[k] for k in ("heuristic", "jev", "llm")})
    return m


def task_review(meter, out_dir, limit, build_only) -> dict:
    items = review.build(meter, limit)
    labels = [v for i in items for v in i["labels"].values()]
    counts = {
        "instances": len(items),
        "original_calls": sum(i["key"].startswith("orig:") for i in items),
        "corrupted_copies": sum(not i["key"].startswith("orig:") for i in items),
        "reasons": len(labels),
        "by_category": Counter(v["category"] for v in labels),
    }
    if build_only:
        return {"dataset": counts}
    costs = review.run(meter, items, out_dir, limit)
    m = review.metrics(items, Jsonl(out_dir / "review.jsonl").values(), costs)
    m = {"dataset": counts, **m, "jev_retries": costs.get("jev_retries")}
    m["verdict"], m["why"] = review.verdict(m)
    return m


RUNNERS = {"scoring": task_scoring, "progress": task_progress, "review": task_review}


def report(results: dict) -> str:
    out = [f"# Decision benchmark run {results['name']}", ""]
    out.append(
        f"Spend this invocation: ${results['usd']:.4f}; ledger total ${results['ledger_usd']:.4f}."
    )
    if results.get("aborted"):
        out.append(f"\n**Stopped early:** {results['aborted']}")
    s = results["tasks"].get("scoring")
    if s and "llm" in s:
        L, J = s["llm"], s["jev"]
        out += [
            "",
            "## Task 1: rubric scoring",
            "",
            f"Dataset: `{json.dumps(s['dataset'], default=str)}`",
            "",
        ]
        rows = [
            ("items (base)", L["n_items"], J["n_items"]),
            (
                "scorings / failures",
                f"{L['scorings']} / {L['failures']}",
                f"{J['scorings']} / {J['failures']}",
            ),
            ("MAE vs gold", L["mae"], J["mae"]),
            ("MAE 95% CI", L["mae_ci"], J["mae_ci"]),
            ("band accuracy", L["band_acc"], J["band_acc"]),
            ("band accuracy 95% CI", L["band_acc_ci"], J["band_acc_ci"]),
            (
                "MAE generated / real",
                f"{_fmt(L['mae_gen'])} / {_fmt(L['mae_real'])}",
                f"{_fmt(J['mae_gen'])} / {_fmt(J['mae_real'])}",
            ),
            (
                "band acc. on gold 0 / 0.5 / 1",
                *(" / ".join(_fmt(X[f"band_acc_gold{k}"]) for k in range(3)) for X in (L, J)),
            ),
            ("repeat: mean per-item std", L["repeat_std"], J["repeat_std"]),
            ("repeat std 95% CI", L["repeat_std_ci"], J["repeat_std_ci"]),
            ("repeat: same band all 5", L["repeat_same_band"], J["repeat_same_band"]),
            (
                "paraphrase spread (groups)",
                f"{_fmt(L['paraphrase_spread'])} ({L['paraphrase_groups']})",
                f"{_fmt(J['paraphrase_spread'])} ({J['paraphrase_groups']})",
            ),
            ("paraphrase spread 95% CI", L["paraphrase_spread_ci"], J["paraphrase_spread_ci"]),
            (
                "name-swap spread (groups)",
                f"{_fmt(L['name_spread'])} ({L['name_groups']})",
                f"{_fmt(J['name_spread'])} ({J['name_groups']})",
            ),
            ("name-swap spread 95% CI", L["name_spread_ci"], J["name_spread_ci"]),
            ("name-swap max spread", L["name_spread_max"], J["name_spread_max"]),
            (
                "ASR noise: MAE / band acc (n)",
                *(
                    f"{_fmt(X['asr_mae'])} / {_fmt(X['asr_band_acc'])} ({X['asr_n']})"
                    for X in (L, J)
                ),
            ),
            (
                "Hinglish: MAE / band acc (n)",
                *(
                    f"{_fmt(X['hinglish_mae'])} / {_fmt(X['hinglish_band_acc'])} ({X['hinglish_n']})"
                    for X in (L, J)
                ),
            ),
            (
                '"I\'d rather not say": MAE (n)',
                *(f"{_fmt(X['declined_mae'])} ({X['declined_n']})" for X in (L, J)),
            ),
            ("injection inflation (>0.66)", L["injection_inflation"], J["injection_inflation"]),
            (
                "injection inflation 95% CI",
                L["injection_inflation_ci"],
                J["injection_inflation_ci"],
            ),
            ("injection mean score", L["injection_mean_score"], J["injection_mean_score"]),
            ("band acc., 80% most confident", L["confident80_band_acc"], J["confident80_band_acc"]),
            (
                "latency p50 / p95 (s)",
                f"{_fmt(L['p50_s'])} / {_fmt(L['p95_s'])}",
                f"{_fmt(J['p50_s'])} / {_fmt(J['p95_s'])}",
            ),
            ("USD per 1,000 scorings", L["usd_per_1000"], J["usd_per_1000"]),
        ]
        out += _table(rows, ("metric", "LLM (production scorer)", "Jev"))
        out += ["", "Paired differences (Jev − LLM, 95% bootstrap CI):", ""]
        out += _table(
            [(k, v["diff"], v["ci"], v["n"]) for k, v in s["paired"].items()],
            ("metric", "diff", "CI", "n"),
        )
        out += ["", f"**Verdict (pre-registered rule): {s['verdict']}**", ""] + [
            f"- {w}" for w in s["why"]
        ]
        out += ["", f"Jev versions: {s['jev_versions']}; Jev retries: {s['jev_retries']}", ""]
        out += ["### End to end (stored evidence, scorer swapped, reviewer skipped)", ""]
        e = s["e2e"]
        out += _table(
            [
                (
                    k,
                    v["n"],
                    v.get("errors"),
                    v["accuracy"],
                    v["false_rejection_rate"],
                    v["needs_review_rate"],
                    f"{v['fairness_same_decision']}/{v['fairness_runs']}",
                )
                for k, v in e.items()
            ],
            (
                "scorer",
                "calls",
                "errors",
                "decision accuracy",
                "false rejections",
                "needs review",
                "fairness runs same decision",
            ),
        )
    p = results["tasks"].get("progress")
    if p and "llm" in p:
        out += [
            "",
            "## Task 2: question progress",
            "",
            f"Dataset: `{json.dumps(p['dataset'], default=str)}`",
            "",
        ]
        out += _table(
            [
                (
                    k,
                    p[k]["n"],
                    p[k]["errors"],
                    p[k]["accuracy"],
                    p[k]["accuracy_ci"],
                    p[k]["macro_f1"],
                    p[k]["false_tick_rate"],
                    p[k]["p50_s"],
                    p[k]["p95_s"],
                    p[k]["usd_per_1000"],
                )
                for k in ("heuristic", "jev", "llm")
            ],
            (
                "contender",
                "turns",
                "errors",
                "accuracy",
                "95% CI",
                "macro-F1",
                "false ticks on none",
                "p50 s",
                "p95 s",
                "USD/1000",
            ),
        )
        out += ["", f"**Verdict (pre-registered rule): {p['verdict']}**", ""] + [
            f"- {w}" for w in p["why"]
        ]
    r = results["tasks"].get("review")
    if r and "llm" in r:
        out += [
            "",
            "## Task 3: reviewer checks",
            "",
            f"Dataset: `{json.dumps(r['dataset'], default=str)}`",
            "",
        ]
        out += _table(
            [
                (
                    k,
                    r[k]["should_flag"],
                    r[k]["correct"],
                    r[k]["catch_rate"],
                    r[k]["catch_rate_ci"],
                    r[k]["false_flag_rate"],
                    r[k]["false_flag_rate_ci"],
                    r[k]["p50_s"],
                    r[k]["p95_s"],
                    r[k]["latency_unit"],
                    r[k]["usd_per_1000_checks"],
                    r[k]["errors"],
                    r[k]["missing_checks"],
                )
                for k in ("llm", "jev")
            ],
            (
                "contender",
                "should flag",
                "correct",
                "catch rate",
                "95% CI",
                "false flags",
                "95% CI",
                "p50 s",
                "p95 s",
                "latency is",
                "USD/1000 checks",
                "errors",
                "missing checks",
            ),
        )
        out += ["", "Catches by category (caught/total):", ""]
        cats = sorted(set(r["llm"]["catch_by_category"]) | set(r["jev"]["catch_by_category"]))
        out += _table(
            [
                (
                    c,
                    *(
                        ("/".join(map(str, r[k]["catch_by_category"].get(c, [0, 0]))))
                        for k in ("llm", "jev")
                    ),
                )
                for c in cats
            ],
            ("category", "LLM", "Jev"),
        )
        out += ["", f"Paired (Jev − LLM): `{json.dumps(r['paired'])}`"]
        out += ["", f"**Verdict (pre-registered rule): {r['verdict']}**", ""] + [
            f"- {w}" for w in r["why"]
        ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="decision-bench", description=__doc__)
    p.add_argument("--task", choices=(*TASKS, "all"), default="all")
    p.add_argument(
        "--budget-usd",
        type=float,
        default=5.0,
        help="total spend allowed across the ledger (evals/bench/data/spend.jsonl)",
    )
    p.add_argument("--limit", type=int, default=None, help="at most N items per stage (smoke)")
    p.add_argument("--run-dir", type=Path, default=None, help="resume this run directory")
    p.add_argument("--build-only", action="store_true", help="build the datasets, run nothing")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    left = args.budget_usd - spent()
    if left <= 0:
        p.error(f"the ledger already shows ${spent():.4f} of a ${args.budget_usd:.2f} budget")
    meter = Meter(left)
    out_dir = args.run_dir or RUNS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = out_dir / "results.json"
    results = json.loads(prev.read_text()) if prev.exists() else {"tasks": {}}
    results |= {"name": out_dir.name, "aborted": None}
    tasks = TASKS if args.task == "all" else (args.task,)
    try:
        with meter.tracking():
            for t in tasks:
                print(f"{t}: ${meter.usd:.4f} spent so far (${left:.2f} left at start)")
                results["tasks"][t] = RUNNERS[t](meter, out_dir, args.limit, args.build_only)
    except BudgetExceeded as e:
        results["aborted"] = str(e)
    finally:
        record_spend(f"{args.task}{' build' if args.build_only else ''} -> {out_dir.name}", meter)
    results |= {
        "usd": meter.usd,
        "tokens": meter.tokens(),
        "ledger_usd": spent(),
        "ledger": read_jsonl(Path("evals/bench/data/spend.jsonl")),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out_dir / "report.md").write_text(report(results))
    print(f"Wrote {out_dir}/report.md · spent ${meter.usd:.4f} (ledger ${spent():.4f})")
    for t, m in results["tasks"].items():
        if "verdict" in m:
            print(f"  {t}: {m['verdict']}")
    if results["aborted"]:
        print(f"Stopped early: {results['aborted']}")
        return BUDGET_EXIT
    return 0
