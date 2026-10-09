"""Turns criteria results and call records into the numbers the report shows. Pure
functions over plain dicts (the shapes in results.json), so they are easy to test and a
past run's results can be re-read.

A call record: {"job", "persona" (Persona dump), "run", "call" {"end_reason", "stopped_by",
"callback_at", "turns", "transcript"}, "assessment" {"result", "score", "answers",
"agent_review", ...} or None, "judgement" (CallJudgement dump) or None, "message"
{..., "judgement"} or None, "calibrated" {"result", "score"} or None, "cost_usd",
"error"}.
"""

from collections import defaultdict
from statistics import mean

from voiceagent.criteria.schema import Criteria, Requirement, Rule
from voiceagent.evals.jobs import RefJob
from voiceagent.evals.judges import CHECKS, ILLEGAL
from voiceagent.evals.personas import as_number

FAIRNESS_TOLERANCE = 0.05


def _rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _avg(xs: list[float]) -> float | None:
    return round(mean(xs), 4) if xs else None


def _norm(v):
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int | float):
        return ("num", round(float(v), 6))
    if isinstance(v, str):
        return ("str", v.strip().casefold())
    if isinstance(v, list):
        return ("list", frozenset(_norm(x) for x in v))
    return ("other", repr(v))


def same_rule(a: Rule | None, b: Rule | None) -> bool:
    """Same operator and value; "1-3 years" read as `between [1, 3]` is not `>= 1`."""
    if a is None or b is None:
        return a is b
    return a.op == b.op and _norm(a.value) == _norm(b.value)


def criteria_metrics(
    job: RefJob, generated: Criteria, warnings: list[str], rounds: int, pairs: list[tuple]
) -> dict:
    labelled = {r.id: r for r in job.criteria.requirements}
    gen = {r.id: r for r in generated.requirements}
    legit = [(g, lab) for g, lab in pairs if lab in labelled]
    illegal_hits = sorted({lab for _, lab in pairs if lab.startswith(ILLEGAL)})
    hard_pairs = [(gen[g], labelled[lab]) for g, lab in legit if labelled[lab].hard]
    return {
        "job": job.slug,
        "generated": len(gen),
        "labelled": len(labelled),
        "recall": _rate([lab in {x for _, x in legit} for lab in labelled]),
        "precision": _rate([g in {x for x, _ in legit} for g in gen]) if gen else 0.0,
        "hard_flag_accuracy": _rate([gen[g].hard == labelled[lab].hard for g, lab in legit]),
        "rule_correctness": _rate(
            [g.hard and same_rule(g.rule, lab.rule) for g, lab in hard_pairs]
        ),
        "illegal_dropped": (
            _rate([f"{ILLEGAL}{i}" not in illegal_hits for i in range(len(job.illegal))])
            if job.illegal
            else None
        ),
        "rounds": rounds,
        "grounding_removed": sum("removed (ungrounded" in w for w in warnings),
        "pairs": [list(p) for p in pairs],
        "unmatched_labelled": sorted(set(labelled) - {lab for _, lab in legit}),
        "unmatched_generated": sorted(set(gen) - {g for g, _ in legit}),
    }


def _boolean(v) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s in ("yes", "true") else False if s in ("no", "false") else None


def answer_correct(r: Requirement, truth, declined: bool, answer: dict | None, text_ok) -> bool:
    """Whether the assessment captured the persona's true answer. Numbers within 10% (at
    least 0.5), booleans and choices exactly, text by the call judge (`text_ok`)."""
    if answer is None:
        return False
    if declined:
        return bool(answer.get("declined"))
    got = answer.get("value")
    if got is None:
        return False
    if r.kind == "number":
        a, b = as_number(got), as_number(truth)
        return a is not None and b is not None and abs(a - b) <= max(0.5, 0.1 * abs(b))
    if r.kind == "boolean":
        return _boolean(got) is not None and _boolean(got) == _boolean(truth)
    if r.kind == "choice":
        return str(got).strip().casefold() == str(truth).strip().casefold()
    return bool(text_ok)


def _screened(rec: dict) -> bool:
    """A call that went through the questions as the persona intended."""
    return (
        rec.get("error") is None
        and rec["persona"]["expected_end"] == "completed"
        and rec["call"]["end_reason"] == "completed"
        and rec.get("assessment") is not None
    )


def capture_metrics(jobs: dict[str, RefJob], records: list[dict]) -> dict:
    by_kind: dict[str, list[bool]] = defaultdict(list)
    missed, reasked = [], []
    for rec in filter(_screened, records):
        persona = rec["persona"]
        answers = {a["requirement_id"]: a for a in rec["assessment"]["answers"]}
        judged = rec.get("judgement") or {}
        text_ok = {t["requirement_id"]: t["agrees"] for t in judged.get("text_answers", [])}
        for r in jobs[rec["job"]].criteria.requirements:
            declined = r.id in persona["declined"]
            truth = persona["truth"].get(r.id)
            if truth is None and not declined:
                continue  # no ground truth to compare with
            ok = answer_correct(r, truth, declined, answers.get(r.id), text_ok.get(r.id))
            by_kind["declined" if declined else r.kind].append(ok)
        if judged:
            ids = [r.id for r in jobs[rec["job"]].criteria.requirements]
            missed.append(len(set(ids) - set(judged.get("asked", []))))
            reasked.append(len(judged.get("reasked", [])))
    every = [x for xs in by_kind.values() for x in xs]
    return {
        "accuracy": _rate(every),
        "by_kind": {k: _rate(v) for k, v in sorted(by_kind.items())},
        "missed_per_call": _avg(missed),
        "reasked_per_call": _avg(reasked),
    }


def decision_metrics(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("error") is None]
    # a call cut short before its decision (no assessment) counts as a wrong decision
    decided = [r for r in ok if r["persona"]["expected_decision"]]
    pairs = [
        (r["persona"]["expected_decision"], (r.get("assessment") or {}).get("result"))
        for r in decided
    ]
    not_rejectable = [got for want, got in pairs if want != "not_suitable" and got is not None]
    groups: dict[tuple, list[str | None]] = defaultdict(list)
    for r, (_, got) in zip(decided, pairs, strict=True):
        groups[(r["job"], r["persona"]["key"])].append(got)
    # with --runs > 1: personas whose repeated calls all got the same decision
    repeated = [len(set(v)) == 1 for v in groups.values() if len(v) > 1]
    return {
        "n": len(pairs),
        "accuracy": _rate([want == got for want, got in pairs]),
        "false_rejection_rate": _rate([got == "not_suitable" for got in not_rejectable]),
        "needs_review_rate": _rate([got == "needs_review" for _, got in pairs]),
        "end_reason_accuracy": _rate(
            [r["call"]["end_reason"] == r["persona"]["expected_end"] for r in ok]
        ),
        "consistency": _rate(repeated),
    }


def reviewer_stats(records: list[dict]) -> dict:
    """The reviewer challenges rejections and invites that would go out unattended (and weak
    verdicts). Disagreeing sends the candidate to a human: a catch when the challenged
    verdict was wrong for the persona, a needless escalation when it was right."""
    out = {}
    for verdict, name in (("not_suitable", "rejections"), ("suitable", "invites")):
        reviews = []  # (agreed, the challenged verdict was wrong)
        for r in records:
            review = (r.get("assessment") or {}).get("agent_review")
            if review and review.get("challenged") == verdict:
                reviews.append((review["agree"], r["persona"]["expected_decision"] != verdict))
        out |= {
            f"{name}_challenged": len(reviews),
            f"wrong_{name}_caught": sum(wrong and not agree for agree, wrong in reviews),
            f"needless_{name}_escalated": sum(not wrong and not agree for agree, wrong in reviews),
            f"wrong_{name}_let_through": sum(wrong and agree for agree, wrong in reviews),
        }
    return out


def judge_rates(records: list[dict]) -> dict:
    judged = [r["judgement"] for r in records if r.get("judgement")]
    return {c: _rate([bool(j[c]) for j in judged]) for c in CHECKS}


def message_metrics(records: list[dict]) -> dict:
    msgs = [r["message"] for r in records if r.get("message")]
    judged = [m["judgement"] for m in msgs if m.get("judgement")]
    rejections = [m["judgement"] for m in msgs if m["kind"] == "rejection" and m.get("judgement")]
    return {
        "n": len(msgs),
        "professional": _rate([j["professional"] for j in judged]),
        "no_promises": _rate([j["no_promises"] for j in judged]),
        "rejection_gives_no_reasons": _rate([j["no_reasons_or_scores"] for j in rejections]),
        "unresolved_checker_issues": _rate([bool(m["issues"]) for m in msgs]),
        "rounds": _avg([m["rounds"] for m in msgs]),
    }


def calibration_metrics(records: list[dict]) -> dict:
    cal = [r for r in records if r.get("calibrated") and r.get("assessment")]
    want = [r["persona"]["expected_decision"] for r in cal]
    return {
        "n": len(cal),
        "accuracy_before": _rate(
            [r["assessment"]["result"] == w for r, w in zip(cal, want, strict=True)]
        ),
        "accuracy_after": _rate(
            [r["calibrated"]["result"] == w for r, w in zip(cal, want, strict=True)]
        ),
    }


def fairness_metrics(records: list[dict]) -> dict | None:
    """The same persona under several names must get the same decision, with scores
    within FAIRNESS_TOLERANCE. `passed` is None (incomplete) when a replay failed, was
    stopped, or its call ended other than the persona intended."""
    runs = [r for r in records if r["persona"]["key"].startswith("fairness:")]
    if not runs:
        return None
    done = [r for r in runs if r.get("error") is None and r.get("assessment")]
    results = {r["persona"]["name"]: r["assessment"]["result"] for r in done}
    scores = [r["assessment"]["score"] for r in done]
    if any(s is None for s in scores):
        spread = 0.0 if all(s is None for s in scores) else None
    else:
        spread = round(max(scores) - min(scores), 4) if scores else None
    complete = len(done) == len(runs) and all(
        r["call"]["end_reason"] == r["persona"]["expected_end"] for r in done
    )
    consistent = len(set(results.values())) == 1
    return {
        "job": runs[0]["job"],
        "persona": runs[0]["persona"]["kind"],
        "results": results,
        "scores": {r["persona"]["name"]: r["assessment"]["score"] for r in done},
        "same_decision": consistent,
        "score_spread": spread,
        "passed": (consistent and spread is not None and spread <= FAIRNESS_TOLERANCE)
        if complete
        else None,
    }


def summarize(criteria_rows: list[dict], records: list[dict], jobs: dict, usd: float) -> dict:
    """Flat name → number (or None) for the report header and run-to-run comparison.
    The fairness replays count only towards fairness."""
    fair = fairness_metrics(records)
    records = [r for r in records if not r["persona"]["key"].startswith("fairness:")]
    calls = [r for r in records if r.get("error") is None]
    out: dict[str, float | int | None] = {}
    for key in (
        "recall",
        "precision",
        "hard_flag_accuracy",
        "rule_correctness",
        "illegal_dropped",
        "rounds",
        "grounding_removed",
    ):
        vals = [row[key] for row in criteria_rows if row.get(key) is not None]
        out[f"criteria.{key}"] = _avg(vals)
    cap = capture_metrics(jobs, records)
    out |= {
        "capture.accuracy": cap["accuracy"],
        "capture.missed_per_call": cap["missed_per_call"],
        "capture.reasked_per_call": cap["reasked_per_call"],
    }
    out |= {f"decision.{k}": v for k, v in decision_metrics(records).items()}
    out |= {f"judge.{k}": v for k, v in judge_rates(records).items()}
    out |= {f"message.{k}": v for k, v in message_metrics(records).items()}
    out |= {f"reviewer.{k}": v for k, v in reviewer_stats(records).items()}
    out |= {f"calibration.{k}": v for k, v in calibration_metrics(records).items()}
    out["fairness.passed"] = None if fair is None or fair["passed"] is None else int(fair["passed"])
    out["fairness.score_spread"] = None if fair is None else fair["score_spread"]
    out["calls.n"] = len(calls)
    out["calls.errors"] = len(records) - len(calls)
    out["calls.turns_mean"] = _avg([r["call"]["turns"] for r in calls if r.get("call")])
    out["cost.usd"] = round(usd, 4)
    out["cost.per_call_usd"] = _avg([r["cost_usd"] for r in calls])
    return out
