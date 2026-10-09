"""Turns criteria results and call records into the numbers the report shows. Pure
functions over plain dicts (the shapes in results.json), so they are easy to test and a
past run's results can be re-read.

A call record: {"job", "persona" (Persona dump), "run", "call" {"end_reason",
"callback_at", "turns", "transcript"}, "assessment" {"result", "score", "answers",
"agent_review", ...} or None, "judgement" (CallJudgement dump) or None, "message"
{..., "judgement"} or None, "calibrated" {"result", "score"} or None, "cost_usd",
"error"}.
"""

from statistics import mean

from voiceagent.criteria.schema import Criteria, Rule
from voiceagent.evals.jobs import RefJob
from voiceagent.evals.judges import ILLEGAL

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
