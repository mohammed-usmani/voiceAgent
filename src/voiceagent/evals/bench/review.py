"""Task 3: checking an assessment's reasons before a recruiter sees them.

Contenders: the production reviewer (assessment.REVIEWER + Review/ReasonCheck on the
reviewer model), run once per call as production runs it and scored per reason (flagged =
not supported or not matching the résumé); and Jev, two noul questions per reason
(supported, matches the résumé) on a state with that reason, the requirement, the
transcript and the résumé (flagged = either below 0.5).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from voiceagent.agents.assessment import REVIEWER, Review, format_transcript
from voiceagent.agents.tools import untrusted
from voiceagent.evals.bench import stats
from voiceagent.evals.bench.common import (
    DATA,
    JEV_WORKERS,
    LLM_WORKERS,
    SEED,
    Jsonl,
    add_costs,
    eval_calls,
    jobs,
    pmap,
    production_models,
    timed,
    tokens_delta,
    usd,
    write_jsonl,
)
from voiceagent.evals.bench.jev import Jev
from voiceagent.evals.cost import Meter
from voiceagent.evals.metrics import answer_correct
from voiceagent.evals.personas import rule_value
from voiceagent.rules import apply_rule, decide

ORIGINAL_CALLS = 50
PER_CORRUPTION = 20
CORRUPTIONS = ("wrong_value", "unsupported_quote", "unit_mixup", "resume_contradiction")
FLAG_BELOW = 0.5
# months read as years, months read as days, a monthly figure read as annual
UNIT_FACTOR = {"years": 12, "days": 30, "LPA": 12}


def _screened() -> list[tuple[str, int, dict]]:
    return [
        (run, idx, rec)
        for run, idx, rec in eval_calls()
        if rec.get("error") is None
        and rec.get("call")
        and rec["call"]["end_reason"] == "completed"
        and rec["persona"]["expected_end"] == "completed"
        and (rec.get("assessment") or {}).get("reasons")
    ]


def _req(job, rid):
    return next(r for r in job.criteria.requirements if r.id == rid)


def label_reason(job, rec: dict, reason: dict) -> tuple[bool, str] | None:
    """(should flag, category) where the persona's ground truth decides it, else None."""
    p = rec["persona"]
    r = _req(job, reason["requirement_id"])
    answers = {a["requirement_id"]: a for a in rec["assessment"]["answers"]}
    truth, declined = p["truth"].get(r.id), r.id in p["declined"]
    v = reason["verdict"]
    if r.hard and v in ("pass", "fail"):
        hard_numbers = [x for x in job.criteria.requirements if x.hard and x.kind == "number"]
        if p["kind"] == "overstates" and hard_numbers and hard_numbers[0].id == r.id:
            return True, "real_resume_contradiction"
        want = apply_rule(r, truth, declined=declined)
        if want == "unknown":
            return None
        ok = v == want and answer_correct(r, truth, declined, answers.get(r.id), None)
        return (False, "correct") if ok else (True, "real_wrong")
    if not r.hard and v == "declined":
        return (False, "correct") if declined else (True, "real_wrong")
    if not r.hard and v == "scored" and r.kind == "boolean" and isinstance(truth, bool):
        ok = (reason["score"] >= 0.5) == truth
        return (False, "correct") if ok else (True, "real_wrong")
    return None


def _header(job, rec: dict, reasons: list[dict], changed: bool) -> tuple[str, list[str]]:
    """The verdict and summary lines the reviewer sees: the stored ones before review, or,
    when a corruption changed a hard verdict, recomputed as the verdict step would."""
    a = rec["assessment"]
    result = (a.get("agent_review") or {}).get("challenged") or a["result"]
    lines = [x for x in a["summary"].split("\n") if x and not x.startswith("The reviewer")]
    if not changed:
        return result, lines
    labels = {r.id: r.label for r in job.criteria.requirements}
    hard = {
        x["requirement_id"]: x["verdict"] for x in reasons if _req(job, x["requirement_id"]).hard
    }
    rest = [x for x in lines if not x.startswith("Must-have")]
    result = decide(list(hard.values()), a["score"], job.criteria.pass_threshold)
    if any(x.startswith("Not assessed") for x in rest) and "fail" not in hard.values():
        result = "needs_review"
    must = [f"Must-have not met: {labels[k]}" for k, v in hard.items() if v == "fail"]
    must += [f"Must-have unconfirmed: {labels[k]}" for k, v in hard.items() if v == "unknown"]
    return result, must + rest


def _note(r, value, verdict) -> str:
    unit = f" {r.unit}" if r.unit else ""
    return f"{value!r}{unit} vs rule {r.rule.op} {r.rule.value!r}: {verdict}"


def corrupt(kind: str, job, rec: dict, reason: dict) -> tuple[dict, str] | None:
    """(the corrupted reason, the résumé) or None when this kind doesn't apply."""
    r = _req(job, reason["requirement_id"])
    value = {a["requirement_id"]: a for a in rec["assessment"]["answers"]}[r.id]["value"]
    resume = rec["persona"]["resume"]
    out = dict(reason)
    if kind == "wrong_value":
        if r.kind == "boolean":
            new = not value
        else:
            new = rule_value(r, "fail" if reason["verdict"] == "pass" else "pass")
        if new is None or apply_rule(r, new) == reason["verdict"]:
            return None
        v = apply_rule(r, new)
        out |= {"verdict": v, "note": _note(r, new, v)}
    elif kind == "unsupported_quote":
        first = next(t["text"] for t in rec["call"]["transcript"] if t["role"] == "candidate")
        if first.strip() == (reason["quote"] or "").strip():
            return None
        out["quote"] = first
    elif kind == "unit_mixup":
        if r.kind != "number" or r.unit not in UNIT_FACTOR or not isinstance(value, int | float):
            return None
        new = round(value * UNIT_FACTOR[r.unit], 2)
        v = apply_rule(r, new)
        out |= {"verdict": v, "note": _note(r, new, v)}
    elif kind == "resume_contradiction":
        if r.kind == "number":
            alt = rule_value(r, "fail" if reason["verdict"] == "pass" else "pass")
            if alt is None:
                return None
            line = f"{r.label}: {alt:g} {r.unit or ''}".rstrip() + "."
        elif r.kind == "boolean":
            line = f"{r.label}: {'no' if value else 'yes'}."
        else:
            other = [o for o in r.options or [] if str(o).casefold() != str(value).casefold()]
            if not other:
                return None
            line = f"{r.label}: {other[-1]}."
        resume = f"{resume.rstrip()} {line}"
    return out, resume


def build(meter: Meter | None = None, limit: int | None = None) -> list[dict]:
    """Call instances to review: a sample of original calls with their ground-truth labels,
    and copies with one correct hard reason corrupted. Deterministic (no model calls)."""
    rng = random.Random(SEED)
    instances, candidates = [], []
    for run, idx, rec in _screened():
        job = jobs()[rec["job"]]
        labels = {}
        for reason in rec["assessment"]["reasons"]:
            lab = label_reason(job, rec, reason)
            if lab is not None:
                labels[reason["requirement_id"]] = {"flag": lab[0], "category": lab[1]}
                if lab == (False, "correct") and _req(job, reason["requirement_id"]).hard:
                    candidates.append((run, idx, rec, reason))
        if labels:
            instances.append(
                {
                    "key": f"orig:{run}:{idx}",
                    "run": run,
                    "idx": idx,
                    "labels": labels,
                    "reasons": rec["assessment"]["reasons"],
                    "resume": rec["persona"]["resume"],
                    "changed": False,
                }
            )
    # every call with a real wrong reason, then a random sample of the rest
    wrong = [i for i in instances if any(v["flag"] for v in i["labels"].values())]
    rest = [i for i in instances if i not in wrong]
    originals = wrong + rng.sample(rest, max(0, min(len(rest), ORIGINAL_CALLS - len(wrong))))
    corrupted = []
    for kind in CORRUPTIONS:
        pool = candidates[:]
        rng.shuffle(pool)
        n = 0
        for run, idx, rec, reason in pool:
            if n == PER_CORRUPTION:
                break
            job = jobs()[rec["job"]]
            got = corrupt(kind, job, rec, reason)
            if got is None:
                continue
            new, resume = got
            rid = reason["requirement_id"]
            reasons = [
                new if x["requirement_id"] == rid else x for x in rec["assessment"]["reasons"]
            ]
            corrupted.append(
                {
                    "key": f"{kind}:{run}:{idx}:{rid}",
                    "run": run,
                    "idx": idx,
                    "labels": {rid: {"flag": True, "category": kind}},
                    "reasons": reasons,
                    "resume": resume,
                    "changed": new["verdict"] != reason["verdict"],
                }
            )
            n += 1
    out = sorted(originals, key=lambda i: i["key"]) + corrupted
    write_jsonl(DATA / "review_items.jsonl", out)
    return out[:limit] if limit else out


def _records() -> dict[tuple[str, int], dict]:
    return {(run, idx): rec for run, idx, rec in eval_calls()}


def reviewer_user(job, rec: dict, inst: dict) -> str:
    """The review step's user message (assessment.run_assessment.review), verbatim."""
    result, summary = _header(job, rec, inst["reasons"], inst["changed"])
    convo = untrusted("transcript", format_transcript(rec["call"]["transcript"]))
    resume_block = f"Résumé:\n{untrusted('resume', inst['resume'] or '(none)')}"
    return (
        f"Verdict: {result}\n"
        + "\n".join(summary)
        + "\n\nReasons:\n"
        + json.dumps(inst["reasons"], indent=1, ensure_ascii=False)
        + "\n\nRequirements:\n"
        + json.dumps([r.model_dump(mode="json") for r in job.criteria.requirements], indent=1)
        + f"\n\nTranscript:\n{convo}\n\n{resume_block}"
    )


def jev_request(job, rec: dict, inst: dict, rid: str) -> tuple[str, dict]:
    r = _req(job, rid)
    reason = next(x for x in inst["reasons"] if x["requirement_id"] == rid)
    keep = ("id", "label", "kind", "unit", "options", "rule", "source")
    req = {k: v for k, v in r.model_dump(mode="json").items() if k in keep and v is not None}
    state = (
        f"Requirement: {json.dumps(req, ensure_ascii=False)}\n"
        f"Reason to check: {json.dumps(reason, ensure_ascii=False)}\n\n"
        f"Transcript:\n{format_transcript(rec['call']['transcript'])}\n\n"
        f"Résumé:\n{inst['resume'] or '(none)'}"
    )
    questions = {
        "supported": {
            "type": "noul",
            "instructions": "The transcript clearly supports this reason: its quote is the "
            "candidate's own answer to this requirement, the value and its unit are read "
            "correctly from it, and the verdict follows from the requirement's rule.",
        },
        "matches_resume": {
            "type": "noul",
            "instructions": "The résumé does not state anything different from the candidate's "
            "answer for this requirement.",
        },
    }
    return state, questions


def run(meter: Meter, items: list[dict], out_dir: Path, limit: int | None = None) -> dict:
    todo = items[:limit] if limit else items
    recs = _records()
    rows = Jsonl(out_dir / "review.jsonl")
    reviewer = production_models().reviewer
    jev = Jev(meter)

    def llm(inst) -> None:
        key = f"llm:{inst['key']}"
        if key in rows:
            return
        rec = recs[(inst["run"], inst["idx"])]
        job = jobs()[rec["job"]]
        row = {
            "key": key,
            "who": "llm",
            "inst": inst["key"],
            "error": None,
            "reasons_reviewed": len(inst["reasons"]),
        }
        try:
            out, row["seconds"] = timed(
                reviewer.structured, Review, REVIEWER, reviewer_user(job, rec, inst)
            )
            checks = {c.requirement_id: c for c in out.checks}
            row["flags"] = {
                rid: (
                    None
                    if rid not in checks
                    else (not checks[rid].supported) or (not checks[rid].matches_resume)
                )
                for rid in inst["labels"]
            }
            row["findings"] = {rid: checks[rid].finding for rid in inst["labels"] if rid in checks}
        except Exception as e:
            row["error"] = repr(e)[:300]
        rows.add(row)

    def jv(t) -> None:
        inst, rid = t
        key = f"jev:{inst['key']}:{rid}"
        if key in rows:
            return
        rec = recs[(inst["run"], inst["idx"])]
        job = jobs()[rec["job"]]
        row = {"key": key, "who": "jev", "inst": inst["key"], "rid": rid, "error": None}
        try:
            answers, row["seconds"], _ = jev.ask(*jev_request(job, rec, inst, rid))
            row["supported"] = answers["supported"]["noul"]
            row["matches_resume"] = answers["matches_resume"]["noul"]
            row["flag"] = min(row["supported"], row["matches_resume"]) < FLAG_BELOW
        except Exception as e:
            row["error"] = repr(e)[:300]
        rows.add(row)

    costs = {}
    pairs = [(i, rid) for i in todo for rid in i["labels"]]
    for name, fn, work, workers in (
        ("jev", jv, pairs, JEV_WORKERS),
        ("llm", llm, todo, LLM_WORKERS),
    ):
        before = meter.tokens()
        print(f"  review: {name} on {len(work)} {'reasons' if name == 'jev' else 'calls'}")
        pmap(fn, work, workers=workers, meter=meter, what=f"review/{name}")
        costs[name] = add_costs(out_dir, f"review/{name}", tokens_delta(before, meter.tokens()))
    costs["jev_retries"] = jev.retries
    return costs


def outcomes(items: list[dict], rows: list[dict]) -> dict[str, dict[tuple[str, str], bool | None]]:
    """Per contender, (instance, requirement) → flagged (None: no answer for it)."""
    out = {"llm": {}, "jev": {}}
    for r in rows:
        if r["error"]:
            continue
        if r["who"] == "llm":
            for rid, f in r["flags"].items():
                out["llm"][(r["inst"], rid)] = f
        else:
            out["jev"][(r["inst"], r["rid"])] = r["flag"]
    return out


def metrics(items: list[dict], rows: list[dict], costs: dict) -> dict:
    labels = {(i["key"], rid): v for i in items for rid, v in i["labels"].items()}
    got = outcomes(items, rows)
    m = {}
    for who in ("llm", "jev"):
        flags = got[who]
        should = [k for k, v in labels.items() if v["flag"] and k in flags]
        fine = [k for k, v in labels.items() if not v["flag"] and k in flags]
        caught = [bool(flags[k]) for k in should]
        false = [bool(flags[k]) for k in fine]
        cats = defaultdict(list)
        for k in should:
            cats[labels[k]["category"]].append(bool(flags[k]))
        secs = [r["seconds"] for r in rows if r["who"] == who and r.get("seconds") is not None]
        checks = sum(
            r["reasons_reviewed"] if who == "llm" else 1
            for r in rows
            if r["who"] == who and not r["error"]
        )
        tokens = costs.get(who, {})
        ci_c = stats.bootstrap(caught, stats.mean, seed=SEED)
        ci_f = stats.bootstrap(false, stats.mean, seed=SEED)
        m[who] = {
            "should_flag": len(should),
            "correct": len(fine),
            "missing_checks": sum(1 for k in labels if k in flags and flags[k] is None),
            "errors": sum(1 for r in rows if r["who"] == who and r["error"]),
            "catch_rate": round(stats.mean(caught), 4) if caught else None,
            "catch_rate_ci": [round(x, 4) for x in ci_c] if ci_c else None,
            "false_flag_rate": round(stats.mean(false), 4) if false else None,
            "false_flag_rate_ci": [round(x, 4) for x in ci_f] if ci_f else None,
            "catch_by_category": {c: [sum(v), len(v)] for c, v in sorted(cats.items())},
            "p50_s": round(stats.percentile(secs, 50), 3) if secs else None,
            "p95_s": round(stats.percentile(secs, 95), 3) if secs else None,
            "latency_unit": "per call (all its reasons)" if who == "llm" else "per reason",
            "usd": round(usd(tokens), 6),
            "usd_per_1000_checks": round(usd(tokens) / checks * 1000, 4) if checks else None,
        }
    both = [k for k in labels if got["llm"].get(k) is not None and got["jev"].get(k) is not None]

    def diff_ci(keys):
        pairs = [(bool(got["llm"][k]), bool(got["jev"][k])) for k in keys]
        if not pairs:
            return None

        def d(ps):
            return stats.mean([j for _, j in ps]) - stats.mean([x for x, _ in ps])

        ci = stats.bootstrap(pairs, d, seed=SEED)
        return {
            "diff": round(d(pairs), 4),
            "ci": [round(ci[0], 4), round(ci[1], 4)],
            "n": len(pairs),
        }

    m["paired"] = {
        "catch_rate": diff_ci([k for k in both if labels[k]["flag"]]),
        "false_flag_rate": diff_ci([k for k in both if not labels[k]["flag"]]),
    }
    return m


def verdict(m: dict) -> tuple[str, list[str]]:
    """Pre-registered: Jev is better if its catch rate ≥ the LLM's and its false-flag rate ≤
    the LLM's, with at least one of the two paired differences (Jev − LLM) significant in
    Jev's favour (95% CI excluding 0); the LLM is better in the mirror case; otherwise no
    clear winner."""
    c, f = m["paired"]["catch_rate"], m["paired"]["false_flag_rate"]
    why = [
        f"catch rate: Jev {m['jev']['catch_rate']} vs LLM {m['llm']['catch_rate']} "
        f"(paired diff {c})",
        f"false-flag rate: Jev {m['jev']['false_flag_rate']} vs LLM "
        f"{m['llm']['false_flag_rate']} (paired diff {f})",
    ]
    if c is None or f is None:
        return "not enough data", why
    jev_sig = c["ci"][0] > 0 or f["ci"][1] < 0
    llm_sig = c["ci"][1] < 0 or f["ci"][0] > 0
    if c["diff"] >= 0 and f["diff"] <= 0 and jev_sig:
        return "Jev", why
    if c["diff"] <= 0 and f["diff"] >= 0 and llm_sig:
        return "LLM", why
    return "no clear winner", why
