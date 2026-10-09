"""One evaluation run: criteria generation on every reference job, then simulated calls
through the real agent, assessment and message agents, judged and summarised into a
report. Sequential, so the budget check between steps is exact.
# ponytail: one call at a time (~1-2 min each); run personas concurrently if runs grow.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from livekit.agents import llm

from voiceagent.agents.assessment import Assessment, run_assessment
from voiceagent.agents.criteria import run_criteria_agent
from voiceagent.agents.messages import run_message_agent
from voiceagent.evals.cost import BudgetExceeded, Meter
from voiceagent.evals.jobs import JOBS_DIR, RefJob, load_jobs
from voiceagent.evals.judges import judge_call, judge_message, match_criteria
from voiceagent.evals.metrics import criteria_metrics, fairness_metrics, summarize
from voiceagent.evals.offline import OfflineAgentLLM, OfflineModels, OfflineSimulator
from voiceagent.evals.personas import FAIRNESS_NAMES, Persona, make_persona, persona_keys, renamed
from voiceagent.evals.personas import NAMES as PERSONA_NAMES
from voiceagent.evals.report import RUNS_DIR, compare, previous_run, run_name, write_run
from voiceagent.evals.simulate import Simulator, simulate_call
from voiceagent.llm import Models
from voiceagent.worker.assess import MESSAGE_KIND, NOT_ASSESSED, answered_consent

TZ = ZoneInfo("Asia/Kolkata")
# Re-assessed with the job's synthetic recruiter overrides, to measure calibration.
CALIBRATION_KINDS = ("clear_fit", "borderline")
# The candidate is played by the voice model: cheap, fast, no reasoning.
SIMULATOR_MODEL = "Qwen/Qwen3.5-9B"


@dataclass
class EvalConfig:
    jobs: list[str] | None = None  # None: all
    personas: int = 4
    runs: int = 1
    out: Path = RUNS_DIR
    compare: bool = True
    budget_usd: float | None = 2.0
    offline: bool = False
    jobs_dir: Path = JOBS_DIR


@dataclass
class Backend:
    models_for: Callable[[RefJob, Persona | None], Models]
    agent_llm: Callable[[RefJob], llm.LLM]
    backup_llm: Callable[[RefJob], llm.LLM]
    simulator: Simulator
    now: Callable[[], datetime] = field(default=lambda: datetime.now(TZ))


def offline_backend() -> Backend:
    now = datetime(2026, 10, 9, 14, 30, tzinfo=TZ)
    return Backend(
        models_for=OfflineModels,
        agent_llm=lambda job: OfflineAgentLLM(job, now),
        backup_llm=lambda job: OfflineAgentLLM(job, now),
        simulator=OfflineSimulator(),
        now=lambda: now,
    )


def real_backend(meter: Meter) -> Backend:
    """Together AI: the backend agents and judges on the default models, the call on the
    production voice model, the candidate on SIMULATOR_MODEL."""
    from langchain_openai import ChatOpenAI

    from voiceagent.evals.simulate import LLMSimulator
    from voiceagent.llm import TOGETHER_BASE_URL, default_models
    from voiceagent.replies import NO_REASONING, backup_llm, main_llm
    from voiceagent.settings import get_settings

    settings = get_settings()
    os.environ.setdefault("TOGETHER_API_KEY", settings.together_api_key)
    models = default_models(settings)
    chat = ChatOpenAI(
        model=SIMULATOR_MODEL,
        api_key=settings.together_api_key,
        base_url=TOGETHER_BASE_URL,
        temperature=0,
        max_tokens=300,
        timeout=60,
        max_retries=2,
        extra_body=NO_REASONING,
    )
    return Backend(
        models_for=lambda job, persona: models,
        agent_llm=lambda job: meter.watch(main_llm()),
        backup_llm=lambda job: meter.watch(backup_llm()),
        simulator=LLMSimulator(chat),
    )


def _error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:500]


def eval_criteria(models: Models, job: RefJob) -> dict:
    """The criteria agent on the job's description, scored against its labels. No
    retrieval: the eval is hermetic, so similar past jobs never leak in."""
    res = run_criteria_agent(
        models,
        title=job.title,
        description=job.description,
        notes=job.notes,
        similar_lookup=lambda query: [],
    )
    pairs = match_criteria(models, job, res.criteria)
    return criteria_metrics(job, res.criteria, res.warnings, res.rounds, pairs) | {
        "warnings": res.warnings
    }


def overrides_lookup(job: RefJob) -> Callable:
    """The job's synthetic recruiter overrides as (id, text), described the way the scorer
    sees real ones."""
    labels = {r.id: r.label for r in job.criteria.requirements}

    def lookup(requirement) -> list[tuple[str, str]]:
        return [
            (
                f"{job.slug}:override:{i}",
                f'{job.title} — {labels[o.requirement_id]}: the candidate said "{o.quote}". '
                f"The agent scored it {o.agent_score:.2f} and judged the candidate "
                f"{o.agent_result}; the recruiter decided {o.recruiter_result}: {o.note}",
            )
            for i, o in enumerate(job.overrides)
            if o.requirement_id == requirement.id
        ]

    return lookup


def _assessment(a: Assessment) -> dict:
    return {
        "result": a.result,
        "score": a.score,
        "hard_failures": a.hard_failures,
        "summary": a.summary,
        "reasons": a.reasons,
        "answers": [e.model_dump(mode="json") for e in a.answers],
        "agent_review": a.agent_review,
    }


async def eval_persona(
    backend: Backend, meter: Meter, job: RefJob, persona: Persona, run: int, records: list
) -> dict:
    """Call, assess, draft the message and judge one persona. The record is appended first,
    so whatever stops it (an error, the budget) leaves what it got."""
    rec = {
        "job": job.slug,
        "persona": persona.model_dump(mode="json"),
        "run": run,
        "call": None,
        "assessment": None,
        "calibrated": None,
        "message": None,
        "judgement": None,
        "cost_usd": 0.0,
        "error": None,
    }
    records.append(rec)
    before = meter.usd
    reqs = job.criteria.requirements
    try:
        sim = await simulate_call(
            job,
            persona,
            agent_llm=backend.agent_llm(job),
            backup_llm=backend.backup_llm(job),
            simulator=backend.simulator,
            now=backend.now(),
            meter=meter,
        )
        rec["call"] = {
            "end_reason": sim.end_reason,
            "callback_at": sim.callback_at,
            "turns": sim.candidate_turns,
            "transcript": sim.transcript,
        }
        models = backend.models_for(job, persona)
        extracted: dict = {}
        if sim.end_reason == "candidate_hung_up" and not answered_consent(sim.transcript):
            # as postcall does: no consent, nothing assessed, a recruiter decides
            rec["assessment"] = {
                "result": "needs_review",
                "score": None,
                "hard_failures": [],
                "summary": "hung up before answering the consent question",
                "reasons": [],
                "answers": [],
                "agent_review": None,
            }
        elif sim.end_reason not in NOT_ASSESSED:

            def assess(lookup) -> Assessment:
                meter.check()
                return run_assessment(
                    models,
                    requirements=reqs,
                    pass_threshold=job.criteria.pass_threshold,
                    transcript=sim.transcript,
                    resume=persona.resume,
                    end_reason=sim.end_reason,
                    overrides_lookup=lookup,
                    auto_send=True,  # jobs auto-send invites by default: the reviewer checks them
                )

            # the fairness replays run as production would, with the recruiter's overrides
            fairness_replay = persona.key.startswith("fairness:")
            a = assess(overrides_lookup(job) if fairness_replay else lambda r: [])
            rec["assessment"] = _assessment(a)
            extracted = {e.requirement_id: e.value for e in a.answers}
            if (
                persona.kind in CALIBRATION_KINDS
                and job.overrides
                and run == 0
                and not fairness_replay
            ):
                c = assess(overrides_lookup(job))
                rec["calibrated"] = {"result": c.result, "score": c.score}
            if a.result in MESSAGE_KIND:
                meter.check()
                kind = MESSAGE_KIND[a.result]
                d = run_message_agent(
                    models,
                    kind=kind,
                    candidate_name=persona.name,
                    job_title=job.title,
                    company_name=job.company,
                    job_facts=job.criteria.job_facts.model_dump(mode="json"),
                    requirement_labels=[r.label for r in reqs],
                )
                meter.check()
                rec["message"] = {
                    "kind": kind,
                    "subject": d.subject,
                    "body": d.body,
                    "issues": d.issues,
                    "rounds": d.rounds,
                    "judgement": judge_message(models, kind, d.subject, d.body).model_dump(),
                }
        meter.check()
        rec["judgement"] = judge_call(models, job, persona, sim.transcript, extracted).model_dump()
    except BudgetExceeded as e:
        rec["error"] = f"stopped: {e}"
        raise
    except Exception as e:  # one persona failing never stops the run
        rec["error"] = _error(e)
    finally:
        rec["cost_usd"] = round(meter.usd - before, 6)
    return rec


def _failed_persona(job: RefJob, key: str, name: str, error: str) -> dict:
    persona = {
        "key": key,
        "kind": key.partition(":")[0],
        "name": name,
        "truth": {},
        "declined": [],
        "expected_end": None,
        "expected_decision": None,
    }
    return {
        "job": job.slug,
        "persona": persona,
        "run": 0,
        "call": None,
        "assessment": None,
        "calibrated": None,
        "message": None,
        "judgement": None,
        "cost_usd": 0.0,
        "error": f"persona: {error}",
    }


async def run_eval(cfg: EvalConfig, backend: Backend, meter: Meter) -> dict:
    """Run, write evals/runs/<name>/ and return the results."""
    started = datetime.now(UTC)
    name = run_name(started)
    jobs = load_jobs(cfg.jobs, cfg.jobs_dir)
    criteria_rows: list[dict] = []
    records: list[dict] = []
    fairness_base: Persona | None = None
    aborted = None
    try:
        for job in jobs:
            meter.check()
            try:
                criteria_rows.append(eval_criteria(backend.models_for(job, None), job))
            except BudgetExceeded:
                raise
            except Exception as e:
                criteria_rows.append({"job": job.slug, "error": _error(e)})
        for i, job in enumerate(jobs):
            for n, key in enumerate(persona_keys(job, cfg.personas, offset=i)):
                meter.check()
                pname = PERSONA_NAMES[(i + n) % len(PERSONA_NAMES)]
                try:
                    persona = make_persona(backend.models_for(job, None), job, key, pname)
                except BudgetExceeded:
                    raise
                except Exception as e:
                    records.append(_failed_persona(job, key, pname, _error(e)))
                    continue
                if i == 0 and persona.kind == "borderline":
                    fairness_base = persona
                for run in range(cfg.runs):
                    await eval_persona(backend, meter, job, persona, run, records)
        if fairness_base is not None:
            for fname in FAIRNESS_NAMES:
                await eval_persona(
                    backend, meter, jobs[0], renamed(fairness_base, fname), 0, records
                )
    except BudgetExceeded as e:
        aborted = str(e)
    by_slug = {j.slug: j for j in jobs}
    summary = summarize(criteria_rows, records, by_slug, meter.usd)
    prev = previous_run(cfg.out, name) if cfg.compare else None
    results = {
        "name": name,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "config": {
            "jobs": [j.slug for j in jobs],
            "personas": cfg.personas,
            "runs": cfg.runs,
            "budget_usd": cfg.budget_usd,
            "offline": cfg.offline,
        },
        "aborted": aborted,
        "summary": summary,
        "criteria": criteria_rows,
        "calls": records,
        "fairness": fairness_metrics(records),
        "comparison": {
            "previous": prev["name"] if prev else None,
            "rows": compare(summary, prev) if prev else [],
        },
        "cost": {"usd": meter.usd, "tokens": meter.tokens()},
    }
    write_run(cfg.out / name, results)
    return results
