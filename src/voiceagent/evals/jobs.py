"""Reference jobs: real-looking job descriptions with hand-labelled criteria
(evals/jobs/<slug>.yaml). The labels are what the criteria agent is scored against and
what every simulated call screens on, so call results don't depend on generation."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from voiceagent.agents.tools import verify_quote
from voiceagent.criteria.schema import Criteria, check_criteria, normalize_weights

JOBS_DIR = Path("evals/jobs")


class Override(BaseModel):
    """A synthetic recruiter correction, fed to soft scoring to measure calibration."""

    requirement_id: str
    quote: str
    agent_score: float
    agent_result: str
    recruiter_result: str
    note: str


class RefJob(BaseModel):
    slug: str
    title: str
    company: str
    description: str
    notes: str = ""
    criteria: Criteria
    # Lines of the description no screening may use; the criteria agent must drop them.
    illegal: list[str] = Field(default_factory=list)
    overrides: list[Override] = Field(default_factory=list)


def load_job(path: Path) -> RefJob:
    data = yaml.safe_load(path.read_text())
    job = RefJob.model_validate({"slug": path.stem, **data})
    errors, _ = check_criteria(job.criteria)
    text = f"{job.description}\n{job.notes}"
    errors += [
        f"{r.id}: source not in the description"
        for r in job.criteria.requirements
        if not verify_quote(r.source, text)
    ]
    errors += [
        f"illegal line not in the description: {line!r}"
        for line in job.illegal
        if line not in job.description
    ]
    ids = {r.id for r in job.criteria.requirements}
    errors += [
        f"override for unknown requirement {o.requirement_id!r}"
        for o in job.overrides
        if o.requirement_id not in ids
    ]
    if errors:
        raise ValueError(f"{path.name}: " + "; ".join(errors))
    return job.model_copy(update={"criteria": normalize_weights(job.criteria)})


def load_jobs(slugs: list[str] | None = None, directory: Path = JOBS_DIR) -> list[RefJob]:
    """All reference jobs (sorted by slug), or the named ones; an unknown slug raises."""
    paths = {p.stem: p for p in sorted(directory.glob("*.yaml"))}
    unknown = sorted(set(slugs or []) - set(paths))
    if unknown:
        raise ValueError(f"unknown jobs: {', '.join(unknown)} (have {', '.join(paths)})")
    return [load_job(paths[s]) for s in (slugs or list(paths))]
