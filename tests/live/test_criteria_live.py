"""Runs the criteria agent against the real models. Costs a few LLM calls:
RUN_LIVE=1 uv run pytest tests/live -v"""

import os

import pytest

from voiceagent.agents.criteria import run_criteria_agent
from voiceagent.agents.tools import verify_quote
from voiceagent.criteria.schema import check_criteria
from voiceagent.llm import default_models
from voiceagent.settings import get_settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1"),
]

JD = """Senior Backend Engineer — Fintech, Bengaluru (hybrid, 3 days a week in office)

You'll design and run payment APIs in Python (FastAPI) on PostgreSQL and Kafka.

Must have:
- 4+ years building backend services in Python
- Production experience with PostgreSQL
- Willing to work from our Bengaluru office 3 days a week

Nice to have: Kafka, AWS, payments domain.
Budget: 25–35 LPA. Immediate joiners or up to 30 days notice preferred.
Candidates under 30 preferred."""


def test_criteria_agent_on_a_real_job_description():
    models = default_models(get_settings())
    result = run_criteria_agent(
        models,
        title="Senior Backend Engineer",
        description=JD,
        notes="",
        similar_lookup=lambda q: [],
    )
    c = result.criteria
    print(c.model_dump_json(indent=1), *result.warnings, f"rounds={result.rounds}", sep="\n")

    assert 3 <= len(c.requirements) <= 8
    assert check_criteria(c)[0] == []
    assert all(verify_quote(r.source, JD) for r in c.requirements)
    hard = {r.id: r for r in c.requirements if r.hard}
    assert any(r.kind == "number" and r.rule.op in (">=", ">") for r in hard.values()), (
        "the 4+ years must-have should be a hard number rule"
    )
    text = " ".join(f"{r.label} {r.question}".lower() for r in c.requirements)
    assert "age" not in text.split() and "how old" not in text, "age must never be screened"
    salary = [r for r in c.requirements if "salary" in r.label.lower() or r.unit == "LPA"]
    assert all(r.sensitive for r in salary)
