from datetime import datetime
from zoneinfo import ZoneInfo

from voiceagent.agents.assessment import run_assessment
from voiceagent.evals.jobs import load_jobs
from voiceagent.evals.offline import OfflineAgentLLM, OfflineModels, OfflineSimulator
from voiceagent.evals.personas import make_persona
from voiceagent.evals.simulate import simulate_call

JOB = load_jobs(["staff-nurse"])[0]
NOW = datetime(2026, 10, 9, 14, 30, tzinfo=ZoneInfo("Asia/Kolkata"))


async def call(key):
    persona = make_persona(OfflineModels(JOB), JOB, key, "Meera Pillai")
    model = OfflineAgentLLM(JOB, NOW)
    sim = await simulate_call(
        JOB, persona, agent_llm=model, backup_llm=model, simulator=OfflineSimulator(), now=NOW
    )
    return persona, sim


async def test_an_offline_call_asks_every_question_and_closes():
    persona, sim = await call("clear_fit")
    assert sim.end_reason == "completed"
    agent_lines = [t["text"] for t in sim.transcript if t["role"] == "agent"]
    for r in JOB.criteria.requirements:
        assert r.question in agent_lines
    a = run_assessment(
        OfflineModels(JOB, persona),
        requirements=JOB.criteria.requirements,
        pass_threshold=0.6,
        transcript=sim.transcript,
        resume=persona.resume,
        end_reason=sim.end_reason,
        overrides_lookup=lambda r: [],
    )
    assert a.result == "suitable"  # every quote verified against the candidate's words
    assert all(r["confidence"] >= 0.6 for r in a.reasons)


async def test_offline_refusals_and_callbacks_end_the_call_like_the_real_agent():
    _, refused = await call("refuses_consent")
    assert refused.end_reason == "declined_consent"
    _, busy = await call("bad_time")
    assert (busy.end_reason, busy.callback_at) == (
        "callback_requested",
        "2026-10-10T17:00:00+05:30",
    )


async def test_offline_models_answer_unknown_schemas_with_an_empty_report():
    from pydantic import BaseModel

    class Check(BaseModel):
        issues: list[str] = []

    assert OfflineModels(JOB).structured(Check, "", "").issues == []
