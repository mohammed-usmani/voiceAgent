from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from tests.fakes import ScriptedLLM, tool_call
from voiceagent.call.prompt import greeting
from voiceagent.evals.jobs import load_jobs
from voiceagent.evals.personas import Persona
from voiceagent.evals.simulate import END, LLMSimulator, eval_job_row, simulate_call

JOB = load_jobs(["software-engineer"])[0]
NOW = datetime(2026, 10, 9, 14, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
PERSONA = Persona(
    key="clear_fit",
    kind="clear_fit",
    name="Ananya Rao",
    resume="Backend engineer, 5 years of Python at a payments startup.",
    truth={"python_years": 5, "hybrid_bengaluru": True, "expected_ctc": None},
    declined=["expected_ctc"],
    behaviour="You are cooperative.",
    expected_end="completed",
    expected_decision="suitable",
)


class Lines:
    """A candidate who says these lines in order, then hangs up; keeps what they heard."""

    def __init__(self, *lines):
        self.lines, self.heard = list(lines), []

    def reply(self, persona, job, said):
        self.heard.append([t["text"] for t in said])
        return self.lines.pop(0) if self.lines else None


async def call(agent_script, candidate, **kw):
    model = ScriptedLLM(agent_script)
    return await simulate_call(
        JOB, PERSONA, agent_llm=model, backup_llm=model, simulator=candidate, now=NOW, **kw
    )


async def test_the_real_agent_talks_to_the_simulated_candidate_until_end_call():
    candidate = Lines("Yes, now is fine.", "Five years.")
    result = await call(
        [
            "How many years have you worked on backend Python?",
            ["Thanks, that's all.", tool_call("end_call", reason="completed")],
        ],
        candidate,
    )
    assert result.end_reason == "completed" and result.candidate_turns == 2
    greet = greeting(eval_job_row(JOB), SimpleNamespace(name="Ananya Rao"))
    assert [t["role"] for t in result.transcript] == ["agent", "candidate"] * 2 + ["agent"]
    assert result.transcript[0]["text"] == greet
    assert candidate.heard[0] == [greet]  # the candidate answers what the agent said


async def test_a_call_that_never_ends_is_cut_off_at_the_turn_cap():
    candidate = Lines(*["Hmm."] * 50)
    result = await call(["Go on."] * 50, candidate, max_turns=3)
    assert result.candidate_turns == 3
    assert result.end_reason == "candidate_hung_up"


async def test_the_candidate_can_hang_up():
    result = await call(["Hello?"], Lines())
    assert (result.end_reason, result.candidate_turns) == ("candidate_hung_up", 0)


async def test_a_callback_keeps_its_time():
    result = await call(
        [
            [
                "Sure, tomorrow at 5 pm.",
                tool_call("end_call", reason="callback_requested", callback_at="2026-10-10T17:00"),
            ]
        ],
        Lines("Busy now, call me tomorrow at 5 pm?"),
    )
    assert result.end_reason == "callback_requested"
    assert result.callback_at == "2026-10-10T17:00:00+05:30"


class FakeChat:
    def __init__(self, reply):
        self.reply, self.messages = reply, None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.reply)


def test_the_simulator_plays_the_persona_from_its_side_of_the_call():
    chat = FakeChat(' "Yes, five years." ')
    said = [
        {"role": "agent", "text": "Hi, is now a good time?"},
        {"role": "candidate", "text": "Yes."},
        {"role": "agent", "text": "How many years of Python?"},
    ]
    assert LLMSimulator(chat).reply(PERSONA, JOB, said) == "Yes, five years."
    system = chat.messages[0][1]
    assert "Ananya Rao" in system and "You are cooperative." in system
    assert "- Backend Python experience: 5" in system
    assert "- Expected salary: you'd rather not say" in system
    assert [role for role, _ in chat.messages[1:]] == ["human", "ai", "human"]


def test_the_simulator_hangs_up_on_the_end_marker():
    assert LLMSimulator(FakeChat(f"Bye! {END}")).reply(PERSONA, JOB, []) is None
    assert LLMSimulator(FakeChat("  ")).reply(PERSONA, JOB, []) is None
