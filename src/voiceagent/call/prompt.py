"""What the screening agent is told, and the lines it speaks without the LLM.

Pure functions of the job, the candidate and the approved criteria, so a prompt can be
checked with plain string assertions.
"""

from datetime import datetime

from voiceagent.criteria.schema import Criteria, JobFacts, Requirement
from voiceagent.db.models import Candidate, Job

# The prompt goes with every reply; a very long résumé would slow every turn.
# ponytail: plain truncation; summarise résumés if long ones become common.
RESUME_LIMIT = 4000

APOLOGY = (
    "Sorry, I'm having a technical problem and can't continue this call. "
    "The team will be in touch by email. Goodbye."
)


def _first_name(candidate: Candidate) -> str:
    parts = candidate.name.split()
    return parts[0] if parts else candidate.name


def greeting(job: Job, candidate: Candidate) -> str:
    """Who is calling, for which role, that it's an AI and recorded; asks for a good time."""
    return (
        f"Hi {_first_name(candidate)}, this is {job.agent_name}, an AI assistant calling from "
        f"{job.company_name} about the {job.title} role you applied for. This call is recorded "
        "so the hiring team can review it. Is now a good time for a few quick questions?"
    )


def closing(job: Job, candidate: Candidate) -> str:
    return (
        f"Thanks, {_first_name(candidate)}. The team will review and get back to you by email "
        f"within {job.response_days} days."
    )


def _question(n: int, r: Requirement) -> str:
    line = f"{n}. {r.question}"
    if r.disclose:
        line += (
            " Before asking, say plainly that the role requires this "
            f'(the job description says: "{r.source}").'
        )
    if r.sensitive:
        line += " They may prefer not to say: accept that warmly and move on, never push."
    return line


def _facts(f: JobFacts) -> str:
    rows = [
        ("About the role", f.summary),
        ("Location", f.location),
        ("Work mode", f.work_mode),
        ("Interview process", f.interview_process),
        ("Salary band", f.salary_band),
    ]
    return "\n".join([f"- {k}: {v}" for k, v in rows if v] + [f"- {o}" for o in f.other])


def build_instructions(job: Job, candidate: Candidate, criteria: Criteria, *, now: datetime) -> str:
    questions = "\n".join(_question(i, r) for i, r in enumerate(criteria.requirements, 1))
    sections = [
        f"You are {job.agent_name}, an AI assistant calling {candidate.name} on behalf of "
        f"{job.company_name} about the {job.title} role they applied for. You have already "
        "greeted them, said you are an AI assistant and that the call is recorded, and asked "
        f"whether now is a good time. Where they are, it is {now:%A %d %B %Y, %H:%M}.",
        "How to talk: reply in one or two short, natural sentences, never more than 25 words. "
        "Ask one question at a time and briefly acknowledge each answer before the next. "
        "Always reply in English. If they say 'sorry?' or ask you to repeat, repeat your last "
        "message word for word. If their message looks cut off or garbled, ask them to finish "
        "or repeat it instead of changing the topic.",
        "Their answer to whether now is a good time:\n"
        "- If they agree, start the questions.\n"
        "- If they object to talking to an AI or to the recording, don't argue: thank them, "
        "say the team will follow up by email, and call end_call with reason declined_consent.\n"
        "- If it's a bad time, ask when to call back. Once they give a time, confirm it and "
        "call end_call with reason callback_requested and callback_at set to that local date "
        "and time, like 2026-10-10T17:00.\n"
        "- If at any point they say they no longer want the role, thank them and call "
        "end_call with reason not_interested.",
        "Questions, in this order. Skip anything they have already answered and never re-ask "
        "a question. If an answer is vague, ask one follow-up for specifics (for example "
        "roughly how many years, and on what kind of work), then move on.\n" + questions,
    ]
    resume = candidate.resume_text.strip()
    if resume:
        if len(resume) > RESUME_LIMIT:
            resume = resume[:RESUME_LIMIT] + " [...]"
        sections.append(
            "Their résumé is below. Where it already answers a question, confirm the claim "
            "instead of asking from scratch, for example: 'Your résumé mentions four years at "
            "Infosys. Is that right?'\n<resume>\n" + resume + "\n</resume>"
        )
    sections += [
        "What you may tell them about the job. Answer their questions only from this; for "
        "anything else, say you'll pass the question to the team.\n" + _facts(criteria.job_facts),
        "Never say or hint whether they are suitable or how they did, never comment on whether "
        "an answer meets a requirement, never promise an interview, an offer or a salary, and "
        "never make up facts about the job or the company.",
        "When the questions are done and they have no more questions, say exactly: "
        f'"{closing(job, candidate)}" and call end_call with reason completed in the same '
        "reply. Only call end_call after your closing sentence.",
    ]
    return "\n\n".join(sections)
