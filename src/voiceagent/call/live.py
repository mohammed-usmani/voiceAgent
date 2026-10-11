"""What the agent publishes into the call's room for the live view (the recruiter's
frontend joins as a hidden viewer): which approved question it is asking, on
screening.progress, and each turn-taking record's timings, on screening.turns. Never
anything the candidate said. Publishing runs in the background and never breaks a call.

Which question a turn asks comes from Jev when it's configured (99% accurate on the
decision benchmark's 1,285 eval turns), else, or when Jev fails or is slow, from word
overlap (91%)."""

import asyncio
import json
import logging
import re

from livekit import rtc
from livekit.agents import AgentSession

from voiceagent.criteria.schema import Requirement
from voiceagent.jev import Jev

logger = logging.getLogger("voiceagent")
PROGRESS_TOPIC = "screening.progress"
TURNS_TOPIC = "screening.turns"
# Words that say nothing about which question is being asked.
STOPWORDS = frozenset(
    "about and any are been but can could did does for from had has have how its just let "
    "our that the their them then there this was were what when where which who will with "
    "would you your".split()
)
# The share of a question's words an agent turn must contain to count as asking it. On the
# evaluation transcripts, questions reworded around the résumé score 0.5-0.7, and closing
# lines and answers to the candidate's questions score below 0.35.
MIN_OVERLAP = 0.5
TEXT_FIELDS = ("transcript", "agent_before")
NONE = "none"
# Jev's answer must arrive within this, or the turn falls back to word overlap.
JEV_TIMEOUT_S = 2.0


def words(text: str) -> set[str]:
    """Content words, cut to five letters so "graduate" meets "graduation".

    ponytail: prefix stemming and plain overlap, no model call; a question asked in unusual
    words ticks late or not at all, so the UI shows a guide, not a record. Embeddings if
    that matters.
    """
    found = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:5] for w in found if len(w) > 2 and w not in STOPWORDS}


def asked_question(questions: dict[str, set[str]], text: str) -> str | None:
    """The requirement whose question this agent turn asks, or None."""
    said = words(text)
    best, best_score = None, 0.0
    for rid, question in questions.items():
        score = len(question & said) / len(question) if question else 0.0
        if score > best_score:
            best, best_score = rid, score
    return best if best_score >= MIN_OVERLAP else None


def progress_request(requirements: list[Requirement], text: str) -> tuple[str, dict]:
    """Jev's state and its one choice question: which requirement this agent turn asks."""
    criteria = {r.id: f"{r.label}: {r.question}" for r in requirements}
    criteria[NONE] = (
        "None of these: a greeting, consent or recording disclosure, acknowledgement, "
        "answer to the candidate's question, callback arrangement, or closing"
    )
    q = {
        "type": "choice",
        "instructions": "Which of the job's screening questions does the agent's turn ask "
        "the candidate? A reworded question, or confirming a résumé fact for it, counts.",
        "criteria": criteria,
    }
    return f"Agent turn: {text}", {"q": q}


def progress_choice(answers: dict) -> str | None:
    choice = answers["q"]["choice"]
    return None if choice == NONE else choice


class Progress:
    """Which approved questions the agent has asked so far, in order."""

    def __init__(self, requirements: list[Requirement], decider: Jev | None = None) -> None:
        self._requirements = requirements
        self._questions = {r.id: words(r.question) for r in requirements}
        self._decider = decider
        self.asked: list[str] = []

    def agent_said(self, text: str) -> dict:
        return self._record(asked_question(self._questions, text))

    async def agent_said_async(self, text: str) -> dict:
        """Jev's answer, or word overlap when there's no Jev or it fails or is slow."""
        if self._decider is not None:
            try:
                answers = await self._decider.ask_async(
                    *progress_request(self._requirements, text), timeout_s=JEV_TIMEOUT_S
                )
                asking = progress_choice(answers)
                if asking is None or asking in self._questions:
                    return self._record(asking)
                logger.warning("Jev chose an unknown question %r", asking)
            except Exception as e:
                logger.warning("Jev progress failed, using word overlap: %r", e)
        return self.agent_said(text)

    def _record(self, asking: str | None) -> dict:
        if asking is not None and asking not in self.asked:
            self.asked.append(asking)
        return {"asking": asking, "asked": list(self.asked)}


class RoomPublisher:
    """Sends JSON on a data topic to everyone in the room, in the background."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._pending: set[asyncio.Task] = set()  # keeps each send alive until it's done

    def publish(self, topic: str, data: dict) -> None:
        try:
            task = asyncio.get_running_loop().create_task(self._send(topic, json.dumps(data)))
        except Exception:
            logger.exception("couldn't publish on %s", topic)
            return
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _send(self, topic: str, payload: str) -> None:
        try:
            await self._room.local_participant.publish_data(payload, reliable=True, topic=topic)
        except Exception:
            logger.exception("couldn't publish on %s", topic)


def turn_record(record: dict) -> dict:
    """A decision-log record (decision_log.Point) without its words."""
    return {k: v for k, v in record.items() if k not in TEXT_FIELDS}


def publish_progress(
    session: AgentSession,
    publisher: RoomPublisher,
    requirements: list[Requirement],
    decider: Jev | None = None,
) -> None:
    """After each agent turn, publish which approved questions have been asked. With Jev,
    each turn is asked about in the background, one turn at a time, in order."""
    progress = Progress(requirements, decider)
    lock = asyncio.Lock()
    pending: set[asyncio.Task] = set()

    async def ask(text: str) -> None:
        try:
            async with lock:
                publisher.publish(PROGRESS_TOPIC, await progress.agent_said_async(text))
        except Exception:
            logger.exception("couldn't work out the question progress")

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        if getattr(ev.item, "role", None) != "assistant":
            return
        text = ev.item.text_content or ""
        if decider is None:
            publisher.publish(PROGRESS_TOPIC, progress.agent_said(text))
            return
        try:
            task = asyncio.get_running_loop().create_task(ask(text))
        except Exception:
            logger.exception("couldn't work out the question progress")
            return
        pending.add(task)
        task.add_done_callback(pending.discard)
