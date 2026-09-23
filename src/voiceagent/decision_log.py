"""Records every turn-taking decision the agent makes during a call.

A *pause* point opens when the caller stops talking and resolves as ``respond``
(the agent commits the turn first) or ``wait`` (the caller speaks again first).
An *overlap* point opens when the caller talks over the agent and resolves as
``interrupt`` or ``ignore`` when that agent speech ends. Likely mistakes are
flagged in ``suspect`` for the reviewer; nothing here decides correctness.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from livekit.agents import AgentSession

from voiceagent.backchannel import is_backchannel, words

LOG_DIR = Path("logs/decisions")
CUT_OFF_WINDOW_S = 1.5  # caller speaking again this soon after a commit means they weren't done
SLOW_RESPONSE_MS = 1200  # turn commit slower than this after a pause feels like dead air


@dataclass
class Point:
    i: int
    kind: str  # "pause" | "overlap"
    t: float
    transcript: str = ""
    agent_before: str = ""
    outcome: str = "unresolved"
    eot_delay_ms: int | None = None
    reply_delay_ms: int | None = None
    paused_then_resumed: bool = False
    suspect: str | None = None
    # bookkeeping, not written
    _abs_t: float = field(default=0.0, repr=False)
    _resolved_at: float | None = field(default=None, repr=False)
    _written: bool = field(default=False, repr=False)
    _line_final: bool = field(default=False, repr=False)

    def record(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


def suspect_for(p: Point) -> str | None:
    if p.kind == "pause" and p.outcome == "respond":
        if p.suspect == "cut_off":
            return "cut_off"
        if p.eot_delay_ms is not None and p.eot_delay_ms > SLOW_RESPONSE_MS:
            return "slow_response"
    if p.kind == "overlap":
        n = len(words(p.transcript))
        if p.outcome == "interrupt" and n <= 2 and is_backchannel(p.transcript):
            return "false_stop"
        if p.outcome == "ignore" and n >= 3 and not is_backchannel(p.transcript):
            return "missed_barge_in"
    return None


class DecisionTracker:
    def __init__(self, out: TextIO, t0: float) -> None:
        self._out = out
        self._t0 = t0
        self._points: list[Point] = []
        self._pause: Point | None = None  # open pause
        self._committed: Point | None = None  # latest respond, for cut-off / reply delay
        self._overlap: Point | None = None
        self._agent_speaking = False
        self._agent_line = ""
        self._finals: list[str] = []
        self._interim = ""

    @classmethod
    def open(cls, room: str, log_dir: Path = LOG_DIR) -> "DecisionTracker":
        log_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        stamp = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"{stamp}-{room}.jsonl"
        return cls(path.open("a", buffering=1), t0=now)

    # -- caller ---------------------------------------------------------------

    def user_started(self, t: float) -> None:
        self._flush(t)
        if self._pause is not None:
            self._resolve(self._pause, "wait", t)
            self._pause = None
        c = self._committed
        if c is not None and c._resolved_at is not None and t - c._resolved_at <= CUT_OFF_WINDOW_S:
            c.suspect = "cut_off"
        if self._agent_speaking and self._overlap is None:
            self._overlap = self._new("overlap", t)
            self._finals, self._interim = [], ""

    def user_stopped(self, t: float) -> Point | None:
        self._flush(t)
        if self._agent_speaking or self._overlap is not None:
            return None
        self._pause = self._new("pause", t)
        self._pause.transcript = self._text()
        return self._pause

    def transcript(self, text: str, is_final: bool, t: float) -> None:
        self._flush(t)
        if is_final:
            self._finals.append(text)
            self._interim = ""
        else:
            self._interim = text
        target = self._overlap or self._pause
        if target is not None:
            target.transcript = self._text()

    # -- agent ----------------------------------------------------------------

    def agent_state(self, state: str, t: float) -> None:
        self._flush(t)
        if state == "thinking" and self._pause is not None:
            p = self._pause
            p.eot_delay_ms = int((t - p._abs_t) * 1000)
            self._resolve(p, "respond", t)
            self._committed, self._pause = p, None
            self._finals, self._interim = [], ""
        elif state == "speaking":
            self._agent_speaking = True
            c = self._committed
            if c is not None and c.reply_delay_ms is None and not c._written:
                c.reply_delay_ms = int((t - c._abs_t) * 1000)
        else:
            self._agent_speaking = False

    def agent_speech_done(self, interrupted: bool, t: float) -> None:
        self._flush(t)
        if self._overlap is not None:
            self._resolve(self._overlap, "interrupt" if interrupted else "ignore", t)
            self._overlap = None
            self._finals, self._interim = [], ""

    def false_interruption(self, resumed: bool, t: float) -> None:
        if self._overlap is not None and resumed:
            self._overlap.paused_then_resumed = True

    def agent_said(self, text: str) -> None:
        self._agent_line = text
        # The spoken line is only added once its speech ends, i.e. after the overlap resolved.
        for p in self._points:
            if p.kind == "overlap" and not p._written and not p._line_final:
                p.agent_before = text
                p._line_final = True

    def close(self, t: float) -> None:
        for p in (self._pause, self._overlap):
            if p is not None and p._resolved_at is None:
                p._resolved_at = t
        self._flush(float("inf"))
        self._out.close()

    # -- internals ------------------------------------------------------------

    def _new(self, kind: str, t: float) -> Point:
        p = Point(
            i=len(self._points),
            kind=kind,
            t=round(t - self._t0, 2),
            agent_before=self._agent_line,
            _abs_t=t,
        )
        self._points.append(p)
        return p

    def _resolve(self, p: Point, outcome: str, t: float) -> None:
        p.outcome = outcome
        p._resolved_at = t

    def _text(self) -> str:
        return " ".join([*self._finals, self._interim]).strip()

    def _flush(self, now: float) -> None:
        for p in self._points:
            if p._written or p._resolved_at is None or now < p._resolved_at + CUT_OFF_WINDOW_S:
                continue
            p.suspect = suspect_for(p)
            self._out.write(json.dumps(p.record()) + "\n")
            p._written = True


def watch(session: AgentSession, tracker: DecisionTracker) -> None:
    """Feed session events into the tracker."""

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        if ev.new_state == "speaking":
            tracker.user_started(ev.created_at)
        elif ev.old_state == "speaking":
            tracker.user_stopped(ev.created_at)

    @session.on("user_input_transcribed")
    def _transcribed(ev) -> None:
        tracker.transcript(ev.transcript, ev.is_final, ev.created_at)

    @session.on("agent_state_changed")
    def _agent_state(ev) -> None:
        tracker.agent_state(ev.new_state, ev.created_at)

    @session.on("speech_created")
    def _speech(ev) -> None:
        ev.speech_handle.add_done_callback(
            lambda h: tracker.agent_speech_done(h.interrupted, time.time())
        )

    @session.on("agent_false_interruption")
    def _false_interruption(ev) -> None:
        tracker.false_interruption(ev.resumed, ev.created_at)

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        if getattr(ev.item, "role", None) == "assistant":
            tracker.agent_said(ev.item.text_content or "")
