import io
import json

from voiceagent.decision_log import DecisionTracker


class Sink(io.StringIO):
    """Keeps its contents readable after the tracker closes it."""

    was_closed = False

    def close(self) -> None:
        self.was_closed = True


def tracker() -> tuple[DecisionTracker, Sink]:
    out = Sink()
    return DecisionTracker(out, t0=100.0), out


def records(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_pause_then_caller_resumes_is_wait():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.transcript("my laptop keeps", True, 100.5)
    tr.user_stopped(101.0)
    tr.user_started(101.6)
    tr.close(110.0)
    [rec] = records(out)
    assert rec["kind"] == "pause" and rec["outcome"] == "wait"
    assert rec["t"] == 1.0 and rec["transcript"] == "my laptop keeps"
    assert rec["suspect"] is None


def test_respond_records_delays_and_flags_slow_response():
    tr, out = tracker()
    tr.agent_said("What seems to be the problem?")
    tr.user_started(100.0)
    tr.transcript("it won't boot", True, 100.8)
    tr.user_stopped(101.0)
    tr.agent_state("thinking", 102.5)
    tr.agent_state("speaking", 103.0)
    tr.close(110.0)
    [rec] = records(out)
    assert rec["outcome"] == "respond"
    assert rec["eot_delay_ms"] == 1500 and rec["reply_delay_ms"] == 2000
    assert rec["agent_before"] == "What seems to be the problem?"
    assert rec["suspect"] == "slow_response"


def test_caller_resuming_right_after_commit_is_cut_off():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.user_stopped(101.0)
    tr.agent_state("thinking", 101.4)
    tr.user_started(102.0)  # 0.6 s after commit
    tr.close(110.0)
    assert records(out)[0]["suspect"] == "cut_off"


def test_resume_after_window_is_not_cut_off_and_point_is_flushed():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.user_stopped(101.0)
    tr.agent_state("thinking", 101.4)
    tr.agent_state("speaking", 101.9)
    tr.agent_state("listening", 105.0)
    assert records(out)[0]["suspect"] is None  # flushed before close


def test_overlap_backchannel_interrupt_is_false_stop():
    tr, out = tracker()
    tr.agent_state("speaking", 100.0)
    tr.user_started(101.0)
    tr.transcript("yeah", True, 101.3)
    tr.agent_speech_done(True, 101.5)
    tr.close(110.0)
    [rec] = records(out)
    assert rec["kind"] == "overlap" and rec["outcome"] == "interrupt"
    assert rec["suspect"] == "false_stop"


def test_overlap_real_barge_in_ignored_is_missed():
    tr, out = tracker()
    tr.agent_state("speaking", 100.0)
    tr.user_started(101.0)
    tr.transcript("no wait that's not it", True, 101.8)
    tr.user_stopped(102.0)  # no pause point while the agent is talking
    tr.agent_speech_done(False, 104.0)
    tr.close(110.0)
    [rec] = records(out)
    assert rec["outcome"] == "ignore" and rec["suspect"] == "missed_barge_in"


def test_false_interruption_resumed_is_ignore_with_flag():
    tr, out = tracker()
    tr.agent_state("speaking", 100.0)
    tr.user_started(101.0)
    tr.transcript("mm-hmm", True, 101.2)
    tr.false_interruption(True, 102.0)
    tr.agent_speech_done(False, 104.0)
    tr.close(110.0)
    [rec] = records(out)
    assert rec["outcome"] == "ignore" and rec["paused_then_resumed"] is True
    assert rec["suspect"] is None


def test_pause_without_transcript():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.user_stopped(100.4)
    tr.user_started(100.9)
    tr.close(110.0)
    assert records(out)[0]["transcript"] == ""


def test_close_marks_unresolved():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.user_stopped(101.0)
    tr.close(101.2)
    assert records(out)[0]["outcome"] == "unresolved"
    assert out.was_closed


def test_interim_transcripts_replace_finals_accumulate():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.transcript("my", False, 100.1)
    tr.transcript("my laptop", True, 100.3)
    tr.transcript("keeps", False, 100.5)
    tr.transcript("keeps crashing", False, 100.6)
    tr.user_stopped(101.0)
    tr.user_started(101.5)
    tr.close(110.0)
    assert records(out)[0]["transcript"] == "my laptop keeps crashing"


def test_open_writes_named_file(tmp_path):
    tr = DecisionTracker.open("call-_+91", log_dir=tmp_path)
    tr.close(tr._t0)
    [path] = tmp_path.iterdir()
    assert path.name.endswith("-call-_+91.jsonl")


def test_overlap_shows_the_line_being_interrupted():
    tr, out = tracker()
    tr.agent_said("What seems to be the problem?")
    tr.agent_state("speaking", 100.0)
    tr.user_started(101.0)
    tr.transcript("no wait", True, 101.2)
    tr.agent_speech_done(True, 101.5)
    tr.agent_said("Okay, so your laptop won't")  # interrupted line lands after speech ends
    tr.close(110.0)
    assert records(out)[0]["agent_before"] == "Okay, so your laptop won't"


def test_record_fields():
    tr, out = tracker()
    tr.user_started(100.0)
    tr.user_stopped(101.0)
    tr.user_started(101.5)
    tr.close(110.0)
    assert set(records(out)[0]) == {
        "i",
        "kind",
        "t",
        "transcript",
        "agent_before",
        "outcome",
        "eot_delay_ms",
        "reply_delay_ms",
        "paused_then_resumed",
        "suspect",
    }
