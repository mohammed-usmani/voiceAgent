# voiceagent

A LiveKit phone voice agent that runs job pre-screening calls, and logs every
turn-taking decision it makes, so you can see exactly where it cuts callers off,
leaves dead air, or mishandles interruptions.

Aria, a recruiter at a fictional company, calls a shortlisted applicant (or takes their
call), confirms their current role, experience, notice period, salary and work setup, and
books a slot for their final interview.

## Results

Reply time is measured from the moment the caller stops speaking to the agent's first
audio.

**At the agent** (from its decision log):

| | Median reply | Slowest reply | Replies |
|---|---|---|---|
| Starting point: LiveKit defaults, Gemini 2.5 Flash | 2.44 s | 3.42 s | 38 |
| Now | 1.81 s | 2.58 s | 63 (10 calls) |

**On the phone:** about 2.8 s median on a recorded call. The caller reaches the agent
over SIP from a free softphone on mobile internet, and that path adds roughly 1–1.5 s
that the agent never sees. It's the next bottleneck.

What moved it:

- **The LLM stopped thinking before speaking.** Gemini 2.5 Flash thinks by default:
  ~1.1 s to the first token; ~0.6–0.9 s with thinking off.
- **Endpointing tuned.** A 1.3 s maximum wait was fast but cut callers off while they
  paused to think; dynamic endpointing with a 2.0 s cap didn't.
- **A bare yes or no to a question ends the turn.** On phone audio, the audio turn
  detector often hears a flat "No." as unfinished.
- **The greeting is spoken without the LLM.**

A faster model (Qwen3.8-27B on Groq, ~0.16 s to first token) was tried and dropped: on
real conversations its replies came back empty or cut off mid-sentence, with no error.

Scale: one caller, ten test calls.

## Turn-taking

LiveKit's recommended turn-taking setup
([Tuning turn-taking](https://docs.livekit.io/agents/logic/turns/tuning/)), with
endpointing tuned and one rule added:

| Piece | Setting |
|---|---|
| End of turn | `inference.TurnDetector()`: LiveKit's audio turn detector, default version, plus a rule: a bare yes/no right after a question ends the turn ([`short_answers.py`](src/voiceagent/short_answers.py)) |
| Interruptions | adaptive (backchannel-aware), `min_duration=0.5`, `min_words=0` |
| Endpointing | Dynamic (learns each caller's mid-sentence pauses), 2.0 s maximum (2.5 s stalled one-word answers, 1.3 s cut off long answers) |
| VAD | Silero, default settings |
| Noise cancellation | `BVCTelephony`, since calls arrive over SIP |
| Preemptive generation | LiveKit default (on): the reply is drafted while the turn is being decided |

A test pins this configuration so it can't drift silently.

## Decision log

Each call writes `logs/decisions/<time>-<room>.jsonl`, one line per decision:

- **pause**: the caller stopped talking. Outcome `respond` (the agent took the
  turn, with the delay in ms) or `wait` (the caller carried on).
- **overlap**: the caller spoke while the agent was talking. Outcome `interrupt`
  or `ignore`.

Likely mistakes are pre-flagged in `suspect`:

| Flag | Meaning |
|---|---|
| `cut_off` | the caller started talking again within 1.5 s of the agent taking the turn |
| `slow_response` | the agent took over 1.2 s to take the turn |
| `false_stop` | the agent stopped for a backchannel like "yeah" or "uh-huh" |
| `missed_barge_in` | the caller said 3+ real words and the agent kept talking |

```json
{"i": 5, "kind": "pause", "t": 32.83, "transcript": "No.", "outcome": "respond",
 "eot_delay_ms": 2503, "suspect": "slow_response", ...}
```

## Reliability

A voice agent that gets no reply text says nothing, and the caller hears dead air.
[`replies.py`](src/voiceagent/replies.py) guards against that:

- If the main model's reply comes back empty or fails, the same turn is answered by a
  backup model on a different provider.
- The history sent to the model is cleaned first: a long answer split into several
  caller messages is joined into one, agent replies cut off after a few words are
  dropped, and provider-specific metadata is removed.
- Every reply stage (created, first token, first audio, state changes, errors) is
  logged, so a stall shows where it stopped.

## Stack

| Layer | Choice |
|---|---|
| Transport / orchestration | LiveKit Agents, with calls over SIP |
| STT | Deepgram Nova-3, Indian English (`en-IN`) |
| LLM | Gemini 2.5 Flash with thinking off (~0.5 s to first token, ~1.1 s with thinking on); gpt-oss-20b on Groq answers if a reply comes back empty or fails |
| TTS | Cartesia Sonic-3.6, voice Jacqueline, English |

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env    # fill in keys
uv sync
uv run voiceagent dev
```

To have the agent call you instead, with the agent running:

```bash
uv run voiceagent-call you@sip.linphone.org   # or set CALL_TO in .env
```

The phone rings showing "Nimbus Labs Hiring", and the agent joins once you answer. A free
[linphone.org](https://www.linphone.org) account works as the callee, so no phone number
has to be bought. The LiveKit outbound trunk is created on first use, over TCP: the
Linphone server didn't answer over UDP.

## Layout

| File | What it does |
|---|---|
| [`agent.py`](src/voiceagent/agent.py) | Prompt, speech models, turn-taking settings, call entrypoint |
| [`replies.py`](src/voiceagent/replies.py) | LLMs, history cleanup, fallback to the backup model, reply-stage logging |
| [`short_answers.py`](src/voiceagent/short_answers.py) | Turn detector wrapper that ends the turn on a bare yes/no |
| [`decision_log.py`](src/voiceagent/decision_log.py) | Per-call log of every pause and overlap, with suspected mistakes flagged |
| [`backchannel.py`](src/voiceagent/backchannel.py) | Which utterances are backchannels ("yeah", "uh-huh") |
| [`outbound.py`](src/voiceagent/outbound.py) | `voiceagent-call`: have the agent call a SIP address |

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
