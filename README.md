# voiceagent

A LiveKit phone voice agent that runs job pre-screening calls and logs every turn-taking decision it makes,
so you can see exactly where it cuts callers off, leaves dead air, or mishandles
interruptions.

## Turn-taking

The agent runs LiveKit's recommended turn-taking setup unchanged
([Tuning turn-taking](https://docs.livekit.io/agents/logic/turns/tuning/)):

| Piece | Setting |
|---|---|
| End of turn | `inference.TurnDetector()`: LiveKit's audio turn detector, default version |
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

## Stack

| Layer | Choice |
|---|---|
| Transport / orchestration | LiveKit Agents, with calls over SIP |
| STT | Deepgram Nova-3, Indian English (`en-IN`) |
| LLM | Qwen3.8-27B on Groq with thinking off (~0.1 s to first token, vs ~1.1 s for Gemini 2.5 Flash), replies in the caller's language |
| TTS | Cartesia Sonic-3.6, voice Jacqueline, English |

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env    # fill in keys
uv sync
uv run voiceagent dev
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```
