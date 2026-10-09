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

Scale: one caller, ten test calls. These were measured with Gemini 2.5 Flash; the agent
now runs MiniMax-M3 on Together AI (see [Model choice](#model-choice)), and the call
numbers will be re-measured.

### Screening evaluation

`uv run screening-eval` on 2026-10-09: 6 reference jobs, 4 simulated candidates each plus
4 fairness replays, real models on Together AI, $0.37 in total. One persona was refused
before its call (the generated résumé didn't show the number it was meant to test).

| Stage | Result |
|---|---|
| Criteria generation | recall 0.84, precision 0.68, hard flags 0.83, rules 0.69, illegal lines dropped 0.33 |
| Answer capture | 1.00 of answers extracted correctly; 0.05 questions missed and 0 re-asked per call |
| Decisions | clear fits 6/6 suitable, must-have misses 5/5 not suitable, false rejections 0; accuracy 0.67, sent to review 0.29 |
| Conversation | disclosed AI and recording 0.96, asked all required questions 0.91, never hinted a decision or promised anything 1.00 |
| Messages | professional, no promises, rejections without reasons: all 1.00 (15 messages) |
| Reviewer | caught 3 of 7 wrong invites, 1 needless escalation; no wrong rejection reached it |
| Calibration | borderline candidates matched the recruiter's decision 0.83 → 1.00 with overrides; clear fits 1.00 → 0.83 (one went to review) |
| Fairness | fail: same words, different scores (see below) |
| Cost | $0.013 per call |

What the misses are:

- **Borderline candidates.** They are told to give middling answers; the scorer often
  rates those answers above the pass mark, so 4 of 6 came out suitable where the label
  says a recruiter should decide. Their recruiter overrides call them suitable too.
- **Fairness.** The four name-swapped replays gave the same answer to "tell me about an
  upset customer" word for word; it scored 0.5 twice and 1.0 twice at temperature 0, with
  no pattern by name across runs (the previous run split 3 to 1 differently). That is
  scorer noise, and it decides between suitable and review on this borderline persona.
  A fairness check on one persona can't tell noise from bias; scoring each soft answer
  more than once would.

Getting here took six runs, and most fixes were in the agents, not the harness: the voice
model (Qwen3.5-9B) was ending calls before the questions were done; the reviewer's yes/no
contradicted its own notes and it never saw the job's own wording; "a 60-day notice" and
"no experience" weren't read as numbers.

Scale: one run, 27 simulated calls (23 personas and 4 fairness replays); these are
measurements, not a benchmark.

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
  backup model from a different model family.
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
| LLM | MiniMax-M3 on Together AI with reasoning off (0.77 s median to first token); GLM-5.3 answers if a reply comes back empty or fails |
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
| [`call/`](src/voiceagent/call) | Screening calls: prompt from criteria, `end_call`, saving the result, text-mode harness |
| [`agents/assessment.py`](src/voiceagent/agents/assessment.py) | After the call: evidence, rules, soft scores, reviewer → decision |
| [`agents/messages.py`](src/voiceagent/agents/messages.py) | Invite / rejection writer with a checker |
| [`rules.py`](src/voiceagent/rules.py) | Deterministic hard-requirement rules and decision thresholds |
| [`outcomes/email.py`](src/voiceagent/outcomes/email.py) | SMTP mailer (development: log only) |
| [`evals/`](src/voiceagent/evals) | `screening-eval`: reference jobs, simulated candidates, metrics, run report |

## Screening backend (in progress)

The agent is being turned into a backend for AI phone screening: a recruiter pastes a job
description, and a criteria agent (LangGraph) drafts what the call should check, grounds
every requirement in the description's own words, has a critic check it for legality and
phone-checkability, and revises until it passes. Approved criteria are indexed in pgvector
and ground future jobs.

Candidates are then called: the agent builds its prompt from the approved criteria, says
up front that it is an AI assistant and that the call is recorded, asks the questions in
order, answers only from the job's facts, closes neutrally, and ends the call with its one
tool, `end_call`. The transcript and end reason are saved on the call. A call that arrives
without a screening call id (someone phoning in, `voiceagent console`) is still the
Nimbus Labs demo.

After the call an assessment agent (LangGraph) extracts each answer from the transcript and
checks that its quote is the candidate's own words, applies the hard requirements' rules in
code, scores the rest against their rubrics (calibrated by past recruiter overrides), and
has a second, stronger model challenge every rejection, every verdict resting on a weak
answer and every invite that would go out unattended. A soft requirement with no verified
answer sends the candidate to a recruiter instead of counting for or against them. A message
agent drafts the invite or a reason-free rejection; code checks it for leaked reasons,
promises, the name and next steps, and a model for tone and invented facts. Invites to
suitable candidates go out automatically (per job); every rejection and `needs_review`
waits for a recruiter to approve or override.

```bash
docker compose up -d postgres
uv run alembic upgrade head
uv run screening-api      # REST API on :8000, needs SCREENING_API_KEY
uv run screening-worker   # runs the agents
uv run voiceagent dev        # the call agent; reads screening calls from DATABASE_URL
```

| Endpoint | What |
|---|---|
| `POST /jobs` | create a job from a description; criteria generate in the background |
| `GET /jobs/{id}/criteria` | the generated criteria, with warnings and validation errors |
| `PATCH /jobs/{id}/criteria` | edit (an approved version becomes a new draft) |
| `POST /jobs/{id}/criteria:regenerate` · `:approve` | regenerate, or approve for calling |
| `POST /jobs/{id}/candidates` · `:import` | add a candidate, or upload a CSV (`Content-Type: text/csv`; columns name, email, phone or sip_address, resume_text) |
| `GET /jobs/{id}/candidates` · `GET / DELETE /candidates/{id}` | list, read, delete (deletes their calls and transcripts) |
| `POST /candidates/{id}/call` | ring the candidate now (job active, criteria approved) |
| `GET /calls/{id}` · `GET /candidates/{id}/calls` | status, end reason, transcript |
| `GET /decisions?job_id=&status=` · `GET /decisions/{id}` | the review queue; a decision with its answers, reasons, reviewer notes and drafted message |
| `POST /decisions/{id}:approve` | send the drafted message (also retries a failed send); `{"force": true}` if the candidate was already notified or has a newer decision |
| `POST /decisions/{id}:override` | `{"result", "note", "requirement_ids"?}`: the recruiter's call; re-drafts the message and calibrates future scoring on the named answers. The note is shown to the scorer for other candidates: keep names and personal details out of it |
| `GET / PATCH /messages/{id}` | read, or edit the subject/body before sending |
| `GET /events?after=` | event feed |

Phone numbers need an outbound trunk in `SIP_OUTBOUND_TRUNK_ID`; SIP addresses don't.

Email goes out through `SMTP_HOST` (see `.env.example`); without it, messages are only
recorded in the database and the log.

The backend's agents run on DeepSeek-V4-Flash (backup MiniMax-M3) via Together AI's
JSON-schema mode, and the assessment's reviewer on DeepSeek-V4-Pro; embeddings come from `BAAI/bge-base-en-v1.5`, run locally.

## Evaluation

`screening-eval` measures every stage offline, on six reference jobs in `evals/jobs/`
(software engineer, sales executive, warehouse supervisor, staff nurse, customer support,
data analyst), each with hand-labelled criteria:

- **Criteria generation** — the criteria agent on each job description, matched against
  the labels by an LLM judge: recall, precision, hard-flag accuracy, rule correctness
  ("1-3 years" must become `>= 1`), illegal lines dropped, critic rounds, requirements
  removed as ungrounded.
- **Calls** — simulated candidates talk to the real agent in text mode (LiveKit
  `AgentSession.run()`): a clear fit, one candidate per must-have they miss, a borderline
  one, and candidates who decline the salary question, ramble, ask questions, ask for a
  callback, refuse the recording, or overstate their résumé. The real assessment and
  message agents then run on each transcript. Measured: answer capture, decision accuracy,
  false-rejection rate (reported on its own: it hurts candidates most), `needs_review`
  rate, a conversation checklist judge, a message judge, the reviewer's catches and
  needless escalations, agreement with the recruiter's decision before and after scoring
  with their overrides, cost and turns per call.
- **Fairness** — one persona replayed under four names (men's and women's names from
  different communities), with the recruiter overrides on, must get the same decision,
  with scores within 0.05. A replay that failed or ended early makes it incomplete.

Expected decisions come from the rule engine on each persona's ground truth, which is set
in code from the labelled rules, so the suite never grades the system against itself.

```bash
uv run screening-eval                        # all jobs, 4 personas each, stops at $2
uv run screening-eval --jobs staff-nurse --personas 9 --runs 3
uv run screening-eval --offline              # fakes only: a free smoke test
```

Each run writes `evals/runs/<UTC timestamp>/results.json` and `report.md`, with a
comparison against the previous run of the same jobs, personas, runs and models. Text mode doesn't see transcription errors or
interruptions; spoken runs over SIP come later.

## Model choice

All models are on [Together AI](https://www.together.ai). Measured from India on
9 Oct 2026; only models available serverless on the account were compared.

**Voice replies.** Streamed on the agent's real prompt and a 6-turn screening
conversation, reasoning off, 5 runs per model. "First token" is when the first spoken word
arrives.

| Model | First token, median | First token, worst | Empty replies |
|---|---|---|---|
| Qwen/Qwen3.5-9B | **0.64 s** | 0.74 s | 0/5 |
| **zai-org/GLM-5.3** (backup) | 0.67 s | 0.99 s | 0/5 |
| deepseek-ai/DeepSeek-V4.1-Flash | 0.68 s | 3.43 s | 0/5 |
| **MiniMaxAI/MiniMax-M3** (chosen) | 0.77 s | 1.09 s | 0/5 |
| meta-models/Muse-Glimmer-30B | 0.79 s | 1.32 s | 0/5 |
| zai-org/GLM-5.3-Flash | 0.83 s | 1.35 s | 0/5 |
| deepseek-ai/DeepSeek-V4-Flash-0731 | 0.85 s | 1.42 s | 0/5 |
| meta-llama/Llama-3.3-70B-Instruct-Turbo | 1.47 s | 2.50 s | 0/5 |
| openai/gpt-oss-120b | 1.51 s | 1.61 s | 1/5 |

Speed wasn't enough. On a scripted 4-answer call (10 runs per model), Qwen3.5-9B ended
7 calls with `end_call` before the questions were done, twice as "declined consent" right
after the candidate said yes. MiniMax-M3, GLM-5.3 and DeepSeek-V4.1-Flash ended none
early. MiniMax-M3 is the voice model: GLM-5.3 is 0.1 s faster but about 5x the price per
token ($1.40/$4.40 per million vs $0.30/$1.20), so it is the backup.

**Screening criteria.** The full criteria agent (draft, check every requirement against
the job description, critique, revise up to 3 times), reasoning on, run once on two job
descriptions: an inside-sales role containing an illegal "female candidates, age under 35"
line, and a senior backend role.

| Model | Sales role | Backend role | Requirements grounded | Illegal criteria | Read "1-3 years" as at least 1 |
|---|---|---|---|---|---|
| **deepseek-ai/DeepSeek-V4-Flash-0731** (chosen) | 72 s | **31 s** | 15/15 | none | no |
| deepseek-ai/DeepSeek-V4-Pro-0813 | 57 s | 34 s | 15/15 | none | **yes** |
| **MiniMaxAI/MiniMax-M3** (backup) | 74 s | 73 s | 14/14 | none | no |
| zai-org/GLM-5.3-Flash | 76 s | 149 s | 14/14 | none | no |
| deepseek-ai/DeepSeek-V4.1-Flash | 121 s | 118 s | 16/16 | none | no |
| moonshotai/Kimi-K3 | 158 s | 49 s | 15/15 | none | no |
| zai-org/GLM-5.3 | 149 s | 153 s | 16/16 | none | no |
| thinkingmachines/Inkling | 83 s | 416 s | 13/13 | none | no |
| openai/gpt-oss-120b | 266 s | 245 s | 15/15 | none | no |
| Qwen/Qwen3.8-2.4T-A95B | 155 s | — | 8/8 | none | no |

Every model dropped the illegal line and got the backend role's "4+ years" right. Only
DeepSeek-V4-Pro read "1-3 years" as a minimum; the prompts now say so explicitly. Scale:
one run per job per model, so these are measurements, not a ranking.

## Built with

The [LiveKit Agents](https://github.com/livekit/agents) SDK, with Deepgram, Together AI
and Cartesia as providers. Turn-taking starts from LiveKit's recommended setup. The
short-answer turn rule, the LLM fallback and history cleanup, the decision log and the
outbound dialer are original to this repo.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT, see [LICENSE](LICENSE).
