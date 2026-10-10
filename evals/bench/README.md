# Decision benchmark: Jev vs LLMs on three screening decisions

**Status: pre-registration.** This section was written and committed before any contender
was run. The rules below are not changed after results are in; anything decided after
that is listed under *Amendments*, with the date and the reason.

## Purpose

The screening backend makes three small, frequent decisions with LLMs or heuristics. This
benchmark asks whether TypeSafe's Jev decision model (`jev-latest`, early access, via
`POST https://api.typesafe.ai/v1/systemone`) is a better choice for each, on accuracy,
consistency, robustness, latency and cost:

| Task | What is decided | Production today | Contenders |
|---|---|---|---|
| 1. Rubric scoring | how well a soft answer meets its rubric (0-1) | `assessment.SCORER` + `SoftScores` on DeepSeek-V4-Flash-0731 (JSON-schema mode, reasoning on, MiniMax-M3 backup) | that scorer; Jev `score` with 3 levels |
| 2. Question progress | which approved question an agent turn asks | token-overlap matcher, `call/live.py` | that matcher; Jev `choice`; DeepSeek-V4.1-Flash (reasoning off, structured output) |
| 3. Reviewer checks | whether each assessment reason is supported and matches the résumé | `assessment.REVIEWER` + `Review`/`ReasonCheck` on DeepSeek-V4-Pro-0813 | that reviewer; Jev `noul` |

The benchmark only imports production code; it changes none of it.

## Pre-registered protocol

### Common

- Inputs come from the six evaluation runs of 9 Oct 2026 (`evals/runs/20261009T*/results.json`,
  168 simulated calls over the six reference jobs in `evals/jobs/`) and from answers
  generated for this benchmark. All candidates are synthetic personas; no real person's data
  is used. (The local dev database's 8 real calls are not used.)
- Gold labels come from **zai-org/GLM-5.3** (reasoning on), a different model family from
  every contender (DeepSeek, Jev). Generated answers come from DeepSeek-V4-Flash-0731.
- Seed `20261010` for every sample and shuffle. Bootstrap: 1,000 resamples, 95% percentile
  intervals. Paired differences are always Jev − the other contender, resampled on the
  same units.
- Cost: Together tokens priced by `voiceagent.evals.cost.PRICES` (Together list prices);
  Jev at the vendor's stated $0.042 per million input tokens, output free, from the `usage`
  it returns. Hard budget **5 USD for everything** (generation, labels, contenders), kept
  in `evals/bench/data/spend.jsonl`; the CLI stops cleanly when it is reached.
- Jev: at most 5 requests in flight; 429/5xx/network errors retried with exponential backoff
  (6 attempts); failures recorded, never retried silently into a different answer.
  Latency is wall time per decision including retries, measured with up to 5 (Jev) or 8
  (Together) requests in flight.

### Task 1: rubric scoring

**Dataset.**
- The 8 soft requirements that have a rubric (frustrated_customer, pandas, ab_testing,
  target_record, sql_depth, api_design, ventilator_care, wms_experience).
- Generated: per requirement × intended level (fully / partly / doesn't meet) × 3 candidate
  styles, one generation call of 8 answers → 576 answers (temperature 0.9).
- Real: every distinct answer quote the production scorer scored for those requirements in
  the six runs.
- GLM-5.3 labels every answer as `does_not_meet` / `partly_meets` / `fully_meets`, blind to
  the intended level and the source, in shuffled batches of up to 8 answers to the same
  requirement. **A generated answer is kept only if the label equals its intended level**;
  the number dropped is reported. A real answer gets the label as its gold. Gold scores:
  0 / 0.5 / 1.
- A random 40 of the real answers, with their labels, go to
  `evals/bench/spotcheck/scoring_real_40.md` for a manual check.
- Stress variants, from a stratified random 40 kept generated answers (14/13/13 by level):
  3 paraphrases each, a phone-ASR-noise version and a Hinglish version (written by
  DeepSeek-V4-Flash-0731, then labelled by GLM-5.3 and kept only if the label equals the
  base answer's level); 4 name swaps each (the base answer under the four fairness names
  Arjun Sharma, Priya Iyer, Imran Qureshi, Sarah D'Souza). Plus 40 injection items: a
  random 40 kept "doesn't meet" answers, each with one of 5 injection sentences appended
  (gold 0 by construction). Plus 24 "I'd rather not say" items (3 refusal phrasings × 8
  requirements; gold 0).

**Input, identical content for both contenders.** A 4-turn transcript: the agent greets the
candidate by first name, the candidate confirms with their full name, the agent asks the
requirement's question, the candidate gives the answer. The LLM gets the score step's own
user message (`Requirements to score` block for that one requirement with the answer as
extracted value and quote, confidence 1.0, then the fenced transcript) and the production
`SCORER` system prompt, through production's `FallbackModels` (main model retried, then the
backup). Production batches all answered soft requirements of a call into one scorer call;
here each call scores one requirement. Jev gets the same block (job, requirement, question,
the answer, the rubric) and the same transcript as its state, and one `score` question with
criteria `[doesn't meet, partly meets, fully meets]` and the rubric in its instructions.

**Score mapping.** Jev: expected level index / 2. LLM: its 0-1 `score`. Bands: < 0.34,
0.34-0.66, > 0.66; gold 0 / 0.5 / 1 fall in bands low / mid / high.

**Metrics** (each item scored **5 times** by each contender; per-item values are averaged
over its repeats; CIs resample items, or groups for spreads):
- MAE vs gold and band accuracy, on the base set (kept generated + labelled real), with CIs;
  also split generated / real and by gold level.
- Repeat consistency: mean per-item standard deviation over the 5 repeats (CI), and the share
  of items whose 5 scores all fall in one band.
- Paraphrase spread: per base answer, max − min of the mean scores of the base and its kept
  paraphrases; mean over groups (CI). Name-swap spread: the same over the 4 names (CI).
- ASR noise, Hinglish, "I'd rather not say": MAE and band accuracy (descriptive only).
- Confidence usefulness: band accuracy on the 80% most confident scorings vs all. The
  production schema has no confidence field, so for the LLM this is **n/a**.
- Injection inflation: share of scorings of injected weak answers above 0.66 (CI over items).
- Latency p50/p95 per scoring; USD per 1,000 scorings.

**End to end.** For every assessed call of the six runs with an expected decision, re-run
`run_assessment` with the evidence **stored in the run** (no new extraction), the scorer
swapped (LLM: the production scorer, called fresh; Jev: one score question per answered soft
requirement, in one Jev request whose state lists every answer and the transcript, with the
recruiter's overrides in the state where the run used them), and **the reviewer skipped**
for both. Report decision accuracy vs the persona's expected decision (fairness replays
excluded), false-rejection rate (not_suitable among calls not expected to be rejected),
needs-review rate, and whether each run's 4 fairness replays get one decision. The stored
production decisions (reviewer included) are shown for reference.

**Decision rule.** Jev is better for Task 1 if **all** of:
1. Jev's MAE − the LLM's MAE ≤ 0.05 (point estimates), and
2. Jev's name-swap spread is clearly lower: the paired 95% CI of (Jev − LLM) lies entirely
   below 0, and
3. Jev's repeat consistency is clearly better: the paired 95% CI of the per-item std
   difference lies entirely below 0, and
4. Jev is cheaper (USD per 1,000) or faster (p50).

Otherwise the LLM is better if its MAE is lower by more than 0.05 or the paired MAE CI lies
entirely above 0; else "no clear winner". (If both contenders are perfectly consistent, 3
fails: equal is not better.)

### Task 2: question progress

**Dataset.** Every agent turn of every call in the six runs (1,285). GLM-5.3 labels each
transcript at once: for every numbered agent turn, the list of requirement ids it asks
(reworded, confirming a résumé fact for it, or re-asking it after an unclear answer all
count; empty for greetings, disclosures, acknowledgements, answers to the candidate's
questions, follow-ups that ask for no requirement's answer, callbacks and closings). Ids
not in the job are dropped and counted. A random 50 labelled turns go to
`evals/bench/spotcheck/progress_50.md`.

**Contenders** see only the agent turn and the job's question list (the matcher sees the
questions' words). Jev: one `choice` over the job's requirement ids plus `none`. LLM:
DeepSeek-V4.1-Flash, reasoning off, temperature 0, JSON-schema output restricted to the
ids plus `none`.

**Metrics.** Accuracy (a prediction is right if it names one of the requirements the turn
asks, or `none` for a turn that asks none) with a bootstrap CI; macro-F1 over all labels
(for a turn asking two, a prediction of either counts as that label); false-tick rate on
`none` turns (any requirement predicted); p50/p95 latency per turn; cost.

**Decision rule.** The most accurate option (point estimate) whose p95 latency is under 1.0 s.

### Task 3: reviewer checks

**Dataset.** From the completed screening calls of the six runs, every reason whose
correctness the persona's ground truth decides:
- a hard requirement with verdict pass/fail: correct if the verdict equals the rule applied
  to the persona's true answer and the extracted value matches the truth (numbers within
  10%, at least 0.5; booleans and choices exactly); otherwise should flag. The "overstates"
  persona's overstated number is should-flag (the answer contradicts the résumé). A truth
  the rule can't decide is skipped; "unknown" verdicts are skipped.
- soft: a "declined" reason is correct iff the persona declined it; a scored boolean is
  correct iff (score ≥ 0.5) equals the truth. Other soft reasons are skipped (not
  decidable from ground truth).

Every call with a should-flag reason, plus a random sample up to 50 calls in total, are
reviewed whole. Plus 80 corrupted copies: 20 each of a correct hard reason with
(a) a wrong value and the verdict flipped to match it, (b) the quote replaced by the
candidate's first line of the call, (c) a unit mix-up (years ×12, days ×30, LPA ×12, verdict
recomputed), (d) a line appended to the résumé contradicting the answer. Each copy has one
corrupted reason (should flag); when its verdict changed, the copy's verdict and summary are
recomputed as the verdict step would.

**Contenders.** LLM: the production review step's exact user message and `REVIEWER` prompt
on `models.reviewer` (DeepSeek-V4-Pro-0813, MiniMax-M3 backup), one call per call instance;
a reason is flagged if its check says `supported: false` or `matches_resume: false` (the
production rule that ignores non-failed reasons on a rejection is not applied: every
reason is scored). A missing check counts as not flagged. Jev: per reason, two `noul`
questions (supported; résumé doesn't contradict) on a state with the requirement, the
reason, the transcript and the résumé; flagged if either is below 0.5.

**Metrics.** Catch rate on should-flag reasons and false-flag rate on correct ones, with
CIs and paired difference CIs; catches by category; latency (LLM per call, Jev per reason);
USD per 1,000 reason checks.

**Decision rule.** Jev is better if its catch rate ≥ the LLM's **and** its false-flag rate
≤ the LLM's (point estimates) **and** at least one of the two paired differences is
significant in Jev's favour (95% CI excludes 0). The LLM is better in the mirror case.
Otherwise "no clear winner".

### If the budget runs short

Reductions, in this order, decided before the main run and recorded under *Amendments*:
(1) Task 3 original calls 50 → 30; (2) Task 1 stress variants scored 3 times instead of 5
(the base set keeps 5); (3) Task 2 limited to the first three runs. Metrics and rules
don't change.

## Amendments

None yet.

## Reproduce

```sh
uv sync
# .env with TOGETHER_API_KEY and TYPESAFE_API_KEY
uv run decision-bench --task all --budget-usd 5        # or --task scoring|progress|review
uv run decision-bench --task scoring --limit 5          # smoke test
uv run decision-bench --task all --run-dir evals/bench/runs/<run>   # resume a run
```

The labelled datasets are committed in `evals/bench/data/` (`scoring_items.jsonl`,
`progress_turns.jsonl`, `review_items.jsonl`, with the raw generator and labeler outputs);
with them in place no generation or labelling call is made and a run only pays for the
contenders. Delete `evals/bench/data/spend.jsonl` to start a fresh budget. Raw outputs and a
`report.md` go to `evals/bench/runs/<timestamp>/` (not committed).
