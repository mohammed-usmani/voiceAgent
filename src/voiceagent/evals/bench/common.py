"""Plumbing shared by the benchmark's three tasks: the six evaluation runs, model clients,
a thread pool that stops at the budget, append-only result files and the spend ledger."""

import contextvars
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from voiceagent.evals.cost import BudgetExceeded, Meter
from voiceagent.evals.jobs import RefJob, load_jobs

logger = logging.getLogger(__name__)
BENCH = Path("evals/bench")
DATA = BENCH / "data"
SPOT = BENCH / "spotcheck"
RUNS = BENCH / "runs"
LEDGER = DATA / "spend.jsonl"
EVAL_RUNS = Path("evals/runs")
EVAL_RUN_GLOB = "20261009T*/results.json"  # the six real runs of 9 Oct 2026
GENERATOR = "deepseek-ai/DeepSeek-V4-Flash-0731"
LABELER = "zai-org/GLM-5.3"
SEED = 20261010
LLM_WORKERS = 8
JEV_WORKERS = 5  # the vendor's early-access limit: never more in flight


@cache
def jobs() -> dict[str, RefJob]:
    return {j.slug: j for j in load_jobs()}


def eval_calls() -> list[tuple[str, int, dict]]:
    """(run name, index in the run, call record) for every call of the six runs."""
    out = []
    for path in sorted(EVAL_RUNS.glob(EVAL_RUN_GLOB)):
        for i, rec in enumerate(json.loads(path.read_text())["calls"]):
            out.append((path.parent.name, i, rec))
    return out


def chat(model: str, *, reasoning: bool = True, temperature: float = 0.0, max_tokens=16000):
    """A Together chat model with structured output in JSON-schema mode, as production's."""
    from langchain_openai import ChatOpenAI

    from voiceagent.llm import MODEL_TIMEOUT_S, TOGETHER_BASE_URL, _JsonSchemaChat
    from voiceagent.replies import NO_REASONING
    from voiceagent.settings import get_settings

    return _JsonSchemaChat(
        ChatOpenAI(
            model=model,
            api_key=get_settings().together_api_key,
            base_url=TOGETHER_BASE_URL,
            temperature=temperature,
            max_retries=0,
            timeout=MODEL_TIMEOUT_S * 2,
            max_tokens=max_tokens,
            extra_body=None if reasoning else NO_REASONING,
        )
    )


def one_model(model: str, **kw):
    """FallbackModels retried on the same model: a helper model (generator, labeler) never
    silently becomes a different one."""
    from voiceagent.llm import FallbackModels

    c = chat(model, **kw)
    return FallbackModels(c, c, None, name=model)


def production_models():
    """default_models() without the local embedder (no step here embeds anything)."""
    from voiceagent.llm import FallbackModels
    from voiceagent.settings import get_settings

    s = get_settings()
    backup = chat(s.backup_model)
    reviewer = FallbackModels(chat(s.reviewer_model), backup, None, name=s.reviewer_model)
    return FallbackModels(chat(s.main_model), backup, None, name=s.main_model, reviewer=reviewer)


class Jsonl:
    """Rows keyed by `key`, appended as they arrive, so an interrupted stage resumes."""

    def __init__(self, path: Path, key: str = "key") -> None:
        self.path, self.key = path, key
        self.rows: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row[key]] = row

    def __contains__(self, k: str) -> bool:
        return k in self.rows

    def add(self, row: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows[row[self.key]] = row

    def values(self) -> list[dict]:
        return list(self.rows.values())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _over(meter: Meter) -> bool:
    try:
        meter.check()
    except BudgetExceeded:
        return True
    except RuntimeError:  # usage dict resized by another thread mid-read: check next time
        return False
    return False


def pmap(fn: Callable, items: list, *, workers: int, meter: Meter, what: str) -> int:
    """fn(item) for every item on a thread pool, each in a copy of this context (so the
    meter's LangChain callback sees its calls). Errors are logged and counted; the budget
    stops everything not yet started and raises BudgetExceeded at the end."""
    stop = threading.Event()
    errors = []

    def run(item) -> None:
        if stop.is_set() or _over(meter):
            stop.set()
            return
        try:
            fn(item)
        except BudgetExceeded:
            stop.set()
        except Exception as e:
            logger.warning("%s failed: %r", what, e)
            errors.append(repr(e))

    with ThreadPoolExecutor(workers) as ex:
        for item in items:
            ex.submit(contextvars.copy_context().run, run, item)
    if errors:
        print(f"  {what}: {len(errors)} failed (first: {errors[0][:200]})")
    if stop.is_set():
        raise BudgetExceeded(f"stopped during {what}: ${meter.usd:.4f} spent")
    return len(errors)


def timed(fn: Callable, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t


def spent() -> float:
    return round(sum(r["usd"] for r in read_jsonl(LEDGER)), 6)


def record_spend(what: str, meter: Meter) -> None:
    if meter.usd <= 0:
        return
    row = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "what": what,
        "usd": meter.usd,
        "tokens": meter.tokens(),
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


def tokens_delta(before: dict, after: dict) -> dict[str, tuple[int, int]]:
    out = {}
    for m, (i, o) in after.items():
        bi, bo = before.get(m, (0, 0))
        if i - bi or o - bo:
            out[m] = (i - bi, o - bo)
    return out


def usd(tokens: dict[str, tuple[int, int]]) -> float:
    from voiceagent.evals.cost import price

    return sum(i * price(m)[0] / 1e6 + o * price(m)[1] / 1e6 for m, (i, o) in tokens.items())


def add_costs(out_dir: Path, stage: str, delta: dict) -> dict[str, tuple[int, int]]:
    """Add a stage's tokens to out_dir/costs.json (a resumed run adds to it) and return the
    stage's total."""
    path = out_dir / "costs.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    total = {m: list(t) for m, t in data.get(stage, {}).items()}
    for m, (i, o) in delta.items():
        t = total.setdefault(m, [0, 0])
        t[0] += i
        t[1] += o
    data[stage] = total
    path.write_text(json.dumps(data, indent=1))
    return {m: (t[0], t[1]) for m, t in total.items()}
