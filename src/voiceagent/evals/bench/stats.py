"""The benchmark's metrics: pure functions, no I/O, no models."""

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence

LEVELS = ("does_not_meet", "partly_meets", "fully_meets")
GOLD = {0: 0.0, 1: 0.5, 2: 1.0}  # level index → gold score
BOOTSTRAP = 1000
LOW, HIGH = 0.34, 0.66


def band(score: float) -> int:
    """0 below 0.34, 2 above 0.66, 1 between (both edges inclusive)."""
    return 0 if score < LOW else 2 if score > HIGH else 1


def jev_score01(score: float, levels: int) -> float:
    """Jev's expected level index (0 … levels-1) on the 0-1 scale."""
    return score / (levels - 1)


def agrees(intended: int, label: str | None) -> bool:
    """A generated item is kept only when the labeler put it at the intended level."""
    return label is not None and LEVELS.index(label) == intended


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pstdev(xs: Sequence[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear interpolation between closest ranks; p in 0-100."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def spread(xs: Sequence[float]) -> float:
    return max(xs) - min(xs)


def bootstrap(
    units: Sequence, stat: Callable[[Sequence], float | None], n: int = BOOTSTRAP, seed=0
) -> tuple[float, float] | None:
    """95% percentile interval of `stat` over `n` resamples of `units` with replacement."""
    if not units:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = stat([units[rng.randrange(len(units))] for _ in units])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return percentile(vals, 2.5), percentile(vals, 97.5)


# --- Task 1: per item, a list of 0-1 scores (one per repeat) ---------------------------


def item_mae(scores: Sequence[float], gold: float) -> float:
    return mean([abs(s - gold) for s in scores])


def item_band_acc(scores: Sequence[float], gold: float) -> float:
    return mean([band(s) == band(gold) for s in scores])


def same_band(scores: Sequence[float]) -> bool:
    return len({band(s) for s in scores}) == 1


def group_spread(groups: dict[str, list[list[float]]]) -> list[float]:
    """Per group, max − min of its members' mean scores (groups of one are skipped)."""
    return [spread([mean(m) for m in ms]) for ms in groups.values() if len(ms) > 1]


def confident_accuracy(rows: Sequence[tuple[float, bool]], keep: float = 0.8) -> float | None:
    """Accuracy on the `keep` share of (confidence, correct) rows with the highest
    confidence (ties broken by order)."""
    if not rows:
        return None
    top = sorted(rows, key=lambda r: -r[0])[: max(1, round(len(rows) * keep))]
    return mean([ok for _, ok in top])


# --- Task 2 -----------------------------------------------------------------------------


def correct_progress(gold: Sequence[str], pred: str | None) -> bool:
    """Right when it names a requirement the turn asks, or none for a turn asking none."""
    return pred in gold if gold else pred is None


def macro_f1(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Macro-F1 over every label seen in (gold, predicted) pairs."""
    if not pairs:
        return None
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in pairs:
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    labels = set(tp) | set(fp) | set(fn)
    f1s = []
    for c in labels:
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom else 0.0)
    return mean(f1s)


def progress_pair(gold: Sequence[str], pred: str | None) -> tuple[str, str]:
    """(gold label, predicted label) for F1, crediting any requirement the turn asks."""
    p = pred or "none"
    if not gold:
        return "none", p
    return (p if p in gold else gold[0]), p


# --- Task 3 -----------------------------------------------------------------------------


def rate(xs: Sequence[bool]) -> float | None:
    return mean([bool(x) for x in xs])
