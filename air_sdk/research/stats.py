"""Statistics for the experiments (standard library only, deterministic).

* Wilson score interval for a pass rate (better than the normal approximation for small n
  and rates near 0 or 1).
* Percentile bootstrap interval for a mean (fixed seed -> reproducible).
* Exact McNemar test for paired yes/no outcomes (two profiles on the same items).
* The qualification rule (non-inferiority): a candidate qualifies only if the LOWER bound of
  its interval is no more than a margin (default 5 points) below the baseline's score.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

Z95: Final[float] = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Rate:
    passes: int
    n: int
    low: float
    high: float

    @property
    def value(self) -> float:
        return self.passes / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, object]:
        return {"passes": self.passes, "n": self.n, "rate": round(self.value, 4),
                "ci_low": round(self.low, 4), "ci_high": round(self.high, 4)}


def wilson(passes: int, n: int, z: float = Z95) -> Rate:
    """Wilson score interval for ``passes`` successes out of ``n``."""
    if n < 0 or not 0 <= passes <= max(n, 0):
        raise ValueError("need 0 <= passes <= n")
    if n == 0:
        return Rate(0, 0, 0.0, 1.0)
    p = passes / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return Rate(passes, n, max(0.0, centre - half), min(1.0, centre + half))


NON_INFERIORITY_MARGIN: Final[float] = 0.05


def qualified(candidate: Rate, baseline_score: float, *,
              margin: float = NON_INFERIORITY_MARGIN) -> bool:
    """Non-inferiority: with 95% confidence the candidate is at most ``margin`` worse than
    the baseline (its interval's lower bound >= baseline - margin). A cheaper compressed
    version can qualify; one that is clearly worse cannot. ``margin=0`` demands a lower
    bound at least equal to the baseline (for adapters that must improve on it)."""
    return candidate.n > 0 and candidate.low >= baseline_score - margin


def bootstrap_mean(values: Sequence[float], *, iterations: int = 2000, seed: int = 0,
                   confidence: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) by the percentile bootstrap."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(max(1, iterations)))
    tail = (1 - confidence) / 2
    lo = means[int(tail * (len(means) - 1))]
    hi = means[int((1 - tail) * (len(means) - 1))]
    return (sum(values) / n, lo, hi)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Paired outcomes of two systems on the same items -> (only_a, only_b, two-sided p)."""
    if len(a) != len(b):
        raise ValueError("paired sequences must have the same length")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return (0, 0, 1.0)
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return (only_a, only_b, min(1.0, 2 * tail))
