"""Reward function  R = w1*T + w2*L + w3*M - w4*Q - w5*S - P.

All terms are normalised so weights are comparable:
  T = clip(tps / baseline_tps, 0, 4) - 1          (0 at baseline, >0 faster)
  L = clip(baseline_ttft / ttft, 0, 4) - 1        (0 at baseline, >0 lower latency)
  M = clip(1 - peak_memory / memory_budget, -1, 1) (fraction of budget left free)
  Q = quality_loss / max_quality_loss in [0, 1]
  S = switch_s / (switch_s + request_s) in [0, 1] (share of the request's wall time spent
      reconfiguring, e.g. reloading the model; 0 when the configuration did not change)
  P = failure_penalty on OOM / runtime failure (other terms are then 0)

``intrinsic`` = R without the S term: how good the configuration itself is, independent of
the one-off cost of switching to it. Configuration statistics use the intrinsic value; the
Q-update uses the full reward, so the agent learns that switching has a price.

The baseline is the first successful episode for the context, so rewards are relative to
this machine rather than to arbitrary absolute numbers.
Hard constraints (quality, memory) are enforced *before* execution by action masking; the
penalty is only a backstop for failures the estimates could not predict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .._validate import clamp
from ..config import Objective


@dataclass(frozen=True, slots=True)
class RewardWeights:
    throughput: float
    latency: float
    memory: float
    quality: float
    switching: float = 0.6

    @classmethod
    def for_objective(cls, objective: Objective) -> RewardWeights:
        table = {
            Objective.BALANCED: cls(0.4, 0.3, 0.2, 0.5),
            Objective.THROUGHPUT: cls(0.7, 0.1, 0.1, 0.5),
            Objective.LATENCY: cls(0.2, 0.6, 0.1, 0.5),
            Objective.MEMORY: cls(0.2, 0.1, 0.6, 0.3),
        }
        return table[objective]


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    tokens_per_second: float
    ttft_s: float
    peak_memory_bytes: int
    memory_budget_bytes: int
    quality_loss: float
    failed: bool = False
    switch_s: float = 0.0
    request_s: float = 0.0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "tps": round(self.tokens_per_second, 4),
            "ttft_s": round(self.ttft_s, 6),
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "quality_loss": round(self.quality_loss, 6),
            "switch_s": round(self.switch_s, 4),
            "request_s": round(self.request_s, 4),
            "failed": self.failed,
        }


@dataclass(frozen=True, slots=True)
class Baseline:
    tokens_per_second: float
    ttft_s: float


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    throughput: float
    latency: float
    memory: float
    quality: float
    penalty: float
    switching: float = 0.0  # weighted S term that was subtracted

    @property
    def intrinsic(self) -> float:
        """Reward of the configuration itself, excluding the one-off switching cost."""
        return self.total + self.switching


_EPS = 1e-9


def _finite(x: float, default: float = 0.0) -> float:
    return x if math.isfinite(x) else default


def switching_share(switch_s: float, request_s: float) -> float:
    """Fraction of wall time spent reconfiguring, in [0, 1]."""
    s, r = max(0.0, _finite(switch_s)), max(0.0, _finite(request_s))
    return s / (s + r) if s > 0 else 0.0


def compute_reward(
    m: EpisodeMetrics,
    baseline: Baseline,
    weights: RewardWeights,
    *,
    max_quality_loss: float,
    failure_penalty: float,
) -> RewardBreakdown:
    if m.failed:
        return RewardBreakdown(-failure_penalty, 0.0, 0.0, 0.0, 0.0, failure_penalty)
    t = clamp(_finite(m.tokens_per_second / max(baseline.tokens_per_second, _EPS)), 0.0, 4.0)
    lat = clamp(_finite(baseline.ttft_s / max(m.ttft_s, _EPS)), 0.0, 4.0)
    mem = clamp(1.0 - m.peak_memory_bytes / max(m.memory_budget_bytes, 1), -1.0, 1.0)
    q = clamp(m.quality_loss / max(max_quality_loss, _EPS), 0.0, 1.0)
    switching = weights.switching * switching_share(m.switch_s, m.request_s)
    t_term, l_term = t - 1.0, lat - 1.0
    total = (
        weights.throughput * t_term
        + weights.latency * l_term
        + weights.memory * mem
        - weights.quality * q
        - switching
    )
    return RewardBreakdown(total, t_term, l_term, mem, q, 0.0, switching)
