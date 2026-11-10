"""Sparse tabular Q-function with optional per-action priors for unvisited entries.

Storage: dict[state_key] -> list[float] of length |A| (plus a parallel visit-count list).
  Space: O(|S_visited| * |A|)
  Lookup / set: O(1) average; argmax / max over actions: O(|A|) with |A| = 11.
Reads never insert rows, so querying unseen states does not grow the table.

Priors: an unvisited (s, a) reads as ``priors[a]`` (default 0.0) instead of a fixed constant.
The runtime supplies priors derived from observed per-configuration rewards, which avoids
the "every untried action looks best" random walk that zero-initialisation causes when
typical rewards are negative. Once (s, a) has been updated, its learned value is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..errors import ValidationError
from .actions import N_ACTIONS, Action

Priors = Sequence[float] | None


def _check(priors: Priors) -> None:
    if priors is not None and len(priors) != N_ACTIONS:
        raise ValidationError(f"priors must have {N_ACTIONS} entries")


class QTable:
    __slots__ = ("_n", "_q")

    def __init__(self) -> None:
        self._q: dict[str, list[float]] = {}
        self._n: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self._q)

    def visits(self, state: str, action: Action) -> int:
        row = self._n.get(state)
        return 0 if row is None else row[action]

    def q(self, state: str, action: Action, prior: float = 0.0) -> float:
        row = self._q.get(state)
        if row is None or self.visits(state, action) == 0:
            return prior
        return row[action]

    def row(self, state: str, priors: Priors = None) -> tuple[float, ...]:
        _check(priors)
        return tuple(self.q(state, a, priors[a] if priors else 0.0) for a in Action)

    def set(self, state: str, action: Action, value: float, visits: int) -> None:
        if not math.isfinite(value):
            raise ValidationError("Q-value must be finite")
        if visits < 0:
            raise ValidationError("visit count must be >= 0")
        if state not in self._q:
            self._q[state] = [0.0] * N_ACTIONS
            self._n[state] = [0] * N_ACTIONS
        self._q[state][action] = value
        self._n[state][action] = visits

    def best(self, state: str, mask: Sequence[bool], priors: Priors = None) -> Action | None:
        """Greedy action among allowed ones; ties resolve to the lowest index
        (MAINTAIN first), which keeps the policy stable."""
        _check(priors)
        best_a: Action | None = None
        best_v = -math.inf
        for a in Action:
            if not mask[a]:
                continue
            v = self.q(state, a, priors[a] if priors else 0.0)
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def max_q(self, state: str, mask: Sequence[bool], priors: Priors = None) -> float:
        a = self.best(state, mask, priors)
        if a is None:
            return 0.0
        return self.q(state, a, priors[a] if priors else 0.0)

    def update(
        self,
        state: str,
        action: Action,
        reward: float,
        next_state: str,
        next_mask: Sequence[bool],
        *,
        alpha: float,
        gamma: float,
        prior: float = 0.0,
        next_priors: Priors = None,
    ) -> tuple[float, float]:
        """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).

        ``prior`` is the value of Q(s,a) if it has never been visited.
        Returns (new_q, td_error).
        """
        if not math.isfinite(reward):
            raise ValidationError("reward must be finite")
        old = self.q(state, action, prior)
        target = reward + gamma * self.max_q(next_state, next_mask, next_priors)
        td_error = target - old
        new = old + alpha * td_error
        self.set(state, action, new, self.visits(state, action) + 1)
        return new, td_error
