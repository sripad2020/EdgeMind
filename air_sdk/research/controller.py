"""Profile selection policies, evaluated by exact replay of the logged results (RQ4, RQ5).

Because every profile answered every item, any policy can be replayed offline: a cascade
"tries" profile A (paying its logged time), looks at A's logged *online check*, and if it
failed moves on to profile B, and so on. The final answer's *gold* score says whether the
policy was right. No policy ever sees gold scores when choosing.

Policies:
    fixed(p)              - always profile p
    cascade(order)        - try profiles in order, stop at the first passed check
    threshold(tau)        - per task, start at the cheapest profile whose check pass rate
                            (on the profile split) is >= tau, then escalate      (RQ5 baseline)
    q_learning            - tabular Q-learning picks the starting point per state (RQ5)

Learning/tuning uses the ``profile`` split; every reported number uses the ``router``
split, which nothing was tuned on.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .stats import bootstrap_mean, wilson
from .store import Result, ResultStore

LAMBDA: Final[float] = 0.05  # reward cost per second of generation


@dataclass(frozen=True, slots=True)
class Outcome:
    correct: bool | None
    seconds: float
    tried: int


@dataclass(frozen=True, slots=True)
class PolicyReport:
    policy: str
    task: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    mean_seconds: float
    seconds_per_correct: float | None
    mean_tried: float

    def as_dict(self) -> dict[str, object]:
        return {"policy": self.policy, "task": self.task, "n": self.n,
                "accuracy": round(self.accuracy, 4), "ci_low": round(self.ci_low, 4),
                "ci_high": round(self.ci_high, 4), "mean_seconds": round(self.mean_seconds, 3),
                "seconds_per_correct": None if self.seconds_per_correct is None
                else round(self.seconds_per_correct, 3),
                "mean_tried": round(self.mean_tried, 3)}


Table = dict[str, dict[str, Result]]  # item_id -> profile -> result


def table(store: ResultStore, task: str, split: str, profiles: Sequence[str]) -> Table:
    """Items answered by *every* listed profile (so replays are paired and fair)."""
    out: Table = defaultdict(dict)
    for r in store.results(task=task, split=split):
        if r.profile in profiles:
            out[r.item_id][r.profile] = r
    return {k: v for k, v in out.items() if len(v) == len(profiles)}


def run_cascade(order: Sequence[str], row: dict[str, Result]) -> Outcome:
    spent = 0.0
    for i, name in enumerate(order, 1):
        r = row[name]
        spent += r.seconds
        if r.check_ok or i == len(order):
            return Outcome(r.gold_ok, spent, i)
    raise ValueError("empty cascade")


def summarize(policy: str, task: str, outcomes: list[Outcome]) -> PolicyReport:
    scored = [o for o in outcomes if o.correct is not None]
    correct = sum(1 for o in scored if o.correct)
    rate = wilson(correct, len(scored))
    secs = sum(o.seconds for o in outcomes)
    return PolicyReport(policy, task, len(outcomes), rate.value, rate.low, rate.high,
                        secs / len(outcomes) if outcomes else 0.0,
                        secs / correct if correct else None,
                        sum(o.tried for o in outcomes) / len(outcomes) if outcomes else 0.0)


def cost_order(store: ResultStore, task: str, profiles: Sequence[str]) -> list[str]:
    """Profiles from cheapest to most expensive (mean seconds on the profile split)."""
    secs: dict[str, list[float]] = defaultdict(list)
    for r in store.results(task=task, split="profile"):
        if r.profile in profiles:
            secs[r.profile].append(r.seconds)
    return sorted(profiles, key=lambda p: sum(secs[p]) / len(secs[p]) if secs[p] else 1e9)


def threshold_start(store: ResultStore, task: str, order: Sequence[str], tau: float) -> int:
    for i, name in enumerate(order):
        rows = store.results(profile=name, task=task, split="profile")
        if rows and sum(r.check_ok for r in rows) / len(rows) >= tau:
            return i
    return len(order) - 1


# ---- tabular Q-learning --------------------------------------------------------------------


@dataclass
class QRouter:
    """State = (task, did the previous request of this task need escalation).
    Action = index in the cost-ordered cascade where to start. Reward = 1 if the final
    online check passed, minus LAMBDA x seconds spent. One-step episodes (gamma = 0)."""

    n_actions: int
    alpha: float = 0.2
    epsilon: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        self.q: dict[tuple[str, bool], list[float]] = defaultdict(
            lambda: [0.0] * self.n_actions)
        self._rng = random.Random(self.seed)

    def act(self, state: tuple[str, bool], *, explore: bool) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        values = self.q[state]
        best = max(values)
        return values.index(best)  # ties -> cheapest start

    def learn(self, state: tuple[str, bool], action: int, reward: float) -> None:
        values = self.q[state]
        values[action] += self.alpha * (reward - values[action])


def q_reward(order: Sequence[str], row: dict[str, Result], start: int,
             lam: float) -> tuple[float, Outcome]:
    out = run_cascade(order[start:], row)
    final = row[order[start + out.tried - 1]]
    return (1.0 if final.check_ok else 0.0) - lam * out.seconds, out


def train_q(order: Sequence[str], train: Table, task: str, *, epochs: int = 20,
            lam: float = LAMBDA, seed: int = 0) -> QRouter:
    router = QRouter(len(order), seed=seed)
    rng = random.Random(seed)
    ids = sorted(train)
    for _ in range(epochs):
        rng.shuffle(ids)
        escalated = False
        for item_id in ids:
            state = (task, escalated)
            a = router.act(state, explore=True)
            reward, out = q_reward(order, train[item_id], a, lam)
            router.learn(state, a, reward)
            escalated = out.tried > 1
    return router


def evaluate_q(router: QRouter, order: Sequence[str], test: Table, task: str,
               lam: float = LAMBDA) -> list[Outcome]:
    outcomes: list[Outcome] = []
    escalated = False
    for item_id in sorted(test):
        start = router.act((task, escalated), explore=False)
        _r, out = q_reward(order, test[item_id], start, lam)
        outcomes.append(out)
        escalated = out.tried > 1
    return outcomes


# ---- the comparison --------------------------------------------------------------------------


def compare(store: ResultStore, task: str, profiles: Sequence[str], *,
            taus: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
            lam: float = LAMBDA) -> list[PolicyReport]:
    """Every policy on the router split of ``task``; tuning only on the profile split."""
    order = cost_order(store, task, profiles)
    train = table(store, task, "profile", order)
    test = table(store, task, "router", order)
    if not test:
        return []
    reports = [summarize(f"fixed:{p}", task, [run_cascade([p], test[i]) for i in sorted(test)])
               for p in order]
    reports.append(summarize("cascade:cheapest-first", task,
                             [run_cascade(order, test[i]) for i in sorted(test)]))
    best_tau, best_val = taus[0], float("-inf")
    for tau in taus:  # tune the threshold on the profile split (reward = same as Q-learning)
        s = threshold_start(store, task, order, tau)
        if not train:
            break
        val = sum(q_reward(order, row, s, lam)[0] for row in train.values())
        if val > best_val:
            best_tau, best_val = tau, val
    s = threshold_start(store, task, order, best_tau)
    reports.append(summarize(f"threshold:tau={best_tau}", task,
                             [run_cascade(order[s:], test[i]) for i in sorted(test)]))
    if train:
        router = train_q(order, train, task, lam=lam)
        reports.append(summarize("q_learning", task, evaluate_q(router, order, test, task, lam)))
    return reports


def cost_interval(outcomes: list[Outcome]) -> tuple[float, float, float]:
    return bootstrap_mean([o.seconds for o in outcomes])
