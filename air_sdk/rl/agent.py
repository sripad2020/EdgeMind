"""Tabular Q-learning agent with masked epsilon-greedy exploration and persistence.

* Exploration is *safe*: random actions are drawn only from the mask of validated actions.
* Epsilon decays per completed episode and is persisted, so a restarted process resumes
  with the same exploration level and the restored Q-table (policy restoration).
* The RNG is ``random.Random`` (seedable, reproducible). It is not used for anything
  security-sensitive, so a CSPRNG is unnecessary.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .._validate import require_float
from ..errors import ValidationError
from .actions import N_ACTIONS, Action
from .qtable import Priors, QTable
from .reward import Baseline
from .store import AgentMeta, EpisodeRecord, ExperienceStore


@dataclass(frozen=True, slots=True)
class QLearningParams:
    alpha: float = 0.2
    # Configuration selection is close to a contextual bandit: long horizons add variance
    # without adding information, so the default discount is moderate.
    gamma: float = 0.5
    # On real backends every exploratory move can force a model reload (~1 s), so the
    # default exploration is modest and decays quickly.
    epsilon_start: float = 0.15
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.95
    failure_penalty: float = 5.0

    def __post_init__(self) -> None:
        require_float("alpha", self.alpha, min_value=1e-6, max_value=1.0)
        require_float("gamma", self.gamma, min_value=0.0, max_value=0.999)
        require_float("epsilon_start", self.epsilon_start, min_value=0.0, max_value=1.0)
        require_float("epsilon_min", self.epsilon_min, min_value=0.0, max_value=1.0)
        require_float("epsilon_decay", self.epsilon_decay, min_value=0.5, max_value=1.0)
        require_float("failure_penalty", self.failure_penalty, min_value=0.0, max_value=1e3)
        if self.epsilon_min > self.epsilon_start:
            raise ValidationError("epsilon_min must be <= epsilon_start")


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    explored: bool


@dataclass(frozen=True, slots=True)
class UpdateResult:
    new_q: float
    td_error: float


class QLearningAgent:
    def __init__(
        self,
        store: ExperienceStore,
        context_key: str,
        *,
        params: QLearningParams,
        description: str,
        seed: int | None = None,
    ) -> None:
        self._store = store
        self._ctx = context_key
        self._p = params
        self._desc = description
        self._rng = random.Random(seed)
        self.q = QTable()
        for state, action, value, visits in store.load_q_values(context_key):
            self.q.set(state, action, value, visits)
        meta = store.load_meta(context_key)
        self._episodes = meta.episodes if meta else 0
        self._epsilon = meta.epsilon if meta else params.epsilon_start
        self._baseline: Baseline | None = (
            Baseline(meta.baseline_tps, meta.baseline_ttft)
            if meta and meta.baseline_tps and meta.baseline_ttft
            else None
        )

    # ---- properties -------------------------------------------------------------------------

    @property
    def context_key(self) -> str:
        return self._ctx

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def params(self) -> QLearningParams:
        return self._p

    @property
    def baseline(self) -> Baseline | None:
        return self._baseline

    def set_baseline(self, baseline: Baseline) -> None:
        if baseline.tokens_per_second <= 0 or baseline.ttft_s <= 0:
            raise ValidationError("baseline values must be positive")
        self._baseline = baseline
        self._store.save_meta(self._ctx, self._meta(), self._desc)

    def _meta(self) -> AgentMeta:
        b = self._baseline
        return AgentMeta(
            self._episodes, self._epsilon,
            b.tokens_per_second if b else None, b.ttft_s if b else None,
        )

    # ---- policy -----------------------------------------------------------------------------

    @staticmethod
    def _check_mask(mask: Sequence[bool]) -> None:
        if len(mask) != N_ACTIONS:
            raise ValidationError(f"action mask must have {N_ACTIONS} entries")

    def select(self, state: str, mask: Sequence[bool], priors: Priors = None) -> Decision:
        """Masked epsilon-greedy. O(|A|)."""
        self._check_mask(mask)
        valid = [a for a in Action if mask[a]]
        if not valid:
            return Decision(Action.MAINTAIN, explored=False)
        if self._rng.random() < self._epsilon:
            return Decision(self._rng.choice(valid), explored=True)
        best = self.q.best(state, mask, priors)
        return Decision(best if best is not None else Action.MAINTAIN, explored=False)

    def greedy(self, state: str, mask: Sequence[bool], priors: Priors = None) -> Action:
        self._check_mask(mask)
        best = self.q.best(state, mask, priors)
        return best if best is not None else Action.MAINTAIN

    def update(
        self,
        state: str,
        action: Action,
        reward: float,
        next_state: str,
        next_mask: Sequence[bool],
        *,
        config_key: str,
        metrics: dict[str, float | int | bool],
        failed: bool,
        prior: float = 0.0,
        next_priors: Priors = None,
    ) -> UpdateResult:
        """One Q-learning step + epsilon decay, persisted atomically."""
        self._check_mask(next_mask)
        new_q, td = self.q.update(
            state, action, reward, next_state, next_mask, alpha=self._p.alpha,
            gamma=self._p.gamma, prior=prior, next_priors=next_priors,
        )
        self._episodes += 1
        self._epsilon = max(self._p.epsilon_min, self._epsilon * self._p.epsilon_decay)
        record = EpisodeRecord(
            state=state, action=action, reward=reward, next_state=next_state,
            config_key=config_key, metrics=metrics, failed=failed, new_q=new_q,
            visits=self.q.visits(state, action), episodes=self._episodes, epsilon=self._epsilon,
        )
        self._store.record_episode(self._ctx, record, self._meta(), self._desc)
        return UpdateResult(new_q, td)
