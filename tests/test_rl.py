from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from air_sdk.config import Objective
from air_sdk.errors import ValidationError
from air_sdk.rl.actions import N_ACTIONS, Action
from air_sdk.rl.agent import QLearningAgent, QLearningParams
from air_sdk.rl.qtable import QTable
from air_sdk.rl.reward import Baseline, EpisodeMetrics, RewardWeights, compute_reward
from air_sdk.rl.store import ExperienceStore

ALL = [True] * N_ACTIONS


def test_q_update_formula() -> None:
    q = QTable()
    q.set("s2", Action.MORE_THREADS, 2.0, 1)
    new, td = q.update("s1", Action.MAINTAIN, 1.0, "s2", ALL, alpha=0.5, gamma=0.9)
    # Q = 0 + 0.5 * (1 + 0.9*2 - 0) = 1.4
    assert new == pytest.approx(1.4)
    assert td == pytest.approx(2.8)
    assert q.visits("s1", Action.MAINTAIN) == 1


def test_max_q_respects_mask() -> None:
    q = QTable()
    q.set("s", Action.LOWER_PRECISION, 10.0, 1)
    mask = [True] * N_ACTIONS
    mask[Action.LOWER_PRECISION] = False
    assert q.best("s", mask) is Action.MAINTAIN  # tie at 0 -> lowest index
    assert q.max_q("s", mask) == 0.0


def test_reads_do_not_grow_table() -> None:
    q = QTable()
    q.q("unseen", Action.MAINTAIN)
    q.best("unseen", ALL)
    assert len(q) == 0


def test_non_finite_rejected() -> None:
    q = QTable()
    with pytest.raises(ValidationError):
        q.update("s", Action.MAINTAIN, float("nan"), "s", ALL, alpha=0.1, gamma=0.9)


def test_agent_only_explores_valid_actions(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "x.db")
    agent = QLearningAgent(store, "ctx", params=QLearningParams(epsilon_start=1.0,
                           epsilon_min=1.0, epsilon_decay=1.0), description="t", seed=3)
    mask = [False] * N_ACTIONS
    mask[Action.MORE_GPU_LAYERS] = True
    mask[Action.SMALLER_BATCH] = True
    picks = {agent.select("s", mask).action for _ in range(200)}
    assert picks <= {Action.MORE_GPU_LAYERS, Action.SMALLER_BATCH}
    assert agent.select("s", [False] * N_ACTIONS).action is Action.MAINTAIN
    store.close()


def test_policy_restored_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    store = ExperienceStore(db)
    agent = QLearningAgent(store, "ctx", params=QLearningParams(), description="t", seed=0)
    agent.set_baseline(Baseline(10.0, 0.5))
    for _ in range(5):
        agent.update("s", Action.MORE_THREADS, 1.0, "s", ALL, config_key="c",
                     metrics={"tps": 1.0}, failed=False)
    q_before, eps_before = agent.q.q("s", Action.MORE_THREADS), agent.epsilon
    store.close()

    store2 = ExperienceStore(db)
    agent2 = QLearningAgent(store2, "ctx", params=QLearningParams(), description="t", seed=0)
    assert agent2.q.q("s", Action.MORE_THREADS) == pytest.approx(q_before)
    assert agent2.epsilon == pytest.approx(eps_before)
    assert agent2.episodes == 5
    assert agent2.baseline == Baseline(10.0, 0.5)
    assert agent2.greedy("s", ALL) is Action.MORE_THREADS
    # Contexts are isolated (different model/hardware never share learning).
    other = QLearningAgent(store2, "other", params=QLearningParams(), description="t")
    assert len(other.q) == 0
    store2.close()


def test_tampered_rows_are_ignored(tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    ExperienceStore(db).close()
    con = sqlite3.connect(db)
    con.execute("INSERT INTO q_values VALUES ('ctx','s',999,1.0,1)")  # bad action
    con.execute("INSERT INTO q_values VALUES ('ctx','s',1,'NaN',1)")  # bad q
    con.execute("INSERT INTO q_values VALUES ('ctx','s',2,0.5,1)")  # good
    con.commit()
    con.close()
    store = ExperienceStore(db)
    rows = store.load_q_values("ctx")
    assert [(r[1], r[2]) for r in rows] == [(Action(2), 0.5)]
    store.close()


def test_blacklist(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "x.db", failure_ttl_s=3600)
    store.mark_failed("ctx", "cfgA", "out_of_memory")
    assert store.blacklist("ctx") == frozenset({"cfgA"})
    assert store.blacklist("other") == frozenset()
    store.close()


def test_reward_shape() -> None:
    w = RewardWeights.for_objective(Objective.BALANCED)
    base = Baseline(10.0, 1.0)
    same = compute_reward(EpisodeMetrics(10.0, 1.0, 50, 100, 0.0), base, w,
                          max_quality_loss=0.05, failure_penalty=5.0)
    faster = compute_reward(EpisodeMetrics(20.0, 0.5, 50, 100, 0.0), base, w,
                            max_quality_loss=0.05, failure_penalty=5.0)
    lossy = compute_reward(EpisodeMetrics(10.0, 1.0, 50, 100, 0.05), base, w,
                           max_quality_loss=0.05, failure_penalty=5.0)
    failed = compute_reward(EpisodeMetrics(0, 0, 0, 100, 0, failed=True), base, w,
                            max_quality_loss=0.05, failure_penalty=5.0)
    assert faster.total > same.total > lossy.total > failed.total
    assert failed.total == -5.0
    assert same.throughput == 0.0
    assert same.latency == 0.0
