"""SQLite persistence for Q-values, agent metadata, experience log and failed configs.

Security
  * Every statement is parameterised - no string-built SQL with external data.
  * Rows read back are treated as untrusted and validated (a tampered DB cannot inject
    NaN Q-values or out-of-range actions; bad rows are skipped).
  * JSON is written with ``allow_nan=False``; nothing is ever pickled.
  * On POSIX the DB file is created with mode 0600 (owner-only).

Performance
  * WAL journal + ``synchronous=NORMAL``; each episode is a single short transaction.
  * The experience log is pruned to ``max_experience_rows`` per context (amortised O(1)),
    so disk usage is bounded.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Concatenate, Final, ParamSpec, TypeVar

from ..errors import PersistenceError
from .actions import N_ACTIONS, Action

SCHEMA_VERSION: Final[int] = 1
_PRUNE_EVERY: Final[int] = 256
# Config rankings average the *intrinsic* reward (excluding one-off switching cost), falling
# back to the raw reward for rows written before the intrinsic value was recorded.

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS q_values (
    context TEXT NOT NULL, state TEXT NOT NULL, action INTEGER NOT NULL,
    q REAL NOT NULL, visits INTEGER NOT NULL,
    PRIMARY KEY (context, state, action)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS agent_meta (
    context TEXT PRIMARY KEY, episodes INTEGER NOT NULL, epsilon REAL NOT NULL,
    baseline_tps REAL, baseline_ttft REAL, description TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experience (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, context TEXT NOT NULL,
    state TEXT NOT NULL, action INTEGER NOT NULL, reward REAL NOT NULL,
    next_state TEXT NOT NULL, config_key TEXT NOT NULL, metrics TEXT NOT NULL,
    failed INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_ctx_id ON experience(context, id);
CREATE INDEX IF NOT EXISTS idx_exp_ctx_cfg ON experience(context, config_key);
CREATE TABLE IF NOT EXISTS skill_results (
    family TEXT NOT NULL, skill TEXT NOT NULL, precision TEXT NOT NULL,
    passes INTEGER NOT NULL, fails INTEGER NOT NULL, updated REAL NOT NULL,
    PRIMARY KEY (family, skill, precision)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS quality (
    variant_key TEXT PRIMARY KEY, reference_key TEXT NOT NULL, agreement REAL NOT NULL,
    measured_loss REAL NOT NULL, prompts INTEGER NOT NULL, updated REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS calibration (
    scope TEXT PRIMARY KEY, ratio_max REAL NOT NULL, samples INTEGER NOT NULL,
    updated REAL NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS failed_configs (
    context TEXT NOT NULL, config_key TEXT NOT NULL, kind TEXT NOT NULL,
    count INTEGER NOT NULL, last_ts REAL NOT NULL,
    PRIMARY KEY (context, config_key)
) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class AgentMeta:
    episodes: int
    epsilon: float
    baseline_tps: float | None
    baseline_ttft: float | None


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    state: str
    action: Action
    reward: float
    next_state: str
    config_key: str
    metrics: dict[str, float | int | bool]
    failed: bool
    new_q: float
    visits: int
    episodes: int
    epsilon: float


@dataclass(frozen=True, slots=True)
class ConfigScore:
    config_key: str
    mean_reward: float
    samples: int


def _secure_create(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix" and not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _locked(fn: Callable[Concatenate[ExperienceStore, _P], _R]) -> Callable[
    Concatenate[ExperienceStore, _P], _R
]:
    """Serialise a store method on the instance lock (thread-safe shared connection)."""

    def wrapper(self: ExperienceStore, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


class ExperienceStore:
    """Single-connection store. Intended for use from the runtime's control thread."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_experience_rows: int = 100_000,
        failure_ttl_s: float = 24 * 3600.0,
        permanent_failure_count: int = 3,
    ) -> None:
        self._path = Path(path)
        self._max_rows = max_experience_rows
        self._failure_ttl = failure_ttl_s
        self._permanent = permanent_failure_count
        self._inserts = 0
        self._lock = threading.RLock()
        try:
            if str(path) != ":memory:":
                _secure_create(self._path)
            # One connection shared across threads (web servers call from worker threads);
            # every public method serialises access through self._lock.
            self._db = sqlite3.connect(str(path), timeout=5.0, isolation_level=None,
                                       check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)
            row = self._db.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                self._db.execute("INSERT INTO schema_version (version) VALUES (?)",
                                 (SCHEMA_VERSION,))
            elif row[0] != SCHEMA_VERSION:
                raise PersistenceError(f"unsupported store schema version {row[0]}")
        except sqlite3.Error as exc:
            raise PersistenceError(f"cannot open experience store at {path}: {exc}") from exc

    @_locked
    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._db.close()

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")
        except sqlite3.Error as exc:
            raise PersistenceError(f"experience store transaction failed: {exc}") from exc

    # ---- Q-values ---------------------------------------------------------------------------

    @_locked
    def load_q_values(self, context: str) -> list[tuple[str, Action, float, int]]:
        rows = self._db.execute(
            "SELECT state, action, q, visits FROM q_values WHERE context = ?", (context,)
        ).fetchall()
        result: list[tuple[str, Action, float, int]] = []
        for state, action, q, visits in rows:
            qf = _finite_or_none(q)
            if (
                not isinstance(state, str)
                or isinstance(action, bool) or not isinstance(action, int)
                or not 0 <= action < N_ACTIONS
                or qf is None
                or isinstance(visits, bool) or not isinstance(visits, int) or visits < 0
            ):
                continue  # untrusted row: skip rather than poison the policy
            result.append((state, Action(action), qf, visits))
        return result

    # ---- agent metadata ---------------------------------------------------------------------

    @_locked
    def load_meta(self, context: str) -> AgentMeta | None:
        row = self._db.execute(
            "SELECT episodes, epsilon, baseline_tps, baseline_ttft FROM agent_meta"
            " WHERE context = ?",
            (context,),
        ).fetchone()
        if row is None:
            return None
        episodes, epsilon = row[0], _finite_or_none(row[1])
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0:
            return None
        if epsilon is None or not 0.0 <= epsilon <= 1.0:
            return None
        tps, ttft = _finite_or_none(row[2]), _finite_or_none(row[3])
        if (tps is not None and tps <= 0) or (ttft is not None and ttft <= 0):
            tps = ttft = None
        return AgentMeta(episodes, epsilon, tps, ttft)

    @_locked
    def save_meta(self, context: str, meta: AgentMeta, description: str) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO agent_meta (context, episodes, epsilon, baseline_tps,"
                " baseline_ttft, description) VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(context) DO UPDATE SET episodes=excluded.episodes,"
                " epsilon=excluded.epsilon, baseline_tps=excluded.baseline_tps,"
                " baseline_ttft=excluded.baseline_ttft",
                (context, meta.episodes, meta.epsilon, meta.baseline_tps, meta.baseline_ttft,
                 description[:2048]),
            )

    # ---- episodes ---------------------------------------------------------------------------

    @_locked
    def record_episode(self, context: str, rec: EpisodeRecord, meta: AgentMeta,
                       description: str) -> None:
        """Atomically persist the Q update, the experience row and agent metadata."""
        metrics_json = json.dumps(rec.metrics, sort_keys=True, allow_nan=False)
        with self._tx() as db:
            db.execute(
                "INSERT INTO q_values (context, state, action, q, visits) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(context, state, action) DO UPDATE SET q=excluded.q,"
                " visits=excluded.visits",
                (context, rec.state, int(rec.action), rec.new_q, rec.visits),
            )
            db.execute(
                "INSERT INTO experience (ts, context, state, action, reward, next_state,"
                " config_key, metrics, failed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), context, rec.state, int(rec.action), rec.reward, rec.next_state,
                 rec.config_key, metrics_json, int(rec.failed)),
            )
            db.execute(
                "INSERT INTO agent_meta (context, episodes, epsilon, baseline_tps,"
                " baseline_ttft, description) VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(context) DO UPDATE SET episodes=excluded.episodes,"
                " epsilon=excluded.epsilon, baseline_tps=excluded.baseline_tps,"
                " baseline_ttft=excluded.baseline_ttft",
                (context, meta.episodes, meta.epsilon, meta.baseline_tps, meta.baseline_ttft,
                 description[:2048]),
            )
            self._inserts += 1
            if self._inserts % _PRUNE_EVERY == 0:
                db.execute(
                    "DELETE FROM experience WHERE context = ? AND id <= ("
                    " SELECT id FROM experience WHERE context = ?"
                    " ORDER BY id DESC LIMIT 1 OFFSET ?)",
                    (context, context, self._max_rows),
                )

    @_locked
    def recent_rewards(self, context: str, limit: int) -> list[float]:
        rows = self._db.execute(
            "SELECT reward FROM experience WHERE context = ? ORDER BY id DESC LIMIT ?",
            (context, limit),
        ).fetchall()
        values = [_finite_or_none(r[0]) for r in rows]
        return [v for v in reversed(values) if v is not None]

    @_locked
    def top_configs(self, context: str, *, min_samples: int, limit: int) -> list[ConfigScore]:
        rows = self._db.execute(
            "SELECT config_key, AVG(COALESCE(json_extract(metrics, '$.intrinsic'), reward)),"
            " COUNT(*) FROM experience"
            " WHERE context = ? AND failed = 0 GROUP BY config_key HAVING COUNT(*) >= ?"
            " ORDER BY AVG(reward) DESC LIMIT ?",
            (context, min_samples, limit),
        ).fetchall()
        out: list[ConfigScore] = []
        for key, mean, count in rows:
            m = _finite_or_none(mean)
            if isinstance(key, str) and m is not None and isinstance(count, int):
                out.append(ConfigScore(key, m, count))
        return out

    @_locked
    def config_stats(self, context: str) -> list[ConfigScore]:
        """Per-configuration reward mean/count over successful episodes. O(rows) via index."""
        rows = self._db.execute(
            "SELECT config_key, AVG(COALESCE(json_extract(metrics, '$.intrinsic'), reward)),"
            " COUNT(*) FROM experience"
            " WHERE context = ? AND failed = 0 GROUP BY config_key",
            (context,),
        ).fetchall()
        out: list[ConfigScore] = []
        for key, mean, count in rows:
            m = _finite_or_none(mean)
            if isinstance(key, str) and m is not None and isinstance(count, int) and count > 0:
                out.append(ConfigScore(key, m, count))
        return out

    # ---- verified skills ----------------------------------------------------------------------

    @_locked
    def record_skill_result(self, family: str, skill: str, precision: str, ok: bool) -> None:
        """Count one verified (ok) or failed check for a model family / skill / precision."""
        with self._tx() as db:
            db.execute(
                "INSERT INTO skill_results (family, skill, precision, passes, fails, updated)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(family, skill, precision) DO UPDATE"
                " SET passes = passes + excluded.passes, fails = fails + excluded.fails,"
                " updated = excluded.updated",
                (family, skill[:64], precision[:32], int(ok), int(not ok), time.time()),
            )

    @_locked
    def skill_results(self, family: str, skill: str) -> dict[str, tuple[int, int]]:
        """precision -> (passes, fails); corrupt rows are ignored."""
        rows = self._db.execute(
            "SELECT precision, passes, fails FROM skill_results WHERE family = ? AND skill = ?",
            (family, skill),
        ).fetchall()
        out: dict[str, tuple[int, int]] = {}
        for prec, passes, fails in rows:
            if (isinstance(prec, str) and isinstance(passes, int) and isinstance(fails, int)
                    and not isinstance(passes, bool) and passes >= 0 and fails >= 0):
                out[prec] = (passes, fails)
        return out

    # ---- measured quality ---------------------------------------------------------------------

    @_locked
    def record_quality(self, variant_key: str, reference_key: str, agreement: float,
                       measured_loss: float, prompts: int) -> None:
        if not (math.isfinite(agreement) and math.isfinite(measured_loss)):
            return
        with self._tx() as db:
            db.execute(
                "INSERT INTO quality (variant_key, reference_key, agreement, measured_loss,"
                " prompts, updated) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(variant_key) DO"
                " UPDATE SET reference_key=excluded.reference_key,"
                " agreement=excluded.agreement, measured_loss=excluded.measured_loss,"
                " prompts=excluded.prompts, updated=excluded.updated",
                (variant_key, reference_key, agreement, measured_loss, prompts, time.time()),
            )

    @_locked
    def quality_overrides(self, variant_keys: list[str]) -> dict[str, float]:
        """Measured quality loss per variant (validated; corrupt rows ignored)."""
        out: dict[str, float] = {}
        for key in variant_keys:
            row = self._db.execute(
                "SELECT measured_loss FROM quality WHERE variant_key = ?", (key,)
            ).fetchone()
            loss = None if row is None else _finite_or_none(row[0])
            if loss is not None and 0.0 <= loss <= 1.0:
                out[key] = loss
        return out

    # ---- memory-estimate calibration ----------------------------------------------------------

    @_locked
    def load_calibration(self, scope: str) -> tuple[float, int] | None:
        """(max observed overhead ratio, sample count) for a backend+hardware scope."""
        row = self._db.execute(
            "SELECT ratio_max, samples FROM calibration WHERE scope = ?", (scope,)
        ).fetchone()
        if row is None:
            return None
        ratio, samples = _finite_or_none(row[0]), row[1]
        if ratio is None or ratio < 0 or isinstance(samples, bool) or not isinstance(
                samples, int) or samples < 1:
            return None  # untrusted / corrupt row
        return ratio, samples

    @_locked
    def record_calibration(self, scope: str, ratio: float) -> None:
        """Keep the *maximum* observed ratio (conservative): estimates may only shrink as far
        as the worst measurement on this machine justifies."""
        if not math.isfinite(ratio) or ratio < 0:
            return
        with self._tx() as db:
            db.execute(
                "INSERT INTO calibration (scope, ratio_max, samples, updated)"
                " VALUES (?, ?, 1, ?) ON CONFLICT(scope) DO UPDATE SET"
                " ratio_max = MAX(ratio_max, excluded.ratio_max), samples = samples + 1,"
                " updated = excluded.updated",
                (scope, ratio, time.time()),
            )

    # ---- failure blacklist ------------------------------------------------------------------

    @_locked
    def mark_failed(self, context: str, config_key: str, kind: str) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO failed_configs (context, config_key, kind, count, last_ts)"
                " VALUES (?, ?, ?, 1, ?) ON CONFLICT(context, config_key) DO UPDATE SET"
                " count = count + 1, kind = excluded.kind, last_ts = excluded.last_ts",
                (context, config_key, kind[:64], time.time()),
            )

    @_locked
    def blacklist(self, context: str) -> frozenset[str]:
        """Configs that failed recently (within TTL) or repeatedly (permanent)."""
        cutoff = time.time() - self._failure_ttl
        rows = self._db.execute(
            "SELECT config_key FROM failed_configs WHERE context = ?"
            " AND (last_ts >= ? OR count >= ?)",
            (context, cutoff, self._permanent),
        ).fetchall()
        return frozenset(r[0] for r in rows if isinstance(r[0], str))
