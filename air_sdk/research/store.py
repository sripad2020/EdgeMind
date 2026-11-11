"""Experiment results: one SQLite file per experiment, and the profile grid built from it.

Tables:
    profiles(name, base, kind, precision, adapter, groups, size_bytes, baseline, meta)
    results(profile, item_id, task, split, check_ok, gold_ok, gold_detail, seconds,
            ttft_s, completion_tokens, prompt_tokens, answer, created)

Every (profile, item) pair has at most one row, so a run can be resumed or repeated.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .stats import Rate, qualified, wilson

MAX_ANSWER: Final[int] = 8_000

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS profiles (
    name TEXT PRIMARY KEY, base TEXT NOT NULL, kind TEXT NOT NULL, precision TEXT,
    adapter TEXT, groups TEXT, size_bytes INTEGER, baseline INTEGER NOT NULL DEFAULT 0,
    meta TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS results (
    profile TEXT NOT NULL, item_id TEXT NOT NULL, task TEXT NOT NULL, split TEXT NOT NULL,
    check_ok INTEGER NOT NULL, gold_ok INTEGER, gold_detail TEXT, seconds REAL NOT NULL,
    ttft_s REAL, completion_tokens INTEGER, prompt_tokens INTEGER, answer TEXT,
    created REAL NOT NULL, PRIMARY KEY (profile, item_id));
CREATE INDEX IF NOT EXISTS results_task ON results (task, split);
"""


@dataclass(frozen=True, slots=True)
class Profile:
    name: str  # the runnable Ollama model tag
    base: str  # the base model it derives from
    kind: str  # "model" | "uniform" | "sensitivity" | "mix" | "adapter"
    precision: str = ""
    adapter: str = ""
    groups: str = ""  # layer groups kept at higher precision / quantized (sensitivity)
    size_bytes: int = 0
    baseline: bool = False
    meta: str = "{}"


@dataclass(frozen=True, slots=True)
class Result:
    profile: str
    item_id: str
    task: str
    split: str
    check_ok: bool
    gold_ok: bool | None
    gold_detail: str
    seconds: float
    ttft_s: float
    completion_tokens: int
    prompt_tokens: int
    answer: str


@dataclass(frozen=True, slots=True)
class Cell:
    """One grid cell: a profile's performance on one task (one split)."""

    profile: str
    task: str
    check: Rate
    gold: Rate  # items whose gold score is available
    mean_seconds: float
    seconds_per_correct: float | None
    qualified: bool

    def as_dict(self) -> dict[str, object]:
        return {"profile": self.profile, "task": self.task, "check": self.check.as_dict(),
                "gold": self.gold.as_dict(), "mean_seconds": round(self.mean_seconds, 3),
                "seconds_per_correct": None if self.seconds_per_correct is None
                else round(self.seconds_per_correct, 3), "qualified": self.qualified}


class ResultStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- profiles --------------------------------------------------------------------------

    def add_profile(self, p: Profile) -> None:
        with self._lock, self._db:
            if p.baseline:  # one baseline per base model
                self._db.execute("UPDATE profiles SET baseline = 0 WHERE base = ?", (p.base,))
            self._db.execute(
                "INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE "
                "SET base=excluded.base, kind=excluded.kind, precision=excluded.precision, "
                "adapter=excluded.adapter, groups=excluded.groups, "
                "size_bytes=excluded.size_bytes, baseline=excluded.baseline, meta=excluded.meta",
                (p.name, p.base, p.kind, p.precision, p.adapter, p.groups, p.size_bytes,
                 int(p.baseline), p.meta))

    def profiles(self) -> list[Profile]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM profiles ORDER BY base, name").fetchall()
        return [Profile(r[0], r[1], r[2], r[3] or "", r[4] or "", r[5] or "", r[6] or 0,
                        bool(r[7]), r[8]) for r in rows]

    def profile(self, name: str) -> Profile | None:
        return next((p for p in self.profiles() if p.name == name), None)

    # -- results ---------------------------------------------------------------------------

    def done(self, profile: str) -> set[str]:
        with self._lock:
            return {r[0] for r in self._db.execute(
                "SELECT item_id FROM results WHERE profile = ?", (profile,))}

    def add_result(self, r: Result) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.profile, r.item_id, r.task, r.split, int(r.check_ok),
                 None if r.gold_ok is None else int(r.gold_ok), r.gold_detail[:500],
                 r.seconds, r.ttft_s, r.completion_tokens, r.prompt_tokens,
                 r.answer[:MAX_ANSWER], time.time()))

    def results(self, *, profile: str | None = None, task: str | None = None,
                split: str | None = None) -> list[Result]:
        sql = ("SELECT profile, item_id, task, split, check_ok, gold_ok, gold_detail, seconds, "
               "ttft_s, completion_tokens, prompt_tokens, answer FROM results WHERE 1=1")
        args: list[object] = []
        for col, val in (("profile", profile), ("task", task), ("split", split)):
            if val is not None:
                sql += f" AND {col} = ?"
                args.append(val)
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY profile, item_id", args).fetchall()
        return [Result(r[0], r[1], r[2], r[3], bool(r[4]), None if r[5] is None else bool(r[5]),
                       r[6] or "", r[7], r[8] or 0.0, r[9] or 0, r[10] or 0, r[11] or "")
                for r in rows]

    # -- grid ------------------------------------------------------------------------------

    def grid(self, split: str | None = "profile") -> list[Cell]:
        """Every (profile, task) cell, qualified against the base model's baseline profile
        on gold accuracy (non-inferiority: lower bound of the 95% Wilson interval at most
        5 points below the baseline score)."""
        cells: dict[tuple[str, str], list[Result]] = {}
        for r in self.results(split=split):
            cells.setdefault((r.profile, r.task), []).append(r)
        profiles = {p.name: p for p in self.profiles()}
        raw: dict[tuple[str, str], Cell] = {}
        for (name, task), rows in cells.items():
            scored = [r for r in rows if r.gold_ok is not None]
            correct = sum(1 for r in scored if r.gold_ok)
            secs = sum(r.seconds for r in rows)
            raw[(name, task)] = Cell(
                name, task, wilson(sum(r.check_ok for r in rows), len(rows)),
                wilson(correct, len(scored)), secs / len(rows),
                secs / correct if correct else None, False)
        out: list[Cell] = []
        for (name, task), cell in sorted(raw.items()):
            p = profiles.get(name)
            base = next((b for b in profiles.values() if p and b.base == p.base and b.baseline),
                        None)
            ref = raw.get((base.name, task)) if base else None
            ok = bool(ref and base and name != base.name and qualified(cell.gold, ref.gold.value))
            out.append(Cell(cell.profile, task, cell.check, cell.gold, cell.mean_seconds,
                            cell.seconds_per_correct, ok))
        return out

    def export(self, path: Path) -> None:
        payload = {"profiles": [_asdict(p) for p in self.profiles()],
                   "grid": [c.as_dict() for c in self.grid()]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _asdict(p: Profile) -> dict[str, object]:
    return {f: getattr(p, f) for f in Profile.__dataclass_fields__}
