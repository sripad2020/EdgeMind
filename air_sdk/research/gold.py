"""Online checks (what the system sees) and gold scoring (the truth, offline only).

Online check per task - the deterministic check EdgeMind uses at answer time:
    sql: compiles against the schema | math: a final number is present
    json: parses and has the required keys | code: parses

Gold score per task - used only to evaluate, never to choose an answer:
    sql:  execution accuracy - predicted and gold queries are run on the read-only database
          and their results compared
    math: the final number equals the gold number
    json: every gold field is present with the same value
    code: the item's assert tests pass, run inside a locked-down Docker container
          (no network, read-only filesystem, memory/CPU/process limits, non-root,
          5 s timeout). Without Docker the gold score is *unavailable* (None), never run
          in a plain subprocess.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..skills.verify import check_json, check_math, check_python, check_sql
from .items import Item

SQL_TIME_BUDGET_S: Final[float] = 2.0
SQL_MAX_ROWS: Final[int] = 10_000
CODE_TIMEOUT_S: Final[int] = 5
DOCKER_IMAGE: Final[str] = "python:3.12-alpine"


@dataclass(frozen=True, slots=True)
class Score:
    ok: bool | None  # None = could not be scored (e.g. no sandbox available)
    detail: str


# ---- online checks ---------------------------------------------------------------------------


def online_check(item: Item, answer: str) -> Score:
    if item.task == "sql":
        c = check_sql(answer, item.context)
    elif item.task == "math":
        c = check_math(answer, None)
    elif item.task == "json":
        c = check_json(answer, item.context)
    else:
        c = check_python(answer, None)
    return Score(c.ok, c.detail)


# ---- gold: SQL ---------------------------------------------------------------------------------

_READ_ONLY_ACTIONS: Final = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                                       sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE})


def _read_only(action: int, *_args: object) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ONLY_ACTIONS else sqlite3.SQLITE_DENY


def _norm(value: object) -> object:
    if isinstance(value, float):
        return round(value, 2) if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value.strip()
    return value


def run_query(db_path: Path, query: str) -> list[tuple[object, ...]]:
    """Run one read-only query with a time budget and a row cap."""
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        db.set_authorizer(_read_only)
        deadline = time.monotonic() + SQL_TIME_BUDGET_S
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cur = db.execute(query)  # sqlite3 refuses more than one statement
        rows = cur.fetchmany(SQL_MAX_ROWS + 1)
        if len(rows) > SQL_MAX_ROWS:
            raise sqlite3.OperationalError("result too large")
        return [tuple(_norm(v) for v in row) for row in rows]
    finally:
        db.close()


def gold_sql(item: Item, answer: str, db_dir: Path) -> Score:
    db_path = db_dir / f"{item.gold['db']}.sqlite"
    if not db_path.is_file():
        return Score(None, f"database {db_path.name} not found")
    c = check_sql(answer, item.context)
    if not c.extracted:
        return Score(False, "no SQL in the answer")
    try:
        predicted = run_query(db_path, c.extracted.rstrip().rstrip(";"))
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return Score(False, f"query failed: {exc}")
    expected = run_query(db_path, str(item.gold["query"]))
    ordered = bool(re.search(r"\border\s+by\b", str(item.gold["query"]), re.I))
    same = predicted == expected if ordered else sorted(map(repr, predicted)) == sorted(
        map(repr, expected))
    return Score(same, "result matches" if same else
                 f"result differs ({len(predicted)} vs {len(expected)} rows)")


# ---- gold: maths and JSON -------------------------------------------------------------------


def gold_math(item: Item, answer: str) -> Score:
    c = check_math(answer, str(item.gold["answer"]))
    return Score(c.ok, c.detail)


def _same(a: object, b: object) -> bool:
    if (isinstance(a, (int, float)) and not isinstance(a, bool)) or \
            (isinstance(b, (int, float)) and not isinstance(b, bool)):
        try:
            return abs(float(str(a).replace(",", "")) - float(str(b).replace(",", ""))) < 1e-6
        except ValueError:
            return False
    return str(a).strip().casefold() == str(b).strip().casefold()


def gold_json(item: Item, answer: str) -> Score:
    c = check_json(answer, None)
    if not c.ok:
        return Score(False, c.detail)
    try:
        data = json.loads(c.extracted)
    except json.JSONDecodeError:
        return Score(False, "invalid JSON")
    fields = item.gold["fields"]
    if not isinstance(data, dict) or not isinstance(fields, dict):
        return Score(False, "not a JSON object")
    wrong = [k for k, v in fields.items() if k not in data or not _same(data[k], v)]
    return Score(not wrong, "all fields match" if not wrong else
                 "wrong or missing: " + ", ".join(wrong))


# ---- gold: code, in a Docker sandbox --------------------------------------------------------

_RUNNER: Final[str] = """import signal, sys
signal.alarm({timeout})
ns = {{}}
exec(compile(open('/w/answer.py').read(), 'answer.py', 'exec'), ns)
exec(compile(open('/w/tests.py').read(), 'tests.py', 'exec'), ns)
print('ALL_TESTS_PASSED')
"""


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            ["docker", "image", "inspect", DOCKER_IMAGE],  # noqa: S607
            capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def gold_code(item: Item, answer: str, *, sandbox: bool) -> Score:
    c = check_python(answer, None)
    if not c.ok:
        return Score(False, c.detail)
    if not sandbox:
        return Score(None, f"no sandbox (needs Docker with image {DOCKER_IMAGE})")
    name = f"edgemind-gold-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="edgemind-gold-") as tmp:
        work = Path(tmp)
        (work / "answer.py").write_text(c.extracted, encoding="utf-8")
        (work / "tests.py").write_text(str(item.gold["tests"]), encoding="utf-8")
        (work / "run.py").write_text(_RUNNER.format(timeout=CODE_TIMEOUT_S), encoding="utf-8")
        cmd = ["docker", "run", "--rm", "--name", name, "--pull", "never",
               "--network", "none", "--read-only", "--memory", "256m", "--cpus", "1",
               "--pids-limit", "64", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--user", "65534:65534",
               "--tmpfs", "/tmp:size=16m",  # noqa: S108 - inside the container
               "-v", f"{work}:/w:ro", DOCKER_IMAGE, "python", "-I", "/w/run.py"]
        try:
            # Fixed argument list (no shell); the untrusted code only runs in the container.
            proc = subprocess.run(cmd, capture_output=True, text=True,  # noqa: S603
                                  timeout=CODE_TIMEOUT_S + 25, check=False)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(  # noqa: S603 - fixed argument list, no shell
                    ["docker", "kill", name], capture_output=True,  # noqa: S607
                    timeout=20, check=False)
            return Score(False, "timed out")
    if "ALL_TESTS_PASSED" in proc.stdout:
        return Score(True, "tests pass")
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["failed"]
    return Score(False, tail[0][:200])


def gold_score(item: Item, answer: str, *, db_dir: Path, sandbox: bool) -> Score:
    if item.task == "sql":
        return gold_sql(item, answer, db_dir)
    if item.task == "math":
        return gold_math(item, answer)
    if item.task == "json":
        return gold_json(item, answer)
    return gold_code(item, answer, sandbox=sandbox)
