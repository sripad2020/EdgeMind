"""Test items: one JSONL file per task, with a deterministic profile / router split.

File format (one JSON object per line; the optional first line is metadata)::

    {"_meta": {"task": "math", "source": "GSM8K test split", "license": "MIT"}}
    {"id": "math-001", "task": "math", "prompt": "What is 17 * 23?", "gold": {"answer": "391"}}

``gold`` per task:

* ``sql``  - ``{"query": "<gold SQL>", "db": "<database name>"}`` and ``context`` = schema
* ``math`` - ``{"answer": "<number>"}``
* ``json`` - ``{"fields": {...}}``; ``context`` = required keys
* ``code`` - ``{"tests": "<assert lines>", "entry": "<function name>"}``

The split is by a hash of the item id (not by position), so it never changes when items
are added, and the router never sees profile items.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from ..errors import ValidationError

TASKS: Final[frozenset[str]] = frozenset({"sql", "math", "json", "code"})
MAX_ITEMS: Final[int] = 50_000
MAX_LINE: Final[int] = 200_000
PROFILE_SHARE: Final[float] = 0.8


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    task: str
    prompt: str
    gold: dict[str, object]
    context: str | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def split(self) -> str:
        """'profile' (80%) or 'router' (20%), fixed by the id's hash."""
        h = int.from_bytes(hashlib.sha256(self.id.encode()).digest()[:8], "big")
        return "profile" if (h % 10_000) / 10_000 < PROFILE_SHARE else "router"

    def to_json(self) -> str:
        out: dict[str, object] = {"id": self.id, "task": self.task, "prompt": self.prompt,
                                  "gold": self.gold}
        if self.context is not None:
            out["context"] = self.context
        return json.dumps(out, ensure_ascii=False)


def _required(gold: dict[str, object], task: str) -> tuple[str, ...]:
    return {"sql": ("query", "db"), "math": ("answer",), "json": ("fields",),
            "code": ("tests", "entry")}[task]


def parse_item(obj: object) -> Item:
    if not isinstance(obj, dict):
        raise ValidationError("item must be a JSON object")
    iid, task, prompt, gold = obj.get("id"), obj.get("task"), obj.get("prompt"), obj.get("gold")
    if not isinstance(iid, str) or not 1 <= len(iid) <= 128:
        raise ValidationError("item id must be a string of 1-128 characters")
    if task not in TASKS:
        raise ValidationError(f"{iid}: task must be one of {sorted(TASKS)}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError(f"{iid}: prompt must be a non-empty string")
    if not isinstance(gold, dict):
        raise ValidationError(f"{iid}: gold must be an object")
    missing = [k for k in _required(gold, str(task)) if k not in gold]
    if missing:
        raise ValidationError(f"{iid}: gold is missing {missing}")
    context = obj.get("context")
    if context is not None and not isinstance(context, str):
        raise ValidationError(f"{iid}: context must be a string")
    return Item(iid, str(task), prompt, dict(gold), context)


def read_items(path: Path) -> tuple[dict[str, object], list[Item]]:
    """Read a JSONL file -> (metadata, items). Ids must be unique."""
    meta: dict[str, object] = {}
    items: list[Item] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if len(line) > MAX_LINE:
                raise ValidationError(f"{path.name}:{n}: line too long")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path.name}:{n}: invalid JSON ({exc.msg})") from exc
            if isinstance(obj, dict) and "_meta" in obj:
                if isinstance(obj["_meta"], dict):
                    meta = dict(obj["_meta"])
                continue
            item = parse_item(obj)
            if item.id in seen:
                raise ValidationError(f"{path.name}:{n}: duplicate id {item.id}")
            seen.add(item.id)
            items.append(item)
            if len(items) > MAX_ITEMS:
                raise ValidationError(f"{path.name}: more than {MAX_ITEMS} items")
    return meta, items


def write_items(path: Path, items: Iterable[Item], meta: dict[str, object]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"_meta": meta}) + "\n")
        for item in items:
            fh.write(item.to_json() + "\n")
            n += 1
    tmp.replace(path)
    return n


def select(items: Iterable[Item], *, split: str | None = None,
           limit: int | None = None) -> Iterator[Item]:
    """Items of one split (or all), at most ``limit`` (in file order)."""
    count = 0
    for item in items:
        if split is not None and item.split != split:
            continue
        if limit is not None and count >= limit:
            return
        count += 1
        yield item
