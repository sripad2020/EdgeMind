"""Sectioned memory: one section per piece of work, plus a small global memory.

A conversation that moves between tasks ("the orders database", "a Python helper",
"a budget calculation") keeps each task's facts and turns in its own **section**.
Only the active section is ever sent to the model, so switching topics never drags the
other topics' facts into the prompt.

Routing a message (deterministic, in this order):

1. explicit cue "new task: ..."            -> a new section
2. explicit cue "back to / switch to X"    -> the section that best matches X
3. shared identifiers (table / function / defined term names) - weighted 3 each,
   shared keywords 0.5 each (max 3), +1 for the active section; best >= ``ROUTE_MIN`` wins
   (ties go to the most recently used)
4. a short follow-up, or the same task as the active section -> the active section
5. otherwise                                -> a new section

Lifecycle: a section untouched for ``DORMANT_AFTER`` turns is *dormant* (kept, not sent);
after ``EXPIRE_AFTER`` turns it *expires* (deleted). At most ``MAX_SECTIONS`` exist; the
least recently used one is removed first. Turns per section, facts per section and
preferences are all capped, so memory per session is bounded.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Final

from .detect import Detection, is_follow_up, opens_follow_up
from .facts import (
    MAX_DEFINITIONS,
    MAX_NAMES,
    MAX_PREFERENCES,
    BoundedDict,
    MessageFacts,
    Schema,
    extract_code_names,
    keywords,
)

MAX_SECTIONS: Final[int] = 8
MAX_TURNS: Final[int] = 24
MAX_KEYWORDS: Final[int] = 256
DORMANT_AFTER: Final[int] = 6
EXPIRE_AFTER: Final[int] = 60
ROUTE_MIN: Final[float] = 2.0
MAX_TURN_CHARS: Final[int] = 8_000

_NEW_CUE: Final = re.compile(
    r"^\s*(?:new\s+(?:task|topic|question|section|problem)\b\s*[:\-,.]?|"
    r"(?:let'?s\s+)?(?:switch|change)\s+(?:the\s+)?(?:topic|subject)\b\s*[:\-,.]?|"
    r"different\s+(?:question|topic)\b\s*[:\-,.]?)", re.I)
_BACK_CUE: Final = re.compile(
    r"\b(?:(?:go(?:ing)?\s+)?back\s+to|switch(?:ing)?\s+(?:back\s+)?to|return(?:ing)?\s+to)"
    r"\s+(?:the\s+|my\s+|our\s+)?([^.,;:?!\n]{1,60})", re.I)


@dataclass(frozen=True, slots=True)
class Turn:
    user: str
    assistant: str
    skill: str


@dataclass(slots=True)
class Section:
    id: int
    title: str
    skill: str
    created_tick: int
    last_tick: int
    schema: Schema = field(default_factory=Schema)
    definitions: BoundedDict = field(default_factory=lambda: BoundedDict(MAX_DEFINITIONS))
    code_names: BoundedDict = field(default_factory=lambda: BoundedDict(MAX_NAMES))
    json_keys: tuple[str, ...] = ()
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    keywords: Counter[str] = field(default_factory=Counter)
    corrections: int = 0

    @property
    def label(self) -> str:
        return f"{self.title} ({self.skill})"

    def identifiers(self) -> set[str]:
        return ({t.lower() for t in self.schema.tables} | {n.lower() for n in self.code_names}
                | set(self.definitions))

    def add_keywords(self, words: list[str]) -> None:
        self.keywords.update(words)
        if len(self.keywords) > MAX_KEYWORDS:  # keep the most frequent ones
            self.keywords = Counter(dict(self.keywords.most_common(MAX_KEYWORDS // 2)))

    def apply(self, facts: MessageFacts) -> int:
        """Store a message's facts in this section; returns how many facts changed."""
        changed = self.schema.apply(facts.ddl)
        for term, meaning in facts.definitions.items():
            self.definitions.put(term, meaning)
        for name, sig in facts.code_names.items():
            self.code_names.put(name, sig)
        if facts.json_keys:
            self.json_keys = facts.json_keys
        self.add_keywords(facts.words)
        return changed + len(facts.definitions) + len(facts.code_names) + bool(facts.json_keys)

    def as_dict(self, state: str) -> dict[str, object]:
        return {
            "id": self.id, "title": self.title, "skill": self.skill, "state": state,
            "turns": len(self.turns), "tables": sorted(self.schema.tables),
            "schema_version": self.schema.version, "definitions": dict(self.definitions),
            "code_names": list(self.code_names), "json_keys": list(self.json_keys),
            "corrections": self.corrections,
        }


def strip_cue(text: str) -> str:
    """The message without a leading routing cue ("New task: ..."), for the model."""
    m = _NEW_CUE.match(text)
    return (text[m.end():].strip() or text) if m else text


def _title(text: str, facts: MessageFacts) -> str:
    tables = [n for n in facts.identifiers if any(n in s.lower() for s in facts.ddl)]
    if tables:
        return "Tables: " + ", ".join(sorted(tables)[:4])
    if facts.code_names:
        return "Code: " + ", ".join(list(facts.code_names)[:3])
    if facts.math is not None:
        return f"Maths: {facts.math[0][:40]}"
    words = keywords(strip_cue(text), limit=5)
    return " ".join(words).capitalize() if words else "Chat"


class SessionMemory:
    """All memory of one conversation: global preferences plus sections.

    Thread-safe: methods take ``lock``; callers that need several steps to be atomic
    (prepare -> generate -> commit) can hold it too (it is re-entrant)."""

    def __init__(self, *, max_sections: int = MAX_SECTIONS) -> None:
        self.lock = threading.RLock()
        self.max_sections = max(1, max_sections)
        self.preferences: BoundedDict = BoundedDict(MAX_PREFERENCES)
        self.sections: dict[int, Section] = {}
        self.active: int | None = None
        self.tick = 0
        self._next_id = 1
        self.version = 0  # bumps on every change (for optimistic checks by a server store)
        self.touched = time.monotonic()

    # -- lifecycle --------------------------------------------------------------------------

    def state(self, section: Section) -> str:
        if section.id == self.active:
            return "active"
        return "dormant" if self.tick - section.last_tick >= DORMANT_AFTER else "recent"

    def _expire(self) -> None:
        for sid in [s.id for s in self.sections.values()
                    if s.id != self.active and self.tick - s.last_tick >= EXPIRE_AFTER]:
            del self.sections[sid]
        while len(self.sections) > self.max_sections:
            victims = [s for s in self.sections.values() if s.id != self.active]
            if not victims:
                break
            del self.sections[min(victims, key=lambda s: s.last_tick).id]

    def _new(self, text: str, facts: MessageFacts, skill: str) -> Section:
        s = Section(self._next_id, _title(text, facts), skill, self.tick, self.tick)
        self._next_id += 1
        self.sections[s.id] = s
        return s

    # -- routing ----------------------------------------------------------------------------

    def _score(self, s: Section, idents: set[str], words: set[str], *, bonus: bool) -> float:
        shared = len(idents & s.identifiers())
        kw = min(3.0, 0.5 * sum(1 for w in words if w in s.keywords))
        return 3.0 * shared + kw + (1.0 if bonus and s.id == self.active else 0.0)

    def _best(self, idents: set[str], words: set[str]) -> tuple[Section, float] | None:
        """Highest-scoring section; ties go to the most recently used."""
        best: tuple[Section, float] | None = None
        for s in self.sections.values():
            sc = self._score(s, idents, words, bonus=True)
            if best is None or sc > best[1] or (
                    sc == best[1] and s.last_tick > best[0].last_tick):
                best = (s, sc)
        return best

    def route(self, text: str, facts: MessageFacts, detection: Detection) -> tuple[Section, str]:
        """Pick (or create) the section for a message and make it active."""
        with self.lock:
            self.tick += 1
            self.version += 1
            self.touched = time.monotonic()
            section, why = self._route(text, facts, detection)
            section.last_tick = self.tick
            self.active = section.id
            self._expire()
            return section, why

    def _route(self, text: str, facts: MessageFacts, d: Detection) -> tuple[Section, str]:
        words = set(facts.words)
        known = set().union(*(s.identifiers() for s in self.sections.values()))
        idents = facts.identifiers | (facts.tokens & known)
        if _NEW_CUE.match(text) or not self.sections:
            return self._new(text, facts, d.skill), "new task"
        back = _BACK_CUE.search(text)
        if back:
            target = set(keywords(back.group(1)))
            found = None
            for s in self.sections.values():
                pool = s.identifiers() | set(s.keywords) | set(keywords(s.title))
                hits = len(target & pool)
                if hits and (found is None or hits > found[1]):
                    found = (s, hits)
            if found:
                return found[0], f"back to '{found[0].title}'"
        def task_change(s: Section) -> bool:
            # Clearly a different task, with no shared names and not phrased as a
            # continuation: shared everyday words are not enough to stay.
            return (d.strong and d.skill != s.skill and not (idents & s.identifiers())
                    and not opens_follow_up(text))

        best = self._best(idents, words)
        active = self.sections.get(self.active) if self.active is not None else None
        if best and (best[1] >= ROUTE_MIN or (idents & best[0].identifiers())) \
                and not task_change(best[0]):
            return best[0], "same task" if best[0] is active else "matched names/keywords"
        if active is not None and not task_change(active) and (
                is_follow_up(text) or d.skill == active.skill
                or (not d.strong and active.skill == "general")):
            return active, "follow-up"
        return self._new(text, facts, d.skill), "new task"

    # -- updates ----------------------------------------------------------------------------

    def remember_preferences(self, prefs: list[str]) -> None:
        with self.lock:
            for p in prefs:
                self.preferences.put(p.lower(), p)

    def commit(self, section_id: int, user: str, assistant: str, skill: str) -> None:
        """Store a finished exchange in its section (names in the answer become facts)."""
        with self.lock:
            s = self.sections.get(section_id)
            if s is None:  # expired meanwhile
                return
            s.turns.append(Turn(user[:MAX_TURN_CHARS], assistant[:MAX_TURN_CHARS], skill))
            for name, sig in extract_code_names(assistant).items():
                s.code_names.put(name, sig)
            s.add_keywords(keywords(assistant, limit=32))
            self.version += 1

    def forget(self) -> None:
        with self.lock:
            self.sections.clear()
            self.preferences.clear()
            self.active = None
            self.version += 1

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "active": self.active,
                "preferences": list(self.preferences.values()),
                "sections": [s.as_dict(self.state(s)) for s in sorted(
                    self.sections.values(), key=lambda s: -s.last_tick)],
            }
