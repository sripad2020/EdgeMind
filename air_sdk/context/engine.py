"""Context engine: from a raw message to a small prompt plus the facts to check against.

    message --> extract facts --> detect task --> route to a section --> store facts
            --> build a budgeted prompt (active section only, relevant facts first)
            --> model answers (verified skills check against the FULL stored facts)
            --> commit the exchange to the section

Everything here is rules and counting; no model is trained and nothing is executed.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Final

from .._validate import require_str
from ..errors import ValidationError
from ..skills.profiles import get_skill
from .budget import PromptPlan, build_prompt
from .calc import format_number
from .detect import Detection, detect
from .facts import extract
from .sections import SessionMemory, strip_cue

AUTO: Final[str] = "auto"
MAX_MESSAGE: Final[int] = 100_000
MAX_CONTEXT: Final[int] = 20_000


def is_auto(skill: str | None) -> bool:
    return skill is None or (isinstance(skill, str) and skill.strip().lower() == AUTO)


@dataclass(frozen=True, slots=True)
class Prepared:
    """A message ready for the model, and what the answer will be checked against."""

    messages: list[dict[str, object]]
    skill: str
    auto: bool  # the task was detected (not given)
    detection: Detection
    section_id: int
    section: str  # label, e.g. "Tables: customers, orders (sql)"
    route: str  # why this section ("new task", "follow-up", "back to ...")
    check_context: str | None  # full facts for the check (schema / keys / expected value)
    bonus_schema: str | None  # schema for checking SQL inside a general answer
    plan: PromptPlan
    correction: bool  # the message corrects the previous answer
    question: str
    prompt_context: str | None = None  # labelled context for the skill (JSON keys)

    def as_dict(self) -> dict[str, object]:
        return {"skill": self.skill, "auto": self.auto, "section": self.section,
                "section_id": self.section_id, "route": self.route,
                "correction": self.correction, "budget": self.plan.as_dict()}


class ContextEngine:
    """Stateless helper; all state lives in the :class:`SessionMemory` passed in."""

    def prepare(
        self,
        memory: SessionMemory,
        text: str,
        *,
        skill: str | None = AUTO,
        context: str | None = None,
        system: str | None = None,
        context_limit: int = 4096,
        max_tokens: int = 512,
    ) -> Prepared:
        text = require_str("message", text, max_len=MAX_MESSAGE)
        if context is not None:
            context = require_str("context", context, max_len=MAX_CONTEXT, allow_empty=True)
        auto = is_auto(skill)
        explicit = None if auto else get_skill(skill).name  # validates the name
        facts = extract(text)
        # Caller-given context (e.g. a schema) is stored as facts but does not steer routing.
        given = extract("", context) if context else None
        with memory.lock:
            known = frozenset(n.lower() for s in memory.sections.values()
                              for n in s.schema.tables) | (given.identifiers if given else set())
            d = detect(text, facts, frozenset(known))
            section, why = memory.route(text, facts, d)
            if explicit is not None:
                name = explicit
            elif d.strong or not section.turns:
                name = d.skill
            else:
                name = section.skill  # weak signal: keep the section's task (hysteresis)
            section.skill = name
            section.apply(facts)
            if given is not None:
                section.apply(given)
            memory.remember_preferences(facts.preferences)
            correction = facts.correction and bool(section.turns)
            if correction:
                section.corrections += 1
            question = strip_cue(text)  # routing cues are for EdgeMind, not the model
            plan = build_prompt(
                question=question, system=system, preferences=list(memory.preferences.values()),
                section=section, facts=facts, context_limit=context_limit,
                max_tokens=max_tokens, include_keys=name == "json")
            schema = section.schema.full_sql() or None
            if name == "sql":
                check: str | None = schema or (context or None)
            elif name == "json":
                keys = facts.json_keys or section.json_keys
                check = context or (", ".join(keys) if keys else None)
            elif name == "math":
                check = context or (format_number(facts.math[1]) if facts.math else None)
            else:
                check = None
            return Prepared(plan.messages, name, auto, d, section.id,
                            f"{section.title} ({name})", why, check, schema, plan,
                            correction, question,
                            context if name == "json" and context else None)

    @staticmethod
    def commit(memory: SessionMemory, prepared: Prepared, answer: str) -> None:
        memory.commit(prepared.section_id, prepared.question, answer, prepared.skill)

    def replay(self, memory: SessionMemory, messages: list[dict[str, object]],
               *, max_messages: int = 200) -> None:
        """Rebuild memory from a plain chat history (user/assistant pairs), e.g. when a
        client sends the whole conversation instead of a conversation id."""
        pending: str | None = None
        for m in messages[-max_messages:]:
            role, content = m.get("role"), m.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "user":
                pending = content
            elif role == "assistant" and pending is not None:
                self.commit(memory, self.prepare(memory, pending), content)
                pending = None


# ---- per-user memory for servers -------------------------------------------------------------

_SESSION_ID: Final = re.compile(r"[A-Za-z0-9_-]{8,64}")


class SessionStore:
    """In-memory conversation memories keyed by a client-chosen id, for a web server.

    Bounded: idle sessions expire after ``ttl_s``; at most ``max_sessions`` are kept (the
    least recently used is dropped). Each :class:`SessionMemory` has its own lock, so users
    never block each other here. (A multi-process deployment would put this in Redis or
    SQLite behind the same interface.)"""

    def __init__(self, *, ttl_s: float = 1800.0, max_sessions: int = 1000) -> None:
        self._ttl = ttl_s
        self._max = max(1, max_sessions)
        self._lock = threading.Lock()
        self._items: dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            raise ValidationError("conversation id must be 8-64 letters, digits, '-' or '_'")
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            mem = self._items.get(session_id)
            if mem is None:
                while len(self._items) >= self._max:  # full: drop the least recently used
                    del self._items[min(self._items, key=lambda k: self._items[k].touched)]
                mem = self._items[session_id] = SessionMemory()
            mem.touched = now
            return mem

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._items.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._items)

    def _sweep(self, now: float) -> None:
        for sid in [k for k, m in self._items.items() if now - m.touched > self._ttl]:
            del self._items[sid]
