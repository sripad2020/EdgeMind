"""Task auto-detection: which skill a message needs, from fixed scoring rules.

Each skill has weighted cues (words, code shapes, facts found in the message, tables known
in the conversation). The highest score wins; below ``STRONG`` the detection is "weak" and
the conversation keeps its current task (hysteresis), so a short follow-up such as "now sort
it by date" stays SQL instead of flipping to general chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from .facts import MessageFacts

STRONG: Final[float] = 2.0

_CUES: Final[dict[str, tuple[tuple[re.Pattern[str], float], ...]]] = {
    "sql": (
        (re.compile(r"\bsql(?:ite)?\b|\bquery\b|\bqueries\b", re.I), 2.0),
        (re.compile(r"\b(?:select|join|group\s+by|order\s+by|where\s+clause|inner\s+join"
                    r"|left\s+join|having)\b", re.I), 1.5),
        (re.compile(r"\b(?:table|tables|column|columns|database|db|rows?)\b", re.I), 1.0),
    ),
    "python": (
        (re.compile(r"\bpython\b|```py(?:thon)?\b", re.I), 2.5),
        (re.compile(r"\b(?:function|def|class|script|code|program|implement|method|"
                    r"algorithm|list comprehension|decorator|regex)\b", re.I), 1.0),
        (re.compile(r"^\s*(?:def|class|import|from)\s+\w", re.M), 2.0),
    ),
    "json": (
        (re.compile(r"\bjson\b", re.I), 2.5),
        (re.compile(r"\b(?:keys?|fields?)\b", re.I), 0.5),
    ),
    "math": (
        (re.compile(r"\b(?:calculate|compute|solve|evaluate|how much is|sum of|product of|"
                    r"percent(?:age)?|average of)\b", re.I), 1.5),
        (re.compile(r"\bwhat\s+is\s+[-\d(]", re.I), 1.0),
    ),
    "summarise": (
        (re.compile(r"\bsummari[sz]e\b|\bsummary\b|\btl;?dr\b|\bkey points\b|\bin short\b",
                    re.I), 3.0),
    ),
}
_OPENER: Final = re.compile(
    r"^\s*(?:and|also|now|then|but|ok(?:ay)?|so|what about|how about|same|instead|again|"
    r"make it|change it|do it|fix it|rewrite it|add|remove|only|sort|order|filter)\b", re.I)
_REFERS: Final = re.compile(
    r"\b(?:it|this|them|those|these|same|previous|above|instead)\b", re.I)


@dataclass(frozen=True, slots=True)
class Detection:
    skill: str
    score: float
    scores: dict[str, float]

    @property
    def strong(self) -> bool:
        return self.score >= STRONG


def detect(text: str, facts: MessageFacts, known_tables: frozenset[str] = frozenset()) -> Detection:
    """Score every skill for one message; ``general`` when nothing scores."""
    t = text[:4000]
    scores = {name: sum(w for p, w in cues if p.search(t)) for name, cues in _CUES.items()}
    if facts.ddl:
        scores["sql"] += 2.0
    if known_tables and known_tables & facts.tokens:
        scores["sql"] += 1.5
    if facts.code_names:
        scores["python"] += 1.0
    if facts.json_keys and scores["json"] > 0:
        scores["json"] += 0.5
    if facts.math is not None:
        scores["math"] += 2.0
    if len(text) > 1500 and scores["summarise"] == 0 and max(scores.values()) < STRONG:
        scores["summarise"] += 1.0  # a long pasted text with no other task
    best = max(scores, key=lambda k: scores[k])
    if scores[best] <= 0:
        return Detection("general", 0.0, scores)
    return Detection(best, scores[best], scores)


def opens_follow_up(text: str) -> bool:
    """Starts like a continuation: "and ...", "now ...", "make it ...", "only ..."."""
    return bool(_OPENER.match(text[:300]))


def is_follow_up(text: str) -> bool:
    """Short message that continues or refers back ("now sort it", "add tests to it")."""
    return len(text.split()) <= 14 and (opens_follow_up(text) or
                                        bool(_REFERS.search(text[:300])))
