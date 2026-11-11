"""Prompt building under a token budget: send only what matters for this message.

Storage is cheap, prompt tokens are not (they cost time on every reply and must fit the
context). So the prompt gets:

* the **global** memory (preferences)                       - at most ``GLOBAL_CAP`` tokens
* the **active section's** facts, ranked by relevance        - at most ``SECTION_CAP`` tokens
  and at most ``FACT_SHARE`` of the context:
    rank 0  tables named in the message (compact form) and terms it uses
    rank 1  tables linked to those by a foreign key
    rank 2  everything else in the section (other tables, terms, code names)
    rank 3  just the names of tables that did not fit
* the **active section's** recent turns, newest first, in whatever room is left
* the message itself (never cut)

Safety ladder, used only when it does not fit: (1) drop rank >= 2 facts, (2) drop older
turns, (3) the runtime grows the active context up to its maximum (budgeting is done
against that maximum), (4) last resort: the bare question with a notice.

Answers are checked against the *full* stored facts, not this trimmed prompt.
Token counts are estimated (about 3 characters per token, deliberately on the high side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from .facts import MessageFacts
from .sections import Section, Turn

FACT_SHARE: Final[float] = 0.25
GLOBAL_CAP: Final[int] = 100
SECTION_CAP: Final[int] = 900
RESERVE: Final[int] = 48  # chat template overhead
_MIN_TURN_ROOM: Final[int] = 64


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3) + 1 if text else 0


@dataclass(frozen=True, slots=True)
class Fact:
    rank: int
    kind: str  # "table" | "tablename" | "term" | "code" | "keys" | "preference"
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text) + 1


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: list[dict[str, object]]
    facts: tuple[Fact, ...]  # facts that were sent
    facts_dropped: int
    turns_sent: int
    turns_dropped: int
    prompt_tokens: int  # estimate
    budget: int  # tokens available for the prompt
    rung: str  # "full" | "ranked facts" | "dropped facts" | "trimmed turns" | "minimal"
    notice: str

    def as_dict(self) -> dict[str, object]:
        return {"facts_sent": len(self.facts), "facts_dropped": self.facts_dropped,
                "turns_sent": self.turns_sent, "turns_dropped": self.turns_dropped,
                "prompt_tokens": self.prompt_tokens, "budget": self.budget,
                "rung": self.rung, "notice": self.notice}


def rank_facts(section: Section, facts: MessageFacts, *, include_keys: bool) -> list[Fact]:
    """The section's facts, most relevant to this message first."""
    words = facts.tokens | {i.lower() for i in facts.identifiers}
    text_l = facts.lowered
    out: list[Fact] = []
    tables = section.schema.tables
    lower = {n.lower(): n for n in tables}
    named = {lower[w] for w in words if w in lower}
    named |= {lower[w[:-1]] for w in words if w.endswith("s") and w[:-1] in lower}
    named |= {lower[w + "s"] for w in words if w + "s" in lower}
    linked = set().union(*(section.schema.neighbours(n) for n in named)) - named if named \
        else set()
    for name, table in tables.items():
        if name in named:
            out.append(Fact(0, "table", table.compact()))
        elif name in linked:
            out.append(Fact(1, "table", table.compact()))
        elif not named:  # nothing named: every table may matter, most recent first
            out.append(Fact(2, "table", table.compact()))
        else:  # unrelated to this question: its name is enough
            out.append(Fact(3, "tablename", name))
    for term, meaning in section.definitions.items():
        out.append(Fact(0 if term in text_l else 2, "term", f'"{term}" means {meaning}'))
    for name, sig in section.code_names.items():
        out.append(Fact(0 if name.lower() in words else 2, "code", sig))
    if include_keys and section.json_keys:
        out.append(Fact(0, "keys", "required keys: " + ", ".join(section.json_keys)))
    out.sort(key=lambda f: f.rank)  # stable: keeps table order within a rank
    return out


def _render(prefs: list[Fact], facts: list[Fact], skipped_tables: list[str]) -> str:
    lines: list[str] = []
    if prefs:
        lines.append("User preferences:")
        lines += [f"- {p.text}" for p in prefs]
    tables = [f.text for f in facts if f.kind == "table"]
    others = [f.text for f in facts if f.kind == "tablename"] + skipped_tables
    if tables or others:
        lines.append("Database schema (tables relevant to this question):")
        lines += tables
        if others:
            lines.append("Other tables: " + ", ".join(others))
    other = [f for f in facts if f.kind not in ("table", "tablename")]
    if other:
        lines.append("Known facts:")
        lines += [f"- {f.text}" for f in other]
    return "\n".join(lines)


def _pick(items: list[Fact], cap: int) -> tuple[list[Fact], list[Fact]]:
    used, dropped, spent = [], [], 0
    for f in items:
        if spent + f.tokens <= cap:
            used.append(f)
            spent += f.tokens
        else:
            dropped.append(f)
    return used, dropped


def build_prompt(
    *,
    question: str,
    system: str | None,
    preferences: list[str],
    section: Section,
    facts: MessageFacts,
    context_limit: int,
    max_tokens: int,
    include_keys: bool = True,
) -> PromptPlan:
    budget = max(0, context_limit - max_tokens - RESERVE)
    fixed = estimate_tokens(question) + estimate_tokens(system or "")
    fact_cap = min(GLOBAL_CAP + SECTION_CAP, int(context_limit * FACT_SHARE))

    prefs, _ = _pick([Fact(0, "preference", p) for p in preferences], min(GLOBAL_CAP, fact_cap))
    ranked = rank_facts(section, facts, include_keys=include_keys)
    room = min(SECTION_CAP, fact_cap - sum(p.tokens for p in prefs))
    used, dropped = _pick(ranked, max(0, room))
    rung = "ranked facts" if dropped else "full"

    def facts_tokens() -> int:
        return estimate_tokens(_render(prefs, used, [])) + 4 * len(dropped)

    # Ladder step 1: not enough room for any turns -> drop the less relevant facts.
    if budget - fixed - facts_tokens() < _MIN_TURN_ROOM and any(f.rank >= 2 for f in used):
        dropped += [f for f in used if f.rank >= 2]
        used = [f for f in used if f.rank < 2]
        rung = "dropped facts"

    skipped = [f.text.split("(", 1)[0] for f in dropped if f.kind == "table"][:24]
    block = _render(prefs, used, skipped)
    room_for_turns = budget - fixed - estimate_tokens(block)

    # Step 4 (last resort): even the essentials do not fit -> bare question.
    if room_for_turns < 0:
        return PromptPlan(
            _assemble(system, "", [], question), (), len(ranked) + len(prefs), 0,
            len(section.turns), fixed, budget, "minimal",
            "The conversation memory did not fit the model's context; answered without it.")

    # Step 2: newest turns first, as many as fit.
    turns: list[Turn] = []
    spent = 0
    for t in reversed(section.turns):
        cost = estimate_tokens(t.user) + estimate_tokens(t.assistant) + 8
        if spent + cost > room_for_turns:
            break
        turns.append(t)
        spent += cost
    turns.reverse()
    turns_dropped = len(section.turns) - len(turns)
    if turns_dropped and rung == "full":
        rung = "trimmed turns"
    return PromptPlan(_assemble(system, block, turns, question), tuple(prefs + used),
                      len(dropped), len(turns), turns_dropped,
                      fixed + estimate_tokens(block) + spent, budget, rung, "")


def _assemble(system: str | None, block: str, turns: list[Turn],
              question: str) -> list[dict[str, object]]:
    sys_text = "\n\n".join(p for p in (system or "", block) if p)
    msgs: list[dict[str, object]] = []
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})
    for t in turns:
        msgs.append({"role": "user", "content": t.user})
        msgs.append({"role": "assistant", "content": t.assistant})
    msgs.append({"role": "user", "content": question})
    return msgs
