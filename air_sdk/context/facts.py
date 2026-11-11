"""Fact extraction: turn chat text into small, structured facts (rules only, no AI).

* **Schema**      - CREATE / ALTER / DROP TABLE statements, applied in order to a private
                    in-memory SQLite database, so the newest definition always wins
                    (versioned). Kept in compact form: ``orders(id, customer_id->customers)``.
* **Definitions** - "X means Y", "define X as Y", "by X I mean Y".
* **Code names**  - Python ``def`` / ``class`` names from questions and answers.
* **JSON keys**   - "with keys title, author and year".
* **Maths**       - the expected answer, computed by the safe calculator (``calc.py``).
* **Preferences** - "always ...", "never ...", "from now on ..." (global for the session).
* **Corrections** - "no, ...", "that's wrong", "actually ..." on the previous answer.

Security and bounds: every input is truncated; DDL statements run one at a time in an
in-memory database with ATTACH / DETACH / PRAGMA denied and a time budget; the number of
tables, columns, definitions and names kept is capped.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from .calc import expected_answer

MAX_TEXT: Final[int] = 50_000
MAX_STMT_CHARS: Final[int] = 8_000
MAX_DDL_PER_MESSAGE: Final[int] = 64
MAX_TABLES: Final[int] = 64
MAX_COLUMNS: Final[int] = 128
MAX_DEFINITIONS: Final[int] = 32
MAX_NAMES: Final[int] = 32
MAX_KEYS: Final[int] = 16
MAX_PREFERENCES: Final[int] = 8
_DDL_TIME_BUDGET_S: Final[float] = 0.5

_STOP_TEXT: Final[str] = """
a an the and or but if then else of to in on at by for with from as is are was were be been
it its this that these those there here what which who whom whose how why when where can
could should would will shall may might must do does did done have has had not no yes you
your yours me my mine we our ours they them their i he she his her him please thanks thank
also just now new some any all each every more most less few many much very so such than
too into onto over under about above below between give show tell write make get use using
want need like one two out up down only same other another again let lets okay ok hi hello
"""
_STOP: Final[frozenset[str]] = frozenset(_STOP_TEXT.split())

_WORD: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_TOKEN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
MAX_TOKENS_SCAN: Final[int] = 4_000


def keywords(text: str, limit: int = 64) -> list[str]:
    """Meaningful lower-case words of a text (stop words removed), first ``limit``."""
    out: list[str] = []
    for m in _WORD.finditer(text[:MAX_TEXT]):
        w = m.group(0).lower()
        if w not in _STOP:
            out.append(w)
            if len(out) >= limit:
                break
    return out


# ---- schema ----------------------------------------------------------------------------------

_DDL_START: Final = re.compile(
    r"\b(create\s+(?:temp(?:orary)?\s+)?table|alter\s+table|drop\s+table)\b", re.I)
_STMT_END: Final = re.compile(r"[;\n]")
_CREATE_NAME: Final = re.compile(
    r"\s*create\s+(?:temp(?:orary)?\s+)?table\s+(?:if\s+not\s+exists\s+)?[\"`\[]?(\w+)", re.I)


def extract_ddl(text: str) -> list[str]:
    """CREATE TABLE (balanced parentheses), ALTER TABLE and DROP TABLE statements."""
    t = text[:MAX_TEXT]
    out: list[str] = []
    pos = 0
    while len(out) < MAX_DDL_PER_MESSAGE:
        m = _DDL_START.search(t, pos)
        if m is None:
            break
        if m.group(1).lower().startswith("create"):
            open_ = t.find("(", m.end())
            semi = t.find(";", m.end())
            if open_ == -1 or (semi != -1 and semi < open_):
                pos = m.end()
                continue
            depth, end = 0, -1
            for i in range(open_, min(len(t), open_ + MAX_STMT_CHARS)):
                if t[i] == "(":
                    depth += 1
                elif t[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                pos = m.end()
                continue
        else:
            e = _STMT_END.search(t, m.end())
            end = e.start() if e else len(t)
        stmt = t[m.start():end].strip()
        if len(stmt) <= MAX_STMT_CHARS:
            out.append(stmt)
        pos = end
    return out


@dataclass(frozen=True, slots=True)
class Table:
    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    references: tuple[tuple[str, str], ...]  # (column, referenced table)

    def compact(self) -> str:
        """``orders(id PK, customer_id->customers, amount)`` - few tokens, full meaning."""
        refs = dict(self.references)
        parts = []
        for c in self.columns:
            if c in refs:
                parts.append(f"{c}->{refs[c]}")
            elif c in self.primary_key:
                parts.append(f"{c} PK")
            else:
                parts.append(c)
        return f"{self.name}({', '.join(parts)})"


def _deny_dangerous(action: int, *_args: object) -> int:
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _time_budget(deadline: float) -> Callable[[], int]:
    """SQLite progress handler: a non-zero return interrupts the statement."""
    def handler() -> int:
        return int(time.monotonic() > deadline)
    return handler


class Schema:
    """Current tables of one section. Statements are applied in order (newest wins); the
    stored form is always the database's own CREATE statements, so it stays small."""

    __slots__ = ("_ddl", "_tables", "rejected", "version")

    def __init__(self) -> None:
        self._ddl: list[str] = []
        self._tables: dict[str, Table] = {}
        self.version = 0
        self.rejected: list[str] = []  # last statements that did not apply (with reason)

    def __bool__(self) -> bool:
        return bool(self._tables)

    @property
    def tables(self) -> dict[str, Table]:
        return self._tables

    def apply(self, statements: list[str]) -> int:
        """Apply DDL statements; returns how many changed the schema."""
        if not statements:
            return 0
        self.rejected = []
        db = sqlite3.connect(":memory:", isolation_level=None)  # explicit savepoints only
        try:
            db.set_authorizer(_deny_dangerous)
            for stmt in self._ddl:
                db.execute(stmt)
            before = list(self._ddl)
            applied = 0
            for stmt in statements:
                db.set_progress_handler(_time_budget(time.monotonic() + _DDL_TIME_BUDGET_S),
                                        1000)
                db.execute("SAVEPOINT stmt")  # each statement applies fully or not at all
                try:
                    redefined = _CREATE_NAME.match(stmt)
                    if redefined and not re.search(r"\bif\s+not\s+exists\b", stmt, re.I):
                        # A new definition of an existing table replaces it (newest wins).
                        db.execute(f'DROP TABLE IF EXISTS "{redefined.group(1)}"')
                    db.execute(stmt)  # one statement only: sqlite3 refuses several
                    if self._count(db) > MAX_TABLES:
                        raise sqlite3.DatabaseError(f"more than {MAX_TABLES} tables")
                except (sqlite3.Error, sqlite3.Warning) as exc:
                    db.execute("ROLLBACK TO stmt")
                    db.execute("RELEASE stmt")
                    self.rejected.append(f"{stmt[:60]}... ({exc})")
                    continue
                db.execute("RELEASE stmt")
                applied += 1
            db.set_progress_handler(None, 0)
            db.set_authorizer(None)
            if applied:
                self._load(db)
                if self._ddl != before:
                    self.version += 1
                else:
                    applied = 0  # same definitions again: nothing changed
            return applied
        finally:
            db.close()

    @staticmethod
    def _count(db: sqlite3.Connection) -> int:
        return int(db.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0])

    def _load(self, db: sqlite3.Connection) -> None:
        rows = db.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                          "AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall()
        self._ddl = [str(sql) for _, sql in rows if sql]
        tables: dict[str, Table] = {}
        for name, _ in rows:
            info = db.execute("SELECT name, pk FROM pragma_table_info(?)", (name,)).fetchall()
            fks = db.execute('SELECT "from", "table" FROM pragma_foreign_key_list(?)',
                             (name,)).fetchall()
            cols = tuple(str(c) for c, _ in info[:MAX_COLUMNS])
            pks = tuple(str(c) for c, pk in info if pk)
            tables[str(name)] = Table(str(name), cols, pks,
                                      tuple((str(a), str(b)) for a, b in fks))
        self._tables = tables

    def full_sql(self) -> str:
        """Every current CREATE statement: what answers are checked against."""
        return ";\n".join(self._ddl) + (";" if self._ddl else "")

    def neighbours(self, name: str) -> set[str]:
        """Tables linked to ``name`` by a foreign key, in either direction."""
        t = self._tables.get(name)
        out = {ref for _, ref in t.references} if t else set()
        out |= {o.name for o in self._tables.values() if any(r == name for _, r in o.references)}
        out.discard(name)
        return {n for n in out if n in self._tables}


# ---- other facts ------------------------------------------------------------------------------

_SENT: Final = r"(?:^|(?<=[.!?\n])\s*)"
_DEFINITION: Final = (
    re.compile(_SENT + r"(?:let\s+)?[\"'`]?([A-Za-z][\w\- ]{0,40}?)[\"'`]?\s+"
               r"(?:means|is defined as|refers to|stands for)\s+([^.\n]{2,200})", re.I | re.M),
    re.compile(r"\bdefine\s+[\"'`]?([A-Za-z][\w\- ]{0,40}?)[\"'`]?\s+as\s+([^.\n]{2,200})",
               re.I),
    re.compile(r"\bby\s+[\"'`]?([A-Za-z][\w\- ]{0,40}?)[\"'`]?\s+I\s+mean\s+([^.\n]{2,200})",
               re.I),
)
_CODE_NAME: Final = re.compile(r"^[ \t]*(?:async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)[ \t]*"
                               r"(\([^)\n]{0,120}\))?", re.M)
_KEYS: Final = re.compile(r"\b(?:keys?|fields?|properties)\b\s*(?:named|called|like|such as|"
                          r"[:=\-])?\s*([^.?!\n]{1,200})", re.I)
_KEY_SPLIT: Final = re.compile(r"\s*(?:,|\band\b|&|/|\s)\s*", re.I)
_PREFERENCE: Final = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*((?:please\s+)?(?:always|never|from now on|in future|going forward"
    r"|i prefer|i'd prefer|i would prefer)\b[^.!?\n]{3,160})", re.I)
_CORRECTION: Final = re.compile(
    r"^\s*(?:no[,.!\s]|nope\b|wrong\b|incorrect\b|that'?s\s+(?:wrong|incorrect|not\s+"
    r"(?:right|correct|what))|that\s+is\s+(?:wrong|incorrect)|actually[,\s]|it\s+should\s+be\b"
    r"|you\s+(?:made\s+a\s+)?mistake)", re.I)


def _clean_term(term: str) -> str:
    words = term.strip().lower().split()
    while words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join(words)


def extract_definitions(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in _DEFINITION:
        for m in pattern.finditer(text[:MAX_TEXT]):
            term = _clean_term(m.group(1))
            if term and len(term.split()) <= 4 and term not in _STOP:
                out[term] = m.group(2).strip()
            if len(out) >= MAX_DEFINITIONS:
                return out
    return out


def extract_code_names(text: str) -> dict[str, str]:
    """``{name: "def name(args)"}`` for Python functions and classes in the text."""
    out: dict[str, str] = {}
    for m in _CODE_NAME.finditer(text[:MAX_TEXT]):
        out[m.group(2)] = f"{m.group(1)} {m.group(2)}{m.group(3) or ''}"
        if len(out) >= MAX_NAMES:
            break
    return out


def extract_json_keys(text: str) -> tuple[str, ...]:
    for m in _KEYS.finditer(text[:MAX_TEXT]):
        keys: list[str] = []
        for raw in _KEY_SPLIT.split(m.group(1)):
            k = raw.strip("\"'`()[]{} ")
            if not re.fullmatch(r"[A-Za-z_][\w\-]{0,40}", k) or k.lower() in _STOP:
                if keys:
                    break  # the list ended
                continue
            if k not in keys:
                keys.append(k)
            if len(keys) >= MAX_KEYS:
                break
        if keys:
            return tuple(keys)
    return ()


def extract_preferences(text: str) -> list[str]:
    return [m.group(1).strip() for m in _PREFERENCE.finditer(text[:MAX_TEXT])][:MAX_PREFERENCES]


def is_correction(text: str) -> bool:
    return bool(_CORRECTION.match(text[:400]))


# ---- one message ------------------------------------------------------------------------------


@dataclass(slots=True)
class MessageFacts:
    """Everything extracted from one user message (and its optional context)."""

    ddl: list[str] = field(default_factory=list)
    definitions: dict[str, str] = field(default_factory=dict)
    code_names: dict[str, str] = field(default_factory=dict)
    json_keys: tuple[str, ...] = ()
    math: tuple[str, float] | None = None
    preferences: list[str] = field(default_factory=list)
    correction: bool = False
    words: list[str] = field(default_factory=list)  # meaningful words (for keywords)
    tokens: set[str] = field(default_factory=set)  # every identifier-like token (for names)
    lowered: str = ""  # the message, lower-case, truncated (for multi-word terms)

    @property
    def identifiers(self) -> set[str]:
        """Names that tie a message to a section: tables in DDL, code names, terms."""
        names = {n.lower() for n in self.code_names} | set(self.definitions)
        for stmt in self.ddl:
            m = re.search(r"\btable\s+(?:if\s+(?:not\s+)?exists\s+)?[\"'`\[]?(\w+)", stmt, re.I)
            if m:
                names.add(m.group(1).lower())
        return names


def extract(text: str, context: str | None = None) -> MessageFacts:
    """Facts from a user message; ``context`` (e.g. a pasted schema) counts as part of it."""
    full = text if not context else f"{context}\n{text}"
    return MessageFacts(
        ddl=extract_ddl(full),
        definitions=extract_definitions(full),
        code_names=extract_code_names(full),
        json_keys=extract_json_keys(text),
        math=expected_answer(text),
        preferences=extract_preferences(text),
        correction=is_correction(text),
        words=keywords(text),
        tokens={m.group(0).lower() for m in _TOKEN.finditer(text[:MAX_TOKENS_SCAN])},
        lowered=text[:MAX_TOKENS_SCAN].lower(),
    )


class BoundedDict(OrderedDict[str, str]):
    """Insertion-ordered dict that keeps only the newest ``limit`` entries."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def put(self, key: str, value: str) -> None:
        self.pop(key, None)
        self[key] = value
        while len(self) > self.limit:
            self.popitem(last=False)
