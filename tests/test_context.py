"""Sectioned context: facts, detection, routing, budget, safety ladder, verification."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from air_sdk import AdaptiveRuntime, ValidationError
from air_sdk.backends.base import GenerationResult
from air_sdk.backends.simulated import SimulatedBackend
from air_sdk.context import (
    ContextEngine,
    Schema,
    SessionMemory,
    SessionStore,
    build_prompt,
    detect,
    evaluate,
    expected_answer,
    extract,
)
from air_sdk.context.detect import is_follow_up
from air_sdk.context.facts import extract_ddl
from air_sdk.context.sections import MAX_SECTIONS
from air_sdk.easy import LLM

from .conftest import small_machine, small_safety

SHOP = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY,
                     customer_id INTEGER REFERENCES customers(id), amount REAL);
CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT, price REAL);
CREATE TABLE suppliers (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE audit_log (id INTEGER PRIMARY KEY, event TEXT);
"""


# ---- safe calculator -------------------------------------------------------------------------


def test_calculator_computes_questions() -> None:
    assert expected_answer("What is 17 * 23?") == ("17 * 23", 391.0)
    found = expected_answer("Calculate 15% of 840 please")
    assert found is not None
    assert found[1] == pytest.approx(126.0)
    found = expected_answer("1250 divided by 25")
    assert found is not None
    assert found[1] == 50.0
    assert expected_answer("I have 3 cats") is None  # a lone number is not a calculation


def test_calculator_is_safe_and_bounded() -> None:
    assert evaluate("__import__('os').system('x')") is None
    assert evaluate("(1).__class__") is None
    assert evaluate("2 ** 1000") is None  # exponent too large
    assert evaluate("9 ** 9 ** 9") is None
    assert evaluate("(-8) ** 0.5") is None  # not a real number
    assert evaluate("1 / 0") is None
    assert evaluate("1" * 500) is None  # too long
    assert evaluate("(2 + 3) * 4") == 20.0


# ---- facts --------------------------------------------------------------------------------------


def test_schema_versions_newest_wins() -> None:
    s = Schema()
    assert s.apply(extract_ddl("CREATE TABLE orders (id INTEGER, amount REAL);")) == 1
    assert s.version == 1
    s.apply(extract_ddl("ALTER TABLE orders ADD COLUMN status TEXT"))
    assert list(s.tables["orders"].columns) == ["id", "amount", "status"]
    s.apply(extract_ddl("CREATE TABLE orders (order_id INTEGER, total REAL)"))  # redefined
    assert list(s.tables["orders"].columns) == ["order_id", "total"]
    v = s.version
    assert s.apply(extract_ddl("CREATE TABLE orders (order_id INTEGER, total REAL)")) == 0
    assert s.version == v  # same definition again: no new version
    s.apply(["DROP TABLE orders"])
    assert not s


def test_failed_redefinition_keeps_the_old_table() -> None:
    s = Schema()
    s.apply(["CREATE TABLE t (a INTEGER)"])
    s.apply(["CREATE TABLE t (a INTEGER, a INTEGER)"])  # invalid: duplicate column
    assert s.tables["t"].columns == ("a",)
    assert s.rejected


def test_schema_is_sandboxed() -> None:
    s = Schema()
    assert extract_ddl("ATTACH DATABASE 'x.db' AS y") == []
    s.apply(["CREATE TABLE a (x INTEGER); DROP TABLE b"])  # two statements: refused
    assert not s


def test_compact_schema_and_foreign_keys() -> None:
    s = Schema()
    s.apply(extract_ddl(SHOP))
    assert s.tables["orders"].compact() == "orders(id PK, customer_id->customers, amount)"
    assert s.neighbours("customers") == {"orders"}
    assert "CREATE TABLE" in s.full_sql()


def test_extracts_definitions_names_keys_preferences_corrections() -> None:
    f = extract("Active customer means someone who ordered in the last 30 days. "
                "Always use ANSI SQL.")
    assert f.definitions == {"active customer": "someone who ordered in the last 30 days"}
    assert f.preferences == ["Always use ANSI SQL"]
    f = extract("Give JSON with keys title, author and year.")
    assert f.json_keys == ("title", "author", "year")
    f = extract("```python\ndef slugify(title: str) -> str:\n    ...\n```")
    assert f.code_names == {"slugify": "def slugify(title: str)"}
    assert extract("No, that's wrong - use LEFT JOIN").correction
    assert not extract("Nothing like that please").correction


# ---- detection ----------------------------------------------------------------------------------


def test_task_detection() -> None:
    def skill(text: str) -> str:
        return detect(text, extract(text)).skill

    assert skill("Write a SQL query for the top 5 customers by spend") == "sql"
    assert skill("Write a Python function that reverses a list") == "python"
    assert skill("What is 17 * 23?") == "math"
    assert skill("Return a JSON object for a book") == "json"
    assert skill("Summarise this article for me") == "summarise"
    assert skill("Tell me a joke about cats") == "general"
    assert not detect("now sort it by amount", extract("now sort it by amount")).strong
    assert is_follow_up("now sort it by amount")


# ---- sections -----------------------------------------------------------------------------------


def _say(engine: ContextEngine, memory: SessionMemory, text: str, **kw: object) -> str:
    p = engine.prepare(memory, text, **kw)  # type: ignore[arg-type]
    engine.commit(memory, p, f"answer to {text[:20]}")
    return p.section


def test_routing_keeps_work_in_separate_sections() -> None:
    eng, mem = ContextEngine(), SessionMemory()
    a = _say(eng, mem, f"Here is my schema:\n{SHOP}\nWrite a SQL query: total amount per "
                       "customer name.")
    assert "sql" in a
    assert _say(eng, mem, "now sort it by the total") == a  # follow-up stays
    b = _say(eng, mem, "Write a Python function to slugify a blog title")
    assert b != a
    assert "python" in b
    assert _say(eng, mem, "add type hints to it") == b
    assert _say(eng, mem, "Back to the orders query: only customers in Paris") == a
    c = _say(eng, mem, "New task: tell me a fun fact about owls")
    assert c not in (a, b)
    snap = mem.snapshot()
    sections = snap["sections"]
    assert isinstance(sections, list)
    assert len(sections) == 3
    sql = next(s for s in sections if s["skill"] == "sql")
    assert "orders" in sql["tables"]


def test_clear_task_change_is_not_captured_by_shared_words() -> None:
    # Found live: "title" is a column in the schema, and "that" looked like a follow-up.
    eng, mem = ContextEngine(), SessionMemory()
    a = _say(eng, mem, f"{SHOP}\nWrite a SQL query: total order amount per customer name.")
    b = _say(eng, mem, "Write a Python function that turns a blog title into a URL slug.")
    assert b != a
    assert "python" in b
    p = eng.prepare(mem, "New task: a JSON object for a book with keys title and year")
    assert p.question.startswith("a JSON object")  # the cue is not sent to the model
    assert "routing" not in p.question


def test_task_hysteresis_and_explicit_skill() -> None:
    eng, mem = ContextEngine(), SessionMemory()
    p = eng.prepare(mem, f"{SHOP}\nSQL query: number of orders per city")
    eng.commit(mem, p, "SELECT ...")
    p = eng.prepare(mem, "and only for 2024")  # weak signal -> keeps SQL
    assert p.skill == "sql"
    assert p.auto
    p = eng.prepare(mem, "and only for 2024", skill="general")  # explicit wins
    assert p.skill == "general"
    assert not p.auto
    with pytest.raises(ValidationError):
        eng.prepare(mem, "x", skill="sqll")


def test_section_limits_and_expiry() -> None:
    eng, mem = ContextEngine(), SessionMemory()
    for i in range(MAX_SECTIONS + 3):
        _say(eng, mem, f"New task: topic number {i} about subject{i}")
    assert len(mem.sections) == MAX_SECTIONS
    assert mem.active in mem.sections  # the active section is never evicted


def test_only_active_section_and_relevant_tables_are_sent() -> None:
    eng, mem = ContextEngine(), SessionMemory()
    eng.commit(mem, eng.prepare(mem, f"Our database:\n{SHOP}\nSQL: count customers"), "ok")
    p = eng.prepare(mem, "SQL: total order amount per customer")
    system = str(p.messages[0]["content"])
    assert "orders(id PK, customer_id->customers, amount)" in system
    assert "customers(" in system  # linked by foreign key
    assert "suppliers(" not in system  # not relevant: only its name may appear
    # The check uses the FULL schema, not the trimmed prompt.
    assert p.check_context is not None
    assert "suppliers" in p.check_context
    # A different task does not carry the schema along.
    p2 = eng.prepare(mem, "New task: write a Python function that adds two numbers")
    assert all("customers" not in str(m["content"]) for m in p2.messages)


def test_maths_expected_answer_is_checked_not_sent() -> None:
    eng, mem = ContextEngine(), SessionMemory()
    p = eng.prepare(mem, "What is 17 * 23?")
    assert p.skill == "math"
    assert p.check_context == "391"
    assert all("391" not in str(m["content"]) for m in p.messages)


# ---- budget and safety ladder ---------------------------------------------------------------


def _big_section(n_tables: int, n_turns: int) -> tuple[SessionMemory, ContextEngine]:
    eng, mem = ContextEngine(), SessionMemory()
    ddl = "\n".join(f"CREATE TABLE t{i} (id INTEGER, value_{i} TEXT, other_{i} REAL);"
                    for i in range(n_tables))
    p = eng.prepare(mem, f"{ddl}\nSQL: count rows in t1")
    eng.commit(mem, p, "SELECT count(*) FROM t1")
    for i in range(n_turns):
        p = eng.prepare(mem, f"and now t1 filtered by value_1 = {i} " + "detail " * 40)
        eng.commit(mem, p, "SELECT * FROM t1 WHERE value_1 = 1 " + "x " * 80)
    return mem, eng


def test_budget_keeps_prompt_inside_context() -> None:
    mem, eng = _big_section(40, 12)
    p = eng.prepare(mem, "and t1 again", context_limit=1024, max_tokens=256)
    assert p.plan.prompt_tokens <= p.plan.budget
    assert p.plan.turns_dropped > 0 or p.plan.facts_dropped > 0
    assert "t1(" in str(p.messages[0]["content"])  # the named table always survives
    assert p.check_context is not None
    assert "t39" in p.check_context


def test_ladder_last_resort_is_minimal_prompt_with_notice() -> None:
    mem, eng = _big_section(5, 2)
    huge = "and t1 " + "word " * 900
    p = eng.prepare(mem, huge, context_limit=1024, max_tokens=256)
    assert p.plan.rung == "minimal"
    assert p.plan.notice
    assert p.messages[-1]["content"] == huge  # the question itself is never cut


def test_build_prompt_includes_preferences() -> None:
    mem = SessionMemory()
    eng = ContextEngine()
    p = eng.prepare(mem, "Always answer in British English. What is a CPU?")
    assert "Always answer in British English" in str(p.messages[0]["content"])
    section = mem.sections[p.section_id]
    plan = build_prompt(question="q", system=None, preferences=[], section=section,
                        facts=extract("q"), context_limit=4096, max_tokens=256)
    assert plan.rung == "full"


# ---- server session store ---------------------------------------------------------------------


def test_session_store_isolation_and_limits() -> None:
    store = SessionStore(ttl_s=60, max_sessions=2)
    a, b = store.get("user-aaaa1"), store.get("user-bbbb2")
    assert a is not b
    assert store.get("user-aaaa1") is a
    store.get("user-cccc3")  # over the limit: least recently used dropped
    assert len(store) == 2
    with pytest.raises(ValidationError):
        store.get("../etc")
    a.touched = time.monotonic() - 120
    store.get("user-dddd4")
    assert "user-aaaa1" not in store._items  # expired


# ---- end to end through the easy API ---------------------------------------------------------


class RecordingBackend(SimulatedBackend):
    """Answers SQL questions with SQL, everything else with plain text; records prompts."""

    prompts: list[str]

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 seed: int | None, on_token: object = None) -> GenerationResult:
        base = super().generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                seed=seed)
        self.prompts.append(prompt)
        last = prompt.rsplit("user:", 1)[-1].lower()
        if "expert sql" in prompt.lower() and "list suppliers" in last:
            text = "```sql\nSELECT name FROM supplier_list\n```"  # wrong table
        elif "expert sql" in prompt.lower():
            text = "```sql\nSELECT c.name, SUM(o.amount) FROM orders o JOIN customers c " \
                   "ON c.id = o.customer_id GROUP BY c.name\n```"
        else:
            text = "Owls can rotate their heads about 270 degrees."
        return GenerationResult(text, base.prompt_tokens, base.completion_tokens, base.ttft_s,
                                base.total_s, base.peak_ram_bytes, base.peak_vram_bytes)


def _llm(tmp_path: Path, model_dir: Path) -> tuple[AdaptiveRuntime, LLM, RecordingBackend]:
    hw, sim = small_machine(vram_mib=0)
    backend = RecordingBackend(sim.hardware, seed=1, noise=0.0)
    backend.prompts = []
    rt = AdaptiveRuntime(backend=backend, data_dir=tmp_path / "db", safety=small_safety(),
                         hardware=hw, seed=3)
    model = rt.load_model(model_dir / "tiny-f16.gguf", discover=True)
    model.optimize(n_ctx=1024)
    return rt, LLM(rt, model, "tiny", {"n_ctx": 1024}), backend


def test_conversation_end_to_end(tmp_path: Path, model_dir: Path) -> None:
    rt, llm, backend = _llm(tmp_path, model_dir)
    with rt:
        chat = llm.conversation()
        r = chat.reply(f"{SHOP}\nWrite a SQL query: total amount per customer name")
        assert r.skill == "sql"
        assert r.auto
        assert r.verified is True
        assert r.section.endswith("(sql)")
        assert "Section:" in r.label
        assert "check passed" in r.label
        r = chat.reply("New task: tell me a fun fact about owls")
        assert r.skill == "general"
        assert r.verified is None
        assert "customers(" not in backend.prompts[-1]  # other section not sent
        # Back to SQL: an answer using a table that does not exist fails the check against
        # the full stored schema (even though the prompt only carried relevant tables).
        r = chat.reply("Back to the customers query: list suppliers")
        assert r.skill == "sql"
        assert r.verified is False
        assert "supplier_list" in r.check
        assert len(chat.sections()["sections"]) == 2  # type: ignore[arg-type]


def test_correction_counts_against_the_precision(tmp_path: Path, model_dir: Path) -> None:
    rt, llm, _ = _llm(tmp_path, model_dir)
    with rt:
        chat = llm.conversation()
        r = chat.reply(f"{SHOP}\nWrite a SQL query: total amount per customer name")
        assert r.verified is True
        before = rt.store.skill_results(llm.model.analyze().family_key, "sql")
        chat.reply("No, that's wrong: use the city instead of the name")
        after = rt.store.skill_results(llm.model.analyze().family_key, "sql")
        assert after[r.precision][1] == before[r.precision][1] + 1


def test_single_question_is_auto_detected(tmp_path: Path, model_dir: Path) -> None:
    rt, llm, _ = _llm(tmp_path, model_dir)
    with rt:
        r = llm.reply("Write a SQL query: total amount per customer name", context=SHOP)
        assert r.skill == "sql"
        assert r.verified is True
        assert llm.skill == "auto"
        llm.use("general")
        assert llm.reply("Write a SQL query please", context=SHOP).skill == "general"
