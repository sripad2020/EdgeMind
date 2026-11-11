"""Verified skills: deterministic checks, routing, and verify-and-escalate."""

from __future__ import annotations

from pathlib import Path

import pytest

from air_sdk import AdaptiveRuntime, ValidationError
from air_sdk.backends.base import GenerationResult
from air_sdk.backends.simulated import SimulatedBackend
from air_sdk.skills import (
    SkillEngine,
    check_json,
    check_math,
    check_python,
    check_sql,
    get_skill,
    start_precision,
)

from .conftest import small_machine, small_safety

SCHEMA = "CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL);"


# ---- checks ---------------------------------------------------------------------------------


def test_sql_check_against_schema() -> None:
    good = "```sql\nSELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id;\n```"
    assert check_sql(good, SCHEMA).ok
    bad_col = "```sql\nSELECT customer, SUM(amount) FROM orders GROUP BY customer;\n```"
    res = check_sql(bad_col, SCHEMA)
    assert not res.ok
    assert "no such column" in res.detail
    assert not check_sql("Sorry, I cannot help.", SCHEMA).ok
    # Two statements are rejected (no smuggling a second command).
    assert not check_sql("SELECT 1; DROP TABLE orders;", SCHEMA).ok


def test_sql_check_never_runs_queries_and_blocks_attach(tmp_path: Path) -> None:
    target = tmp_path / "evil.db"
    res = check_sql(f"ATTACH DATABASE '{target}' AS x", SCHEMA)
    assert not res.ok
    hostile_schema = f"ATTACH DATABASE '{target}' AS x;"
    res = check_sql("SELECT 1", hostile_schema)
    assert not res.ok
    assert not res.retryable  # bad schema is the caller's problem, not the model's
    assert not target.exists()


def test_sql_without_schema_checks_completeness() -> None:
    assert check_sql("SELECT a FROM t WHERE x = 1", None).ok
    assert not check_sql("no sql here", None).ok


def test_python_check_parses_without_running() -> None:
    assert check_python("```python\ndef f(x):\n    return x * 2\n```", None).ok
    res = check_python("```python\ndef f(x)\n    return x\n```", None)
    assert not res.ok
    assert "syntax error" in res.detail.lower()
    # Code that would be dangerous if executed only has to *parse*; nothing runs.
    assert check_python("```python\nimport os\nos.remove('x')\n```", None).ok


def test_json_check_with_required_keys() -> None:
    assert check_json('```json\n{"name": "a", "age": 3}\n```', "name, age").ok
    res = check_json('{"name": "a"}', "name, age")
    assert not res.ok
    assert "age" in res.detail
    assert check_json('Here: {"a": 1}', None).ok
    assert not check_json("{not json}", None).ok


def test_math_check() -> None:
    assert check_math("12*7 = 84\nAnswer: 84", "84").ok
    assert not check_math("Answer: 85", "84").ok
    assert check_math("so the result is 1,234.5", None).ok
    assert not check_math("I don't know", None).ok


def test_skill_lookup_and_aliases() -> None:
    assert get_skill(None).name == "general"
    assert get_skill("Programming").name == "python"
    with pytest.raises(ValidationError, match="Did you mean 'sql'"):
        get_skill("sqll")


def test_routing_starts_fast_and_skips_failing_precisions() -> None:
    ladder = ("F16", "Q8_0", "Q4_0")
    assert start_precision(ladder, {}) == "Q4_0"  # untested: try the fastest
    assert start_precision(ladder, {"Q4_0": (1, 1)}) == "Q4_0"  # too few samples
    assert start_precision(ladder, {"Q4_0": (1, 4)}) == "Q8_0"  # Q4 keeps failing
    assert start_precision(ladder, {"Q4_0": (0, 5), "Q8_0": (0, 5)}) == "F16"
    assert start_precision(ladder, {"Q4_0": (9, 1)}) == "Q4_0"  # Q4 proven good


# ---- verify-and-escalate end to end ------------------------------------------------------------


class PrecisionDependentBackend(SimulatedBackend):
    """Q4_0 writes broken SQL; higher precisions write valid SQL."""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 seed: int | None, on_token: object = None) -> GenerationResult:
        base = super().generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                seed=seed)
        cfg = self.loaded_config
        assert cfg is not None
        text = ("```sql\nSELECT custmer FROM orderz\n```" if cfg.precision == "Q4_0"
                else "```sql\nSELECT customer_id FROM orders\n```")
        return GenerationResult(text, base.prompt_tokens, base.completion_tokens, base.ttft_s,
                                base.total_s, base.peak_ram_bytes, base.peak_vram_bytes)


def _engine(tmp_path: Path, model_dir: Path) -> tuple[AdaptiveRuntime, SkillEngine]:
    hw, sim = small_machine(vram_mib=0)
    backend = PrecisionDependentBackend(sim.hardware, seed=1, noise=0.0)
    rt = AdaptiveRuntime(backend=backend, data_dir=tmp_path / "db", safety=small_safety(),
                         hardware=hw, seed=3)
    model = rt.load_model(model_dir / "tiny-f16.gguf", discover=True)
    model.optimize(n_ctx=1024)
    return rt, SkillEngine(model, rt.store)


def test_escalates_from_fast_to_accurate_precision(tmp_path: Path, model_dir: Path) -> None:
    rt, engine = _engine(tmp_path, model_dir)
    with rt:
        escalations: list[tuple[str, str]] = []
        out = engine.run([{"role": "user", "content": "customers?"}], "sql", context=SCHEMA,
                         max_tokens=32, on_escalate=lambda a, b, _r: escalations.append((a, b)))
        assert [a.precision for a in out.attempts] == ["Q4_0", "Q8_0"]
        assert [a.ok for a in out.attempts] == [False, True]
        assert out.verified is True
        assert out.escalated is True
        assert escalations == [("Q4_0", "Q8_0")]
        assert "customer_id" in out.response.text


def test_routing_learns_to_skip_the_failing_precision(tmp_path: Path, model_dir: Path) -> None:
    rt, engine = _engine(tmp_path, model_dir)
    with rt:
        for _ in range(3):
            engine.run([{"role": "user", "content": "q"}], "sql", context=SCHEMA, max_tokens=16)
        out = engine.run([{"role": "user", "content": "q"}], "sql", context=SCHEMA,
                         max_tokens=16)
        assert [a.precision for a in out.attempts] == ["Q8_0"]  # starts where it passes
        assert out.escalated is False
        report = engine.report("sql")
        assert report["starts_on"] == "Q8_0"
        precisions = report["precisions"]
        assert isinstance(precisions, list)
        rows = {r["precision"]: r for r in precisions}
        assert rows["Q4_0"]["fails"] == 3
        assert rows["Q8_0"]["passes"] == 4


class AlwaysWrongBackend(SimulatedBackend):
    """Every precision writes SQL referencing a column that does not exist."""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 seed: int | None, on_token: object = None) -> GenerationResult:
        base = super().generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                seed=seed)
        return GenerationResult("```sql\nSELECT city FROM orders\n```", base.prompt_tokens,
                                base.completion_tokens, base.ttft_s, base.total_s,
                                base.peak_ram_bytes, base.peak_vram_bytes)


def test_failures_on_every_precision_do_not_blame_the_fast_one(tmp_path: Path,
                                                               model_dir: Path) -> None:
    hw, sim = small_machine(vram_mib=0)
    with AdaptiveRuntime(backend=AlwaysWrongBackend(sim.hardware, seed=1, noise=0.0),
                         data_dir=tmp_path / "db", safety=small_safety(), hardware=hw) as rt:
        model = rt.load_model(model_dir / "tiny-f16.gguf", discover=True)
        model.optimize(n_ctx=1024)
        engine = SkillEngine(model, rt.store)
        out = engine.run([{"role": "user", "content": "q"}], "sql", context=SCHEMA,
                         max_tokens=16)
        assert out.verified is False
        assert [a.precision for a in out.attempts] == ["Q4_0", "Q8_0", "F16"]
        stats = rt.store.skill_results(engine.family, "sql")
        # Only the most accurate attempt is blamed: the task is beyond the model.
        assert stats == {"F16": (0, 1)}
        assert engine.report("sql")["starts_on"] == "Q4_0"


def test_unverified_skill_runs_once_with_instruction(tmp_path: Path, model_dir: Path) -> None:
    rt, engine = _engine(tmp_path, model_dir)
    with rt:
        out = engine.run([{"role": "user", "content": "long text"}], "summarise",
                         max_tokens=16)
        assert out.verified is None
        assert len(out.attempts) == 1
    msgs = SkillEngine.build_messages(get_skill("sql"), [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "q"}], SCHEMA)
    assert msgs[0]["role"] == "system"
    assert "SQL" in str(msgs[0]["content"])
    assert SCHEMA in str(msgs[0]["content"])
    assert "Be brief." in str(msgs[0]["content"])
    assert len(msgs) == 2
