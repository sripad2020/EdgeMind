"""Research harness: statistics, items, gold scoring, mixed precision, grid, policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from air_sdk.analysis.gguf import read_gguf
from air_sdk.errors import ValidationError
from air_sdk.quantize import QuantTarget, quantize_gguf
from air_sdk.research.build import group_override, layer_of
from air_sdk.research.controller import compare, run_cascade
from air_sdk.research.datasets import build_shop_db, json_items, math_items, sql_items
from air_sdk.research.gold import gold_code, gold_json, gold_math, gold_sql, online_check
from air_sdk.research.items import Item, parse_item, read_items, write_items
from air_sdk.research.report import build_report
from air_sdk.research.runner import Generation, messages_for, profile_items
from air_sdk.research.stats import bootstrap_mean, mcnemar_exact, qualified, wilson
from air_sdk.research.store import Profile, Result, ResultStore
from air_sdk.testing import SyntheticModel, write_gguf

# ---- statistics ---------------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    r = wilson(8, 10)
    assert r.value == 0.8
    assert r.low == pytest.approx(0.4902, abs=1e-3)
    assert r.high == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0).high == 1.0  # no evidence: the whole range
    with pytest.raises(ValueError, match="passes"):
        wilson(3, 2)


def test_qualification_is_non_inferiority_on_the_lower_bound() -> None:
    assert qualified(wilson(95, 100), 0.80)  # lower bound ~0.89, well above
    assert qualified(wilson(188, 200), 0.90)  # 94%, lower bound ~0.90 >= 0.90 - 0.05
    assert not qualified(wilson(176, 200), 0.90)  # 88%, lower bound ~0.83 < 0.85
    assert not qualified(wilson(9, 10), 0.80)  # 90% but too few items: lower bound ~0.60
    assert not qualified(wilson(188, 200), 0.90, margin=0.0)  # strict: lower bound < 0.90
    assert not qualified(wilson(0, 0), 0.0)


def test_mcnemar_and_bootstrap() -> None:
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 10
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (0, 10)
    assert p == pytest.approx(2 / 1024)
    assert mcnemar_exact(a, a)[2] == 1.0
    assert bootstrap_mean([1.0, 2.0, 3.0], seed=1) == bootstrap_mean([1.0, 2.0, 3.0], seed=1)


# ---- items -----------------------------------------------------------------------------------


def test_items_validate_and_split_deterministically(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="gold is missing"):
        parse_item({"id": "m1", "task": "math", "prompt": "1+1?", "gold": {}})
    with pytest.raises(ValidationError, match="task"):
        parse_item({"id": "x", "task": "chess", "prompt": "p", "gold": {}})
    items = math_items()
    assert len(items) >= 200
    share = sum(i.split == "profile" for i in items) / len(items)
    assert 0.7 < share < 0.9
    path = tmp_path / "m.jsonl"
    write_items(path, items, {"task": "math"})
    meta, back = read_items(path)
    assert meta["task"] == "math"
    assert [i.split for i in back] == [i.split for i in items]
    path.write_text(path.read_text() + items[0].to_json() + "\n")
    with pytest.raises(ValidationError, match="duplicate"):
        read_items(path)


def test_builtin_sets_are_large_enough() -> None:
    assert len(sql_items()) >= 200
    assert len(math_items()) >= 200
    assert len(json_items()) >= 200


# ---- gold scoring ----------------------------------------------------------------------------


@pytest.fixture
def shop(tmp_path: Path) -> Path:
    build_shop_db(tmp_path / "shop.sqlite")
    return tmp_path


def test_gold_sql_execution_accuracy(shop: Path) -> None:
    item = sql_items()[0]  # How many customers live in Paris?
    good = "```sql\nSELECT count(id) FROM customers WHERE city = 'Paris';\n```"
    assert gold_sql(item, good, shop).ok is True
    wrong = "```sql\nSELECT COUNT(*) FROM customers\n```"  # all cities, not Paris
    assert gold_sql(item, wrong, shop).ok is False
    # Writes and multiple statements are refused by the read-only connection.
    assert gold_sql(item, "```sql\nDELETE FROM customers\n```", shop).ok is False
    assert gold_sql(item, "SELECT 1; DROP TABLE customers;", shop).ok is False
    assert gold_sql(item, good, shop).ok is True  # the database is untouched


def test_gold_math_json_and_code_without_sandbox() -> None:
    m = Item("m", "math", "q", {"answer": "391"})
    assert gold_math(m, "17*23 = 391\nAnswer: 391").ok
    assert not gold_math(m, "Answer: 390").ok
    j = Item("j", "json", "q", {"fields": {"name": "Asha Rao", "age": 31}}, "name, age")
    assert gold_json(j, '```json\n{"name": "asha rao", "age": "31"}\n```').ok
    assert not gold_json(j, '{"name": "Asha Rao", "age": 30}').ok
    c = Item("c", "code", "q", {"tests": "assert f(1) == 2", "entry": "f"})
    score = gold_code(c, "```python\ndef f(x):\n    return x + 1\n```", sandbox=False)
    assert score.ok is None  # never run outside the sandbox
    assert gold_code(c, "```python\ndef f(x)\n```", sandbox=False).ok is False


def test_online_check_uses_the_schema() -> None:
    item = sql_items()[0]
    assert online_check(item, "```sql\nSELECT COUNT(*) FROM customers\n```").ok
    assert not online_check(item, "```sql\nSELECT COUNT(*) FROM clients\n```").ok
    assert "Database schema" in str(messages_for(item)[0]["content"])


# ---- mixed precision -------------------------------------------------------------------------


def test_mixed_precision_quantizes_only_the_chosen_groups(tmp_path: Path) -> None:
    src = write_gguf(tmp_path / "m.gguf", SyntheticModel(n_layers=4, n_embd=64, n_ff=128))
    dst = tmp_path / "mix.gguf"
    # Group size 2: layers 0-1 = group 0, layers 2-3 = group 1. Keep group 1 at Q8_0.
    quantize_gguf(src, dst, QuantTarget.Q4_0,
                  override=group_override({1}, 2, QuantTarget.Q8_0))
    types = {t.name: t.ggml_type for t in read_gguf(dst).tensors}
    assert types["blk.0.attn_q.weight"] == QuantTarget.Q4_0.ggml_type
    assert types["blk.1.ffn_up.weight"] == QuantTarget.Q4_0.ggml_type
    assert types["blk.2.attn_q.weight"] == QuantTarget.Q8_0.ggml_type
    assert types["blk.3.ffn_down.weight"] == QuantTarget.Q8_0.ggml_type
    assert types["blk.0.attn_norm.weight"] == 0  # 1-D norms are never quantized
    assert dst.stat().st_size < src.stat().st_size


def test_layer_of() -> None:
    from air_sdk.analysis.gguf import TensorInfo  # noqa: PLC0415
    assert layer_of(TensorInfo("blk.12.attn_q.weight", (4, 4), 1, 0, 16, 32)) == 12
    assert layer_of(TensorInfo("token_embd.weight", (4, 4), 1, 0, 16, 32)) is None


# ---- store, grid, policies, report ------------------------------------------------------------


def _fill(store: ResultStore, items: list[Item]) -> None:
    """Synthetic study: Q4 is fast but often wrong on sql; Q8 is slower and right."""
    store.add_profile(Profile("base", "base", "model", "F16", size_bytes=700, baseline=True))
    store.add_profile(Profile("q8", "base", "uniform", "Q8_0", size_bytes=380))
    store.add_profile(Profile("q4", "base", "uniform", "Q4_0", size_bytes=220))
    for n, item in enumerate(items):
        q4_ok = n % 3 != 0
        for name, ok, secs in (("base", n % 2 == 0, 3.0), ("q8", True, 2.0), ("q4", q4_ok, 1.0)):
            store.add_result(Result(name, item.id, item.task, item.split, ok, ok, "", secs,
                                    0.1, 20, 50, "answer"))


def test_grid_qualification_and_policies(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.sqlite")
    items = sql_items()
    _fill(store, items)
    grid = {(c.profile, c.task): c for c in store.grid()}
    assert grid[("q8", "sql")].qualified  # 100% vs a 50% baseline
    assert not grid[("base", "sql")].qualified  # the baseline itself
    reports = {r.policy: r for r in compare(store, "sql", ["base", "q8", "q4"])}
    cascade = reports["cascade:cheapest-first"]
    assert cascade.accuracy == 1.0  # failed Q4 checks escalate to Q8
    assert cascade.mean_seconds < reports["fixed:q8"].mean_seconds + 1.0
    assert "q_learning" in reports
    assert reports["q_learning"].accuracy >= reports["fixed:q4"].accuracy
    md = build_report(store, tmp_path)
    text = md.read_text(encoding="utf-8")
    assert "RQ4 / RQ5" in text
    assert "RQ6" in text
    assert json.loads((tmp_path / "report.json").read_text())["grid"]
    store.close()


def test_cascade_stops_at_first_passed_check() -> None:
    rows = {"a": Result("a", "i", "sql", "router", False, False, "", 1.0, 0, 0, 0, ""),
            "b": Result("b", "i", "sql", "router", True, True, "", 2.0, 0, 0, 0, "")}
    out = run_cascade(["a", "b"], rows)
    assert (out.correct, out.seconds, out.tried) == (True, 3.0, 2)


def test_profile_items_resumes(tmp_path: Path, shop: Path) -> None:
    store = ResultStore(tmp_path / "r.sqlite")
    calls: list[str] = []

    def fake(model: str, messages: list[dict[str, object]], *, max_tokens: int) -> Generation:
        calls.append(model)
        return Generation("```sql\nSELECT COUNT(*) FROM customers WHERE city = 'Paris'\n```",
                          0.5, 0.1, 12, 80)

    items = sql_items()[:3]
    assert profile_items(store, fake, "m", items, data_dir=shop, sandbox=False) == 3
    assert profile_items(store, fake, "m", items, data_dir=shop, sandbox=False) == 0
    assert len(calls) == 3
    results = store.results(profile="m")
    assert results[0].gold_ok is True  # item 1 is the Paris count
    assert all(r.check_ok for r in results)
    store.close()
