"""Smarter starting choice (prediction + probe), automatic context size, shape keys."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from air_sdk import AdaptiveRuntime
from air_sdk._validate import GiB, MiB
from air_sdk.analysis.profile import analyze_model, discover_variants
from air_sdk.config import DevicePlacement, Objective, SafetyPolicy, shape_key
from air_sdk.engines.space import MemoryBudget
from air_sdk.hardware.profiler import GPUInfo, HardwareProfile
from air_sdk.planner import ExecutionPlanner
from air_sdk.quality import evaluate_against_reference, word_similarity
from air_sdk.rl.agent import QLearningParams

from .conftest import small_machine, small_safety

BIG = MemoryBudget(10**12, 10**12)


def _planner(model_dir: Path, objective: Objective, hw: HardwareProfile) -> ExecutionPlanner:
    variants = discover_variants(analyze_model(model_dir / "tiny-f16.gguf"))
    return ExecutionPlanner(variants, hw, SafetyPolicy(max_quality_loss=0.06), objective,
                            n_ctx=1024, placement=DevicePlacement.AUTO, pinned_precision=None)


CPU_ONLY = HardwareProfile("cpu", 4, 8, 16 * GiB, 12 * GiB, "x")


def test_cpu_only_starts_on_faster_precision(model_dir: Path) -> None:
    for objective in (Objective.THROUGHPUT, Objective.BALANCED):
        p = _planner(model_dir, objective, CPU_ONLY)
        start = p.heuristic(BIG, frozenset())
        # Smaller weights read fewer bytes per token: never the largest (F16) on CPU-only.
        assert p.space.variant(start).quant.name in ("Q8_0", "Q4_0"), objective
        assert p.space.thread_levels[start.threads] == 4  # physical cores


def test_memory_objective_starts_smallest(model_dir: Path) -> None:
    p = _planner(model_dir, Objective.MEMORY, CPU_ONLY)
    start = p.heuristic(BIG, frozenset())
    safe = [q for q in p.space.all_positions() if p.space.check(q, BIG, frozenset()) is None]
    assert p.space.estimate(start).total == min(p.space.estimate(q).total for q in safe)


def test_gpu_machine_prefers_offload(model_dir: Path) -> None:
    hw = replace(CPU_ONLY, gpus=(GPUInfo(0, "gpu", 8 * GiB, 8 * GiB, (8, 6)),))
    p = _planner(model_dir, Objective.THROUGHPUT, hw)
    start = p.heuristic(BIG, frozenset())
    assert p.space.n_gpu_layers(start) > 0


def test_predicted_cost_orders_variants(model_dir: Path) -> None:
    p = _planner(model_dir, Objective.THROUGHPUT, CPU_ONLY)
    costs = {p.space.variant(q).quant.name: p.predicted_token_cost(q)
             for q in p.space.all_positions()}
    assert costs["Q4_0"] < costs["Q8_0"] < costs["F16"]


def _rt(tmp_path: Path, **kw: object) -> AdaptiveRuntime:
    hw, backend = small_machine(vram_mib=0)
    return AdaptiveRuntime(backend=backend, data_dir=tmp_path / "db", safety=small_safety(),
                           hardware=hw, seed=3, rl_params=QLearningParams(), **kw)  # type: ignore[arg-type]


def test_probe_measures_each_version(tmp_path: Path, model_dir: Path) -> None:
    with _rt(tmp_path) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024, probe=True)
        assert plan.source == "probe"
        probe = m.performance_report()["probe"]
        assert isinstance(probe, list)
        assert {row["precision"] for row in probe} == {"F16", "Q8_0", "Q4_0"}
        assert sum(bool(row["chosen"]) for row in probe) == 1
        chosen = next(row for row in probe if row["chosen"])
        assert chosen["precision"] == plan.config.precision
        # Simulated speed is bandwidth bound, so the smallest weights measure fastest.
        assert max(probe, key=lambda r: r["tokens_per_second"])["precision"] == "Q4_0"


def test_context_starts_small_and_grows(tmp_path: Path, model_dir: Path) -> None:
    with _rt(tmp_path) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=4096)
        assert plan.config.n_ctx == 1024  # starts small
        r = m.generate("hi", max_tokens=32)
        assert r.config.n_ctx == 1024
        long = "word " * 1500  # ~1.9k tokens by the conservative estimate
        r = m.generate(long, max_tokens=64)
        assert r.config.n_ctx == 4096
        assert r.switch_s == 0.0  # growth reload is not charged to the agent
        perf = m.performance_report()
        assert perf["context_growths"] == 1
        assert perf["context_active"] == 4096


def test_fixed_context_when_auto_off(tmp_path: Path, model_dir: Path) -> None:
    with _rt(tmp_path) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        assert m.optimize(n_ctx=2048, auto_context=False).config.n_ctx == 2048


def test_shape_key_ignores_context() -> None:
    a = '{"n_ctx":1024,"n_threads":4,"precision":"Q4_0"}'
    b = '{"n_ctx":4096,"n_threads":4,"precision":"Q4_0"}'
    assert shape_key(a) == shape_key(b)
    assert shape_key("not json") == "not json"


def test_word_similarity() -> None:
    assert word_similarity("a b c", "a b c") == 1.0
    assert word_similarity("", "") == 1.0
    assert word_similarity("a b c d", "a b x y") == pytest.approx(0.5)


def test_evaluate_against_reference() -> None:
    answers = {("ref", "p"): "one two three four", ("q8", "p"): "one two three four",
               ("q4", "p"): "one two five six"}
    released: list[str] = []
    out = evaluate_against_reference(lambda m, p: answers[(m, p)], "ref", ["q8", "q4"],
                                     ["p"], release=released.append)
    assert [r.agreement for r in out] == [1.0, 0.5]
    assert out[1].measured_loss == pytest.approx(0.05)
    assert released == ["ref", "q8", "q4"]


def test_tiny_budget_still_safe(model_dir: Path) -> None:
    p = _planner(model_dir, Objective.THROUGHPUT, CPU_ONLY)
    tight = MemoryBudget(int(1.5 * MiB), 0)
    with pytest.raises(Exception):  # noqa: B017, PT011 - planner reports cannot-fit
        p.heuristic(tight, frozenset())
