from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from air_sdk.analysis.profile import analyze_model, discover_variants
from air_sdk.analysis.quant import KVCacheType
from air_sdk.config import DevicePlacement, Objective
from air_sdk.engines.placement import estimate_memory, max_fitting_layers
from air_sdk.engines.precision import build_precision_candidates
from air_sdk.engines.space import MemoryBudget
from air_sdk.errors import InsufficientResourcesError
from air_sdk.planner import ExecutionPlanner
from air_sdk.rl.actions import Action
from air_sdk.testing import SyntheticModel, write_gguf

from .conftest import small_machine, small_safety


def _planner(model_dir: Path, placement: DevicePlacement = DevicePlacement.AUTO,
             vram_mib: int = 4) -> ExecutionPlanner:
    primary = analyze_model(model_dir / "tiny-f16.gguf")
    hw, _ = small_machine(vram_mib=vram_mib)
    return ExecutionPlanner(
        discover_variants(primary), hw, small_safety(), Objective.BALANCED,
        n_ctx=1024, placement=placement, pinned_precision=None,
    )


def test_memory_estimate_monotone_in_gpu_layers(model_path: Path) -> None:
    p = analyze_model(model_path)
    vram = [
        estimate_memory(p, n_gpu_layers=k, kv_type=KVCacheType.F16, n_ctx=1024, n_batch=256,
                        flash_attn=False).vram_bytes
        for k in range(p.n_layers + 2)
    ]
    assert vram == sorted(vram)
    assert vram[0] == 0
    budget = vram[2]
    assert max_fitting_layers(p, vram_budget=budget, kv_type=KVCacheType.F16, n_ctx=1024,
                              n_batch=256, flash_attn=False) == 2


def test_quality_filter_and_ordering(model_dir: Path) -> None:
    planner = _planner(model_dir)
    names = [v.quant.name for v in planner.space.precision.variants]
    assert names == ["F16", "Q8_0", "Q4_0"]  # largest (highest precision) first
    for v in range(len(names)):
        for kv in range(len(planner.space.precision.kv_types)):
            ok = planner.space.precision.is_quality_ok(v, kv)
            assert ok == (planner.space.precision.quality_loss(v, kv) <= 0.06 + 1e-12)


def test_cpu_placement_has_no_gpu_levels(model_dir: Path) -> None:
    planner = _planner(model_dir, DevicePlacement.CPU)
    assert all(levels == (0,) for levels in planner.space.gpu_levels)


def test_action_mask_respects_budget(model_dir: Path) -> None:
    planner = _planner(model_dir)
    tight = MemoryBudget(ram_bytes=10**12, vram_bytes=0)
    pos = planner.conservative(tight, frozenset())
    mask = planner.action_mask(pos, tight, frozenset())
    assert not mask[Action.MORE_GPU_LAYERS]  # zero VRAM budget => never offered
    for a in Action:
        if mask[a]:
            nxt = planner.space.neighbor(pos, a)
            assert nxt is not None
            assert planner.space.check(nxt, tight, frozenset()) is None


def test_blacklisted_config_is_masked(model_dir: Path) -> None:
    planner = _planner(model_dir)
    big = MemoryBudget(10**12, 10**12)
    pos = planner.conservative(big, frozenset())
    nxt = planner.space.neighbor(pos, Action.MORE_THREADS) or planner.space.neighbor(
        pos, Action.FEWER_THREADS)
    assert nxt is not None
    bl = frozenset({planner.space.config_at(nxt).key})
    mask = planner.action_mask(pos, big, bl)
    for a in Action:
        if planner.space.neighbor(pos, a) == nxt:
            assert not mask[a]


def test_model_too_large_is_reported(model_dir: Path) -> None:
    primary = analyze_model(model_dir / "tiny-f16.gguf")
    hw, _ = small_machine(vram_mib=0)
    tiny_hw = replace(hw, total_ram=2 * 1024 * 1024, available_ram=1024 * 1024)
    with pytest.raises(InsufficientResourcesError, match="cannot fit"):
        ExecutionPlanner((primary,), tiny_hw, small_safety(), Objective.BALANCED, n_ctx=1024,
                         placement=DevicePlacement.AUTO, pinned_precision=None)


def test_quality_limit_is_inclusive_despite_float_rounding(tmp_path: Path) -> None:
    # Q4_0 has a 0.05 loss prior; combined with f16 KV it computes to 0.050000000000000044.
    # A limit of exactly 0.05 must accept it (regression: gemma2:2b was wrongly rejected).
    q4 = analyze_model(write_gguf(tmp_path / "q4.gguf", SyntheticModel(file_type=2)))
    cands = build_precision_candidates((q4,), max_quality_loss=0.05,
                                       allowed_kv=(KVCacheType.F16,))
    assert [v.quant.name for v in cands.variants] == ["Q4_0"]
    assert cands.kv_types == (KVCacheType.F16,)
    assert cands.is_quality_ok(0, 0)


def test_signature_is_stable(model_dir: Path) -> None:
    assert _planner(model_dir).signature() == _planner(model_dir).signature()
