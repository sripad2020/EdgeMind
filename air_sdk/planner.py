"""Central Adaptive Execution Planner.

Builds the validated candidate lattice from the deterministic engines, computes live
memory budgets, masks unsafe RL actions, and provides the conservative fallback.

Stability: the lattice is derived from the machine's *capacity* (total RAM/VRAM minus
headroom), not from transient free memory, so lattice coordinates mean the same thing on
every run on the same hardware - a requirement for persisted Q-values to stay valid. Live
free memory is applied at decision time through :meth:`action_mask`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ._validate import clamp
from .analysis.profile import ModelProfile
from .analysis.quant import KV_LADDER, KVCacheType
from .config import (
    DevicePlacement,
    ExecutionConfig,
    MemoryEstimate,
    Objective,
    SafetyPolicy,
    shape_key,
)
from .engines.placement import estimate_memory, max_fitting_layers, placement_levels
from .engines.precision import PrecisionCandidates, build_precision_candidates
from .engines.space import CandidateSpace, MemoryBudget, Position
from .errors import InsufficientResourcesError, ValidationError
from .hardware.profiler import HardwareProfile, TelemetrySample
from .rl.actions import N_ACTIONS, Action
from .rl.reward import RewardWeights
from .rl.store import ConfigScore

_BATCH_LEVELS = (128, 256, 512)
# Typical discrete-GPU memory bandwidth relative to dual-channel system RAM (~300-900 GB/s vs
# ~40-90 GB/s). Only used to *rank* starting candidates; measurements (probe/RL) override it.
_GPU_BANDWIDTH_ADVANTAGE = 8.0
_SPLIT_PENALTY = 0.03


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    config: ExecutionConfig
    position: Position
    estimate: MemoryEstimate
    fallback: ExecutionConfig
    quality_loss: float
    candidates: int
    source: str  # "learned" | "heuristic" | "conservative"

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "estimated_ram_bytes": self.estimate.ram_bytes,
            "estimated_vram_bytes": self.estimate.vram_bytes,
            "quality_loss_prior": round(self.quality_loss, 6),
            "fallback": self.fallback.as_dict(),
            "candidate_count": self.candidates,
            "source": self.source,
        }


def _thread_levels(hw: HardwareProfile) -> tuple[int, ...]:
    levels = {max(1, hw.physical_cores // 2), hw.physical_cores}
    if hw.logical_cores > hw.physical_cores:
        levels.add(hw.logical_cores)
    return tuple(sorted(levels))


class ExecutionPlanner:
    def __init__(
        self,
        variants: tuple[ModelProfile, ...],
        hardware: HardwareProfile,
        safety: SafetyPolicy,
        objective: Objective,
        *,
        n_ctx: int,
        placement: DevicePlacement,
        pinned_precision: str | None,
        kv_types: tuple[KVCacheType, ...] = KV_LADDER,
        quality_overrides: Mapping[str, float] | None = None,
    ) -> None:
        self._hw = hardware
        self._safety = safety
        self._objective = objective
        self._placement = placement
        max_ctx = min(v.context_length for v in variants)
        if not 16 <= n_ctx <= max_ctx:
            raise ValidationError(f"n_ctx must be in [16, {max_ctx}] for this model, got {n_ctx}")
        self._n_ctx = n_ctx
        if placement is DevicePlacement.GPU and hardware.primary_gpu is None:
            raise InsufficientResourcesError("device_placement='gpu' but no GPU was detected")
        precision = build_precision_candidates(
            variants, max_quality_loss=safety.max_quality_loss, pinned_precision=pinned_precision,
            allowed_kv=kv_types, quality_overrides=quality_overrides,
        )
        # The lattice is always built for the *maximum* context (conservative); the active
        # context (see set_context) may be smaller and grows with the conversation.
        self.space = self._build_space(precision)
        self._index: dict[str, Position] | None = None

    @property
    def max_context(self) -> int:
        return self._n_ctx

    def context_levels(self, minimum: int = 1024) -> tuple[int, ...]:
        """Context sizes to grow through: minimum, 2x, 4x ... capped at the maximum."""
        levels: list[int] = []
        level = min(minimum, self._n_ctx)
        while level < self._n_ctx:
            levels.append(level)
            level *= 2
        levels.append(self._n_ctx)
        return tuple(levels)

    def set_context(self, n_ctx: int) -> None:
        """Change the *active* context size used for estimates and loads (<= maximum)."""
        if not 16 <= n_ctx <= self._n_ctx:
            raise ValidationError(f"context must be in [16, {self._n_ctx}]")
        self.space = replace(self.space, n_ctx=n_ctx)

    # ---- predicted performance (deterministic prior for the starting choice) --------------

    def predicted_token_cost(self, pos: Position) -> float:
        """Relative time per generated token. Decoding is memory-bandwidth bound: every token
        reads all weights once. Cost = bytes read from RAM + bytes read from VRAM / GPU
        bandwidth advantage + a fixed penalty for a CPU/GPU split (activations cross PCIe).
        Embedding rows are looked up, not streamed, so they are excluded. O(1)."""
        p = self.space.variant(pos)
        k = self.space.n_gpu_layers(pos)
        gpu_layers = min(k, p.n_layers)
        gpu_bytes = p.layer_suffix_bytes[gpu_layers] + (p.output_bytes if k > p.n_layers else 0)
        cpu_bytes = max(0, p.weight_bytes - p.other_bytes - gpu_bytes)
        split = 0 < k <= p.n_layers
        streamed = max(1, p.weight_bytes - p.other_bytes)
        return (cpu_bytes + gpu_bytes / _GPU_BANDWIDTH_ADVANTAGE
                + (_SPLIT_PENALTY * streamed if split else 0.0))

    def predicted_score(self, pos: Position, reference_cost: float) -> float:
        """Predicted reward of a configuration with the objective's own weights, so the
        starting choice optimises the same thing the RL engine will."""
        w = RewardWeights.for_objective(self._objective)
        speed = clamp(reference_cost / max(self.predicted_token_cost(pos), 1.0), 0.0, 4.0) - 1
        cap = max(1, self.capacity_budget().ram_bytes + self.capacity_budget().vram_bytes)
        memory = clamp(1.0 - self.space.estimate(pos).total / cap, -1.0, 1.0)
        quality = self.space.precision.quality_loss(pos.variant, pos.kv) / max(
            self._safety.max_quality_loss, 1e-9)
        return ((w.throughput + w.latency) * speed + w.memory * memory
                - w.quality * clamp(quality, 0.0, 1.0))

    # ---- budgets ----------------------------------------------------------------------------

    def capacity_budget(self) -> MemoryBudget:
        """Static capacity budget (machine class), used to shape the lattice."""
        ram = self._hw.total_ram - self._safety.ram_headroom(self._hw.total_ram)
        gpu = self._hw.primary_gpu
        if gpu is None or self._placement is DevicePlacement.CPU:
            return MemoryBudget(max(0, ram), 0)
        vram = gpu.total_vram - self._safety.vram_headroom(gpu.total_vram)
        return MemoryBudget(max(0, ram), max(0, vram))

    def live_budget(
        self, sample: TelemetrySample, loaded: MemoryEstimate | None
    ) -> MemoryBudget:
        """Free memory right now, counting what the current config would release on reload."""
        reclaim_ram = min(loaded.ram_bytes, sample.process_rss) if loaded else 0
        ram = sample.ram_available + reclaim_ram - self._safety.ram_headroom(sample.ram_total)
        gpu = self._hw.primary_gpu
        if gpu is None or self._placement is DevicePlacement.CPU:
            return MemoryBudget(max(0, ram), 0)
        if sample.vram_total is not None and sample.vram_used is not None:
            reclaim_vram = min(loaded.vram_bytes, sample.vram_used) if loaded else 0
            free = sample.vram_total - sample.vram_used + reclaim_vram
            vram = free - self._safety.vram_headroom(sample.vram_total)
        else:
            vram = gpu.free_vram - self._safety.vram_headroom(gpu.total_vram)
        return MemoryBudget(max(0, ram), max(0, vram))

    # ---- lattice construction ---------------------------------------------------------------

    def _build_space(self, precision: PrecisionCandidates) -> CandidateSpace:
        cap = self.capacity_budget()
        batches = tuple(b for b in _BATCH_LEVELS if b <= self._n_ctx) or (min(self._n_ctx, 64),)
        compact_kv = precision.kv_types[-1]
        min_batch = batches[0]
        gpu = self._hw.primary_gpu
        reserves = {
            "gpu_context_bytes": self._safety.reserve_gpu_context_bytes,
            "runtime_bytes": self._safety.reserve_runtime_bytes,
        }

        kept: list[ModelProfile] = []
        levels_per_variant: list[tuple[int, ...]] = []
        for p in precision.variants:
            gpu_ok = (
                gpu is not None
                and self._placement is not DevicePlacement.CPU
                and (p.quant.name != "BF16" or gpu.supports_bf16)
            )
            if gpu_ok:
                k_max = max_fitting_layers(
                    p, vram_budget=cap.vram_bytes, kv_type=compact_kv, n_ctx=self._n_ctx,
                    n_batch=min_batch, flash_attn=compact_kv.requires_flash_attention,
                    **reserves,
                )
                levels = placement_levels(p, k_max)
                if self._placement is DevicePlacement.GPU:
                    levels = (k_max,) if k_max > 0 else ()
            else:
                levels = (0,)
            fitting = tuple(
                k for k in levels
                if estimate_memory(
                    p, n_gpu_layers=k, kv_type=compact_kv, n_ctx=self._n_ctx, n_batch=min_batch,
                    flash_attn=compact_kv.requires_flash_attention, **reserves,
                ).ram_bytes <= cap.ram_bytes
            )
            if fitting:
                kept.append(p)
                levels_per_variant.append(fitting)

        if not kept:
            smallest = min(precision.variants, key=lambda v: v.weight_bytes)
            raise InsufficientResourcesError(
                f"model cannot fit this hardware: smallest variant ({smallest.quant.name}) needs"
                f" ~{smallest.weight_bytes / 2**30:.2f} GiB of weights plus KV cache"
                f" (n_ctx={self._n_ctx}); budget is RAM {cap.ram_bytes / 2**30:.2f} GiB,"
                f" VRAM {cap.vram_bytes / 2**30:.2f} GiB after safety headroom"
            )
        return CandidateSpace(
            precision=PrecisionCandidates(tuple(kept), precision.kv_types,
                                          precision.max_quality_loss),
            gpu_levels=tuple(levels_per_variant),
            thread_levels=_thread_levels(self._hw),
            batch_levels=batches,
            n_ctx=self._n_ctx,
            gpu_context_bytes=self._safety.reserve_gpu_context_bytes,
            runtime_bytes=self._safety.reserve_runtime_bytes,
        )

    def signature(self) -> str:
        s = self.space
        raw = repr((
            [v.variant_key for v in s.precision.variants],
            [k.value for k in s.precision.kv_types],
            s.gpu_levels, s.thread_levels, s.batch_levels, s.n_ctx,
        ))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    # ---- safety-filtered decisions ----------------------------------------------------------

    def action_mask(
        self, pos: Position, budget: MemoryBudget, blacklist: frozenset[str]
    ) -> list[bool]:
        """mask[a] is True iff action ``a`` leads to an in-lattice, safe configuration.

        O(|A|) checks, each O(1).
        """
        mask = [False] * N_ACTIONS
        for a in Action:
            nxt = self.space.neighbor(pos, a)
            mask[a] = nxt is not None and self.space.check(nxt, budget, blacklist) is None
        return mask

    def _safe_positions(self, budget: MemoryBudget, blacklist: frozenset[str]) -> list[Position]:
        return [
            p for p in self.space.all_positions()
            if self.space.check(p, budget, blacklist) is None
        ]

    def _thread_distance(self, pos: Position) -> int:
        return abs(self.space.thread_levels[pos.threads] - self._hw.physical_cores)

    def conservative(self, budget: MemoryBudget, blacklist: frozenset[str]) -> Position:
        """Lowest-risk safe configuration: prefer CPU-only, then smallest footprint."""
        safe = self._safe_positions(budget, blacklist) or self._safe_positions(
            budget, frozenset()
        )
        if not safe:
            raise InsufficientResourcesError(
                "no configuration fits the currently free memory (RAM "
                f"{budget.ram_bytes / 2**30:.2f} GiB, VRAM {budget.vram_bytes / 2**30:.2f} GiB"
                " after headroom); close other applications or use a smaller model variant"
            )
        return min(
            safe,
            key=lambda p: (p.gpu > 0, self.space.estimate(p).total, self._thread_distance(p), p),
        )

    def relief(
        self,
        current: Position,
        budget: MemoryBudget,
        blacklist: frozenset[str],
        value_of: Callable[[str], float | None],
    ) -> Position:
        """Memory-pressure relief: when ``current`` no longer fits, pick the best *safe*
        configuration instead of jumping straight to the smallest one.

        Preference: highest learned value (``value_of`` returns the observed mean reward of a
        configuration, or None if never tried); among untried ones, the closest to the
        current configuration (fewest changed settings), then the higher precision.
        O(N) over the lattice (N < ~2,000), each check O(1).
        """
        safe = self._safe_positions(budget, blacklist)
        if not safe:
            return self.conservative(budget, blacklist)

        def distance(p: Position) -> int:
            return (abs(p.variant - current.variant) + abs(p.kv - current.kv)
                    + abs(p.gpu - current.gpu) + abs(p.threads - current.threads)
                    + abs(p.batch - current.batch))

        known = [(v, p) for p in safe
                 if (v := value_of(self.space.config_at(p).shape_key)) is not None]
        if known:
            return max(known, key=lambda vp: (vp[0], -distance(vp[1])))[1]
        return min(safe, key=lambda p: (distance(p), p.variant, p.kv))

    def heuristic(self, budget: MemoryBudget, blacklist: frozenset[str]) -> Position:
        """Deterministic starting point before any learning has happened: the safe
        configuration with the best *predicted* reward on this hardware (bandwidth model +
        quality cost + memory), so e.g. a CPU-only machine starts on a smaller, faster
        precision instead of always the largest. O(N) over the lattice."""
        if self._objective is Objective.MEMORY:
            return self.conservative(budget, blacklist)
        safe = self._safe_positions(budget, blacklist)
        if not safe:
            return self.conservative(budget, blacklist)
        reference = max(self.predicted_token_cost(p) for p in safe)
        return max(
            safe,
            key=lambda p: (
                round(self.predicted_score(p, reference), 6), -self._thread_distance(p),
                p.batch, -p.kv, -p.variant,
            ),
        )

    def best_for_variant(
        self, variant: int, budget: MemoryBudget, blacklist: frozenset[str]
    ) -> Position | None:
        """Best predicted safe configuration that uses a given weight variant (for probing)."""
        safe = [p for p in self._safe_positions(budget, blacklist) if p.variant == variant]
        if not safe:
            return None
        reference = max(self.predicted_token_cost(p) for p in safe)
        return max(safe, key=lambda p: (round(self.predicted_score(p, reference), 6),
                                        -self._thread_distance(p), p.batch, -p.kv))

    def position_of(self, config_key: str) -> Position | None:
        """Lattice position for a stored config key. Matching ignores the context size,
        which changes during a conversation. O(N) once, then O(1)."""
        if self._index is None:
            self._index = {self.space.config_at(p).shape_key: p
                           for p in self.space.all_positions()}
        return self._index.get(shape_key(config_key))

    def plan(
        self,
        budget: MemoryBudget,
        blacklist: frozenset[str],
        learned: list[ConfigScore],
    ) -> ExecutionPlan:
        fallback = self.conservative(budget, blacklist)
        chosen: Position | None = None
        source = "heuristic"
        for score in learned:
            pos = self.position_of(score.config_key)
            if pos is not None and self.space.check(pos, budget, blacklist) is None:
                chosen, source = pos, "learned"
                break
        if chosen is None:
            chosen = self.heuristic(budget, blacklist)
            if self._objective is Objective.MEMORY:
                source = "conservative"
        return self.make_plan(chosen, fallback, source)

    def make_plan(self, pos: Position, fallback: Position, source: str) -> ExecutionPlan:
        return ExecutionPlan(
            config=self.space.config_at(pos),
            position=pos,
            estimate=self.space.estimate(pos),
            fallback=self.space.config_at(fallback),
            quality_loss=self.space.precision.quality_loss(pos.variant, pos.kv),
            candidates=self.space.size(),
            source=source,
        )
