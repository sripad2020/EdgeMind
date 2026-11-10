"""Validated candidate lattice shared by the planner and the RL engine.

A :class:`Position` is a coordinate (variant, kv, gpu level, threads, batch). The space is
built once per ``optimize()`` from the deterministic engines; everything the RL agent can
reach is a point in this lattice.

Two-stage validation
  * Static (build time): a level is kept if it can *possibly* fit (using the most compact
    settings), so the lattice stays small.
  * Dynamic (decision time): :meth:`CandidateSpace.check` re-validates an exact position
    against live free memory, quality and the failure blacklist before it is applied.

Size: |space| = V * K * G * T * B, typically < 2,000 points; full enumeration (used only for
the fallback search) is cheap. Each check is O(1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..analysis.profile import ModelProfile
from ..config import ExecutionConfig, MemoryEstimate
from ..rl.actions import Action
from .placement import estimate_memory
from .precision import PrecisionCandidates


@dataclass(frozen=True, slots=True, order=True)
class Position:
    variant: int
    kv: int
    gpu: int
    threads: int
    batch: int


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    ram_bytes: int
    vram_bytes: int


class Rejection(str, Enum):
    OUT_OF_RANGE = "out_of_range"
    QUALITY = "quality_constraint"
    RAM = "insufficient_ram"
    VRAM = "insufficient_vram"
    BLACKLISTED = "previously_failed"


@dataclass(frozen=True, slots=True)
class CandidateSpace:
    precision: PrecisionCandidates
    gpu_levels: tuple[tuple[int, ...], ...]  # per variant, ascending, first == 0
    thread_levels: tuple[int, ...]
    batch_levels: tuple[int, ...]
    n_ctx: int
    gpu_context_bytes: int
    runtime_bytes: int
    # Live-check calibration (see estimate_memory). Not part of the lattice signature: the
    # lattice itself is always built uncalibrated, so learned coordinates stay stable.
    overhead_scale: float = 1.0

    # ---- lattice geometry ----------------------------------------------------------------

    def variant(self, pos: Position) -> ModelProfile:
        return self.precision.variants[pos.variant]

    def in_range(self, pos: Position) -> bool:
        return (
            0 <= pos.variant < len(self.precision.variants)
            and 0 <= pos.kv < len(self.precision.kv_types)
            and 0 <= pos.gpu < len(self.gpu_levels[pos.variant])
            and 0 <= pos.threads < len(self.thread_levels)
            and 0 <= pos.batch < len(self.batch_levels)
        )

    def n_gpu_layers(self, pos: Position) -> int:
        return self.gpu_levels[pos.variant][pos.gpu]

    def gpu_fraction(self, pos: Position) -> float:
        p = self.variant(pos)
        return min(self.n_gpu_layers(pos), p.n_layers + 1) / (p.n_layers + 1)

    def config_at(self, pos: Position) -> ExecutionConfig:
        p = self.variant(pos)
        kv = self.precision.kv_types[pos.kv]
        return ExecutionConfig(
            model_path=str(p.path),
            precision=p.quant.name,
            kv_type=kv,
            n_gpu_layers=self.n_gpu_layers(pos),
            n_threads=self.thread_levels[pos.threads],
            n_batch=self.batch_levels[pos.batch],
            n_ctx=self.n_ctx,
            flash_attn=kv.requires_flash_attention,
        )

    def estimate(self, pos: Position) -> MemoryEstimate:
        kv = self.precision.kv_types[pos.kv]
        return estimate_memory(
            self.variant(pos),
            n_gpu_layers=self.n_gpu_layers(pos),
            kv_type=kv,
            n_ctx=self.n_ctx,
            n_batch=self.batch_levels[pos.batch],
            flash_attn=kv.requires_flash_attention,
            gpu_context_bytes=self.gpu_context_bytes,
            runtime_bytes=self.runtime_bytes,
            overhead_scale=self.overhead_scale,
        )

    def uncalibrated_estimate(self, pos: Position) -> MemoryEstimate:
        """Estimate with overhead_scale=1.0 (used to derive calibration ratios)."""
        return replace(self, overhead_scale=1.0).estimate(pos)

    def weights_only_estimate(self, pos: Position) -> MemoryEstimate:
        """Estimate with zero overhead: the exact weight bytes per device."""
        return replace(self, overhead_scale=0.0).estimate(pos)

    def _remap_gpu(self, new_variant: int, n_layers_now: int) -> int:
        """Index of the largest level in ``new_variant`` that is <= the current layer count."""
        levels = self.gpu_levels[new_variant]
        best = 0
        for i, k in enumerate(levels):  # O(G), G <= ~6
            if k <= n_layers_now:
                best = i
        return best

    def neighbor(self, pos: Position, action: Action) -> Position | None:
        """Apply a relative action. Returns None if it leaves the lattice. O(1)."""
        v, kv, g, t, b = pos.variant, pos.kv, pos.gpu, pos.threads, pos.batch
        if action is Action.MAINTAIN:
            return pos
        if action in (Action.LOWER_PRECISION, Action.HIGHER_PRECISION):
            nv = v + 1 if action is Action.LOWER_PRECISION else v - 1
            if not 0 <= nv < len(self.precision.variants):
                return None
            new = Position(nv, kv, self._remap_gpu(nv, self.n_gpu_layers(pos)), t, b)
        elif action is Action.MORE_GPU_LAYERS:
            new = Position(v, kv, g + 1, t, b)
        elif action is Action.FEWER_GPU_LAYERS:
            new = Position(v, kv, g - 1, t, b)
        elif action is Action.MORE_THREADS:
            new = Position(v, kv, g, t + 1, b)
        elif action is Action.FEWER_THREADS:
            new = Position(v, kv, g, t - 1, b)
        elif action is Action.LARGER_BATCH:
            new = Position(v, kv, g, t, b + 1)
        elif action is Action.SMALLER_BATCH:
            new = Position(v, kv, g, t, b - 1)
        elif action is Action.LOWER_KV_PRECISION:
            new = Position(v, kv + 1, g, t, b)
        else:  # Action.HIGHER_KV_PRECISION
            new = Position(v, kv - 1, g, t, b)
        return new if self.in_range(new) else None

    # ---- safety ---------------------------------------------------------------------------

    def check(
        self, pos: Position, budget: MemoryBudget, blacklist: frozenset[str]
    ) -> Rejection | None:
        """Deterministic safety rules. Returns None if safe, else the reason. O(1)."""
        if not self.in_range(pos):
            return Rejection.OUT_OF_RANGE
        if not self.precision.is_quality_ok(pos.variant, pos.kv):
            return Rejection.QUALITY
        est = self.estimate(pos)
        if est.ram_bytes > budget.ram_bytes:
            return Rejection.RAM
        if est.vram_bytes > budget.vram_bytes:
            return Rejection.VRAM
        if self.config_at(pos).key in blacklist:
            return Rejection.BLACKLISTED
        return None

    def all_positions(self) -> list[Position]:
        return [
            Position(v, kv, g, t, b)
            for v in range(len(self.precision.variants))
            for kv in range(len(self.precision.kv_types))
            for g in range(len(self.gpu_levels[v]))
            for t in range(len(self.thread_levels))
            for b in range(len(self.batch_levels))
        ]

    def size(self) -> int:
        per_variant = (
            len(self.precision.kv_types) * len(self.thread_levels) * len(self.batch_levels)
        )
        return sum(len(levels) * per_variant for levels in self.gpu_levels)
