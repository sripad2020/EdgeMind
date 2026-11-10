"""Core configuration value types shared by every engine."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from enum import Enum

from ._validate import GiB, MiB, require_float, require_int
from .analysis.quant import KVCacheType


class Objective(str, Enum):
    BALANCED = "balanced"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"


class DevicePlacement(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Deterministic hard constraints. The RL engine can never override these."""

    ram_headroom_fraction: float = 0.10
    ram_headroom_min: int = 1 * GiB
    vram_headroom_fraction: float = 0.08
    vram_headroom_min: int = 256 * MiB
    max_quality_loss: float = 0.05
    thermal_limit_c: int = 85
    max_prompt_chars: int = 1_000_000
    max_new_tokens: int = 32_768
    # Memory reserved for the GPU runtime context and host-side runtime structures.
    reserve_gpu_context_bytes: int = 384 * MiB
    reserve_runtime_bytes: int = 256 * MiB

    def __post_init__(self) -> None:
        require_int("reserve_gpu_context_bytes", self.reserve_gpu_context_bytes, min_value=0)
        require_int("reserve_runtime_bytes", self.reserve_runtime_bytes, min_value=0)
        require_float("ram_headroom_fraction", self.ram_headroom_fraction, min_value=0.0,
                      max_value=0.9)
        require_int("ram_headroom_min", self.ram_headroom_min, min_value=0)
        require_float("vram_headroom_fraction", self.vram_headroom_fraction, min_value=0.0,
                      max_value=0.9)
        require_int("vram_headroom_min", self.vram_headroom_min, min_value=0)
        require_float("max_quality_loss", self.max_quality_loss, min_value=0.0, max_value=1.0)
        require_int("thermal_limit_c", self.thermal_limit_c, min_value=40, max_value=120)
        require_int("max_prompt_chars", self.max_prompt_chars, min_value=1)
        require_int("max_new_tokens", self.max_new_tokens, min_value=1)

    def ram_headroom(self, total_ram: int) -> int:
        return max(self.ram_headroom_min, int(total_ram * self.ram_headroom_fraction))

    def vram_headroom(self, total_vram: int) -> int:
        return max(self.vram_headroom_min, int(total_vram * self.vram_headroom_fraction))


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    ram_bytes: int
    vram_bytes: int

    @property
    def total(self) -> int:
        return self.ram_bytes + self.vram_bytes


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """A fully specified, backend-agnostic execution configuration."""

    model_path: str
    precision: str
    kv_type: KVCacheType
    n_gpu_layers: int
    n_threads: int
    n_batch: int
    n_ctx: int
    flash_attn: bool

    def __post_init__(self) -> None:
        require_int("n_gpu_layers", self.n_gpu_layers, min_value=0, max_value=100_000)
        require_int("n_threads", self.n_threads, min_value=1, max_value=4_096)
        require_int("n_batch", self.n_batch, min_value=1, max_value=1 << 16)
        require_int("n_ctx", self.n_ctx, min_value=16, max_value=1 << 24)

    def as_dict(self) -> dict[str, object]:
        return {
            "model_path": self.model_path,
            "precision": self.precision,
            "kv_type": self.kv_type.value,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "n_ctx": self.n_ctx,
            "flash_attn": self.flash_attn,
        }

    @property
    def key(self) -> str:
        """Canonical, injective string identity (stable across runs)."""
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def shape_key(self) -> str:
        """Identity *without* the context size. The context grows automatically during a
        conversation; learned statistics about a configuration must survive that."""
        return shape_key(self.key)


@functools.lru_cache(maxsize=4096)
def shape_key(config_key: str) -> str:
    """Strip ``n_ctx`` from a canonical config key. Cached: O(1) amortised.
    Keys that are not valid config JSON (e.g. from a tampered DB) map to themselves."""
    try:
        data = json.loads(config_key)
    except (json.JSONDecodeError, TypeError):
        return config_key
    if not isinstance(data, dict):
        return config_key
    data.pop("n_ctx", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
