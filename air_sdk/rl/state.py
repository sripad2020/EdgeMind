"""RL state: discretised hardware/workload conditions plus the current lattice position.

The relative actions ("more GPU layers") only make sense relative to where we are, so the
state includes the current configuration coordinates in addition to the hardware levels in
the design document.

State-space bound (finite by construction, so the Q-table cannot grow without limit):
  4 (RAM) * 4 (VRAM) * 3 (CPU) * 4 (GPU) * 3 (pressure) * 3 (size) * 3 (ctx) * 2 (thermal)
  * |lattice positions|. Only *visited* states are stored (sparse dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..analysis.profile import ModelSizeClass


class Level(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Pressure(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class ContextLevel(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Thermal(str, Enum):
    NORMAL = "normal"
    THROTTLED = "throttled"


@dataclass(frozen=True, slots=True)
class RLState:
    ram: Level
    vram: Level
    cpu: Level
    gpu: Level
    pressure: Pressure
    model_size: ModelSizeClass
    context: ContextLevel
    thermal: Thermal
    variant: int
    kv: int
    gpu_bucket: int
    threads: int
    batch: int

    def key(self) -> str:
        """Canonical text key (used as dict key and DB column; never pickled)."""
        return (
            f"ram={self.ram.value}|vram={self.vram.value}|cpu={self.cpu.value}"
            f"|gpu={self.gpu.value}|mem={self.pressure.value}|size={self.model_size.value}"
            f"|ctx={self.context.value}|therm={self.thermal.value}|v={self.variant}"
            f"|kv={self.kv}|g={self.gpu_bucket}|t={self.threads}|b={self.batch}"
        )


@dataclass(frozen=True, slots=True)
class StateInputs:
    ram_available: int
    ram_percent: float
    cpu_percent: float
    vram_free: int | None  # None => no GPU
    vram_percent: float | None
    gpu_percent: int | None
    temperature_c: int | None
    model_weight_bytes: int
    model_size: ModelSizeClass
    request_tokens: int
    gpu_fraction: float
    thermal_limit_c: int


def _ratio_level(available: int, need: int) -> Level:
    ratio = available / max(1, need)
    if ratio < 0.5:
        return Level.LOW
    if ratio < 1.5:
        return Level.MEDIUM
    return Level.HIGH


def _util_level(pct: float | None) -> Level:
    if pct is None:
        return Level.NONE
    if pct < 30:
        return Level.LOW
    if pct < 75:
        return Level.MEDIUM
    return Level.HIGH


def _gpu_bucket(fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    if fraction < 0.34:
        return 1
    if fraction < 0.67:
        return 2
    if fraction < 1.0:
        return 3
    return 4


def discretize(
    x: StateInputs, *, variant: int, kv: int, threads: int, batch: int
) -> RLState:
    """Map continuous measurements to a discrete state. O(1)."""
    worst_pct = max(x.ram_percent, x.vram_percent or 0.0)
    if worst_pct >= 95:
        pressure = Pressure.CRITICAL
    elif worst_pct >= 85:
        pressure = Pressure.WARNING
    else:
        pressure = Pressure.NORMAL

    if x.request_tokens < 1024:
        context = ContextLevel.SHORT
    elif x.request_tokens < 4096:
        context = ContextLevel.MEDIUM
    else:
        context = ContextLevel.LONG

    hot = x.temperature_c is not None and x.temperature_c >= x.thermal_limit_c - 5
    return RLState(
        ram=_ratio_level(x.ram_available, x.model_weight_bytes),
        vram=Level.NONE if x.vram_free is None else _ratio_level(x.vram_free,
                                                                 x.model_weight_bytes),
        cpu=_util_level(x.cpu_percent),
        gpu=_util_level(None if x.gpu_percent is None else float(x.gpu_percent)),
        pressure=pressure,
        model_size=x.model_size,
        context=context,
        thermal=Thermal.THROTTLED if hot else Thermal.NORMAL,
        variant=variant,
        kv=kv,
        gpu_bucket=_gpu_bucket(x.gpu_fraction),
        threads=threads,
        batch=batch,
    )
