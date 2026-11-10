"""Hardware Profiler -> Hardware Capability Profile (static) and live samples."""

from __future__ import annotations

import hashlib
import os
import platform
import time
from dataclasses import dataclass, field

import psutil

from .._validate import GiB
from . import nvml


@dataclass(frozen=True, slots=True)
class GPUInfo:
    index: int
    name: str
    total_vram: int
    free_vram: int
    compute_capability: tuple[int, int] | None = None
    driver: str = "unknown"

    @property
    def supports_bf16(self) -> bool:
        return self.compute_capability is not None and self.compute_capability >= (8, 0)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Hardware Capability Profile."""

    cpu_name: str
    physical_cores: int
    logical_cores: int
    total_ram: int
    available_ram: int
    os_name: str
    gpus: tuple[GPUInfo, ...] = field(default_factory=tuple)

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def fingerprint(self) -> str:
        """Stable identity of the machine *class* (not of transient free memory)."""
        gpu_part = [(g.name, g.total_vram // GiB, g.compute_capability) for g in self.gpus]
        raw = repr(
            (self.cpu_name, self.physical_cores, self.logical_cores, self.total_ram // GiB,
             self.os_name, gpu_part)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def as_dict(self) -> dict[str, object]:
        return {
            "cpu": self.cpu_name,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "total_ram_gib": round(self.total_ram / GiB, 2),
            "available_ram_gib": round(self.available_ram / GiB, 2),
            "os": self.os_name,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_vram_gib": round(g.total_vram / GiB, 2),
                    "free_vram_gib": round(g.free_vram / GiB, 2),
                    "compute_capability": g.compute_capability,
                }
                for g in self.gpus
            ],
        }


def profile_hardware() -> HardwareProfile:
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    vm = psutil.virtual_memory()
    gpus = tuple(
        GPUInfo(d.index, d.name, d.total_vram, d.free_vram, d.compute_capability, d.driver)
        for d in nvml.describe_devices()
    )
    return HardwareProfile(
        cpu_name=(platform.processor() or platform.machine() or "unknown")[:256],
        physical_cores=max(1, int(physical)),
        logical_cores=max(1, int(logical)),
        total_ram=int(vm.total),
        available_ram=int(vm.available),
        os_name=f"{platform.system()} {platform.release()}"[:128],
        gpus=gpus,
    )


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    timestamp: float
    ram_total: int
    ram_available: int
    ram_percent: float
    process_rss: int
    cpu_percent: float
    vram_used: int | None = None
    vram_total: int | None = None
    gpu_percent: int | None = None
    temperature_c: int | None = None


def take_sample(process: psutil.Process, gpu_index: int | None) -> TelemetrySample:
    """One O(1) telemetry sample. ``cpu_percent(None)`` is non-blocking."""
    vm = psutil.virtual_memory()
    try:
        rss = int(process.memory_info().rss)
    except (psutil.Error, OSError):
        rss = 0
    gpu = nvml.sample(gpu_index) if gpu_index is not None else None
    return TelemetrySample(
        timestamp=time.monotonic(),
        ram_total=int(vm.total),
        ram_available=int(vm.available),
        ram_percent=float(vm.percent),
        process_rss=rss,
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        vram_used=gpu.vram_used if gpu else None,
        vram_total=gpu.vram_total if gpu else None,
        gpu_percent=gpu.utilization_pct if gpu else None,
        temperature_c=gpu.temperature_c if gpu else None,
    )
