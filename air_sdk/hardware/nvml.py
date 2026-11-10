"""Thin, typed, fail-safe wrapper over NVML (``nvidia-ml-py``).

NVML is optional. Every call is wrapped so a missing driver, an unplugged GPU or an
NVML error degrades to "no data" instead of crashing the runtime. All values returned by
the untyped C binding are validated before they enter typed SDK code.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Final

_lock: Final = threading.Lock()
_state: dict[str, ModuleType | bool | None] = {"module": None, "tried": False}


def _nvml() -> ModuleType | None:
    """Initialise NVML once (thread-safe). Returns the module or None if unavailable."""
    with _lock:
        if _state["tried"]:
            mod = _state["module"]
            return mod if isinstance(mod, ModuleType) else None
        _state["tried"] = True
        try:
            import pynvml  # noqa: PLC0415  (optional dependency, imported lazily)

            pynvml.nvmlInit()
        except Exception:  # noqa: BLE001 - any NVML/import failure means "no GPU data"
            return None
        if not isinstance(pynvml, ModuleType):
            return None
        _state["module"] = pynvml
        atexit.register(_shutdown)
        return pynvml


def _shutdown() -> None:
    mod = _state.get("module")
    if isinstance(mod, ModuleType):
        # Best-effort cleanup at interpreter exit; NVML may already be torn down.
        with contextlib.suppress(Exception):
            mod.nvmlShutdown()


def _as_nonneg_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _as_text(value: object, limit: int = 256) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[:limit] if isinstance(value, str) else "unknown"


@dataclass(frozen=True, slots=True)
class NvmlDevice:
    index: int
    name: str
    total_vram: int
    free_vram: int
    compute_capability: tuple[int, int] | None
    driver: str


@dataclass(frozen=True, slots=True)
class NvmlSample:
    index: int
    vram_used: int
    vram_total: int
    utilization_pct: int | None
    temperature_c: int | None


def device_count() -> int:
    mod = _nvml()
    if mod is None:
        return 0
    try:
        return _as_nonneg_int(mod.nvmlDeviceGetCount()) or 0
    except Exception:  # noqa: BLE001
        return 0


def describe_devices() -> tuple[NvmlDevice, ...]:
    mod = _nvml()
    if mod is None:
        return ()
    devices: list[NvmlDevice] = []
    try:
        driver = _as_text(mod.nvmlSystemGetDriverVersion())
    except Exception:  # noqa: BLE001
        driver = "unknown"
    for i in range(device_count()):
        try:
            handle = mod.nvmlDeviceGetHandleByIndex(i)
            mem = mod.nvmlDeviceGetMemoryInfo(handle)
            total = _as_nonneg_int(getattr(mem, "total", None))
            free = _as_nonneg_int(getattr(mem, "free", None))
            if total is None or free is None:
                continue
            try:
                major, minor = mod.nvmlDeviceGetCudaComputeCapability(handle)
                cc: tuple[int, int] | None = (int(major), int(minor))
            except Exception:  # noqa: BLE001
                cc = None
            devices.append(
                NvmlDevice(i, _as_text(mod.nvmlDeviceGetName(handle)), total, free, cc, driver)
            )
        except Exception:  # noqa: BLE001, S112 - skip devices NVML cannot query
            continue
    return tuple(devices)


def sample(index: int) -> NvmlSample | None:
    mod = _nvml()
    if mod is None:
        return None
    try:
        handle = mod.nvmlDeviceGetHandleByIndex(index)
        mem = mod.nvmlDeviceGetMemoryInfo(handle)
        used = _as_nonneg_int(getattr(mem, "used", None))
        total = _as_nonneg_int(getattr(mem, "total", None))
        if used is None or total is None:
            return None
        try:
            util = _as_nonneg_int(getattr(mod.nvmlDeviceGetUtilizationRates(handle), "gpu", None))
        except Exception:  # noqa: BLE001
            util = None
        try:
            temp = _as_nonneg_int(mod.nvmlDeviceGetTemperature(handle, mod.NVML_TEMPERATURE_GPU))
        except Exception:  # noqa: BLE001
            temp = None
        return NvmlSample(index, used, total, util, temp)
    except Exception:  # noqa: BLE001
        return None
