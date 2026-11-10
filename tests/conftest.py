from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from air_sdk._validate import MiB
from air_sdk.backends.simulated import SimulatedBackend, SimulatedHardware
from air_sdk.config import SafetyPolicy
from air_sdk.hardware.profiler import GPUInfo, HardwareProfile
from air_sdk.testing import SyntheticModel, write_gguf

TINY = SyntheticModel()
_GiB = 1024**3


@pytest.fixture(autouse=True)
def _fixed_system_memory(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make tests hermetic: report a fixed 16 GB machine with 8 GB free, regardless of how
    busy the real computer is. Tests marked ``live`` use the real machine."""
    if request.node.get_closest_marker("live") is not None:
        return
    fixed = SimpleNamespace(total=16 * _GiB, available=8 * _GiB, percent=50.0,
                            used=8 * _GiB, free=8 * _GiB)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: fixed)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Written once per session (tests only read model files) to keep disk use minimal.
    d = tmp_path_factory.mktemp("models")
    write_gguf(d / "tiny-f16.gguf", TINY)
    write_gguf(d / "tiny-q8_0.gguf", SyntheticModel(file_type=7))
    write_gguf(d / "tiny-q4_0.gguf", SyntheticModel(file_type=2))
    return d


@pytest.fixture
def model_path(model_dir: Path) -> Path:
    return model_dir / "tiny-f16.gguf"


def small_safety() -> SafetyPolicy:
    return SafetyPolicy(
        ram_headroom_min=1 * MiB, vram_headroom_min=1 * MiB, ram_headroom_fraction=0.05,
        vram_headroom_fraction=0.05, max_quality_loss=0.06,
        reserve_gpu_context_bytes=1 * MiB, reserve_runtime_bytes=2 * MiB,
    )


def small_machine(*, vram_mib: int = 4, gpu_context_mib: int = 1) -> tuple[
    HardwareProfile, SimulatedBackend
]:
    ram = 64 * MiB
    gpus = (GPUInfo(0, "SimGPU", vram_mib * MiB, vram_mib * MiB, (8, 6)),) if vram_mib else ()
    hw = HardwareProfile("SimCPU", 4, 8, ram, ram - 16 * MiB, "SimOS", gpus)
    sim = SimulatedHardware(
        ram_bytes=ram, background_ram_bytes=16 * MiB, vram_bytes=vram_mib * MiB,
        physical_cores=4, logical_cores=8, cpu_bandwidth_gbs=0.05, gpu_bandwidth_gbs=0.5,
        gpu_context_bytes=gpu_context_mib * MiB, runtime_bytes=2 * MiB,
    )
    return hw, SimulatedBackend(sim, seed=1, noise=0.0)
