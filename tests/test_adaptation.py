"""Calibration, memory-pressure relief (per request and while idle), background switching."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from air_sdk import AdaptiveModel, AdaptiveRuntime
from air_sdk._validate import MiB
from air_sdk.backends.simulated import SimulatedBackend, SimulatedHardware
from air_sdk.hardware.profiler import TelemetrySample
from air_sdk.rl.agent import QLearningParams

from .conftest import small_machine, small_safety


class PressureBackend(SimulatedBackend):
    """Simulated machine whose 'other programs' memory can be changed during a test."""

    extra_used = 0

    def adjust_sample(self, sample: TelemetrySample) -> TelemetrySample:
        s = super().adjust_sample(sample)
        avail = max(0, s.ram_available - self.extra_used)
        return replace(s, ram_available=avail,
                       ram_percent=100.0 * (s.ram_total - avail) / s.ram_total)


def _rt(tmp_path: Path, backend: SimulatedBackend, **kw: object) -> AdaptiveRuntime:
    hw, _ = small_machine(vram_mib=0)
    return AdaptiveRuntime(backend=backend, data_dir=tmp_path / "db", safety=small_safety(),
                           hardware=hw, seed=5, rl_params=QLearningParams(), **kw)  # type: ignore[arg-type]


def _squeeze(sdk: AdaptiveRuntime, backend: PressureBackend, m: AdaptiveModel) -> None:
    """Make another 'program' use memory so the live budget lands halfway between what the
    current config needs and the next-smaller configuration (the budget counts the memory
    the current model would release on reload)."""
    space, pos = m._planner.space, m._pos  # type: ignore[union-attr]
    assert pos is not None
    current = space.estimate(pos).ram_bytes
    smaller = sorted({space.estimate(p).ram_bytes for p in space.all_positions()
                      if space.estimate(p).ram_bytes < current})
    assert smaller, "test needs a smaller configuration to move to"
    sample = backend.adjust_sample(sdk.telemetry.sample_now())
    headroom = sdk.safety.ram_headroom(sample.ram_total)
    target_available = (current + smaller[-1]) // 2 - current + headroom
    assert target_available > 0
    backend.extra_used = sample.ram_available - target_available


def _sim(*, runtime_mib: int = 2) -> SimulatedHardware:
    _, base = small_machine(vram_mib=0)
    return replace(base.hardware, runtime_bytes=runtime_mib * MiB)


def test_calibration_shrinks_overestimates(tmp_path: Path, model_dir: Path) -> None:
    # The simulated machine really needs only 0.5 MiB of runtime overhead while the planner
    # assumes 2 MiB: after one load the estimate scale must drop below 1.
    with _rt(tmp_path, SimulatedBackend(_sim(runtime_mib=0), seed=1, noise=0.0)) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        m.optimize(n_ctx=1024)
        scale = m.performance_report()["memory_estimate_scale"]
        assert isinstance(scale, float)
        assert scale < 1.0
    with _rt(tmp_path, SimulatedBackend(_sim(runtime_mib=0), seed=1, noise=0.0)) as sdk2:
        m2 = sdk2.load_model(model_dir / "tiny-f16.gguf", discover=True)
        m2.optimize(n_ctx=1024)  # persisted: applies before the first load
        assert m2.performance_report()["memory_estimate_scale"] == scale


def test_calibration_never_inflates_exact_estimates(tmp_path: Path, model_dir: Path) -> None:
    with _rt(tmp_path, SimulatedBackend(_sim(), seed=1, noise=0.0)) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        m.optimize(n_ctx=1024)
        assert m.performance_report()["memory_estimate_scale"] == 1.0


def test_relief_before_request_keeps_best_safe_config(tmp_path: Path, model_dir: Path) -> None:
    backend = PressureBackend(_sim(), seed=1, noise=0.0)
    with _rt(tmp_path, backend) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024)
        m.generate("hello " * 20, max_tokens=32)
        # Another program grabs memory: the current config no longer fits, smaller ones do.
        _squeeze(sdk, backend, m)
        r = m.generate("hello " * 20, max_tokens=32)
        # Staying put is masked as unsafe, so either the agent picks a safe move itself or
        # the safety relief moves it; both must end on a smaller, safe configuration.
        assert r.relief or r.action != "MAINTAIN"
        assert r.config != plan.config
        space = m._planner.space  # type: ignore[union-attr]
        assert space.estimate(m._pos).ram_bytes < plan.estimate.ram_bytes  # type: ignore[arg-type]


def test_relief_when_agent_cannot_adapt(tmp_path: Path, model_dir: Path) -> None:
    backend = PressureBackend(_sim(), seed=1, noise=0.0)
    with _rt(tmp_path, backend, auto_adapt=False) as sdk:  # agent may not change config
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024)
        m.generate("hello " * 20, max_tokens=32)
        _squeeze(sdk, backend, m)
        r = m.generate("hello " * 20, max_tokens=32)
        assert r.relief  # the safety rule acted even though adaptation is off
        assert not r.learned or r.reward is None  # forced move is not learned from
        assert r.config != plan.config
        assert m.performance_report()["memory_pressure_reliefs"] == 1


def test_idle_watcher_relieves_pressure(tmp_path: Path, model_dir: Path) -> None:
    backend = PressureBackend(_sim(), seed=1, noise=0.0)
    with _rt(tmp_path, backend, background_switching=True, pressure_check_s=0.2) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024)
        m.generate("hello " * 20, max_tokens=32)
        time.sleep(0.3)  # let the background preparation finish
        _squeeze(sdk, backend, m)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not m.performance_report()["events"]:
            time.sleep(0.1)
        perf = m.performance_report()
        events = perf["events"]
        assert isinstance(events, list)
        assert events
        assert events[0]["when"] == "while idle"
        space = m._planner.space  # type: ignore[union-attr]
        assert space.estimate(m._pos).ram_bytes < plan.estimate.ram_bytes  # type: ignore[arg-type]
        # The next request runs on the relieved config without another switch.
        r = m.generate("hello " * 20, max_tokens=32)
        assert not r.rolled_back


def test_background_switching_prepares_next_config(tmp_path: Path, model_dir: Path) -> None:
    backend = SimulatedBackend(_sim(), seed=1, noise=0.0)
    with _rt(tmp_path, backend, background_switching=True) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        m.optimize(n_ctx=1024)
        responses = [m.generate("hello " * 20, max_tokens=32) for _ in range(25)]
        assert all(r.completion_tokens == 32 for r in responses)
        assert sum(r.prefetched for r in responses) == 24  # every reply after the first
        assert not any(r.rolled_back for r in responses)
        report = m.learning_report()
        assert report["episodes"] == 25
