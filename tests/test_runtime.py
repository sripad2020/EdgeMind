from __future__ import annotations

from pathlib import Path

import pytest

from air_sdk import AdaptiveRuntime, ExecutionConfig, ModelLoadError, ValidationError
from air_sdk.backends.simulated import SimulatedBackend
from air_sdk.errors import FailureKind
from air_sdk.rl.agent import QLearningParams

from .conftest import small_machine, small_safety


def _runtime(tmp_path: Path, *, vram_mib: int = 4, gpu_context_mib: int = 1,
             rl: bool = True) -> AdaptiveRuntime:
    hw, backend = small_machine(vram_mib=vram_mib, gpu_context_mib=gpu_context_mib)
    return AdaptiveRuntime(
        "balanced", rl_enabled=rl, auto_adapt=True, backend=backend, data_dir=tmp_path / "db",
        safety=small_safety(), hardware=hw, seed=7,
        rl_params=QLearningParams(epsilon_start=0.5, epsilon_decay=0.95),
    )


def test_end_to_end_learning_loop(tmp_path: Path, model_dir: Path) -> None:
    with _runtime(tmp_path) as sdk:
        model = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        assert len(model.variants) == 3
        plan = model.optimize(n_ctx=1024)
        assert plan.source in {"heuristic", "learned"}
        for _ in range(40):
            r = model.generate("Explain how QUIC works " * 4, max_tokens=64)
            assert r.completion_tokens == 64
            assert r.tokens_per_second > 0
        report = model.learning_report()
        assert report["episodes"] == 40
        assert isinstance(report["q_states"], int)
        assert report["q_states"] > 0
        perf = model.performance_report()
        assert perf["requests"] == 40
        assert sdk.rl_engine.get_learning_report()[0]["episodes"] == 40


def test_learning_persists_across_runtimes(tmp_path: Path, model_dir: Path) -> None:
    with _runtime(tmp_path) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        m.optimize(n_ctx=1024)
        for _ in range(15):
            m.generate("hello world", max_tokens=32)
    with _runtime(tmp_path) as sdk2:
        m2 = sdk2.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m2.optimize(n_ctx=1024)
        assert plan.source == "learned"
        assert m2.learning_report()["episodes"] == 15


def test_optimize_rolls_back_when_estimate_is_optimistic(tmp_path: Path,
                                                         model_dir: Path) -> None:
    # The simulated GPU's true context cost (3 MiB) exceeds the planner's estimate (1 MiB),
    # so the planner's first GPU choice OOMs at load and must be rolled back.
    with _runtime(tmp_path, vram_mib=4, gpu_context_mib=3) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024)
        assert plan.source == "conservative"
        assert plan.config.n_gpu_layers == 0
        assert m.learning_report()["blacklisted_configs"] == 1
        assert all(m.generate("hi", max_tokens=8).completion_tokens == 8 for _ in range(20))


class _OOMOnChangeBackend(SimulatedBackend):
    """Loads succeed until armed; afterwards only the 'home' config can be loaded."""

    home: ExecutionConfig | None = None

    def load(self, config: ExecutionConfig) -> None:
        if self.home is not None and config != self.home:
            self.unload()
            raise ModelLoadError("injected OOM", FailureKind.OUT_OF_MEMORY)
        super().load(config)


def test_generate_oom_penalised_blacklisted_and_rolled_back(tmp_path: Path,
                                                            model_dir: Path) -> None:
    hw, sim = small_machine(vram_mib=0)
    backend = _OOMOnChangeBackend(sim.hardware, seed=1, noise=0.0)
    with AdaptiveRuntime(backend=backend, data_dir=tmp_path / "db", safety=small_safety(),
                         hardware=hw, seed=7,
                         rl_params=QLearningParams(epsilon_start=1.0, epsilon_min=1.0,
                                                   epsilon_decay=1.0)) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024)
        backend.home = plan.config
        responses = [m.generate("hi there", max_tokens=16) for _ in range(30)]
        assert all(r.completion_tokens == 16 for r in responses)  # caller always answered
        rolled = [r for r in responses if r.rolled_back]
        assert rolled
        assert all(r.reward == -5.0 for r in rolled)  # failure penalty applied
        assert all(r.config == plan.config for r in responses)  # always on known-good config
        report = m.learning_report()
        assert isinstance(report["blacklisted_configs"], int)
        assert report["blacklisted_configs"] >= 1


def test_cpu_only_machine(tmp_path: Path, model_dir: Path) -> None:
    with _runtime(tmp_path, vram_mib=0) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf")
        plan = m.optimize(n_ctx=1024)
        assert plan.config.n_gpu_layers == 0
        for _ in range(10):
            assert m.generate("x", max_tokens=8).config.n_gpu_layers == 0


def test_input_validation(tmp_path: Path, model_dir: Path) -> None:
    with _runtime(tmp_path) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf")
        m.optimize(n_ctx=256)
        with pytest.raises(ValidationError):
            m.generate("x", max_tokens=True)
        with pytest.raises(ValidationError):
            m.generate(123)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            m.generate("x", max_tokens=10_000)  # exceeds n_ctx
        with pytest.raises(ValidationError):
            m.optimize(rl_policy="dqn")
        with pytest.raises(ValidationError):
            sdk.load_model(tmp_path / "nope.gguf")


def test_rl_disabled_keeps_config(tmp_path: Path, model_dir: Path) -> None:
    with _runtime(tmp_path, rl=False) as sdk:
        m = sdk.load_model(model_dir / "tiny-f16.gguf", discover=True)
        plan = m.optimize(n_ctx=1024)
        for _ in range(5):
            r = m.generate("x", max_tokens=8)
            assert r.config == plan.config
            assert r.reward is None
