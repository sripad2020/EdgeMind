"""Deterministic *simulated* backend - for development, CI and demos only.

It models a machine with a roofline-style cost model (decode is memory-bandwidth bound,
prefill is compute bound, CPU-GPU splits pay a transfer cost) and raises OOM exactly when
a configuration exceeds the simulated memory. It produces no real text.

Use it to exercise the full RL loop (state -> action -> reward -> Q update -> rollback)
without llama.cpp or a GPU. Never use its numbers as benchmark results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from .._validate import MiB, clamp, require_float, require_int
from ..analysis.profile import ModelProfile, analyze_model
from ..config import ExecutionConfig, MemoryEstimate
from ..engines.placement import estimate_memory
from ..errors import FailureKind, GenerationError, ModelLoadError, ValidationError
from ..hardware.profiler import HardwareProfile, TelemetrySample
from .base import GenerationResult, InferenceBackend, TokenCallback


@dataclass(frozen=True, slots=True)
class SimulatedHardware:
    ram_bytes: int
    background_ram_bytes: int
    vram_bytes: int
    physical_cores: int
    logical_cores: int
    cpu_bandwidth_gbs: float = 40.0
    gpu_bandwidth_gbs: float = 360.0
    cpu_gflops_per_core: float = 40.0
    gpu_tflops: float = 20.0
    transfer_overhead_ms: float = 1.5
    # "True" runtime reserves of the simulated machine; may differ from the planner's
    # estimates, which lets tests exercise OOM -> rollback.
    gpu_context_bytes: int = 384 * MiB
    runtime_bytes: int = 256 * MiB

    def __post_init__(self) -> None:
        require_int("ram_bytes", self.ram_bytes, min_value=MiB)
        require_int("gpu_context_bytes", self.gpu_context_bytes, min_value=0)
        require_int("runtime_bytes", self.runtime_bytes, min_value=0)
        require_int("background_ram_bytes", self.background_ram_bytes, min_value=0,
                    max_value=self.ram_bytes)
        require_int("vram_bytes", self.vram_bytes, min_value=0)
        require_int("physical_cores", self.physical_cores, min_value=1)
        require_int("logical_cores", self.logical_cores, min_value=self.physical_cores)
        for name in ("cpu_bandwidth_gbs", "gpu_bandwidth_gbs", "cpu_gflops_per_core",
                     "gpu_tflops"):
            require_float(name, getattr(self, name), min_value=1e-3)
        require_float("transfer_overhead_ms", self.transfer_overhead_ms, min_value=0.0)

    @classmethod
    def from_profile(cls, hw: HardwareProfile) -> SimulatedHardware:
        gpu = hw.primary_gpu
        return cls(
            ram_bytes=hw.total_ram,
            background_ram_bytes=max(0, hw.total_ram - hw.available_ram),
            vram_bytes=gpu.total_vram if gpu else 0,
            physical_cores=hw.physical_cores,
            logical_cores=hw.logical_cores,
        )


class SimulatedBackend(InferenceBackend):
    name = "simulated"

    def __init__(self, hardware: SimulatedHardware, *, seed: int | None = 0,
                 noise: float = 0.03) -> None:
        self._hw = hardware
        self._noise = require_float("noise", noise, min_value=0.0, max_value=0.5)
        self._rng = random.Random(seed)
        self._profiles: dict[str, ModelProfile] = {}
        self._config: ExecutionConfig | None = None
        self._estimate: MemoryEstimate | None = None

    @property
    def version(self) -> str:
        return "sim-1"

    @property
    def hardware(self) -> SimulatedHardware:
        return self._hw

    @property
    def loaded_config(self) -> ExecutionConfig | None:
        return self._config

    def _profile(self, path: str) -> ModelProfile:
        cached = self._profiles.get(path)
        if cached is None:
            cached = analyze_model(Path(path))
            self._profiles[path] = cached
        return cached

    def load(self, config: ExecutionConfig) -> None:
        self.unload()
        p = self._profile(config.model_path)
        est = estimate_memory(
            p, n_gpu_layers=config.n_gpu_layers, kv_type=config.kv_type, n_ctx=config.n_ctx,
            n_batch=config.n_batch, flash_attn=config.flash_attn,
            gpu_context_bytes=self._hw.gpu_context_bytes, runtime_bytes=self._hw.runtime_bytes,
        )
        if config.n_gpu_layers > 0 and self._hw.vram_bytes == 0:
            raise ModelLoadError("no GPU in simulated hardware", FailureKind.LOAD_FAILED)
        if est.vram_bytes > self._hw.vram_bytes:
            raise ModelLoadError("simulated CUDA out of memory", FailureKind.OUT_OF_MEMORY)
        if est.ram_bytes > self._hw.ram_bytes - self._hw.background_ram_bytes:
            raise ModelLoadError("simulated host out of memory", FailureKind.OUT_OF_MEMORY)
        self._config, self._estimate = config, est

    def unload(self) -> None:
        self._config, self._estimate = None, None

    def measured_footprint(self) -> tuple[int, int] | None:
        est = self._estimate
        return None if est is None else (est.ram_bytes, est.vram_bytes)

    def _jitter(self) -> float:
        return clamp(self._rng.gauss(1.0, self._noise), 0.5, 1.5) if self._noise else 1.0

    def generate(
        self, prompt: str, *, max_tokens: int, temperature: float, seed: int | None,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        cfg, est = self._config, self._estimate
        if cfg is None or est is None:
            raise GenerationError("no configuration loaded")
        p = self._profile(cfg.model_path)
        prompt_tokens = max(1, len(prompt) // 4)
        if prompt_tokens + max_tokens > cfg.n_ctx:
            raise ValidationError(
                f"prompt ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds n_ctx {cfg.n_ctx}"
            )
        hw = self._hw
        gpu_layers = min(cfg.n_gpu_layers, p.n_layers)
        gpu_bytes = p.layer_suffix_bytes[gpu_layers] + (
            p.output_bytes if cfg.n_gpu_layers > p.n_layers else 0
        )
        cpu_bytes = max(0, p.weight_bytes - gpu_bytes - p.other_bytes)  # embeddings: lookup
        gpu_frac = gpu_bytes / max(1, p.weight_bytes - p.other_bytes)

        # Thread scaling: saturates at physical cores; SMT oversubscription costs ~15%.
        eff = min(cfg.n_threads, hw.physical_cores) / hw.physical_cores
        if cfg.n_threads > hw.physical_cores:
            eff *= 0.85
        cpu_bw = hw.cpu_bandwidth_gbs * 1e9 * (0.35 + 0.65 * eff)
        gpu_bw = hw.gpu_bandwidth_gbs * 1e9
        split = 0 < cfg.n_gpu_layers <= p.n_layers
        ctx_tokens = prompt_tokens + max_tokens // 2
        kv_read = p.kv_bytes_per_token(cfg.kv_type) * ctx_tokens
        kv_bw = gpu_bw if gpu_layers == p.n_layers else cpu_bw
        token_s = (
            cpu_bytes / cpu_bw
            + gpu_bytes / gpu_bw
            + kv_read / kv_bw
            + (hw.transfer_overhead_ms / 1e3 if split else 0.0)
        )
        # Prefill: 2 * params FLOPs per token; batching improves utilisation.
        batch_eff = clamp((cfg.n_batch / 512) ** 0.3, 0.3, 1.0)
        flops = 2.0 * p.n_params * prompt_tokens
        cpu_flops = hw.cpu_gflops_per_core * 1e9 * max(1.0, eff * hw.physical_cores)
        prefill_s = (
            flops * (1 - gpu_frac) / (cpu_flops * batch_eff)
            + flops * gpu_frac / (hw.gpu_tflops * 1e12 * batch_eff)
        )
        ttft = (prefill_s + token_s) * self._jitter()
        total = ttft + (max_tokens - 1) * token_s * self._jitter()
        text = f"[simulated completion: {max_tokens} tokens]"
        if on_token is not None:
            on_token(text)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=max_tokens,
            ttft_s=ttft,
            total_s=total,
            peak_ram_bytes=est.ram_bytes,
            peak_vram_bytes=est.vram_bytes,
        )

    def adjust_sample(self, sample: TelemetrySample) -> TelemetrySample:
        """Report the *simulated* machine's memory, so the planner sees consistent numbers."""
        hw, est = self._hw, self._estimate
        used_ram = hw.background_ram_bytes + (est.ram_bytes if est else 0)
        used_vram = est.vram_bytes if est else 0
        return replace(
            sample,
            ram_total=hw.ram_bytes,
            ram_available=max(0, hw.ram_bytes - used_ram),
            ram_percent=100.0 * used_ram / hw.ram_bytes,
            process_rss=est.ram_bytes if est else 0,
            vram_used=used_vram if hw.vram_bytes else None,
            vram_total=hw.vram_bytes if hw.vram_bytes else None,
            gpu_percent=(60 if used_vram else 0) if hw.vram_bytes else None,
        )
