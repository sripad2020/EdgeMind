"""Inference backend contract (Engine 4's execution layer)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from ..analysis.quant import KV_LADDER, KVCacheType
from ..config import ExecutionConfig
from ..hardware.profiler import TelemetrySample

TokenCallback = Callable[[str], None]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    # Backends that know their exact footprint report it; otherwise the runtime uses
    # telemetry peaks.
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    # Exact decode throughput when the backend measures it (e.g. Ollama eval_duration).
    decode_tps: float | None = None

    @property
    def tokens_per_second(self) -> float:
        """Decode throughput (excludes prompt processing / time-to-first-token)."""
        if self.decode_tps is not None:
            return self.decode_tps
        decode_tokens = self.completion_tokens - 1
        decode_time = self.total_s - self.ttft_s
        if decode_tokens > 0 and decode_time > 0:
            return decode_tokens / decode_time
        return self.completion_tokens / self.total_s if self.total_s > 0 else 0.0


class InferenceBackend(ABC):
    """A backend loads exactly one configuration at a time.

    Contract:
      * ``load`` replaces any loaded configuration; it raises ``ModelLoadError`` (with
        ``FailureKind.OUT_OF_MEMORY`` when memory is the cause) and leaves nothing loaded.
      * ``generate``/``chat`` raise ``GenerationError`` for runtime failures and
        ``ValidationError`` for request problems (e.g. prompt longer than the context).
      * ``on_token`` (optional) receives text pieces as they are produced.
      * Backends are not thread-safe; the runtime serialises calls.
    """

    name: str = "abstract"

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def loaded_config(self) -> ExecutionConfig | None: ...

    @abstractmethod
    def load(self, config: ExecutionConfig) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(
        self, prompt: str, *, max_tokens: int, temperature: float, seed: int | None,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult: ...

    def chat(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float,
        seed: int | None, on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        """Default: render a plain transcript and complete it. Backends with native chat
        templates (e.g. Ollama) override this."""
        transcript = "".join(f"{m.role.value}: {m.content}\n" for m in messages)
        return self.generate(transcript + "assistant:", max_tokens=max_tokens,
                             temperature=temperature, seed=seed, on_token=on_token)

    def measured_footprint(self) -> tuple[int, int] | None:
        """(RAM bytes, VRAM bytes) the loaded configuration actually uses, if the backend can
        measure it. Used to calibrate memory estimates. Default: unknown."""
        return None

    def supported_kv_types(self) -> tuple[KVCacheType, ...]:
        """KV-cache types this backend can select per load. Default: all."""
        return KV_LADDER

    def adjust_sample(self, sample: TelemetrySample) -> TelemetrySample:
        """Hook to correct telemetry (simulated hardware, out-of-process servers)."""
        return sample
