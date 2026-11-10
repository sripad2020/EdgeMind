"""llama.cpp backend via ``llama-cpp-python`` (optional dependency).

Security notes
  * The GGUF header has already been validated by AIR-SDK's own parser before the path
    reaches llama.cpp.
  * Only raw text completion is used; GGUF-embedded chat templates (Jinja) are never
    rendered by this backend.
  * Values coming back from the untyped binding are type-checked before use.
"""

from __future__ import annotations

import gc
import time
from typing import Any

from .._validate import require_str
from ..config import ExecutionConfig
from ..errors import (
    BackendUnavailableError,
    FailureKind,
    GenerationError,
    ModelLoadError,
    ValidationError,
)
from .base import GenerationResult, InferenceBackend, TokenCallback

_OOM_MARKERS = ("out of memory", "failed to allocate", "cudamalloc", "alloc", "oom")


def _classify(exc: BaseException) -> FailureKind:
    if isinstance(exc, MemoryError):
        return FailureKind.OUT_OF_MEMORY
    text = str(exc).lower()
    return FailureKind.OUT_OF_MEMORY if any(m in text for m in _OOM_MARKERS) else (
        FailureKind.LOAD_FAILED
    )


class LlamaCppBackend(InferenceBackend):
    name = "llama_cpp"

    def __init__(self) -> None:
        try:
            import llama_cpp  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise BackendUnavailableError(
                "llama-cpp-python is not installed; install with: pip install 'air-sdk[llama]'"
            ) from exc
        self._mod: Any = llama_cpp
        self._llm: Any = None
        self._config: ExecutionConfig | None = None

    @property
    def version(self) -> str:
        v = getattr(self._mod, "__version__", "unknown")
        return v[:64] if isinstance(v, str) else "unknown"

    @property
    def loaded_config(self) -> ExecutionConfig | None:
        return self._config

    def load(self, config: ExecutionConfig) -> None:
        self.unload()
        try:
            self._llm = self._mod.Llama(
                model_path=config.model_path,
                n_gpu_layers=config.n_gpu_layers,
                n_threads=config.n_threads,
                n_threads_batch=config.n_threads,
                n_batch=config.n_batch,
                n_ctx=config.n_ctx,
                type_k=config.kv_type.ggml_type,
                type_v=config.kv_type.ggml_type,
                flash_attn=config.flash_attn,
                use_mmap=True,
                verbose=False,
            )
        except (MemoryError, ValueError, RuntimeError, OSError) as exc:
            self._llm = None
            raise ModelLoadError(f"llama.cpp failed to load: {exc}", _classify(exc)) from exc
        self._config = config

    def unload(self) -> None:
        llm, self._llm, self._config = self._llm, None, None
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            del llm
            gc.collect()

    def generate(
        self, prompt: str, *, max_tokens: int, temperature: float, seed: int | None,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        if self._llm is None:
            raise GenerationError("no configuration loaded")
        start = time.perf_counter()
        ttft: float | None = None
        pieces: list[str] = []
        chunks = 0
        try:
            prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8"), add_bos=True))
            stream = self._llm.create_completion(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature, stream=True,
                seed=seed,
            )
            for chunk in stream:
                text = chunk["choices"][0]["text"]
                if ttft is None:
                    ttft = time.perf_counter() - start
                piece = require_str("completion chunk", text, max_len=1 << 20,
                                    allow_empty=True)
                pieces.append(piece)
                chunks += 1  # llama-cpp-python streams ~one token per chunk
                if on_token is not None and piece:
                    on_token(piece)
        except ValueError as exc:
            # e.g. "Requested tokens exceed context window" - a request problem.
            raise ValidationError(str(exc)) from exc
        except (MemoryError, RuntimeError, KeyError, IndexError, TypeError) as exc:
            kind = _classify(exc)
            raise GenerationError(
                f"generation failed: {exc}",
                kind if kind is FailureKind.OUT_OF_MEMORY else FailureKind.RUNTIME_ERROR,
            ) from exc
        total = time.perf_counter() - start
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=prompt_tokens,
            completion_tokens=chunks,
            ttft_s=ttft if ttft is not None else total,
            total_s=total,
        )
