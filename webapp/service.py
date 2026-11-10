from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from air_sdk import AdaptiveModel, AdaptiveRuntime, AirError, ValidationError
from air_sdk.backends.ollama import OllamaModelInfo

_MODELS_TTL_S = 30.0


class BusyError(AirError):
    """A generation is already in progress."""


class NoSessionError(AirError):
    """No model has been loaded yet."""


@dataclass(frozen=True, slots=True)
class SessionSettings:
    model: str
    device: str
    objective: str
    rl_enabled: bool
    auto_adapt: bool
    n_ctx: int
    adapt_every: int
    probe: bool = True
    auto_context: bool = True


class ChatService:
    def __init__(self, data_dir: Path, ollama_url: str) -> None:
        self._data_dir = data_dir
        self._ollama_url = ollama_url
        self._lock = threading.RLock()  # guards runtime/session swaps
        self._busy = threading.Lock()  # held for the duration of one generation
        self._runtime: AdaptiveRuntime | None = None
        self._objective: str | None = None
        self._model: AdaptiveModel | None = None
        self._settings: SessionSettings | None = None
        self._models_cache: tuple[float, list[OllamaModelInfo]] | None = None

    # ---- lifecycle --------------------------------------------------------------------------

    def _runtime_for(self, objective: str) -> AdaptiveRuntime:
        if self._runtime is None or self._objective != objective:
            if self._runtime is not None:
                self._runtime.close()
                self._model = None
            self._runtime = AdaptiveRuntime(
                objective, backend="ollama", data_dir=self._data_dir,
                ollama_url=self._ollama_url,
                # Interactive chat has idle time between messages: prepare configuration
                # changes then, and watch memory pressure while idle.
                background_switching=True,
            )
            self._objective = objective
        return self._runtime

    def close(self) -> None:
        with self._lock:
            if self._runtime is not None:
                self._runtime.close()
            self._runtime, self._model, self._settings = None, None, None

    # ---- queries ----------------------------------------------------------------------------

    def hardware(self) -> dict[str, object]:
        with self._lock:
            rt = self._runtime_for(self._objective or "balanced")
            return rt.hardware_profile().as_dict()

    def models(self, *, refresh: bool = False) -> list[OllamaModelInfo]:
        with self._lock:
            now = time.monotonic()
            if (not refresh and self._models_cache is not None
                    and now - self._models_cache[0] < _MODELS_TTL_S):
                return self._models_cache[1]
            rt = self._runtime_for(self._objective or "balanced")
            models = rt.list_ollama_models()
            self._models_cache = (now, models)
            return models

    def _require_model(self) -> AdaptiveModel:
        if self._model is None:
            raise NoSessionError("no model loaded; start a session first")
        return self._model

    # ---- session ----------------------------------------------------------------------------

    def start_session(self, s: SessionSettings) -> dict[str, object]:
        if not self._busy.acquire(blocking=False):
            raise BusyError("a generation is in progress")
        try:
            with self._lock:
                allowed = {m.name for m in self.models()}
                if s.model not in allowed:  # allow-list against the Ollama catalogue
                    raise ValidationError(f"unknown or non-chat model {s.model!r}")
                rt = self._runtime_for(s.objective)
                if self._model is not None:
                    rt.release(self._model)
                    self._model = None
                rt.rl_enabled = s.rl_enabled
                rt.auto_adapt = s.auto_adapt
                rt.adapt_every = s.adapt_every
                model = rt.load_ollama_model(s.model, discover=True)
                ctx = min(s.n_ctx, model.analyze().context_length)
                try:
                    plan = model.optimize(device_placement=s.device, n_ctx=ctx,
                                          probe=s.probe, auto_context=s.auto_context)
                except AirError:
                    rt.release(model)
                    raise
                self._model, self._settings = model, s
                return {
                    "settings": asdict(s) | {"n_ctx": ctx},
                    "profile": model.analyze().as_dict(),
                    "variants": [
                        {"precision": v.quant.name, "weight_bytes": v.weight_bytes}
                        for v in model.variants
                    ],
                    "plan": plan.as_dict(),
                    "hardware": rt.hardware_profile().as_dict(),
                    "performance": model.performance_report(),
                    "quantization": rt.quantization_plan(s.model),
                }
        finally:
            self._busy.release()

    # ---- adding models ------------------------------------------------------------------------

    def pull(self, model: str, on_progress: Callable[[str, int | None, int | None], None]
             ) -> dict[str, object]:
        """Download any model into Ollama (library name or hf.co/<user>/<repo>[:quant])."""
        if not self._busy.acquire(blocking=False):
            raise BusyError("a generation is in progress")
        try:
            with self._lock:
                rt = self._runtime_for(self._objective or "balanced")
            name = rt.pull_model(model, on_progress)
            with self._lock:
                self._models_cache = None
            return {"model": name}
        finally:
            self._busy.release()

    # ---- automatic quantization ----------------------------------------------------------

    def _check_known(self, model: str) -> AdaptiveRuntime:
        if model not in {m.name for m in self.models()}:  # allow-list
            raise ValidationError(f"unknown or non-chat model {model!r}")
        return self._runtime_for(self._objective or "balanced")

    def quantization_plan(self, model: str) -> dict[str, object]:
        with self._lock:
            return self._check_known(model).quantization_plan(model)

    def _free_session(self, rt: AdaptiveRuntime) -> None:
        """Unload the chat model so evaluation/quantization has memory to work with."""
        if self._model is not None:
            rt.release(self._model)
            self._model, self._settings = None, None

    def evaluate_quality(self, model: str) -> dict[str, object]:
        """Measure each installed version's answer quality against the original."""
        if not self._busy.acquire(blocking=False):
            raise BusyError("a generation is in progress")
        try:
            with self._lock:
                rt = self._check_known(model)
                self._free_session(rt)
                return rt.evaluate_quality(model)
        finally:
            self._busy.release()

    def quantize(self, model: str) -> dict[str, object]:
        """Create Q8_0/Q4_0 copies, then measure their quality (CPU-heavy, about a minute
        for small models): blocks chat meanwhile and ends the current chat session."""
        if not self._busy.acquire(blocking=False):
            raise BusyError("a generation is in progress")
        try:
            with self._lock:
                rt = self._check_known(model)
                self._free_session(rt)
                t0 = time.monotonic()
                results = rt.auto_quantize(model)
                self._models_cache = None  # new models appear in the catalogue
                quality = rt.evaluate_quality(model)
                return {
                    "quality": quality,
                    "model": model,
                    "seconds": round(time.monotonic() - t0, 1),
                    "results": [
                        {"name": r.model_name, "quant": r.target.value,
                         "size_bytes": r.size_bytes, "seconds": r.seconds,
                         "created": r.created}
                        for r in results
                    ],
                    "plan": rt.quantization_plan(model),
                }
        finally:
            self._busy.release()

    def update_learning(self, *, rl_enabled: bool, auto_adapt: bool) -> None:
        with self._lock:
            if self._runtime is None:
                raise NoSessionError("no session")
            self._runtime.rl_enabled = rl_enabled
            self._runtime.auto_adapt = auto_adapt

    # ---- chat -------------------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
        on_token: Callable[[str], None],
        on_reset: Callable[[], None],
    ) -> dict[str, object]:
        if not self._busy.acquire(blocking=False):
            raise BusyError("a generation is already in progress")
        try:
            with self._lock:
                model = self._require_model()
            r = model.chat(messages, max_tokens=max_tokens, temperature=temperature,
                           on_token=on_token, on_reset=on_reset)
            return {"text": r.text, "metrics": r.as_dict()}
        finally:
            self._busy.release()

    # ---- reports ----------------------------------------------------------------------------

    def reports(self) -> dict[str, object]:
        with self._lock:
            model = self._require_model()
            perf = model.performance_report()
            return {
                "settings": asdict(self._settings) if self._settings else None,
                "plan": model.plan.as_dict() if model.plan else None,
                "learning": model.learning_report(),
                "performance": perf,
                "rewards": model.reward_history(100),
            }

    def policy(self) -> dict[str, object]:
        with self._lock:
            return self._require_model().policy_snapshot()

    def is_busy(self) -> bool:
        return self._busy.locked()
