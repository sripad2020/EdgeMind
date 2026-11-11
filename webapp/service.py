from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from air_sdk import AdaptiveModel, AdaptiveRuntime, AirError, ValidationError
from air_sdk.backends.ollama import OllamaModelInfo
from air_sdk.context import AUTO, ContextEngine, SessionMemory, SessionStore
from air_sdk.skills import SkillEngine

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
        self._sessions = SessionStore()  # per-conversation sectioned memory
        self._context = ContextEngine()

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
        temperature: float | None,
        on_token: Callable[[str], None],
        on_reset: Callable[[], None],
        skill: str = AUTO,
        context: str | None = None,
        on_escalate: Callable[[str, str, str], None] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, object]:
        """One reply to the last user message.

        Context: with a ``conversation_id`` the server keeps that conversation's sectioned
        memory (facts per task, routing, token budget); otherwise the memory is rebuilt from
        the ``messages`` sent. The task is detected unless ``skill`` names one; verified
        skills check the answer against the full stored facts and retry on a more accurate
        precision if the check fails."""
        system = next((str(m["content"]) for m in messages if m.get("role") == "system"), None)
        last = messages[-1]
        if last.get("role") != "user" or not isinstance(last.get("content"), str):
            raise ValidationError("the last message must be the user's")
        text = str(last["content"])
        if not self._busy.acquire(blocking=False):
            raise BusyError("a generation is already in progress")
        try:
            with self._lock:
                model = self._require_model()
                rt = self._runtime_for(self._objective or "balanced")
            if conversation_id is not None:
                memory = self._sessions.get(conversation_id)
            else:
                memory = SessionMemory()
                self._context.replay(memory, [m for m in messages[:-1]
                                              if m.get("role") != "system"])
            engine = SkillEngine(model, rt.store)
            with memory.lock:
                p = self._context.prepare(memory, text, skill=skill, context=context,
                                          system=system, context_limit=model.context_limit,
                                          max_tokens=max_tokens)
                out = engine.run(
                    p.messages, p.skill, context=p.prompt_context,
                    check_context=p.check_context, bonus=True, bonus_schema=p.bonus_schema,
                    max_tokens=max_tokens, temperature=temperature, on_token=on_token,
                    on_reset=on_reset, on_escalate=on_escalate)
                self._context.commit(memory, p, out.response.text)
            return {
                "text": out.response.text,
                "metrics": out.response.as_dict(),
                "skill": {
                    "name": out.skill, "verified": out.verified, "escalated": out.escalated,
                    "check": out.detail, "auto": p.auto,
                    "attempts": [{"precision": a.precision, "ok": a.ok, "detail": a.detail}
                                 for a in out.attempts],
                },
                "context": p.as_dict(),
            }
        finally:
            self._busy.release()

    def conversation_memory(self, conversation_id: str) -> dict[str, object]:
        return self._sessions.get(conversation_id).snapshot()

    def forget_conversation(self, conversation_id: str) -> bool:
        return self._sessions.drop(conversation_id)

    def skills(self) -> list[dict[str, object]]:
        with self._lock:
            model = self._require_model()
            rt = self._runtime_for(self._objective or "balanced")
            return SkillEngine(model, rt.store).catalogue()

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
