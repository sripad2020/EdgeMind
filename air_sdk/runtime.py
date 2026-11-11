"""Public API: AdaptiveRuntime and AdaptiveModel.

One ``generate()`` call is one RL episode (learning happens *between* requests, never while a
response is being produced):

    observe state s -> mask unsafe actions -> agent picks a -> apply at the request boundary
    -> generate -> measure -> reward r -> observe s' -> Q(s,a) update -> persist

Failures (OOM, load errors) are penalised, the configuration is blacklisted, the runtime
rolls back to the last known-good (or conservative) configuration and retries once, so the
caller still gets a response.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ._validate import (
    clamp,
    require_bool,
    require_enum,
    require_float,
    require_int,
    require_str,
)
from .analysis.profile import ModelProfile, analyze_model, discover_variants
from .backends.base import (
    ChatMessage,
    GenerationResult,
    InferenceBackend,
    Role,
    TokenCallback,
)
from .backends.ollama import (
    DEFAULT_BASE_URL,
    OllamaBackend,
    OllamaClient,
    OllamaError,
    OllamaModelInfo,
    validate_model_name,
)
from .backends.simulated import SimulatedBackend, SimulatedHardware
from .config import (
    DevicePlacement,
    ExecutionConfig,
    MemoryEstimate,
    Objective,
    SafetyPolicy,
    shape_key,
)
from .engines.space import MemoryBudget, Position
from .errors import (
    AirError,
    FailureKind,
    GenerationError,
    InsufficientResourcesError,
    ModelLoadError,
    ValidationError,
)
from .hardware.profiler import HardwareProfile, TelemetrySample, profile_hardware
from .hardware.telemetry import TelemetryMonitor, WindowStats
from .planner import ExecutionPlan, ExecutionPlanner
from .quality import evaluate_against_reference
from .quantize import (
    QuantizationError,
    QuantizeResult,
    QuantTarget,
    derived_model_name,
    is_quantizable,
    plan_targets,
    timed_quantize,
)
from .rl.actions import N_ACTIONS, Action
from .rl.agent import QLearningAgent, QLearningParams
from .rl.reward import (
    Baseline,
    EpisodeMetrics,
    RewardBreakdown,
    RewardWeights,
    compute_reward,
)
from .rl.state import RLState, StateInputs, discretize
from .rl.store import ConfigScore, ExperienceStore

_RECENT = 100
_TTFT_REF_TOKENS = 128
_SWITCH_EMA = 0.3
_PROBE_PROMPT = "Write a short paragraph about the ocean."
_PROBE_TOKENS = 48
_CHARS_PER_TOKEN = 3.5  # conservative (English averages ~4); over-estimating need is safe
_CTX_MARGIN = 64
_EMPTY_WINDOW = WindowStats(samples=0, peak_rss=0, peak_vram_used=None, min_ram_available=0,
                            mean_cpu_percent=0.0, max_temperature_c=None)


@dataclass(frozen=True, slots=True)
class _Outcome:
    result: GenerationResult
    window: WindowStats
    switch_s: float  # seconds spent (re)loading a new configuration before this request


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    config: ExecutionConfig
    tokens_per_second: float
    ttft_s: float
    completion_tokens: int
    reward: float | None
    action: str
    explored: bool
    rolled_back: bool
    prompt_tokens: int = 0
    total_s: float = 0.0
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    quality_loss: float = 0.0
    learned: bool = False  # False when learning was off or the reply was too short to score
    switch_s: float = 0.0  # time this reply waited for a configuration (re)load
    prefetched: bool = False  # configuration was prepared in the background
    relief: bool = False  # configuration was changed because memory got tight

    def as_dict(self) -> dict[str, object]:
        return {
            "learned": self.learned,
            "switch_s": round(self.switch_s, 3),
            "prefetched": self.prefetched,
            "relief": self.relief,
            "tokens_per_second": round(self.tokens_per_second, 3),
            "ttft_s": round(self.ttft_s, 4),
            "total_s": round(self.total_s, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "quality_loss_prior": round(self.quality_loss, 6),
            "reward": None if self.reward is None else round(self.reward, 4),
            "action": self.action,
            "explored": self.explored,
            "rolled_back": self.rolled_back,
            "config": self.config.as_dict(),
        }


ResetCallback = Callable[[], None]
_MAX_MESSAGES = 512


@dataclass(frozen=True, slots=True)
class _Request:
    prompt: str | None
    messages: tuple[ChatMessage, ...] | None
    max_tokens: int
    temperature: float
    seed: int | None
    on_token: TokenCallback | None
    on_reset: ResetCallback | None
    # Run on this precision variant (index into the precision ladder) instead of letting the
    # RL policy choose; used by verified skills. Safety checks still apply.
    pin_variant: int | None = None

    @property
    def chars(self) -> int:
        if self.messages is not None:
            return sum(len(m.content) for m in self.messages)
        return len(self.prompt or "")


def _validate_messages(messages: object, max_chars: int) -> tuple[ChatMessage, ...]:
    """Validate an untrusted chat history: roles allow-listed, sizes bounded. O(n)."""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, (list, tuple)):
        raise ValidationError("messages must be a list")
    if not 1 <= len(messages) <= _MAX_MESSAGES:
        raise ValidationError(f"messages must contain 1..{_MAX_MESSAGES} items")
    out: list[ChatMessage] = []
    total = 0
    for i, m in enumerate(messages):
        content: object
        if isinstance(m, ChatMessage):
            role, content = m.role, m.content
        elif isinstance(m, dict):
            role = require_enum(f"messages[{i}].role", m.get("role"), Role)
            content = m.get("content")
        else:
            raise ValidationError(f"messages[{i}] must be a ChatMessage or dict")
        text = require_str(f"messages[{i}].content", content, max_len=max_chars,
                           allow_empty=True)
        total += len(text)
        if total > max_chars:
            raise ValidationError(f"conversation exceeds {max_chars} characters")
        out.append(ChatMessage(role, text))
    if out[-1].role is not Role.USER:
        raise ValidationError("the last message must be from the user")
    return tuple(out)


class _PerfStats:
    """Running aggregates O(1) + bounded recent window O(_RECENT)."""

    __slots__ = ("failures", "prefetched", "recent", "reliefs", "requests", "rollbacks",
                 "sum_switch_s", "sum_tps", "sum_ttft")

    def __init__(self) -> None:
        self.requests = 0
        self.failures = 0
        self.rollbacks = 0
        self.reliefs = 0
        self.prefetched = 0
        self.sum_switch_s = 0.0
        self.sum_tps = 0.0
        self.sum_ttft = 0.0
        self.recent: deque[GenerationResponse] = deque(maxlen=_RECENT)

    def add(self, r: GenerationResponse) -> None:
        self.requests += 1
        self.sum_tps += r.tokens_per_second
        self.sum_ttft += r.ttft_s
        self.rollbacks += int(r.rolled_back)
        self.prefetched += int(r.prefetched)
        self.sum_switch_s += r.switch_s
        self.recent.append(r)


class _ConfigValues:
    """Running mean reward per configuration, used as Q-value priors.

    O(1) update and lookup; O(#configurations seen) memory. The global mean over configs is
    maintained incrementally (sum of per-config means), never recomputed by scanning.
    """

    __slots__ = ("_horizon", "_means", "_n", "_sum_of_means")

    def __init__(self, gamma: float) -> None:
        self._horizon = 1.0 / (1.0 - gamma)  # value of "stay here forever"
        self._means: dict[str, float] = {}
        self._n: dict[str, int] = {}
        self._sum_of_means = 0.0

    def load(self, scores: list[ConfigScore]) -> None:
        """Load stored per-config means, merged across context sizes (weighted by count)."""
        for s in scores:
            key = shape_key(s.config_key)
            n = self._n.get(key, 0)
            old = self._means.get(key, 0.0)
            new = (old * n + s.mean_reward * s.samples) / (n + s.samples)
            self._sum_of_means += new - (old if n else 0.0)
            self._means[key], self._n[key] = new, n + s.samples

    def add(self, key: str, reward: float) -> None:
        n = self._n.get(key, 0)
        old = self._means.get(key, 0.0)
        new = old + (reward - old) / (n + 1)
        self._sum_of_means += new - (old if n else 0.0)
        self._means[key], self._n[key] = new, n + 1

    def mean(self, key: str) -> float | None:
        """Observed mean reward of a configuration, or None if never tried."""
        return self._means.get(key)

    def value(self, key: str) -> float:
        mean = self._means.get(key)
        if mean is None:
            # Unknown config: neutral prior (average of known configs), not optimistic 0.
            mean = self._sum_of_means / len(self._means) if self._means else 0.0
        return mean * self._horizon


@dataclass(frozen=True, slots=True)
class _Pending:
    """A completed (s, a, r) waiting for s' - the state at the start of the next episode."""

    state: str
    action: Action
    reward: float
    prior: float
    config_key: str
    metrics: dict[str, float | int | bool]
    failed: bool


@dataclass(frozen=True, slots=True)
class _Decision:
    """The configuration chosen for one request, and how it was chosen."""

    state: str
    action: Action
    explored: bool
    prior: float
    target: Position
    learn: bool  # False for safety-forced moves (not the agent's choice)
    relief: bool  # True if chosen by memory-pressure relief
    prefetched: bool = False  # True if prepared and loaded in the background


_CAL_MARGIN = 1.15  # safety margin on the worst observed calibration ratio
_CAL_MIN = 0.35
_CAL_MAX = 2.0


def _local_model_name(path: Path) -> str:
    """Ollama-safe name for an imported file: 'Phi-3 mini.Q4.gguf' -> 'phi-3-mini.q4:air-local'.

    Note: Ollama (0.40) registers models tagged exactly ':local' but then cannot resolve
    them, so that tag is never used (see _RESERVED_TAGS)."""
    stem = re.sub(r"[^a-z0-9._-]+", "-", path.stem.lower()).strip("-._")[:80] or "model"
    return f"{stem}:air-local"


_RESERVED_TAGS = frozenset({"local"})


def _check_import_name(name: str) -> str:
    tag = name.rsplit(":", 1)[1] if ":" in name.rsplit("/", 1)[-1] else "latest"
    if tag.lower() in _RESERVED_TAGS:
        raise ValidationError(f"the tag ':{tag}' cannot be used with Ollama; choose another "
                              "(e.g. ':air-local')")
    return name


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]
    return cleaned if cleaned.endswith(".gguf") else f"{cleaned}.gguf"


def _hash(*parts: object) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:32]


class AdaptiveModel:
    """A model managed by AIR-SDK. Obtain via :meth:`AdaptiveRuntime.load_model`."""

    def __init__(
        self, runtime: AdaptiveRuntime, primary: ModelProfile,
        variants: tuple[ModelProfile, ...], backend: InferenceBackend,
    ) -> None:
        self._rt = runtime
        self._primary = primary
        self._variants = variants
        self._backend = backend
        self._lock = threading.Lock()
        self._planner: ExecutionPlanner | None = None
        self._agent: QLearningAgent | None = None
        self._plan: ExecutionPlan | None = None
        self._pos: Position | None = None
        self._last_good: Position | None = None
        self._loaded_est: MemoryEstimate | None = None
        self._capacity_total = 0
        self._stats = _PerfStats()
        self._values = _ConfigValues(runtime.rl_params.gamma)
        self._pending: _Pending | None = None
        # Running estimate of the weighted switching penalty (EMA of observed values).
        self._switch_est = RewardWeights.for_objective(runtime.objective).switching * 0.5
        # Background switching / pressure watcher state.
        self._load_lock = threading.RLock()
        self._next: _Decision | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: FailureKind | None = None
        self._decisions = 0
        self._last_request_tokens = 0
        self._watcher: threading.Thread | None = None
        self._watch_stop = threading.Event()
        self._events: deque[dict[str, object]] = deque(maxlen=50)
        self._calib_scope = f"{backend.name}|{runtime.hardware.fingerprint}"
        self._auto_context = True
        self._ctx_growths = 0
        self._probe_report: list[dict[str, object]] = []
        self._closed = False

    # ---- Engine 1 ---------------------------------------------------------------------------

    def analyze(self) -> ModelProfile:
        return self._primary

    @property
    def variants(self) -> tuple[ModelProfile, ...]:
        return self._variants

    # ---- planning ---------------------------------------------------------------------------

    def optimize(
        self,
        precision: str = "auto",
        device_placement: str | DevicePlacement = "auto",
        rl_policy: str = "q_learning",
        *,
        n_ctx: int | None = None,
        auto_context: bool = True,
        probe: bool = False,
    ) -> ExecutionPlan:
        """Initial optimisation: build candidates, filter unsafe ones, pick, load, validate.

        Args:
            n_ctx: maximum context size (tokens).
            auto_context: start with a small context and grow it as the conversation needs
                (saves RAM); otherwise the full ``n_ctx`` is reserved from the start.
            probe: run a short speed test of each precision version and start on the one
                that measures best on this machine (instead of the predicted best).
        """
        self._check_open()
        precision = require_str("precision", precision, max_len=32)
        placement = require_enum("device_placement", device_placement, DevicePlacement)
        if require_str("rl_policy", rl_policy, max_len=32) != "q_learning":
            raise ValidationError("rl_policy must be 'q_learning' (the only supported policy)")
        ctx = min(4096, self._primary.context_length) if n_ctx is None else require_int(
            "n_ctx", n_ctx, min_value=16)
        pinned = None if precision.lower() == "auto" else precision
        self._auto_context = require_bool("auto_context", auto_context)
        do_probe = require_bool("probe", probe)

        with self._lock:
            rt = self._rt
            self._join_prefetch()
            self._next = None
            self._flush_pending()  # complete the previous context's last transition
            planner = ExecutionPlanner(
                self._variants, rt.hardware, rt.safety, rt.objective,
                n_ctx=ctx, placement=placement, pinned_precision=pinned,
                kv_types=self._backend.supported_kv_types(),
                quality_overrides=rt.store.quality_overrides(
                    [v.variant_key for v in self._variants]),
            )
            if self._auto_context:
                planner.set_context(planner.context_levels()[0])
            self._ctx_growths = 0
            self._probe_report = []
            context_key = _hash(
                "air-sdk/v1", self._primary.family_key,
                tuple(sorted(v.variant_key for v in self._variants)), rt.hardware.fingerprint,
                self._backend.name, self._backend.version, rt.objective.value, placement.value,
                pinned, rt.safety.max_quality_loss, planner.signature(),
            )
            description = (
                f"{self._primary.name} [{', '.join(v.quant.name for v in self._variants)}] on "
                f"{rt.hardware.cpu_name} / "
                f"{rt.hardware.primary_gpu.name if rt.hardware.primary_gpu else 'no GPU'}"
                f" via {self._backend.name} {self._backend.version}, objective={rt.objective.value}"
            )
            agent = QLearningAgent(
                rt.store, context_key, params=rt.rl_params, description=description, seed=rt.seed
            )
            cap = planner.capacity_budget()
            self._capacity_total = cap.ram_bytes + cap.vram_bytes
            self._planner, self._agent = planner, agent
            self._values = _ConfigValues(rt.rl_params.gamma)
            self._values.load(rt.store.config_stats(context_key))
            self._apply_calibration()  # live checks use this machine's calibrated estimates

            with self._load_lock:
                self._backend.unload()
                self._loaded_est = None
            budget = planner.live_budget(self._sample(), None)
            blacklist = rt.store.blacklist(context_key)
            learned = (
                rt.store.top_configs(context_key, min_samples=3, limit=10)
                if rt.rl_enabled else []
            )
            plan = planner.plan(budget, blacklist, learned)
            fallback_pos = planner.conservative(budget, blacklist)
            if do_probe and plan.source != "learned" and len(planner.space.precision.variants) > 1:
                measured = self._probe(planner, budget, blacklist)
                if measured is not None:
                    plan = planner.make_plan(measured, fallback_pos, "probe")
            try:
                self._load(plan.position)
            except ModelLoadError as exc:  # validate-by-loading; roll back to conservative
                rt.store.mark_failed(context_key, plan.config.key, exc.kind.value)
                self._load(fallback_pos)
                plan = planner.make_plan(fallback_pos, fallback_pos, "conservative")
            self._plan = plan
            self._pos = self._last_good = plan.position
            self._start_watcher()
            return plan

    # ---- execution --------------------------------------------------------------------------

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
        seed: int | None = None, *, on_token: TokenCallback | None = None,
        on_reset: ResetCallback | None = None,
    ) -> GenerationResponse:
        """Single-prompt completion (one RL episode)."""
        prompt = require_str("prompt", prompt, max_len=self._rt.safety.max_prompt_chars)
        return self._submit(_Request(prompt, None, *self._common(max_tokens, temperature, seed),
                                     on_token, on_reset))

    def chat(
        self, messages: Sequence[ChatMessage | dict[str, object]], max_tokens: int = 256,
        temperature: float = 0.7, seed: int | None = None, *,
        on_token: TokenCallback | None = None, on_reset: ResetCallback | None = None,
        precision: str | None = None,
    ) -> GenerationResponse:
        """Multi-turn chat using the backend's native chat template (one RL episode).

        ``on_token`` receives streamed text; ``on_reset`` is called if a failure forces a
        rollback-and-retry, so streaming UIs can discard the partial answer.
        ``precision`` (e.g. ``"Q8_0"``) runs this request on that weight precision instead
        of the policy's choice, if it is safe right now; otherwise the policy decides.
        """
        msgs = _validate_messages(messages, self._rt.safety.max_prompt_chars)
        pin = None if precision is None else self._variant_index(precision)
        return self._submit(_Request(None, msgs, *self._common(max_tokens, temperature, seed),
                                     on_token, on_reset, pin))

    def precision_ladder(self) -> tuple[str, ...]:
        """Available weight precisions, highest precision (largest) first."""
        if self._planner is None:
            self.optimize()
        return tuple(v.quant.name for v in self._require_planner().space.precision.variants)

    def _variant_index(self, precision: str) -> int:
        name = require_str("precision", precision, max_len=32).upper()
        ladder = self.precision_ladder()
        if name not in ladder:
            raise ValidationError(f"precision {precision!r} is not available; have {list(ladder)}")
        return ladder.index(name)

    def _common(
        self, max_tokens: int, temperature: float, seed: int | None
    ) -> tuple[int, float, int | None]:
        safety = self._rt.safety
        mt = require_int("max_tokens", max_tokens, min_value=1, max_value=safety.max_new_tokens)
        temp = require_float("temperature", temperature, min_value=0.0, max_value=5.0)
        sd = None if seed is None else require_int("seed", seed, min_value=0,
                                                   max_value=2**31 - 1)
        return mt, temp, sd

    def _submit(self, req: _Request) -> GenerationResponse:
        self._check_open()
        if req.on_token is not None and not callable(req.on_token):
            raise ValidationError("on_token must be callable")
        if self._planner is None:
            self.optimize()
        with self._lock:
            return self._episode(req)

    def _ready(self) -> tuple[ExecutionPlanner, QLearningAgent, Position]:
        """Explicit invariant check (not ``assert``, which ``python -O`` strips)."""
        if self._planner is None or self._agent is None or self._pos is None:
            raise AirError("model is not optimized; call optimize() first")
        return self._planner, self._agent, self._pos

    # ---- decisions ----------------------------------------------------------------------------

    def _decide(self, request_tokens: int, pin_variant: int | None = None) -> _Decision:
        """Observe the state, complete the previous episode's Q-update (this state is its s'),
        and choose the configuration for the next request."""
        planner, agent, pos = self._ready()
        rt = self._rt
        sample = self._sample()
        budget = planner.live_budget(sample, self._loaded_est)
        blacklist = rt.store.blacklist(agent.context_key)
        state = self._state(sample, pos, request_tokens).key()
        mask = planner.action_mask(pos, budget, blacklist)
        priors = self._priors(pos)
        self._flush_pending(state, mask, priors)

        if pin_variant is not None:
            pinned = planner.best_for_variant(pin_variant, budget, blacklist)
            if pinned is not None:
                # A caller-chosen precision (verified skills) is not the agent's decision,
                # so it is not learned from; it still had to pass every safety check.
                return _Decision(state, Action.MAINTAIN, False, priors[Action.MAINTAIN],
                                 pinned, learn=False, relief=False)

        action, explored = Action.MAINTAIN, False
        if rt.rl_enabled and rt.auto_adapt and self._decisions % rt.adapt_every == 0:
            choice = agent.select(state, mask, priors)
            action, explored = choice.action, choice.explored
        self._decisions += 1
        if not mask[action]:
            # The current configuration no longer fits (memory pressure): the safety rule
            # overrides the policy and moves to the best *safe* configuration. Forced moves
            # are not the agent's choice, so they are not learned from.
            target = planner.relief(pos, budget, blacklist, self._values.mean)
            self._note_relief(pos, target, "before request")
            return _Decision(state, action, explored, priors[action], target, learn=False,
                             relief=True)
        target = planner.space.neighbor(pos, action) or pos
        return _Decision(state, action, explored, priors[action], target,
                         learn=rt.rl_enabled, relief=False)

    def _take_decision(
        self, request_tokens: int, pin_variant: int | None = None,
    ) -> tuple[_Decision, float, FailureKind | None]:
        """Return (decision, seconds this request waited for a background load, failure of
        that background load). Without a prepared decision, decide synchronously."""
        prepared, self._next = self._next, None
        if pin_variant is not None:
            if prepared is not None:
                self._join_prefetch()  # the prepared choice is replaced by the pinned one
            return self._decide(request_tokens, pin_variant), 0.0, None
        if prepared is None:
            return self._decide(request_tokens), 0.0, None
        t0 = time.perf_counter()
        failure = self._join_prefetch()
        waited = time.perf_counter() - t0
        if failure is not None:
            return prepared, waited, failure
        planner, agent, _pos = self._ready()
        budget = planner.live_budget(self._sample(), self._loaded_est)
        if planner.space.check(prepared.target, budget,
                               self._rt.store.blacklist(agent.context_key)) is not None:
            # Memory changed since the decision was prepared: re-decide with fresh state.
            return self._decide(request_tokens), waited, None
        return replace(prepared, prefetched=True), waited, None

    def _prepare_next(self) -> None:
        """Background switching: right after a reply, choose the next configuration and load
        it while the user reads/types, so the reload is not charged to the next request."""
        decision = self._decide(self._last_request_tokens)
        self._next = decision
        planner = self._require_planner()
        if planner.space.config_at(decision.target) == self._backend.loaded_config:
            return
        self._prefetch_error = None
        thread = threading.Thread(target=self._prefetch_run, args=(decision.target,),
                                  name="air-prefetch", daemon=True)
        self._prefetch_thread = thread
        thread.start()

    def _prefetch_run(self, pos: Position) -> None:
        try:
            self._load(pos)
        except ModelLoadError as exc:
            self._prefetch_error = exc.kind
        except Exception:  # noqa: BLE001 - a background thread must never die silently
            self._prefetch_error = FailureKind.LOAD_FAILED

    def _join_prefetch(self) -> FailureKind | None:
        thread, self._prefetch_thread = self._prefetch_thread, None
        if thread is not None:
            thread.join()
        failure, self._prefetch_error = self._prefetch_error, None
        return failure

    # ---- episode ------------------------------------------------------------------------------

    def _episode(self, req: _Request) -> GenerationResponse:
        planner, agent, _pos = self._ready()
        rt = self._rt
        self._last_request_tokens = req.chars // 4 + req.max_tokens
        self._grow_context_if_needed(req)
        decision, waited, prefetch_failure = self._take_decision(self._last_request_tokens,
                                                                 req.pin_variant)
        target = decision.target

        outcome: _Outcome | FailureKind = (
            prefetch_failure if prefetch_failure is not None
            else self._execute(target, req, waited_s=waited)
        )
        config = planner.space.config_at(target)
        reward: float | None = None
        learned = False
        rolled_back = isinstance(outcome, FailureKind)

        if isinstance(outcome, FailureKind):
            self._stats.failures += 1
            rt.store.mark_failed(agent.context_key, config.key, outcome.value)
            if decision.learn:
                metrics = EpisodeMetrics(0.0, 0.0, 0, self._capacity_total, 0.0, failed=True)
                reward = self._reward(metrics).total
                learned = True
                self._pending = _Pending(decision.state, decision.action, reward,
                                         decision.prior, config.key, metrics.as_dict(),
                                         failed=True)
            if req.on_reset is not None:
                req.on_reset()
            target, outcome = self._rollback(target, req)
            config = planner.space.config_at(target)
        else:
            if decision.learn:
                reward = self._score(outcome, target, decision)
                learned = reward is not None
            self._last_good = target
        result, window = outcome.result, outcome.window

        self._pos = target
        response = GenerationResponse(
            text=result.text,
            config=config,
            tokens_per_second=result.tokens_per_second,
            ttft_s=result.ttft_s,
            completion_tokens=result.completion_tokens,
            reward=reward,
            action=decision.action.name,
            explored=decision.explored,
            rolled_back=rolled_back,
            prompt_tokens=result.prompt_tokens,
            total_s=result.total_s,
            peak_ram_bytes=result.peak_ram_bytes if result.peak_ram_bytes is not None
            else window.peak_rss,
            peak_vram_bytes=result.peak_vram_bytes if result.peak_vram_bytes is not None
            else window.peak_vram_used,
            quality_loss=planner.space.precision.quality_loss(target.variant, target.kv),
            learned=learned,
            switch_s=outcome.switch_s,
            prefetched=decision.prefetched,
            relief=decision.relief,
        )
        self._stats.add(response)
        if rt.background_switching:
            self._prepare_next()
        return response

    def _execute(
        self, pos: Position, req: _Request, *, waited_s: float = 0.0
    ) -> _Outcome | FailureKind:
        """Load ``pos`` (if needed) and run the request. Request errors (ValidationError)
        propagate; backend failures are returned as a FailureKind. ``waited_s`` is time
        already spent waiting for a background load; it counts as switching cost."""
        try:
            t0 = time.perf_counter()
            switched = self._load(pos)
            switch_s = waited_s + (time.perf_counter() - t0 if switched else 0.0)
            self._rt.telemetry.begin_window()
            try:
                if req.messages is not None:
                    result = self._backend.chat(
                        req.messages, max_tokens=req.max_tokens, temperature=req.temperature,
                        seed=req.seed, on_token=req.on_token,
                    )
                else:
                    result = self._backend.generate(
                        req.prompt or "", max_tokens=req.max_tokens,
                        temperature=req.temperature, seed=req.seed, on_token=req.on_token,
                    )
            finally:
                window = self._rt.telemetry.end_window()
        except (ModelLoadError, GenerationError) as exc:
            return exc.kind
        return _Outcome(result, window, switch_s)

    def _score(self, outcome: _Outcome, target: Position, decision: _Decision) -> float | None:
        """Reward a successful episode and queue its Q-update. Returns the reward, or None
        if the reply was too short to give a reliable measurement (fixed per-request costs
        dominate very short replies); such replies are served but not learned from."""
        if outcome.result.completion_tokens < self._rt.min_learning_tokens:
            return None
        config = self._require_planner().space.config_at(target)
        config_key = config.key
        metrics = self._metrics(outcome, target)
        breakdown = self._reward(metrics)
        self._values.add(config.shape_key, breakdown.intrinsic)
        if breakdown.switching > 0:
            self._switch_est += _SWITCH_EMA * (breakdown.switching - self._switch_est)
        record = metrics.as_dict() | {"intrinsic": round(breakdown.intrinsic, 6)}
        self._pending = _Pending(decision.state, decision.action, breakdown.total,
                                 decision.prior, config_key, record, failed=False)
        return breakdown.total

    def _reward(self, metrics: EpisodeMetrics) -> RewardBreakdown:
        agent = self._agent
        if agent is None:
            raise AirError("model is not optimized; call optimize() first")
        rt = self._rt
        if not metrics.failed and agent.baseline is None:
            # First successful episode on this machine/model defines the reference point.
            agent.set_baseline(Baseline(max(metrics.tokens_per_second, 1e-6),
                                        max(metrics.ttft_s, 1e-6)))
        return compute_reward(
            metrics, agent.baseline or Baseline(1.0, 1.0),
            RewardWeights.for_objective(rt.objective),
            max_quality_loss=rt.safety.max_quality_loss,
            failure_penalty=agent.params.failure_penalty,
        )

    def _rollback(self, failed: Position, req: _Request) -> tuple[Position, _Outcome]:
        """Reload the last known-good config (or the conservative one) and retry once."""
        planner, agent, _pos = self._ready()
        with self._load_lock:
            self._backend.unload()
            self._loaded_est = None
        budget = planner.live_budget(self._sample(), None)
        blacklist = self._rt.store.blacklist(agent.context_key)
        candidates: list[Position] = []
        if self._last_good is not None and self._last_good != failed:
            candidates.append(self._last_good)
        candidates.append(planner.conservative(budget, blacklist))
        last_failure: FailureKind | None = None
        for i, pos in enumerate(dict.fromkeys(candidates)):  # de-duplicate, keep order
            if i > 0 and req.on_reset is not None:
                req.on_reset()
            outcome = self._execute(pos, req)
            if isinstance(outcome, FailureKind):
                self._rt.store.mark_failed(agent.context_key, planner.space.config_at(pos).key,
                                           outcome.value)
                last_failure = outcome
                continue
            self._last_good = pos
            return pos, outcome
        raise GenerationError(
            f"generation failed and rollback was unsuccessful ({last_failure})",
            FailureKind.RUNTIME_ERROR,
        )

    # ---- quick speed test (probe) ---------------------------------------------------------------

    def _probe(
        self, planner: ExecutionPlanner, budget: MemoryBudget, blacklist: frozenset[str]
    ) -> Position | None:
        """Measure each precision version with a short fixed generation and return the one
        with the best reward on *this* machine. Results also seed the configuration values,
        so the RL engine starts from measurements instead of guesses.

        Cost: one load + ~48 generated tokens per version (V <= a few)."""
        candidates: list[tuple[Position, _Outcome]] = []
        for v in range(len(planner.space.precision.variants)):
            pos = planner.best_for_variant(v, budget, blacklist)
            if pos is None:
                continue
            try:
                t0 = time.perf_counter()
                self._load(pos)
                load_s = time.perf_counter() - t0
                result = self._backend.generate(_PROBE_PROMPT, max_tokens=_PROBE_TOKENS,
                                                temperature=0.0, seed=0)
            except (ModelLoadError, GenerationError) as exc:
                if self._agent is not None:
                    self._rt.store.mark_failed(self._agent.context_key,
                                               planner.space.config_at(pos).key, exc.kind.value)
                continue
            if result.completion_tokens >= self._rt.min_learning_tokens:
                candidates.append((pos, _Outcome(result, _EMPTY_WINDOW, 0.0)))
                self._probe_report.append({
                    "precision": planner.space.variant(pos).quant.name,
                    "tokens_per_second": round(result.tokens_per_second, 1),
                    "load_s": round(load_s, 2),
                })
        if not candidates:
            return None
        best: tuple[float, Position] | None = None
        for pos, outcome in candidates:  # variant 0 (highest precision) first => baseline
            breakdown = self._reward(self._metrics(outcome, pos))
            self._values.add(planner.space.config_at(pos).shape_key, breakdown.intrinsic)
            if best is None or breakdown.intrinsic > best[0]:
                best = (breakdown.intrinsic, pos)
        for row, (_pos, _o) in zip(self._probe_report[-len(candidates):], candidates,
                                   strict=True):
            row["chosen"] = best is not None and _pos == best[1]
        return None if best is None else best[1]

    # ---- automatic context size -----------------------------------------------------------------

    def _grow_context_if_needed(self, req: _Request) -> None:
        """Start small and grow the context (KV cache) only when a conversation needs it:
        a short chat does not reserve memory for 8k tokens. Growth is a system decision, so
        its reload is done here and is *not* charged to the agent's action."""
        planner = self._require_planner()
        active = planner.space.n_ctx
        need = int(req.chars / _CHARS_PER_TOKEN) + req.max_tokens + _CTX_MARGIN
        if not self._auto_context or need <= active or active >= planner.max_context:
            return
        new = next((lvl for lvl in planner.context_levels() if lvl >= need),
                   planner.max_context)
        self._join_prefetch()
        self._next = None  # a prepared decision refers to the old context size
        planner.set_context(new)
        self._ctx_growths += 1
        self._events.append({"time": time.time(), "event": "context_grow",
                             "when": "before request", "from": str(active), "to": str(new)})
        pos = self._pos
        if pos is None or self._agent is None:
            return
        budget = planner.live_budget(self._sample(), self._loaded_est)
        if planner.space.check(pos, budget, self._rt.store.blacklist(
                self._agent.context_key)) is None:
            with contextlib.suppress(ModelLoadError):
                self._load(pos)  # if it fails, the normal decision path relieves/rolls back

    # ---- memory-pressure watcher (idle) ---------------------------------------------------------

    def _start_watcher(self) -> None:
        if not self._rt.background_switching or self._watcher is not None:
            return
        self._watch_stop.clear()
        self._watcher = threading.Thread(target=self._watch, name="air-pressure", daemon=True)
        self._watcher.start()

    def _stop_watcher(self) -> None:
        self._watch_stop.set()
        if self._watcher is not None:
            self._watcher.join(timeout=5.0)
            self._watcher = None

    def _watch(self) -> None:
        """While idle, re-check every few seconds that the loaded configuration still fits
        free memory; if another program grew, switch to a safe configuration *before* the
        next request instead of letting the OS start swapping."""
        while not self._watch_stop.wait(self._rt.pressure_check_s):
            if not self._lock.acquire(blocking=False):
                continue  # a request is running; it does its own checks
            try:
                self._relieve_if_needed()
            except AirError:
                pass  # best effort; the next request re-validates anyway
            finally:
                self._lock.release()

    def _relieve_if_needed(self) -> None:
        if self._closed or self._planner is None or self._agent is None or self._pos is None:
            return
        failure = self._join_prefetch()
        if failure is not None:
            self._prefetch_error = failure  # leave it for the next request to handle
            return
        planner, agent = self._planner, self._agent
        current = self._next.target if self._next is not None else self._pos
        budget = planner.live_budget(self._sample(), self._loaded_est)
        blacklist = self._rt.store.blacklist(agent.context_key)
        if planner.space.check(current, budget, blacklist) is None:
            return
        target = planner.relief(current, budget, blacklist, self._values.mean)
        if target == current:
            return
        try:
            self._load(target)
        except ModelLoadError as exc:
            self._rt.store.mark_failed(agent.context_key, planner.space.config_at(target).key,
                                       exc.kind.value)
            return
        self._note_relief(current, target, "while idle")
        self._pos, self._next = target, None

    def _note_relief(self, before: Position, after: Position, when: str) -> None:
        space = self._require_planner().space
        a, b = space.config_at(before), space.config_at(after)
        self._stats.reliefs += 1
        self._events.append({
            "time": time.time(),
            "event": "memory_pressure_relief",
            "when": when,
            "from": f"{a.precision}/kv-{a.kv_type.value}/gpu{a.n_gpu_layers}/b{a.n_batch}",
            "to": f"{b.precision}/kv-{b.kv_type.value}/gpu{b.n_gpu_layers}/b{b.n_batch}",
        })

    # ---- helpers ----------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise AirError("model has been closed")

    def _sample(self) -> TelemetrySample:
        return self._backend.adjust_sample(self._rt.telemetry.sample_now())

    def _require_planner(self) -> ExecutionPlanner:
        if self._planner is None:
            raise AirError("model is not optimized; call optimize() first")
        return self._planner

    def _load(self, pos: Position) -> bool:
        """Ensure ``pos`` is loaded. Returns True if a (re)load was needed. Serialised with
        background loads by ``_load_lock``."""
        with self._load_lock:
            planner = self._require_planner()
            config = planner.space.config_at(pos)
            if self._backend.loaded_config == config:
                return False
            self._loaded_est = None
            self._backend.load(config)  # raises ModelLoadError; backend leaves nothing loaded
            measured = self._backend.measured_footprint()
            if measured is not None:
                self._calibrate(pos, measured)
            self._loaded_est = planner.space.estimate(pos)
            return True

    # ---- memory-estimate calibration --------------------------------------------------------

    def _calibrate(self, pos: Position, measured: tuple[int, int]) -> None:
        """Compare the measured footprint with the estimate. Weights are exact, so the ratio
        is taken on the overhead part only: (measured - weights) / (estimate - weights)."""
        space = self._require_planner().space
        hi = space.uncalibrated_estimate(pos).total
        lo = space.weights_only_estimate(pos).total
        if hi <= lo:
            return
        ratio = max(0, measured[0] + measured[1] - lo) / (hi - lo)
        self._rt.record_calibration(self._calib_scope, ratio)
        self._apply_calibration()

    def _apply_calibration(self) -> None:
        planner = self._planner
        if planner is None:
            return
        scale = self._rt.overhead_scale(self._calib_scope)
        if planner.space.overhead_scale != scale:
            planner.space = replace(planner.space, overhead_scale=scale)

    def _state(self, sample: TelemetrySample, pos: Position, request_tokens: int) -> RLState:
        space = self._require_planner().space
        variant = space.variant(pos)
        gpu = self._rt.hardware.primary_gpu
        if sample.vram_total is not None and sample.vram_used is not None:
            vram_free: int | None = sample.vram_total - sample.vram_used
            vram_pct: float | None = 100.0 * sample.vram_used / max(1, sample.vram_total)
        elif gpu is not None:
            used = self._loaded_est.vram_bytes if self._loaded_est else 0
            vram_free = max(0, gpu.free_vram - used)
            vram_pct = 100.0 * used / max(1, gpu.total_vram)
        else:
            vram_free, vram_pct = None, None
        inputs = StateInputs(
            ram_available=sample.ram_available,
            ram_percent=sample.ram_percent,
            cpu_percent=sample.cpu_percent,
            vram_free=vram_free,
            vram_percent=vram_pct,
            gpu_percent=sample.gpu_percent,
            temperature_c=sample.temperature_c,
            model_weight_bytes=variant.weight_bytes,
            model_size=variant.size_class,
            request_tokens=request_tokens,
            gpu_fraction=space.gpu_fraction(pos),
            thermal_limit_c=self._rt.safety.thermal_limit_c,
        )
        return discretize(inputs, variant=pos.variant, kv=pos.kv, threads=pos.threads,
                          batch=pos.batch)

    def _metrics(self, outcome: _Outcome, pos: Position) -> EpisodeMetrics:
        planner = self._require_planner()
        result, window = outcome.result, outcome.window
        if result.peak_ram_bytes is not None:
            peak = result.peak_ram_bytes + (result.peak_vram_bytes or 0)
        else:
            peak = window.peak_rss + (window.peak_vram_used or 0)
        # Time-to-first-token grows with prompt length (e.g. as a chat history grows), which
        # is not the configuration's fault. Normalise to a 128-token reference prompt so the
        # latency term compares configurations, not conversations.
        ttft = result.ttft_s * _TTFT_REF_TOKENS / max(result.prompt_tokens, _TTFT_REF_TOKENS)
        return EpisodeMetrics(
            tokens_per_second=result.tokens_per_second,
            ttft_s=ttft,
            peak_memory_bytes=peak,
            memory_budget_bytes=max(1, self._capacity_total),
            quality_loss=planner.space.precision.quality_loss(pos.variant, pos.kv),
            switch_s=outcome.switch_s,
            request_s=result.total_s,
        )

    def _priors(self, pos: Position) -> list[float]:
        """Prior Q-value per action = value of the configuration the action leads to, minus
        the expected one-off switching cost for actions that change the configuration.
        O(|A|) lookups, each O(1)."""
        space = self._require_planner().space
        priors = [0.0] * N_ACTIONS
        for a in Action:
            nxt = space.neighbor(pos, a)
            if nxt is not None:
                value = self._values.value(space.config_at(nxt).shape_key)
                priors[a] = value if nxt == pos else value - self._switch_est
        return priors

    def _flush_pending(
        self, next_state: str | None = None, next_mask: list[bool] | None = None,
        next_priors: list[float] | None = None,
    ) -> None:
        """Apply the deferred Q-update for the previous episode.

        With no arguments, s' is observed now (used on close / re-optimise / report).
        """
        pending, agent = self._pending, self._agent
        if pending is None or agent is None:
            return
        self._pending = None
        if next_state is None or next_mask is None or next_priors is None:
            planner, _agent, pos = self._ready()
            sample = self._sample()
            next_state = self._state(sample, pos, 0).key()
            budget = planner.live_budget(sample, self._loaded_est)
            next_mask = planner.action_mask(pos, budget,
                                            self._rt.store.blacklist(agent.context_key))
            next_priors = self._priors(pos)
        agent.update(
            pending.state, pending.action, pending.reward, next_state, next_mask,
            config_key=pending.config_key, metrics=pending.metrics, failed=pending.failed,
            prior=pending.prior, next_priors=next_priors,
        )

    # ---- reports / lifecycle ----------------------------------------------------------------

    @property
    def current_config(self) -> ExecutionConfig | None:
        if self._planner is None or self._pos is None:
            return None
        return self._planner.space.config_at(self._pos)

    @property
    def context_limit(self) -> int:
        """Largest prompt+answer size (tokens) this model may use here: the planner's maximum
        context (the active context grows to it automatically), else a safe default."""
        if self._planner is not None:
            return int(self._planner.max_context)
        return min(4096, self._primary.context_length)

    def performance_report(self) -> dict[str, object]:
        s = self._stats
        ok = s.requests
        return {
            "requests": s.requests,
            "failures": s.failures,
            "rollbacks": s.rollbacks,
            "memory_pressure_reliefs": s.reliefs,
            "background_prepared": s.prefetched,
            "total_wait_for_reloads_s": round(s.sum_switch_s, 3),
            "background_switching": self._rt.background_switching,
            "memory_estimate_scale": (
                round(self._planner.space.overhead_scale, 3) if self._planner else None
            ),
            "events": list(self._events)[-10:],
            "context_active": self._planner.space.n_ctx if self._planner else None,
            "context_max": self._planner.max_context if self._planner else None,
            "auto_context": self._auto_context,
            "context_growths": self._ctx_growths,
            "probe": self._probe_report,
            "mean_tokens_per_second": round(s.sum_tps / ok, 3) if ok else None,
            "mean_ttft_s": round(s.sum_ttft / ok, 4) if ok else None,
            "current_config": self.current_config.as_dict() if self.current_config else None,
            "estimated_memory": (
                {"ram_bytes": self._loaded_est.ram_bytes, "vram_bytes": self._loaded_est.vram_bytes}
                if self._loaded_est else None
            ),
            "recent": [r.as_dict() for r in list(s.recent)[-10:]],
        }

    @property
    def plan(self) -> ExecutionPlan | None:
        return self._plan

    def policy_snapshot(self) -> dict[str, object]:
        """Explain the agent's next decision: current state, and for every action its target
        configuration, safety verdict, Q-value (learned or prior) and visit count.

        O(|A|) safety checks; read-only (does not advance learning).
        """
        with self._lock:
            if self._planner is None or self._agent is None or self._pos is None:
                return {"status": "not optimized"}
            planner, agent, pos = self._planner, self._agent, self._pos
            sample = self._sample()
            budget = planner.live_budget(sample, self._loaded_est)
            blacklist = self._rt.store.blacklist(agent.context_key)
            state = self._state(sample, pos, 0)
            key = state.key()
            priors = self._priors(pos)
            mask = planner.action_mask(pos, budget, blacklist)
            greedy = agent.greedy(key, mask, priors)
            actions: list[dict[str, object]] = []
            for a in Action:
                nxt = planner.space.neighbor(pos, a)
                reason = None if nxt is None else planner.space.check(nxt, budget, blacklist)
                visits = agent.q.visits(key, a)
                actions.append({
                    "action": a.name,
                    "allowed": mask[a],
                    "blocked_reason": (
                        "outside candidate space" if nxt is None
                        else reason.value if reason is not None else None
                    ),
                    "q_value": round(agent.q.q(key, a, priors[a]), 4),
                    "learned": visits > 0,
                    "visits": visits,
                    "target": None if nxt is None else {
                        k: v for k, v in planner.space.config_at(nxt).as_dict().items()
                        if k != "model_path"
                    },
                })
            return {
                "state": key,
                "epsilon": round(agent.epsilon, 4),
                "greedy_action": greedy.name,
                "budget_ram_bytes": budget.ram_bytes,
                "budget_vram_bytes": budget.vram_bytes,
                "actions": actions,
            }

    def reward_history(self, limit: int = 100) -> list[float]:
        agent = self._agent
        if agent is None:
            return []
        n = require_int("limit", limit, min_value=1, max_value=10_000)
        return [round(r, 4) for r in self._rt.store.recent_rewards(agent.context_key, n)]

    def learning_report(self) -> dict[str, object]:
        agent = self._agent
        if agent is None:
            return {"status": "not optimized"}
        with self._lock:
            self._flush_pending()  # include the most recent completed episode
        store = self._rt.store
        rewards = store.recent_rewards(agent.context_key, 200)
        half = len(rewards) // 2
        early = rewards[:half]
        late = rewards[half:]
        top = store.top_configs(agent.context_key, min_samples=1, limit=3)
        return {
            "model": self._primary.name,
            "context": agent.context_key,
            "episodes": agent.episodes,
            "epsilon": round(agent.epsilon, 4),
            "q_states": len(agent.q),
            "mean_reward_early": round(sum(early) / len(early), 4) if early else None,
            "mean_reward_recent": round(sum(late) / len(late), 4) if late else None,
            "blacklisted_configs": len(store.blacklist(agent.context_key)),
            "top_configs": [
                {"mean_reward": round(c.mean_reward, 4), "samples": c.samples,
                 "config": c.config_key}
                for c in top
            ],
        }

    def close(self) -> None:
        self._stop_watcher()
        with self._lock:
            if not self._closed:
                self._join_prefetch()
                self._next = None
                self._flush_pending()  # persist the final transition before shutdown
                with self._load_lock:
                    self._backend.unload()
                    self._loaded_est = None
                self._closed = True


class RLEngineView:
    """Read-only view over the RL engine for all models of a runtime."""

    def __init__(self, runtime: AdaptiveRuntime) -> None:
        self._rt = runtime

    def get_learning_report(self) -> list[dict[str, object]]:
        return [m.learning_report() for m in self._rt.models]


class AdaptiveRuntime:
    """Entry point.

    Args:
        optimization: "balanced" | "throughput" | "latency" | "memory".
        rl_enabled: learn and use the Q-learning policy.
        auto_adapt: allow the policy to change the configuration between requests.
        backend: "llama_cpp", "ollama", "simulated", or an ``InferenceBackend`` instance.
        data_dir: where the experience DB lives (default ``~/.air_sdk``).
        adapt_every: consider a configuration change every N requests (reloads are costly
            on real backends; raise this to amortise them).
        hardware: override the detected hardware profile (e.g. to simulate another machine).
    """

    def __init__(
        self,
        optimization: str | Objective = "balanced",
        rl_enabled: bool = True,
        auto_adapt: bool = True,
        *,
        backend: str | InferenceBackend = "llama_cpp",
        data_dir: str | os.PathLike[str] | None = None,
        safety: SafetyPolicy | None = None,
        rl_params: QLearningParams | None = None,
        adapt_every: int = 1,
        seed: int | None = None,
        hardware: HardwareProfile | None = None,
        telemetry_interval_s: float = 0.25,
        ollama_url: str = DEFAULT_BASE_URL,
        min_learning_tokens: int = 16,
        background_switching: bool = False,
        pressure_check_s: float = 3.0,
        calibrate_memory: bool = True,
    ) -> None:
        self.objective = require_enum("optimization", optimization, Objective)
        self.rl_enabled = require_bool("rl_enabled", rl_enabled)
        self.auto_adapt = require_bool("auto_adapt", auto_adapt)
        self.adapt_every = require_int("adapt_every", adapt_every, min_value=1,
                                       max_value=1_000_000)
        self.seed = None if seed is None else require_int("seed", seed, min_value=0)
        if safety is not None and not isinstance(safety, SafetyPolicy):
            raise ValidationError("safety must be a SafetyPolicy")
        if rl_params is not None and not isinstance(rl_params, QLearningParams):
            raise ValidationError("rl_params must be QLearningParams")
        if hardware is not None and not isinstance(hardware, HardwareProfile):
            raise ValidationError("hardware must be a HardwareProfile")
        if not isinstance(backend, (str, InferenceBackend)):
            raise ValidationError("backend must be a name or an InferenceBackend instance")
        self.safety = safety or SafetyPolicy()
        self.rl_params = rl_params or QLearningParams()
        self._backend_spec = backend
        self.hardware = hardware or profile_hardware()

        root = Path(data_dir) if data_dir is not None else Path.home() / ".air_sdk"
        self.data_dir = root
        self.store = ExperienceStore(root / "experience.db")
        gpu = self.hardware.primary_gpu
        self.telemetry = TelemetryMonitor(
            gpu_index=gpu.index if gpu is not None and hardware is None else None,
            interval_s=telemetry_interval_s,
        )
        self.telemetry.start()
        self.models: list[AdaptiveModel] = []
        self.rl_engine = RLEngineView(self)
        self._ollama_url = require_str("ollama_url", ollama_url, max_len=256)
        # Replies shorter than this are served but not used as learning signal.
        self.min_learning_tokens = require_int("min_learning_tokens", min_learning_tokens,
                                               min_value=1, max_value=4096)
        # Prepare/load the next configuration right after each reply, and watch memory
        # while idle (switch before the OS starts swapping).
        self.background_switching = require_bool("background_switching", background_switching)
        self.pressure_check_s = require_float("pressure_check_s", pressure_check_s,
                                              min_value=0.2, max_value=600.0)
        # Shrink memory overhead estimates to what this machine actually measures.
        self.calibrate_memory = require_bool("calibrate_memory", calibrate_memory)
        self._ollama: OllamaClient | None = None
        self._profile_cache: dict[tuple[str, int, int], ModelProfile] = {}
        self._closed = False

    # ---- memory-estimate calibration --------------------------------------------------------

    def overhead_scale(self, scope: str) -> float:
        """Calibrated overhead factor for a backend+hardware scope (1.0 if uncalibrated).

        * Worst observed ratio < 1 (estimates too high): shrink, but keep a safety margin and
          never go above 1.0 because of the margin alone.
        * Worst observed ratio > 1 (estimates too low): grow by the ratio plus the margin.
        Clamped to [0.35, 2.0].
        """
        if not self.calibrate_memory:
            return 1.0
        cal = self.store.load_calibration(scope)
        if cal is None:
            return 1.0
        ratio = cal[0]
        scale = min(1.0, ratio * _CAL_MARGIN) if ratio <= 1.0 else ratio * _CAL_MARGIN
        return clamp(scale, _CAL_MIN, _CAL_MAX)

    def record_calibration(self, scope: str, ratio: float) -> None:
        if self.calibrate_memory:
            self.store.record_calibration(scope, ratio)

    def hardware_profile(self, refresh: bool = False) -> HardwareProfile:
        if require_bool("refresh", refresh) and isinstance(self._backend_spec, str) \
                and self._backend_spec != "simulated":
            self.hardware = profile_hardware()
        return self.hardware

    def _make_backend(self) -> InferenceBackend:
        spec = self._backend_spec
        if isinstance(spec, InferenceBackend):
            return spec
        if spec == "simulated":
            return SimulatedBackend(SimulatedHardware.from_profile(self.hardware), seed=self.seed)
        if spec == "llama_cpp":
            from .backends.llama_cpp import LlamaCppBackend  # noqa: PLC0415 - optional dep

            return LlamaCppBackend()
        if spec == "ollama":
            return OllamaBackend(self.ollama_client())
        raise ValidationError(
            f"unknown backend {spec!r}; use 'llama_cpp', 'ollama' or 'simulated'"
        )

    # ---- Ollama integration -----------------------------------------------------------------

    def ollama_client(self) -> OllamaClient:
        """Shared client for the Ollama server (only when backend='ollama' or an
        OllamaBackend instance was given)."""
        spec = self._backend_spec
        if isinstance(spec, OllamaBackend):
            return spec.client
        if spec != "ollama":
            raise ValidationError("runtime was not created with the Ollama backend")
        if self._ollama is None:
            self._ollama = OllamaClient(self._ollama_url)
        return self._ollama

    def _ollama_profile(self, info: OllamaModelInfo) -> ModelProfile:
        """Analyse an Ollama GGUF blob, cached by (path, size, mtime). O(1) on cache hit."""
        stat = info.blob_path.stat()
        key = (str(info.blob_path), stat.st_size, stat.st_mtime_ns)
        cached = self._profile_cache.get(key)
        if cached is None:
            cached = analyze_model(info.blob_path, suffixes=("", ".gguf"))
            self._profile_cache[key] = cached
        return cached

    def list_ollama_models(self) -> list[OllamaModelInfo]:
        """Chat-capable models installed in Ollama (embedding-only models excluded)."""
        client = self.ollama_client()
        out: list[OllamaModelInfo] = []
        for name in client.list_models():
            try:
                info = client.show(name)
            except (OllamaError, ValidationError):
                continue
            if info.chat_capable:
                out.append(info)
        return out

    def load_ollama_model(self, name: str, *, discover: bool = True) -> AdaptiveModel:
        """Import an Ollama model by name (e.g. ``"qwen2.5:0.5b"``).

        With ``discover``, other installed tags of the *same* model at different precisions
        (same architecture fingerprint) become precision candidates for the RL engine.
        """
        if self._closed:
            raise AirError("runtime has been closed")
        client = self.ollama_client()
        info = client.show(validate_model_name(name))
        if not info.chat_capable:
            raise ValidationError(f"{name} is not a chat/completion model")
        primary = self._ollama_profile(info)
        backend = self._make_backend()
        if not isinstance(backend, OllamaBackend):
            raise ValidationError("load_ollama_model requires the Ollama backend")
        found: dict[str, ModelProfile] = {primary.variant_key: primary}
        backend.register(info.name, primary.path)
        if require_bool("discover", discover):
            for other in self.list_ollama_models():
                if other.name == info.name:
                    continue
                try:
                    profile = self._ollama_profile(other)
                except (AirError, OSError):
                    continue
                if profile.family_key == primary.family_key:
                    found.setdefault(profile.variant_key, profile)
                    backend.register(other.name, profile.path)
        model = AdaptiveModel(self, primary, tuple(found.values()), backend)
        self.models.append(model)
        return model

    # ---- automatic quantization -------------------------------------------------------------

    def _free_ram_budget(self) -> int:
        s = self.telemetry.sample_now()
        return max(0, s.ram_available - self.safety.ram_headroom(s.ram_total))

    def quantization_plan(self, name: str) -> dict[str, object]:
        """What quantized versions of an Ollama model can be created, and which fit the
        memory that is free *right now* (deterministic rule, no learning)."""
        client = self.ollama_client()
        info = client.show(validate_model_name(name))
        profile = self._ollama_profile(info)
        installed = set(client.list_models())
        budget = self._free_ram_budget()
        targets: list[dict[str, object]] = []
        recommended: str | None = None
        for item in plan_targets(profile, self.safety.max_quality_loss):
            derived = derived_model_name(info.name, item.target)
            # weights + ~25% for KV cache / buffers at a modest context
            fits = int(item.estimated_bytes * 1.25) <= budget
            if fits and recommended is None:
                recommended = derived
            targets.append({
                "quant": item.target.value, "name": derived,
                "estimated_bytes": item.estimated_bytes,
                "quality_loss_prior": item.quality_loss_prior,
                "installed": derived in installed, "fits_free_ram_now": fits,
            })
        return {
            "model": info.name, "source_precision": profile.quant.name,
            "eligible": is_quantizable(profile), "free_ram_budget_bytes": budget,
            "targets": targets, "recommended": recommended,
            "note": None if is_quantizable(profile) else (
                f"{profile.quant.name} models are already quantized; re-quantizing would "
                "compound error. Only F16/BF16/F32 sources are converted."),
        }

    def auto_quantize(
        self, name: str, targets: Sequence[str] | None = None,
    ) -> list[QuantizeResult]:
        """Create quantized copies of an F16/BF16/F32 Ollama model and register them in
        Ollama, so the RL engine can switch precision. Existing copies are skipped.

        Each copy: deterministic re-quantization (numpy) -> validation with the hardened
        GGUF parser -> upload to Ollama's blob store (SHA-256 verified) -> create a model
        that reuses the source's chat template and parameters -> temporary file removed.
        """
        client = self.ollama_client()
        info = client.show(validate_model_name(name))
        profile = self._ollama_profile(info)
        if not is_quantizable(profile):
            raise ValidationError(
                f"{info.name} is {profile.quant.name}; only F16/BF16/F32 models can be quantized"
            )
        wanted = None if targets is None else {
            require_enum("target", t, QuantTarget) for t in targets
        }
        installed = set(client.list_models())
        settings = client.model_settings(info.name)
        workdir = self.data_dir / "quantized"
        workdir.mkdir(parents=True, exist_ok=True)
        results: list[QuantizeResult] = []
        for item in plan_targets(profile, self.safety.max_quality_loss):
            if wanted is not None and item.target not in wanted:
                continue
            derived = derived_model_name(info.name, item.target)
            if derived in installed:
                results.append(QuantizeResult(derived, item.target, item.estimated_bytes,
                                              0.0, created=False))
                continue
            dst = workdir / f"{hashlib.sha256(derived.encode()).hexdigest()[:16]}.gguf"
            try:
                new_profile, seconds = timed_quantize(profile.path, dst, item.target)
                digest = client.push_blob(dst)
                client.create_model(derived, gguf_digest=digest, filename="model.gguf",
                                    settings=settings)
                if not client.show(derived).chat_capable:
                    raise QuantizationError(f"Ollama did not register {derived} as a chat model")
            finally:
                if dst.exists():
                    dst.unlink()  # Ollama keeps its own copy in the blob store
            results.append(QuantizeResult(derived, item.target, new_profile.file_size,
                                          round(seconds, 2), created=True))
        return results

    # ---- bringing in any model ----------------------------------------------------------------

    def import_gguf(self, path: str | os.PathLike[str], name: str | None = None) -> str:
        """Register a local GGUF file with Ollama so it can be used like any other model.

        The file is validated with the hardened GGUF parser *before* anything is sent to
        Ollama. Ollama derives the chat template from the GGUF metadata. Re-importing the
        same file is a no-op (content-addressed). Returns the Ollama model name.
        """
        profile = analyze_model(path)  # validation + architecture check
        client = self.ollama_client()
        model_name = _check_import_name(
            validate_model_name(name or _local_model_name(profile.path)))
        installed = {n.lower() for n in client.list_models()}
        digest = client.push_blob(profile.path)
        if model_name.lower() not in installed:
            client.create_model(model_name, gguf_digest=digest,
                                filename=_safe_filename(profile.path.name), settings={})
        return model_name

    def pull_model(
        self, name: str,
        on_progress: Callable[[str, int | None, int | None], None] | None = None,
    ) -> str:
        """Download a model into Ollama (library name or 'hf.co/<user>/<repo>[:<quant>]')."""
        self.ollama_client().pull(name, on_progress)
        return validate_model_name(name)

    # ---- measured quality -------------------------------------------------------------------

    def evaluate_quality(self, name: str) -> dict[str, object]:
        """Measure how much each installed version of a model changes its answers versus
        the highest-precision version (see ``air_sdk.quality``). Results are stored and
        replace the catalogue quality priors the next time the model is optimised."""
        client = self.ollama_client()
        info = client.show(validate_model_name(name))
        primary = self._ollama_profile(info)
        family: list[tuple[str, ModelProfile]] = [(info.name, primary)]
        for other in self.list_ollama_models():
            if other.name == info.name:
                continue
            try:
                profile = self._ollama_profile(other)
            except (AirError, OSError):
                continue
            if profile.family_key == primary.family_key:
                family.append((other.name, profile))
        if len(family) < 2:
            return {"model": info.name, "reference": None, "results": [], "skipped": [],
                    "note": "only one version installed; create quantized versions first"}
        family.sort(key=lambda item: (-item[1].weight_bytes, item[0]))  # highest precision
        budget = self._free_ram_budget()
        fits = {n: int(p.weight_bytes * 1.3) <= budget for n, p in family}
        ref_name, ref_profile = family[0]
        if not fits[ref_name]:
            raise InsufficientResourcesError(
                f"the reference version {ref_name} does not fit free RAM right now; "
                "close other applications and try again")
        runnable = [n for n, _p in family[1:] if fits[n]]
        keys = dict(family)
        results = evaluate_against_reference(client.greedy_answer, ref_name, runnable,
                                             release=client.release)
        limit = self.safety.max_quality_loss
        for r in results:
            self.store.record_quality(keys[r.model_name].variant_key, ref_profile.variant_key,
                                      r.agreement, r.measured_loss, r.prompts)
        return {
            "model": info.name, "reference": ref_name, "limit": limit,
            "results": [r.as_dict() | {"within_limit": r.measured_loss <= limit + 1e-9}
                        for r in results],
            "skipped": [n for n, _p in family[1:] if not fits[n]],
            "note": None,
        }

    def release(self, model: AdaptiveModel) -> None:
        """Close a model and drop it from this runtime (frees backend memory)."""
        model.close()
        if model in self.models:
            self.models.remove(model)

    def load_model(
        self,
        path: str | os.PathLike[str],
        *,
        variants: list[str | os.PathLike[str]] | None = None,
        discover: bool = False,
    ) -> AdaptiveModel:
        """Import and analyse a GGUF model.

        Args:
            variants: extra GGUF files of the same model at other precisions.
            discover: also scan the model's directory for such variants.
        """
        if self._closed:
            raise AirError("runtime has been closed")
        primary = analyze_model(path)
        found: dict[str, ModelProfile] = {primary.variant_key: primary}
        if require_bool("discover", discover):
            for v in discover_variants(primary):
                found.setdefault(v.variant_key, v)
        for extra in variants or []:
            profile = analyze_model(extra)
            if profile.family_key != primary.family_key:
                raise ValidationError(f"{profile.path} is not a variant of {primary.name}")
            found.setdefault(profile.variant_key, profile)
        model = AdaptiveModel(self, primary, tuple(found.values()), self._make_backend())
        self.models.append(model)
        return model

    def close(self) -> None:
        if self._closed:
            return
        for m in self.models:
            m.close()
        self.telemetry.stop()
        self.store.close()
        if self._ollama is not None:
            self._ollama.close()
        self._closed = True

    def __enter__(self) -> AdaptiveRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
