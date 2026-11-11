"""The simple way to use AIR-SDK: one function, one object.

    import air_sdk

    llm = air_sdk.load("qwen2.5:0.5b")                         # any installed Ollama model
    llm = air_sdk.load("llama3.2:1b", pull=True)               # download it if missing
    llm = air_sdk.load("hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M", pull=True)
    llm = air_sdk.load("C:/models/my-model.Q4_K_M.gguf")       # any local GGUF file

    print(llm.ask("What is QUIC?"))                            # one question -> text

    for piece in llm.stream("Tell me a short story"):          # live, word by word
        print(piece, end="", flush=True)

    chat = llm.conversation(system="You are a friendly tutor.")
    print(chat.say("What is a CPU?"))
    print(chat.say("And how is a GPU different?"))             # remembers the conversation

    print(llm.summary())                                       # what AIR-SDK chose and why
    llm.close()

Everything else - hardware analysis, safe memory planning, choosing precision / CPU-GPU
split / context size, learning from every reply, rollback on failure - happens automatically.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import platform
import queue
import shutil
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final

import httpx
import psutil

from ._validate import GiB, require_bool, require_enum, require_float, require_int, require_str
from .analysis.profile import analyze_model
from .backends.ollama import DEFAULT_BASE_URL, OllamaClient, OllamaError
from .config import Objective, SafetyPolicy
from .context import AUTO, ContextEngine, Prepared, SessionMemory, is_auto
from .errors import (
    AirError,
    InsufficientResourcesError,
    ModelNotFoundError,
    SetupError,
    ValidationError,
)
from .hardware.profiler import profile_hardware
from .runtime import AdaptiveModel, AdaptiveRuntime, GenerationResponse
from .skills import SkillEngine, SkillOutcome, get_skill

__all__ = [
    "LLM",
    "Conversation",
    "DoctorReport",
    "Reply",
    "doctor",
    "list_models",
    "load",
]

ProgressCallback = Callable[[str, int | None, int | None], None]

OLLAMA_INSTALL_HINT: Final[str] = (
    "Install Ollama from https://ollama.com, then start it (open the Ollama app, or run "
    "`ollama serve` in a terminal)."
)


class Goal(str, Enum):
    """What to optimise for."""

    BALANCED = "balanced"
    SPEED = "speed"  # most tokens per second
    LATENCY = "latency"  # fastest first word
    MEMORY = "memory"  # smallest footprint


_GOAL_TO_OBJECTIVE: Final = {
    Goal.BALANCED: Objective.BALANCED,
    Goal.SPEED: Objective.THROUGHPUT,
    Goal.LATENCY: Objective.LATENCY,
    Goal.MEMORY: Objective.MEMORY,
}


class _Source(Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL_FILE = "local_file"


# ---- results ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    """One answer plus what AIR-SDK did to produce it. ``str(reply)`` is the text."""

    text: str
    tokens: int
    tokens_per_second: float
    time_to_first_token_s: float
    precision: str
    device: str
    context_tokens: int
    action: str
    explored: bool
    reward: float | None
    prepared_in_background: bool
    memory_relief: bool
    rolled_back: bool
    # Verified skills: which task, whether the answer passed its check (None = no check),
    # whether a more accurate precision was needed, and every attempt (precision, ok, why).
    skill: str = "general"
    verified: bool | None = None
    escalated: bool = False
    check: str = ""
    attempts: tuple[tuple[str, bool | None, str], ...] = ()
    # Context: the task was detected automatically, which section of the conversation
    # answered, how many stored facts were sent, and a notice if memory had to be cut.
    auto: bool = False
    section: str = ""
    facts_sent: int = 0
    prompt_tokens: int = 0
    notice: str = ""

    def __str__(self) -> str:
        return self.text

    @property
    def label(self) -> str:
        """One line: ``Section: Tables: orders (sql, detected) - check passed``."""
        how = "detected" if self.auto else "chosen"
        check = {True: "check passed", False: "check FAILED", None: "no check"}[self.verified]
        where = f"Section: {self.section} - " if self.section else ""
        return f"{where}task {self.skill} ({how}) - {check}"

    @classmethod
    def _from(cls, r: GenerationResponse, outcome: SkillOutcome | None = None,
              prepared: Prepared | None = None) -> Reply:
        c = r.config
        return cls(
            text=r.text, tokens=r.completion_tokens, tokens_per_second=r.tokens_per_second,
            time_to_first_token_s=r.ttft_s, precision=c.precision,
            device="CPU" if c.n_gpu_layers == 0 else f"GPU ({c.n_gpu_layers} layers)",
            context_tokens=c.n_ctx, action=r.action, explored=r.explored, reward=r.reward,
            prepared_in_background=r.prefetched, memory_relief=r.relief,
            rolled_back=r.rolled_back,
            skill=outcome.skill if outcome else "general",
            verified=outcome.verified if outcome else None,
            escalated=outcome.escalated if outcome else False,
            check=outcome.detail if outcome else "",
            attempts=tuple((a.precision, a.ok, a.detail) for a in outcome.attempts)
            if outcome else (),
            auto=prepared.auto if prepared else False,
            section=prepared.section if prepared else "",
            facts_sent=len(prepared.plan.facts) if prepared else 0,
            prompt_tokens=prepared.plan.prompt_tokens if prepared else 0,
            notice=prepared.plan.notice if prepared else "",
        )


_RECOVER_NOTICE = "\n[recovering: retrying with a safer configuration]\n"
_ESCALATE_PREFIX = "\n[check failed"


# ---- helpers ----------------------------------------------------------------------------------


def _classify(model: object) -> tuple[_Source, str]:
    if isinstance(model, os.PathLike):
        return _Source.LOCAL_FILE, str(os.fspath(model))
    text = require_str("model", model, max_len=1024).strip()
    if text.lower().endswith(".gguf"):
        return _Source.LOCAL_FILE, text
    with contextlib.suppress(OSError, ValueError):
        if Path(text).is_file():
            return _Source.LOCAL_FILE, text
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/",
                   "hf://"):
        if text.lower().startswith(prefix):
            return _Source.HUGGINGFACE, "hf.co/" + text[len(prefix):]
    if text.lower().startswith("hf.co/"):
        return _Source.HUGGINGFACE, text
    return _Source.OLLAMA, text


def _installed_name(client: OllamaClient, name: str) -> str | None:
    """Match a user-typed name to an installed one ('qwen2.5' -> 'qwen2.5:latest'),
    case-insensitively."""
    installed = {n.lower(): n for n in client.list_models()}
    last = name.rsplit("/", 1)[-1]
    candidates = [name] if ":" in last else [name, f"{name}:latest"]
    for c in candidates:
        if c.lower() in installed:
            return installed[c.lower()]
    return None


def _similar_installed(client: OllamaClient, name: str, limit: int = 3) -> list[str]:
    """Installed names that look like what the user typed (same family, or close spelling)."""
    installed = client.list_models()
    base = name.split(":", 1)[0].lower()
    same_family = [n for n in installed if n.lower().split(":", 1)[0] == base]
    close = difflib.get_close_matches(name.lower(), [n.lower() for n in installed], n=limit,
                                      cutoff=0.6)
    lookup = {n.lower(): n for n in installed}
    merged = dict.fromkeys([*same_family, *(lookup[c] for c in close)])
    return list(merged)[:limit]


def _check_ollama(client: OllamaClient) -> str:
    try:
        return client.version()
    except (httpx.HTTPError, OllamaError) as exc:
        raise SetupError(f"Cannot reach Ollama at {client.base_url}.",
                         hint=OLLAMA_INSTALL_HINT) from exc


def _free_model_budget(safety: SafetyPolicy) -> int:
    vm = psutil.virtual_memory()
    return max(0, int(vm.available) - safety.ram_headroom(int(vm.total)))


def _models_that_fit(client: OllamaClient, safety: SafetyPolicy) -> list[str]:
    budget = _free_model_budget(safety)
    fits: list[str] = []
    for name in client.list_models():
        try:
            info = client.show(name)
            if info.chat_capable and analyze_model(
                    info.blob_path, suffixes=("", ".gguf")).weight_bytes * 1.3 <= budget:
                fits.append(name)
        except (AirError, OSError, httpx.HTTPError):
            continue
    return fits


def _fit_hint(client: OllamaClient, safety: SafetyPolicy, message: str) -> str:
    if "no GPU" in message:
        return "No NVIDIA GPU was detected on this machine; use device='auto' or device='cpu'."
    fits = _models_that_fit(client, safety)
    smaller = f" Models that fit right now: {', '.join(fits)}." if fits else ""
    return ("Close other programs to free memory, or pick a smaller model (or a more "
            f"compressed version, e.g. a Q4 tag).{smaller} Advanced: load(..., "
            "reserve_ram_gb=...) changes how much RAM is kept free for other programs.")


# ---- the main object --------------------------------------------------------------------------


class LLM:
    """A loaded, self-optimising language model. Create with :func:`load`."""

    def __init__(self, runtime: AdaptiveRuntime, model: AdaptiveModel, name: str,
                 settings: dict[str, object]) -> None:
        self._rt = runtime
        self._model = model
        self._name = name
        self._settings = settings
        self._lock = threading.Lock()
        self._skill = AUTO
        self._engine = SkillEngine(model, runtime.store)
        self._context = ContextEngine()
        self.last: Reply | None = None

    # -- basics --------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> AdaptiveModel:
        """The underlying :class:`AdaptiveModel` (advanced use)."""
        return self._model

    @property
    def runtime(self) -> AdaptiveRuntime:
        """The underlying :class:`AdaptiveRuntime` (advanced use)."""
        return self._rt

    def __repr__(self) -> str:
        cfg = self._model.current_config
        where = "not loaded" if cfg is None else (
            f"{cfg.precision}, {'CPU' if cfg.n_gpu_layers == 0 else 'GPU'}")
        return f"<LLM {self._name!r} ({where})>"

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Free the model's memory and save what was learned."""
        self._rt.close()

    # -- asking --------------------------------------------------------------------------------

    def ask(self, prompt: str, *, skill: str | None = None, context: str | None = None,
            system: str | None = None, max_tokens: int = 512,
            temperature: float | None = None) -> str:
        """Ask one question, get the answer text.

        The task is detected automatically (``skill="auto"``, the default): a SQL question
        is checked against the schema, a calculation against the computed result, and so
        on. ``skill`` (``"sql"``, ``"python"``, ``"json"``, ``"math"``, ``"summarise"``,
        ``"general"``) forces one; ``context`` gives facts to use and check against (SQL
        schema, required JSON keys, expected number).
        """
        return self.reply(prompt, skill=skill, context=context, system=system,
                          max_tokens=max_tokens, temperature=temperature).text

    def reply(self, prompt: str, *, skill: str | None = None, context: str | None = None,
              system: str | None = None, max_tokens: int = 512,
              temperature: float | None = None) -> Reply:
        """Like :meth:`ask` but returns a :class:`Reply` with speed, decisions and checks."""
        return self._chat(SessionMemory(), prompt, system, max_tokens, temperature,
                          skill=skill, context=context)

    def stream(self, prompt: str, *, skill: str | None = None, context: str | None = None,
               system: str | None = None, max_tokens: int = 512,
               temperature: float | None = None) -> Iterator[str]:
        """Yield the answer piece by piece as it is generated."""
        return self._stream(SessionMemory(), prompt, system, max_tokens, temperature,
                            skill=skill, context=context)

    def conversation(self, system: str | None = None, *, skill: str | None = None,
                     context: str | None = None) -> Conversation:
        """Start a conversation with sectioned memory: each piece of work (a database, a
        piece of code, a calculation...) keeps its own facts, and only the relevant ones
        are sent with each message."""
        if skill is not None and not is_auto(skill):
            get_skill(skill)  # validate early
        return Conversation(self, system, skill=skill, context=context)

    # -- skills ----------------------------------------------------------------------------------

    def use(self, skill: str | None) -> None:
        """Set the task for following questions: ``"auto"`` (default: detect per message),
        ``"sql"``, ``"python"``, ``"json"``, ``"math"``, ``"summarise"`` or ``"general"``.
        Verified skills answer on the fastest precision, check the answer, and retry on a
        more accurate precision only if the check fails."""
        self._skill = AUTO if skill is not None and is_auto(skill) else get_skill(skill).name

    @property
    def skill(self) -> str:
        """The active skill (``"auto"`` = detected per message)."""
        return self._skill

    def skills(self) -> list[dict[str, object]]:
        """All skills with their check, and per-precision pass/fail evidence for this model."""
        return self._engine.catalogue()

    def skill_report(self, skill: str | None = None) -> dict[str, object]:
        """Evidence for one skill (default: the active one): passes and fails per precision,
        and which precision it will start on next."""
        name = skill or self._skill
        if is_auto(name):
            name = self.last.skill if self.last else "general"
        return self._engine.report(name)

    # -- internals -------------------------------------------------------------------------------

    def _exchange(self, memory: SessionMemory, text: str, system: str | None,
                  max_tokens: int, temperature: float | None, *, skill: str | None,
                  context: str | None, previous: Reply | None = None,
                  on_token: Callable[[str], None] | None = None,
                  on_reset: Callable[[], None] | None = None,
                  on_escalate: Callable[[str, str, str], None] | None = None) -> Reply:
        """prepare (facts, task, section, budget) -> answer + check -> remember."""
        max_tokens = require_int("max_tokens", max_tokens, min_value=1, max_value=1 << 16)
        with memory.lock:
            p = self._context.prepare(
                memory, text, skill=skill if skill is not None else self._skill,
                context=context, system=system, context_limit=self._model.context_limit,
                max_tokens=max_tokens)
            with self._lock:
                if p.correction and previous is not None and previous.verified:
                    # "That's wrong": a passed check missed it, so count a fail there.
                    self._engine.record_correction(previous.skill, previous.precision)
                out = self._engine.run(
                    p.messages, p.skill, context=p.prompt_context,
                    check_context=p.check_context, bonus=True, bonus_schema=p.bonus_schema,
                    max_tokens=max_tokens, temperature=temperature, on_token=on_token,
                    on_reset=on_reset, on_escalate=on_escalate)
            self._context.commit(memory, p, out.response.text)
        reply = Reply._from(out.response, out, p)
        self.last = reply
        return reply

    def _chat(self, memory: SessionMemory, text: str, system: str | None, max_tokens: int,
              temperature: float | None, *, skill: str | None = None,
              context: str | None = None, previous: Reply | None = None) -> Reply:
        return self._exchange(memory, text, system, max_tokens, temperature, skill=skill,
                              context=context, previous=previous)

    def _stream(self, memory: SessionMemory, text: str, system: str | None,
                max_tokens: int, temperature: float | None, *, skill: str | None = None,
                context: str | None = None, previous: Reply | None = None) -> Iterator[str]:
        q: queue.Queue[tuple[str, object]] = queue.Queue()

        def work() -> None:
            try:
                r = self._exchange(
                    memory, text, system, max_tokens, temperature, skill=skill,
                    context=context, previous=previous,
                    on_token=lambda t: q.put(("token", t)),
                    on_reset=lambda: q.put(("reset", None)),
                    on_escalate=lambda a, b, why: q.put(("escalate", (a, b, why))))
                q.put(("done", r))
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
                q.put(("error", exc))

        thread = threading.Thread(target=work, name="air-stream", daemon=True)
        thread.start()
        try:
            while True:
                kind, value = q.get()
                if kind == "token" and isinstance(value, str):
                    yield value
                elif kind == "reset":
                    yield _RECOVER_NOTICE
                elif kind == "escalate" and isinstance(value, tuple):
                    _from, to, why = value
                    yield f"{_ESCALATE_PREFIX} on {_from}: {why}; retrying on {to}]\n"
                elif kind == "done" and isinstance(value, Reply):
                    return
                elif kind == "error" and isinstance(value, BaseException):
                    raise value
        finally:
            thread.join()

    # -- insight -------------------------------------------------------------------------------

    def report(self) -> dict[str, object]:
        """Everything AIR-SDK knows: model profile, plan, performance, learning."""
        return {
            "model": self._name,
            "profile": self._model.analyze().as_dict(),
            "plan": self._model.plan.as_dict() if self._model.plan else None,
            "performance": self._model.performance_report(),
            "learning": self._model.learning_report(),
        }

    def summary(self) -> str:
        """Human-readable explanation of what AIR-SDK chose and how it is doing."""
        p = self._model.analyze()
        plan = self._model.plan
        perf = self._model.performance_report()
        learn = self._model.learning_report()
        cfg = self._model.current_config
        hw = self._rt.hardware
        gpu = hw.primary_gpu.name if hw.primary_gpu else "none (CPU only)"
        lines = [
            f"Model      : {self._name}  ({p.architecture}, {p.n_params / 1e9:.2f}B parameters,"
            f" {p.n_layers} layers)",
            f"Hardware   : {hw.physical_cores} cores, {hw.total_ram / GiB:.1f} GB RAM, GPU: {gpu}",
            f"Versions   : {', '.join(v.quant.name for v in self._model.variants)}",
        ]
        if plan is not None:
            chosen_by = {"learned": "what it learned before", "heuristic": "hardware prediction",
                         "probe": "a quick speed test", "conservative": "safety fallback"}
            lines.append(f"Started on : {plan.config.precision} (chosen by "
                         f"{chosen_by.get(plan.source, plan.source)})")
        if cfg is not None:
            where = "CPU" if cfg.n_gpu_layers == 0 else f"GPU ({cfg.n_gpu_layers} layers)"
            lines.append(f"Now using  : {cfg.precision} on {where}, {cfg.n_threads} threads,"
                         f" context {cfg.n_ctx}/{perf['context_max']} tokens")
        probe = perf.get("probe")
        for row in probe if isinstance(probe, list) else []:
            if isinstance(row, dict):
                lines.append(f"Speed test : {row['precision']:7} {row['tokens_per_second']} tok/s"
                             f"{'  <- chosen' if row.get('chosen') else ''}")
        sk = self.skill_report()
        if is_auto(self._skill):
            lines.append("Skill      : auto (task detected per message; SQL, Python, JSON and "
                         "maths answers are checked)")
        elif sk["verified"]:
            lines.append(f"Skill      : {self._skill} (checked answers; starts on "
                         f"{sk['starts_on']}, escalates only if a check fails)")
        else:
            lines.append(f"Skill      : {self._skill} (no automatic check)")
        lines += [
            f"Replies    : {perf['requests']}  (avg {perf['mean_tokens_per_second']} tok/s,"
            f" failures {perf['failures']}, memory switches {perf['memory_pressure_reliefs']})",
            f"Learning   : {learn.get('episodes', 0)} episodes, exploration "
            f"{learn.get('epsilon', '-')}, reward early {learn.get('mean_reward_early')}"
            f" -> recent {learn.get('mean_reward_recent')}",
        ]
        return "\n".join(lines)

    # -- make it better --------------------------------------------------------------------------

    def make_faster(self) -> dict[str, object]:
        """Create smaller, faster versions of this model (Q8_0 / Q4_0) if it is a
        full-precision model, measure their answer quality, and reload so AIR-SDK can use
        them. Takes about a minute for small models."""
        with self._lock:
            plan = self._rt.quantization_plan(self._name)
            if not plan["eligible"]:
                return {"created": [], "note": plan["note"]}
            self._rt.release(self._model)
            created = self._rt.auto_quantize(self._name)
            quality = self._rt.evaluate_quality(self._name)
            self._reload()
        return {"created": [r.model_name for r in created if r.created], "quality": quality}

    def measure_quality(self) -> dict[str, object]:
        """Measure how much each installed version changes the answers (vs the original)."""
        with self._lock:
            self._rt.release(self._model)
            try:
                return self._rt.evaluate_quality(self._name)
            finally:
                self._reload()

    def _reload(self) -> None:
        self._model = self._rt.load_ollama_model(self._name, discover=True)
        self._model.optimize(**self._settings)  # type: ignore[arg-type]
        self._engine = SkillEngine(self._model, self._rt.store)


# ---- conversations ----------------------------------------------------------------------------


@dataclass
class Conversation:
    """A chat with sectioned memory. Create with :meth:`LLM.conversation`.

    Each piece of work (a database, some code, a calculation, ...) gets its own *section*
    with its own facts (schema, definitions, function names, required keys) and turns.
    Messages are routed to the right section automatically ("back to the orders query"
    and "new task: ..." also work); only the active section's relevant facts and recent
    turns are sent, within a token budget. Answers are checked against the full facts.
    """

    llm: LLM
    system: str | None = None
    history: list[dict[str, object]] = field(default_factory=list)  # full transcript
    max_messages: int = 100  # transcript length kept in ``history``
    skill: str | None = None  # None = follow the LLM's active skill ("auto" by default)
    context: str | None = None  # e.g. a SQL schema that applies to the conversation
    memory: SessionMemory = field(default_factory=SessionMemory)
    last: Reply | None = None

    def _record(self, text: str, answer: str) -> None:
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": answer}]
        del self.history[:-self.max_messages]

    def say(self, text: str, *, max_tokens: int = 512,
            temperature: float | None = None) -> str:
        """Send a message and get the reply."""
        return self.reply(text, max_tokens=max_tokens, temperature=temperature).text

    def reply(self, text: str, *, max_tokens: int = 512,
              temperature: float | None = None) -> Reply:
        """Like :meth:`say` but returns the :class:`Reply` (with ``section`` and checks)."""
        r = self.llm._chat(self.memory, text, self.system, max_tokens, temperature,
                           skill=self.skill, context=self.context, previous=self.last)
        self.last = r
        self._record(text, r.text)
        return r

    def stream(self, text: str, *, max_tokens: int = 512,
               temperature: float | None = None) -> Iterator[str]:
        """Like :meth:`say` but yields the reply piece by piece."""
        yield from self.llm._stream(self.memory, text, self.system, max_tokens, temperature,
                                    skill=self.skill, context=self.context,
                                    previous=self.last)
        self.last = self.llm.last
        if self.last is not None:
            self._record(text, self.last.text)

    @property
    def section(self) -> str:
        """Label of the active section ("" before the first message)."""
        with self.memory.lock:
            s = self.memory.sections.get(self.memory.active) \
                if self.memory.active is not None else None
            return s.label if s else ""

    def sections(self) -> dict[str, object]:
        """What the conversation remembers: preferences and every section's facts."""
        return self.memory.snapshot()

    def reset(self) -> None:
        """Forget the conversation (keeps the system message)."""
        self.history.clear()
        self.memory.forget()
        self.last = None


# ---- load ---------------------------------------------------------------------------------------


def load(
    model: str | os.PathLike[str],
    *,
    device: str = "auto",
    goal: str = "balanced",
    learn: bool = True,
    max_context: int = 4096,
    speed_test: bool = True,
    pull: bool = False,
    quality_limit: float = 0.05,
    reserve_ram_gb: float | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    ollama_url: str = DEFAULT_BASE_URL,
    on_progress: ProgressCallback | None = None,
) -> LLM:
    """Load any LLM and get it ready, optimised for this machine.

    Args:
        model: an Ollama model name (``"qwen2.5:0.5b"``), a HuggingFace GGUF
            (``"hf.co/<user>/<repo>[:<quant>]"`` or a huggingface.co URL), or a path to a
            local ``.gguf`` file.
        device: ``"auto"`` (recommended), ``"cpu"`` or ``"gpu"`` (NVIDIA).
        goal: ``"balanced"``, ``"speed"``, ``"latency"`` or ``"memory"``.
        learn: let the reinforcement-learning engine keep improving the setup.
        max_context: longest conversation (in tokens) to support; memory for it is only
            reserved when a conversation actually gets that long.
        speed_test: quickly measure each available precision on load and start on the best.
        pull: download the model if it is not installed yet.
        quality_limit: maximum acceptable quality loss for compressed versions (0-1).
        reserve_ram_gb: RAM always kept free for your other programs. Default: 10% of total
            RAM (at least 1 GB). Lower it only if you know the machine can spare it.
        data_dir: where learning is stored (default ``~/.air_sdk``).
        ollama_url: address of the Ollama server.
        on_progress: ``callback(status, done_bytes, total_bytes)`` while downloading.

    Raises:
        SetupError / ModelNotFoundError: with a *how to fix* hint.
    """
    goal_enum = require_enum("goal", goal, Goal)
    require_enum("device", device, _Device)
    learn = require_bool("learn", learn)
    max_context = require_int("max_context", max_context, min_value=256, max_value=1 << 20)
    speed_test = require_bool("speed_test", speed_test)
    pull = require_bool("pull", pull)
    quality_limit = require_float("quality_limit", quality_limit, min_value=0.0, max_value=1.0)
    if reserve_ram_gb is None:
        safety = SafetyPolicy(max_quality_loss=quality_limit)
    else:
        reserve = int(require_float("reserve_ram_gb", reserve_ram_gb, min_value=0.25,
                                    max_value=1024.0) * GiB)
        safety = SafetyPolicy(max_quality_loss=quality_limit, ram_headroom_min=reserve,
                              ram_headroom_fraction=0.0)

    runtime = AdaptiveRuntime(
        _GOAL_TO_OBJECTIVE[goal_enum], rl_enabled=learn, auto_adapt=learn, backend="ollama",
        data_dir=data_dir, safety=safety, background_switching=True, ollama_url=ollama_url,
    )
    try:
        client = runtime.ollama_client()
        _check_ollama(client)
        name = _resolve(runtime, client, model, pull=pull, on_progress=on_progress)
        info = client.show(name)
        if not info.chat_capable:
            raise SetupError(f"{name} cannot chat (capabilities: {', '.join(info.capabilities)}).",
                             hint="Pick a chat model; see them with air_sdk.list_models().")
        adaptive = runtime.load_ollama_model(name, discover=True)
        settings: dict[str, object] = {
            "device_placement": device,
            "n_ctx": min(max_context, adaptive.analyze().context_length),
            "probe": speed_test,
            "auto_context": True,
        }
        try:
            adaptive.optimize(**settings)  # type: ignore[arg-type]
        except InsufficientResourcesError as exc:
            raise SetupError(str(exc), hint=_fit_hint(client, safety, str(exc))) from exc
        return LLM(runtime, adaptive, name, settings)
    except BaseException:
        runtime.close()
        raise


class _Device(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


def _resolve(runtime: AdaptiveRuntime, client: OllamaClient, model: object, *, pull: bool,
             on_progress: ProgressCallback | None) -> str:
    """Turn any model reference into an installed Ollama model name."""
    source, ref = _classify(model)
    if source is _Source.LOCAL_FILE:
        try:
            return runtime.import_gguf(ref)
        except ValidationError as exc:
            raise ModelNotFoundError(f"Cannot use the file {ref!r}: {exc}",
                                     hint="Check the path; it must be an existing .gguf file."
                                     ) from exc
    installed = _installed_name(client, ref)
    if installed is not None:
        return installed
    if not pull:
        similar = _similar_installed(client, ref)
        did_you_mean = f"Did you mean: {', '.join(similar)}? Otherwise download it" if similar \
            else "Download it"
        raise ModelNotFoundError(
            f"Model {ref!r} is not installed in Ollama.",
            hint=f"{did_you_mean} with `ollama pull {ref}`, or call "
                 f"air_sdk.load({ref!r}, pull=True). See installed models with "
                 "air_sdk.list_models().")
    try:
        runtime.pull_model(ref, on_progress)
    except OllamaError as exc:
        raise ModelNotFoundError(
            f"Could not download {ref!r}: {exc.message}",
            hint="Check the name at https://ollama.com/library (or the HuggingFace repo for "
                 "hf.co/... names) and your internet connection.") from exc
    return _installed_name(client, ref) or ref


# ---- discovery & diagnostics ------------------------------------------------------------------


def list_models(ollama_url: str = DEFAULT_BASE_URL) -> list[dict[str, object]]:
    """Installed Ollama models, whether they can chat, and whether they fit free RAM now."""
    client = OllamaClient(ollama_url)
    try:
        _check_ollama(client)
        safety = SafetyPolicy()
        budget = _free_model_budget(safety)
        out: list[dict[str, object]] = []
        for name in client.list_models():
            try:
                info = client.show(name)
                profile = analyze_model(info.blob_path, suffixes=("", ".gguf"))
            except (AirError, OSError, httpx.HTTPError):
                continue
            out.append({
                "name": name, "architecture": profile.architecture,
                "parameters": info.parameter_size or f"{profile.n_params / 1e9:.2f}B",
                "precision": profile.quant.name,
                "size_gb": round(profile.weight_bytes / 1e9, 2),
                "can_chat": info.chat_capable,
                "fits_now": profile.weight_bytes * 1.3 <= budget,
            })
        return out
    finally:
        client.close()


@dataclass(frozen=True, slots=True)
class _Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Result of :func:`doctor`. ``print(report)`` shows a readable checklist."""

    checks: tuple[_Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def __str__(self) -> str:
        mark = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
        lines = ["AIR-SDK environment check", "-" * 60]
        for c in self.checks:
            lines.append(f"{mark[c.status]} {c.name:<16} {c.detail}")
            if c.hint:
                lines.append(f"{'':7}{'':16} -> {c.hint}")
        lines.append("-" * 60)
        lines.append("Ready to go." if self.ok else "Fix the [FAIL] items above, then try again.")
        return "\n".join(lines)


def doctor(ollama_url: str = DEFAULT_BASE_URL) -> DoctorReport:
    """Check everything AIR-SDK needs and explain how to fix what is missing."""
    from . import __version__  # noqa: PLC0415 - avoid an import cycle at module load

    checks: list[_Check] = []
    py_ok = sys.version_info >= (3, 10)
    checks.append(_Check("Python", "ok" if py_ok else "fail", platform.python_version(),
                         None if py_ok else "Install Python 3.10 or newer."))
    checks.append(_Check("AIR-SDK", "ok", __version__))
    hw = profile_hardware()
    checks.append(_Check("CPU", "ok", f"{hw.physical_cores} cores / {hw.logical_cores} threads"))
    free_gb = hw.available_ram / GiB
    checks.append(_Check(
        "Memory", "ok" if free_gb >= 2 else "warn",
        f"{free_gb:.1f} GB free of {hw.total_ram / GiB:.1f} GB",
        None if free_gb >= 2 else "Little free RAM: close programs or use small models (<1B)."))
    if hw.gpus:
        g = hw.gpus[0]
        checks.append(_Check("GPU", "ok", f"{g.name}, {g.free_vram / GiB:.1f} GB free"))
    else:
        checks.append(_Check("GPU", "warn", "no NVIDIA GPU detected - models run on the CPU",
                             "This is fine. AIR-SDK will pick fast compressed versions for CPU."))
    client = OllamaClient(ollama_url)
    try:
        try:
            version = _check_ollama(client)
            checks.append(_Check("Ollama", "ok", f"version {version} at {client.base_url}"))
            names = client.list_models()
            chat_models = []
            for n in names:
                with contextlib.suppress(AirError, httpx.HTTPError):
                    if client.show(n).chat_capable:
                        chat_models.append(n)
            checks.append(_Check(
                "Models", "ok" if chat_models else "warn",
                f"{len(chat_models)} chat model(s): {', '.join(chat_models[:6])}"
                + ("..." if len(chat_models) > 6 else ""),
                None if chat_models else "Download one: `ollama pull qwen2.5:0.5b`"))
        except SetupError as exc:
            checks.append(_Check("Ollama", "fail", exc.message, exc.hint))
    finally:
        client.close()
    try:
        import llama_cpp  # noqa: F401, PLC0415 - optional

        checks.append(_Check("llama.cpp", "ok", "llama-cpp-python installed (optional)"))
    except ImportError:
        checks.append(_Check("llama.cpp", "ok", "not installed (optional; Ollama is used)"))
    data = Path.home() / ".air_sdk"
    probe_dir = data if data.exists() else Path.home()
    free_disk = shutil.disk_usage(probe_dir).free / GiB
    checks.append(_Check("Disk", "ok" if free_disk >= 2 else "warn", f"{free_disk:.1f} GB free",
                         None if free_disk >= 2 else "Free some disk space for models."))
    return DoctorReport(tuple(checks))
