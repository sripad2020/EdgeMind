"""Ollama backend: drives a local Ollama server through its HTTP API.

Mapping of AIR-SDK configuration -> Ollama runner options
  n_gpu_layers -> num_gpu      (layers offloaded to the GPU; 0 = CPU only)
  n_threads    -> num_thread
  n_batch      -> num_batch
  n_ctx        -> num_ctx
  weight precision -> which Ollama tag is used (e.g. qwen2.5:0.5b-instruct-q4_K_M vs -q8_0)
KV-cache type is a *server-wide* Ollama setting (OLLAMA_KV_CACHE_TYPE), so this backend
advertises exactly one KV type and the RL engine never tries to change it per request.

Measurements come from Ollama itself: decode speed from ``eval_count / eval_duration`` and
the model's RAM/VRAM footprint from ``/api/ps`` (``size`` / ``size_vram``).

Security
  * The base URL is fixed at construction (never taken from end users) and must be http(s).
  * Model names are validated against a strict pattern; the web layer additionally
    allow-lists them against ``/api/tags``.
  * Streamed lines are size-capped and parsed as JSON with type checks on every field used.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx

from .._validate import require_str
from ..analysis.quant import KVCacheType
from ..config import ExecutionConfig
from ..errors import (
    AirError,
    FailureKind,
    GenerationError,
    ModelLoadError,
    ValidationError,
)
from ..hardware.profiler import TelemetrySample
from .base import ChatMessage, GenerationResult, InferenceBackend, TokenCallback

DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:11434"
_MODEL_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/]{0,150}(:[A-Za-z0-9._\-]{1,80})?$")
_MAX_LINE_BYTES: Final[int] = 1 << 20
_MAX_ERROR_CHARS: Final[int] = 500
_OOM_MARKERS: Final = (
    "out of memory", "requires more system memory", "insufficient memory",
    "cudamalloc", "failed to allocate", "unable to allocate", "not enough memory",
)


class OllamaError(AirError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Ollama error {status}: {message}")
        self.status = status
        self.message = message


def validate_model_name(name: object) -> str:
    text = require_str("model name", name, max_len=232)
    if not _MODEL_NAME_RE.fullmatch(text):
        raise ValidationError(f"invalid Ollama model name {text!r}")
    return text


def _is_oom(message: str) -> bool:
    low = message.lower()
    return any(m in low for m in _OOM_MARKERS)


_KNOWN_PARAMETERS: Final = frozenset({
    "num_ctx", "num_batch", "num_gpu", "num_thread", "num_keep", "num_predict", "seed",
    "top_k", "top_p", "min_p", "typical_p", "repeat_last_n", "repeat_penalty", "temperature",
    "presence_penalty", "frequency_penalty", "stop", "penalize_newline", "mirostat",
    "mirostat_tau", "mirostat_eta", "tfs_z", "use_mmap", "numa", "low_vram",
})


def _parse_parameters(text: str) -> dict[str, object]:
    """Parse Ollama's ``parameters`` listing ("stop  \\"<|im_end|>\\"" per line) into the dict
    form ``/api/create`` accepts. Repeated keys (e.g. ``stop``) become lists; numbers are
    converted. Unknown/garbled lines are skipped. O(lines)."""
    out: dict[str, object] = {}
    for line in text.splitlines()[:200]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or parts[0] not in _KNOWN_PARAMETERS:
            continue
        key, raw = parts[0], parts[1].strip()
        value: object
        if len(raw) >= 2 and raw[0] == raw[-1] == '"':
            value = raw[1:-1]
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        if key in out:
            prev = out[key]
            out[key] = [*prev, value] if isinstance(prev, list) else [prev, value]
        elif key == "stop":
            out[key] = [value]
        else:
            out[key] = value
    return out


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True, slots=True)
class OllamaModelInfo:
    name: str
    blob_path: Path
    capabilities: tuple[str, ...]
    family: str
    parameter_size: str
    quantization: str

    @property
    def chat_capable(self) -> bool:
        return "completion" in self.capabilities


@dataclass(frozen=True, slots=True)
class RunningModel:
    name: str
    size: int
    size_vram: int


class OllamaClient:
    """Minimal, typed Ollama HTTP client."""

    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, *, timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parts = urlsplit(require_str("base_url", base_url, max_len=256))
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValidationError("base_url must be an http(s) URL with a host")
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    @staticmethod
    def _error(resp: httpx.Response) -> OllamaError:
        try:
            body = resp.json()
            msg = body.get("error") if isinstance(body, dict) else None
        except (json.JSONDecodeError, ValueError):
            msg = None
        text = msg if isinstance(msg, str) else resp.text
        return OllamaError(resp.status_code, text[:_MAX_ERROR_CHARS])

    def get_json(self, path: str) -> dict[str, object]:
        resp = self._http.get(path)
        if resp.status_code >= 400:
            raise self._error(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise OllamaError(resp.status_code, "unexpected response shape")
        return data

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        resp = self._http.post(path, json=payload)
        if resp.status_code >= 400:
            raise self._error(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise OllamaError(resp.status_code, "unexpected response shape")
        return data

    def stream(self, path: str, payload: dict[str, object]) -> Iterator[dict[str, object]]:
        """Yield NDJSON objects; each line is size-capped before parsing."""
        with self._http.stream("POST", path, json=payload) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise self._error(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                if len(line) > _MAX_LINE_BYTES:
                    raise OllamaError(502, "stream line exceeds size limit")
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise OllamaError(502, "unexpected stream item")
                yield obj

    def version(self) -> str:
        v = self.get_json("/api/version").get("version")
        return v[:32] if isinstance(v, str) else "unknown"

    def list_models(self) -> list[str]:
        models = self.get_json("/api/tags").get("models")
        names: list[str] = []
        if isinstance(models, list):
            for m in models:
                name = m.get("name") if isinstance(m, dict) else None
                if isinstance(name, str) and _MODEL_NAME_RE.fullmatch(name):
                    names.append(name)
        return names

    def show(self, name: str) -> OllamaModelInfo:
        name = validate_model_name(name)
        data = self.post_json("/api/show", {"model": name})
        modelfile = data.get("modelfile")
        if not isinstance(modelfile, str):
            raise OllamaError(502, f"no modelfile for {name}")
        source = next(
            (ln[5:].strip() for ln in modelfile.splitlines() if ln.startswith("FROM ")), None
        )
        if not source:
            raise OllamaError(502, f"no FROM line for {name}")
        caps_raw = data.get("capabilities")
        caps = tuple(c for c in caps_raw if isinstance(c, str)) if isinstance(
            caps_raw, list) else ()
        details = data.get("details")
        d = details if isinstance(details, dict) else {}

        def text(key: str) -> str:
            v = d.get(key)
            return v[:64] if isinstance(v, str) else ""

        return OllamaModelInfo(name, Path(source), caps, text("family"),
                               text("parameter_size"), text("quantization_level"))

    def pull(
        self, name: str,
        on_progress: Callable[[str, int | None, int | None], None] | None = None,
    ) -> None:
        """Download a model into Ollama: library names ('llama3.2:1b') or any GGUF on
        HuggingFace ('hf.co/<user>/<repo>[:<quant>]'). ``on_progress(status, done, total)``."""
        name = validate_model_name(name)
        try:
            for item in self.stream("/api/pull", {"model": name, "stream": True}):
                err = item.get("error")
                if err is not None:
                    raise OllamaError(400, str(err)[:_MAX_ERROR_CHARS])
                status = item.get("status")
                if on_progress is not None and isinstance(status, str):
                    on_progress(status[:200], _as_int(item.get("completed")),
                                _as_int(item.get("total")))
        except httpx.HTTPError as exc:
            raise OllamaError(503, f"download interrupted: {exc}") from exc

    def greedy_answer(self, name: str, prompt: str, *, max_tokens: int = 64,
                      num_ctx: int = 1024) -> str:
        """Deterministic answer (temperature 0, fixed seed) for quality evaluation."""
        data = self.post_json("/api/chat", {
            "model": validate_model_name(name), "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 0, "top_k": 1, "num_predict": max_tokens,
                        "num_ctx": num_ctx},
        })
        msg = data.get("message")
        text = msg.get("content") if isinstance(msg, dict) else None
        return text[:20_000] if isinstance(text, str) else ""

    def release(self, name: str) -> None:
        """Unload a model from Ollama's memory (best effort)."""
        with contextlib.suppress(OllamaError, httpx.HTTPError):
            self.post_json("/api/generate", {"model": validate_model_name(name), "keep_alive": 0})

    # ---- model import (used by automatic quantization) --------------------------------------

    def model_settings(self, name: str) -> dict[str, object]:
        """Chat template, system prompt and runtime parameters of an installed model, so a
        derived model (e.g. a quantized copy) behaves the same way."""
        data = self.post_json("/api/show", {"model": validate_model_name(name)})
        out: dict[str, object] = {}
        template = data.get("template")
        if isinstance(template, str) and template:
            out["template"] = template[:200_000]
        system = data.get("system")
        if isinstance(system, str) and system:
            out["system"] = system[:200_000]
        params = data.get("parameters")
        if isinstance(params, str) and params.strip():
            out["parameters"] = _parse_parameters(params)
        return out

    def push_blob(self, path: Path) -> str:
        """Upload a file to Ollama's content-addressed blob store; returns 'sha256:<hex>'.
        The digest is computed while streaming (O(file) time, O(1) memory) and Ollama
        verifies it on receipt, so a corrupted upload is rejected."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = f"sha256:{h.hexdigest()}"
        head = self._http.head(f"/api/blobs/{digest}")
        if head.status_code == 200:
            return digest

        def body() -> Iterator[bytes]:
            with path.open("rb") as fh:
                yield from iter(lambda: fh.read(1 << 20), b"")

        resp = self._http.post(f"/api/blobs/{digest}", content=body())
        if resp.status_code >= 400:
            raise self._error(resp)
        return digest

    def create_model(
        self, name: str, *, gguf_digest: str, filename: str, settings: dict[str, object],
        adapters: dict[str, str] | None = None,
    ) -> None:
        """Create a model from an uploaded GGUF blob, optionally with LoRA adapter blobs
        (``{filename: digest}``) applied on top of it."""
        name = validate_model_name(name)
        digests = [gguf_digest, *(adapters or {}).values()]
        if not all(re.fullmatch(r"sha256:[0-9a-f]{64}", d) for d in digests):
            raise ValidationError("invalid blob digest")
        payload: dict[str, object] = {
            "model": name, "files": {filename: gguf_digest}, "stream": False, **settings,
        }
        if adapters:
            payload["adapters"] = dict(adapters)
        self.post_json("/api/create", payload)

    def delete_model(self, name: str) -> None:
        """Remove a model from Ollama (its blobs are freed when no other model uses them)."""
        resp = self._http.request("DELETE", "/api/delete",
                                  json={"model": validate_model_name(name)})
        if resp.status_code >= 400:
            raise self._error(resp)

    def running(self) -> list[RunningModel]:
        models = self.get_json("/api/ps").get("models")
        out: list[RunningModel] = []
        if isinstance(models, list):
            for m in models:
                if not isinstance(m, dict):
                    continue
                name, size, vram = m.get("name"), _as_int(m.get("size")), _as_int(
                    m.get("size_vram"))
                if isinstance(name, str) and size is not None:
                    out.append(RunningModel(name, size, vram or 0))
        return out


class OllamaBackend(InferenceBackend):
    name = "ollama"

    def __init__(
        self,
        client: OllamaClient | None = None,
        *,
        kv_type: KVCacheType = KVCacheType.F16,
        keep_alive: str = "10m",
    ) -> None:
        self.client = client or OllamaClient()
        if not isinstance(kv_type, KVCacheType):
            raise ValidationError("kv_type must be a KVCacheType")
        if not re.fullmatch(r"\d{1,5}[smh]?", keep_alive):
            raise ValidationError("keep_alive must look like '10m', '300s' or '0'")
        self._kv = kv_type
        self._keep_alive = keep_alive
        self._names: dict[str, str] = {}
        self._config: ExecutionConfig | None = None
        self._footprint: tuple[int, int] | None = None  # (ram, vram) of the loaded model
        self._version: str | None = None

    # ---- registry -------------------------------------------------------------------------

    def register(self, model_name: str, blob_path: Path) -> None:
        """Map a GGUF path (as used in ExecutionConfig) to its Ollama model name."""
        self._names[str(blob_path)] = validate_model_name(model_name)

    def model_name(self, config: ExecutionConfig) -> str:
        name = self._names.get(config.model_path)
        if name is None:
            raise ModelLoadError(f"no Ollama model registered for {config.model_path}")
        return name

    # ---- contract -------------------------------------------------------------------------

    @property
    def version(self) -> str:
        if self._version is None:
            try:
                self._version = f"ollama-{self.client.version()}"
            except (httpx.HTTPError, OllamaError):
                return "ollama-unknown"
        return self._version

    @property
    def loaded_config(self) -> ExecutionConfig | None:
        return self._config

    def supported_kv_types(self) -> tuple[KVCacheType, ...]:
        return (self._kv,)

    @staticmethod
    def _options(config: ExecutionConfig) -> dict[str, object]:
        return {
            "num_ctx": config.n_ctx,
            "num_batch": config.n_batch,
            "num_thread": config.n_threads,
            "num_gpu": config.n_gpu_layers,
        }

    def load(self, config: ExecutionConfig) -> None:
        name = self.model_name(config)
        previous = self._config
        if previous is not None and self._names.get(previous.model_path) != name:
            self.unload()  # free the old variant before loading another (memory safety)
        self._config = None
        try:
            self.client.post_json("/api/generate", {
                "model": name, "prompt": "", "stream": False,
                "keep_alive": self._keep_alive, "options": self._options(config),
            })
        except OllamaError as exc:
            kind = FailureKind.OUT_OF_MEMORY if _is_oom(exc.message) else FailureKind.LOAD_FAILED
            raise ModelLoadError(f"Ollama could not load {name}: {exc.message}", kind) from exc
        except httpx.HTTPError as exc:
            raise ModelLoadError(f"cannot reach Ollama at {self.client.base_url}: {exc}") from exc
        self._config = config
        self._refresh_footprint(name)

    def unload(self) -> None:
        config, self._config, self._footprint = self._config, None, None
        if config is None:
            return
        name = self._names.get(config.model_path)
        if name is None:
            return
        # Best effort; Ollama also evicts on its own keep_alive timer.
        with contextlib.suppress(OllamaError, httpx.HTTPError):
            self.client.post_json("/api/generate", {"model": name, "keep_alive": 0})

    def generate(
        self, prompt: str, *, max_tokens: int, temperature: float, seed: int | None,
        on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        return self._run("/api/generate", {"prompt": prompt}, max_tokens=max_tokens,
                         temperature=temperature, seed=seed, on_token=on_token)

    def chat(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float,
        seed: int | None, on_token: TokenCallback | None = None,
    ) -> GenerationResult:
        body: dict[str, object] = {
            "messages": [{"role": m.role.value, "content": m.content} for m in messages]
        }
        return self._run("/api/chat", body, max_tokens=max_tokens, temperature=temperature,
                         seed=seed, on_token=on_token)

    def _run(
        self, path: str, body: dict[str, object], *, max_tokens: int, temperature: float,
        seed: int | None, on_token: TokenCallback | None,
    ) -> GenerationResult:
        config = self._config
        if config is None:
            raise GenerationError("no configuration loaded")
        name = self.model_name(config)
        options: dict[str, object] = {
            **self._options(config), "num_predict": max_tokens, "temperature": temperature,
        }
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, object] = {
            "model": name, "stream": True, "keep_alive": self._keep_alive, "options": options,
            **body,
        }
        start = time.perf_counter()
        ttft: float | None = None
        pieces: list[str] = []
        final: dict[str, object] | None = None
        try:
            for chunk in self.client.stream(path, payload):
                err = chunk.get("error")
                if err is not None:
                    raise OllamaError(500, str(err)[:_MAX_ERROR_CHARS])
                piece = self._piece(chunk)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    pieces.append(piece)
                    if on_token is not None:
                        on_token(piece)
                if chunk.get("done") is True:
                    final = chunk
                    break
        except OllamaError as exc:
            kind = FailureKind.OUT_OF_MEMORY if _is_oom(exc.message) else (
                FailureKind.RUNTIME_ERROR)
            raise GenerationError(f"Ollama generation failed: {exc.message}", kind) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Ollama stream failed: {exc}") from exc
        total = time.perf_counter() - start
        stats = final or {}
        eval_count = _as_int(stats.get("eval_count"))
        eval_ns = _as_int(stats.get("eval_duration"))
        decode_tps = eval_count / (eval_ns / 1e9) if eval_count and eval_ns else None
        self._refresh_footprint(name)
        ram, vram = self._footprint or (None, None)
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=_as_int(stats.get("prompt_eval_count")) or 0,
            completion_tokens=eval_count if eval_count is not None else len(pieces),
            ttft_s=ttft if ttft is not None else total,
            total_s=total,
            peak_ram_bytes=ram,
            peak_vram_bytes=vram,
            decode_tps=decode_tps,
        )

    @staticmethod
    def _piece(chunk: dict[str, object]) -> str:
        raw = chunk.get("response")
        if raw is None:
            msg = chunk.get("message")
            raw = msg.get("content") if isinstance(msg, dict) else ""
        if not isinstance(raw, str):
            raise OllamaError(502, "non-text token in stream")
        return raw

    def _refresh_footprint(self, name: str) -> None:
        try:
            for m in self.client.running():
                if m.name == name:
                    self._footprint = (max(0, m.size - m.size_vram), m.size_vram)
                    return
        except (OllamaError, httpx.HTTPError):
            pass
        self._footprint = None

    @property
    def footprint(self) -> tuple[int, int] | None:
        """(RAM bytes, VRAM bytes) of the loaded model as reported by Ollama."""
        return self._footprint

    def measured_footprint(self) -> tuple[int, int] | None:
        return self._footprint

    def adjust_sample(self, sample: TelemetrySample) -> TelemetrySample:
        # The model lives in the Ollama server process, not ours: report its footprint as
        # the reclaimable "process" memory so live budgets account for it correctly.
        if self._footprint is None:
            return sample
        return replace(sample, process_rss=self._footprint[0])
