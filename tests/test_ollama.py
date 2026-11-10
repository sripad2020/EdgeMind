"""Ollama backend tests against an in-process fake Ollama server (httpx.MockTransport)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from air_sdk import AdaptiveRuntime, ValidationError
from air_sdk.analysis.profile import analyze_model
from air_sdk.analysis.quant import KVCacheType
from air_sdk.backends.base import ChatMessage, Role
from air_sdk.backends.ollama import OllamaBackend, OllamaClient, validate_model_name
from air_sdk.config import ExecutionConfig
from air_sdk.errors import FailureKind, ModelLoadError

from .conftest import small_machine, small_safety


class FakeOllama:
    def __init__(self, blob: Path) -> None:
        self.blob = blob
        self.loaded: str | None = None
        self.fail_load_with: str | None = None
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.blobs: dict[str, bytes] = {}
        self.created: dict[str, dict[str, object]] = {}
        self.paths: dict[str, Path] = {}
        self.pullable: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/api/blobs/"):
            digest = path.removeprefix("/api/blobs/")
            if request.method == "HEAD":
                return httpx.Response(200 if digest in self.blobs else 404)
            data = request.read()
            if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
                return httpx.Response(400, json={"error": "digest mismatch"})
            self.blobs[digest] = data
            return httpx.Response(201)
        body = json.loads(request.content) if request.content else {}
        self.requests.append((path, body))
        if path == "/api/pull":
            name = str(body["model"])
            if name not in self.pullable:
                return httpx.Response(200, content=b'{"error":"pull model manifest: file does '
                                                   b'not exist"}\n')
            self.created[name] = {}
            progress = [{"status": "pulling manifest"},
                        {"status": "downloading", "completed": 50, "total": 100},
                        {"status": "success"}]
            return httpx.Response(200,
                                  content="\n".join(json.dumps(x) for x in progress).encode())
        if path == "/api/create":
            files = body.get("files")
            assert isinstance(files, dict)
            assert all(d in self.blobs for d in files.values())
            name = str(body["model"])
            self.created[name] = body
            digest = next(iter(files.values()))
            dest = self.blob.parent / f"created-{hashlib.sha256(name.encode()).hexdigest()[:12]}"
            dest.write_bytes(self.blobs[digest])
            self.paths[name] = dest
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/chat" and body.get("stream") is False:  # greedy evaluation answers
            prompt = body["messages"][-1]["content"]
            words = f"the answer to {prompt} is forty two".split()
            if str(body["model"]).endswith("q4_0"):
                words = [*words[:-2], "something", "else"]  # Q4 changes some words
            return httpx.Response(200, json={"message": {"content": " ".join(words)},
                                             "done": True})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.13"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "tiny:latest"}, {"name": "embed:latest"}, {"name": "bad name!"},
                *({"name": n} for n in self.created),
            ]})
        if path == "/api/show":
            caps = ["embedding"] if body["model"] == "embed:latest" else ["completion"]
            source = self.paths.get(str(body["model"]), self.blob)
            return httpx.Response(200, json={
                "template": "{{ .Prompt }}",
                "parameters": 'stop "<|end|>"\ntemperature 0.2',
                "modelfile": f"# comment\nFROM {source}\nPARAMETER x 1\n",
                "capabilities": caps,
                "details": {"family": "llama", "parameter_size": "1M",
                            "quantization_level": "F16"},
            })
        if path == "/api/ps":
            models = [{"name": self.loaded, "size": 50_000_000, "size_vram": 0}] \
                if self.loaded else []
            return httpx.Response(200, json={"models": models})
        if path == "/api/generate" and body.get("keep_alive") == 0:
            self.loaded = None
            return httpx.Response(200, json={"done": True})
        if path == "/api/generate" and body.get("prompt") == "" and not body.get("stream"):
            if self.fail_load_with:
                return httpx.Response(500, json={"error": self.fail_load_with})
            self.loaded = str(body["model"])
            return httpx.Response(200, json={"done": True})
        if path in ("/api/chat", "/api/generate"):
            lines: list[str] = []
            for word in ["Hello", " there", "!"]:
                item = {"message": {"role": "assistant", "content": word}} \
                    if path == "/api/chat" else {"response": word}
                lines.append(json.dumps({**item, "done": False}))
            lines.append(json.dumps({
                "done": True, "eval_count": 3, "eval_duration": 30_000_000,
                "prompt_eval_count": 12, "prompt_eval_duration": 5_000_000,
                **({"message": {"role": "assistant", "content": ""}} if path == "/api/chat"
                   else {"response": ""}),
            }))
            return httpx.Response(200, content=("\n".join(lines) + "\n").encode())
        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def fake(model_dir: Path) -> FakeOllama:
    return FakeOllama(model_dir / "tiny-f16.gguf")


def _runtime(tmp_path: Path, fake: FakeOllama) -> AdaptiveRuntime:
    client = OllamaClient(transport=httpx.MockTransport(fake))
    hw, _ = small_machine(vram_mib=0)
    return AdaptiveRuntime(backend=OllamaBackend(client), data_dir=tmp_path / "db",
                           safety=small_safety(), hardware=hw, seed=3)


def test_model_name_validation() -> None:
    assert validate_model_name("qwen2.5:0.5b") == "qwen2.5:0.5b"
    for bad in ["../etc/passwd", "a b", "", "x" * 300, "qwen:tag:extra", "-rf"]:
        with pytest.raises(ValidationError):
            validate_model_name(bad)


def test_catalogue_filters_non_chat_and_invalid(tmp_path: Path, fake: FakeOllama) -> None:
    with _runtime(tmp_path, fake) as sdk:
        names = [m.name for m in sdk.list_ollama_models()]
        assert names == ["tiny:latest"]  # embedding model and invalid name excluded


def test_chat_streams_and_uses_ollama_measurements(tmp_path: Path, fake: FakeOllama) -> None:
    with _runtime(tmp_path, fake) as sdk:
        model = sdk.load_ollama_model("tiny:latest")
        plan = model.optimize(n_ctx=1024)
        assert plan.config.kv_type.value == "f16"  # backend advertises one KV type
        tokens: list[str] = []
        r = model.chat([{"role": "user", "content": "hi"}], max_tokens=8, on_token=tokens.append)
        assert tokens == ["Hello", " there", "!"]
        assert r.text == "Hello there!"
        assert r.tokens_per_second == pytest.approx(100.0)  # 3 tokens / 0.03 s
        assert r.prompt_tokens == 12
        assert r.peak_ram_bytes == 50_000_000
        chat_req = next(b for p, b in fake.requests if p == "/api/chat")
        opts = chat_req["options"]
        assert isinstance(opts, dict)
        assert opts["num_gpu"] == plan.config.n_gpu_layers
        assert opts["num_ctx"] == 1024
        assert opts["num_predict"] == 8


def test_chat_message_validation(tmp_path: Path, fake: FakeOllama) -> None:
    with _runtime(tmp_path, fake) as sdk:
        model = sdk.load_ollama_model("tiny:latest")
        model.optimize(n_ctx=1024)
        with pytest.raises(ValidationError):
            model.chat([{"role": "root", "content": "x"}])
        with pytest.raises(ValidationError):
            model.chat([{"role": "assistant", "content": "x"}])  # last must be user
        with pytest.raises(ValidationError):
            model.chat([])
        with pytest.raises(ValidationError):
            model.chat("hello")  # type: ignore[arg-type]
        ok = model.chat([ChatMessage(Role.SYSTEM, "be brief"), ChatMessage(Role.USER, "hi")])
        assert ok.text


def test_load_oom_is_classified(fake: FakeOllama, model_dir: Path) -> None:
    backend = OllamaBackend(OllamaClient(transport=httpx.MockTransport(fake)))
    path = model_dir / "tiny-f16.gguf"
    backend.register("tiny:latest", path)
    analyze_model(path)
    cfg = ExecutionConfig(str(path), "F16", KVCacheType.F16, 0, 2, 128, 1024, False)
    fake.fail_load_with = "model requires more system memory (9.1 GiB) than is available"
    with pytest.raises(ModelLoadError) as info:
        backend.load(cfg)
    assert info.value.kind is FailureKind.OUT_OF_MEMORY
    assert backend.loaded_config is None


def test_auto_quantize_creates_and_registers_variants(tmp_path: Path, model_dir: Path) -> None:
    fake = FakeOllama(_fresh_f16(tmp_path, model_dir))
    with _runtime(tmp_path, fake) as sdk:
        plan = sdk.quantization_plan("tiny:latest")
        assert plan["eligible"] is True
        targets = plan["targets"]
        assert isinstance(targets, list)
        assert [t["quant"] for t in targets] == ["q8_0", "q4_0"]
        results = sdk.auto_quantize("tiny:latest")
        assert [(r.model_name, r.created) for r in results] == [
            ("tiny:latest-air-q8_0", True), ("tiny:latest-air-q4_0", True)]
        req = fake.created["tiny:latest-air-q8_0"]
        assert req["template"] == "{{ .Prompt }}"  # chat template carried over
        assert req["parameters"] == {"stop": ["<|end|>"], "temperature": 0.2}
        assert not list((tmp_path / "db" / "quantized").glob("*"))  # temp files removed
        again = sdk.auto_quantize("tiny:latest")  # idempotent
        assert all(not r.created for r in again)


def test_quality_measurement_overrides_priors(tmp_path: Path, model_dir: Path) -> None:
    fake = FakeOllama(_fresh_f16(tmp_path, model_dir))
    with _runtime(tmp_path, fake) as sdk:
        assert sdk.evaluate_quality("tiny:latest")["note"]  # only one version yet
        sdk.auto_quantize("tiny:latest")
        report = sdk.evaluate_quality("tiny:latest")
        assert report["reference"] == "tiny:latest"
        results = report["results"]
        assert isinstance(results, list)
        by_name = {r["model"]: r for r in results}
        assert by_name["tiny:latest-air-q8_0"]["agreement"] == 1.0  # identical answers
        q4 = by_name["tiny:latest-air-q4_0"]
        assert 0.0 < q4["agreement"] < 1.0
        assert q4["measured_loss"] == pytest.approx((1 - q4["agreement"]) * 0.1, abs=1e-4)
        # The measurement replaces the Q4_0 prior (0.05) when the model is optimised.
        model = sdk.load_ollama_model("tiny:latest", discover=True)
        model.optimize(n_ctx=1024)
        cands = model._planner.space.precision  # type: ignore[union-attr]
        losses = {v.quant.name: v.quant.quality_loss for v in cands.variants}
        assert losses["Q4_0"] == pytest.approx(q4["measured_loss"], abs=1e-4)
        assert losses["Q8_0"] == 0.0


def _fresh_f16(tmp_path: Path, model_dir: Path) -> Path:
    """Private copy of the F16 model so files created next to it don't leak across tests."""
    d = tmp_path / "blobs"
    d.mkdir()
    dst = d / "tiny-f16.gguf"
    dst.write_bytes((model_dir / "tiny-f16.gguf").read_bytes())
    return dst


def test_auto_quantize_rejects_quantized_source(tmp_path: Path, model_dir: Path) -> None:
    fake = FakeOllama(model_dir / "tiny-q4_0.gguf")
    with _runtime(tmp_path, fake) as sdk:
        assert sdk.quantization_plan("tiny:latest")["eligible"] is False
        with pytest.raises(ValidationError, match="only F16"):
            sdk.auto_quantize("tiny:latest")


def test_rejects_bad_base_url() -> None:
    with pytest.raises(ValidationError):
        OllamaClient("file:///etc/passwd")
    with pytest.raises(ValidationError):
        OllamaClient("not a url")
