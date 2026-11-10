"""The simple API (air_sdk.load / LLM / Conversation / doctor / list_models) and the CLI,
against the in-process fake Ollama server."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import air_sdk
from air_sdk import easy, runtime
from air_sdk.analysis.profile import analyze_model
from air_sdk.backends.ollama import OllamaClient
from air_sdk.cli import build_parser, main
from air_sdk.easy import _classify, _Source
from air_sdk.testing import SyntheticModel, write_gguf

from .test_ollama import FakeOllama, _fresh_f16


@pytest.fixture
def fake(tmp_path: Path, model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    server = FakeOllama(_fresh_f16(tmp_path, model_dir))
    transport = httpx.MockTransport(server)

    def client(url: str = "http://127.0.0.1:11434", **_kw: object) -> OllamaClient:
        return OllamaClient(url, transport=transport)

    monkeypatch.setattr(runtime, "OllamaClient", client)
    monkeypatch.setattr(easy, "OllamaClient", client)
    return server


def _load(tmp_path: Path, model: str | Path = "tiny:latest", **kw: object) -> air_sdk.LLM:
    return air_sdk.load(model, data_dir=tmp_path / "data", reserve_ram_gb=0.25, **kw)  # type: ignore[arg-type]


# ---- model references ---------------------------------------------------------------------


@pytest.mark.parametrize(("ref", "kind", "normalised"), [
    ("qwen2.5:0.5b", _Source.OLLAMA, "qwen2.5:0.5b"),
    ("hf.co/bartowski/X-GGUF:Q4_K_M", _Source.HUGGINGFACE, "hf.co/bartowski/X-GGUF:Q4_K_M"),
    ("https://huggingface.co/bartowski/X-GGUF", _Source.HUGGINGFACE, "hf.co/bartowski/X-GGUF"),
    ("C:/models/my model.Q4_K_M.gguf", _Source.LOCAL_FILE, "C:/models/my model.Q4_K_M.gguf"),
])
def test_classify_model_reference(ref: str, kind: _Source, normalised: str) -> None:
    assert _classify(ref) == (kind, normalised)


def test_classify_pathlike(tmp_path: Path) -> None:
    assert _classify(tmp_path / "m.gguf")[0] is _Source.LOCAL_FILE


# ---- loading & using ----------------------------------------------------------------------


def test_load_ask_stream_and_converse(tmp_path: Path, fake: FakeOllama) -> None:
    with _load(tmp_path, "tiny") as llm:  # no tag -> resolves to tiny:latest
        assert llm.name == "tiny:latest"
        assert llm.ask("hi") == "Hello there!"
        assert list(llm.stream("hi")) == ["Hello", " there", "!"]
        assert llm.last is not None
        assert llm.last.tokens_per_second > 0
        assert str(llm.last) == "Hello there!"
        chat = llm.conversation(system="be brief")
        chat.say("first")
        chat.say("second")
        assert [m["role"] for m in chat.history] == ["user", "assistant"] * 2
        last_chat = [b for p, b in fake.requests if p == "/api/chat"][-1]
        sent = last_chat["messages"]
        assert isinstance(sent, list)
        roles = [m["role"] for m in sent]
        assert roles == ["system", "user", "assistant", "user"]  # history was sent
        text = llm.summary()
        assert "Model" in text
        assert "Now using" in text
        assert set(llm.report()) == {"model", "profile", "plan", "performance", "learning"}


def test_missing_model_gives_fix_hint(tmp_path: Path, fake: FakeOllama) -> None:
    with pytest.raises(air_sdk.ModelNotFoundError) as info:
        _load(tmp_path, "tinyy:latest")
    msg = str(info.value)
    assert "ollama pull tinyy:latest" in msg
    assert "Did you mean: tiny:latest" in msg


def test_pull_downloads_missing_model(tmp_path: Path, fake: FakeOllama) -> None:
    fake.pullable.add("hf.co/acme/tiny-GGUF:Q8_0")
    seen: list[str] = []
    with _load(tmp_path, "https://huggingface.co/acme/tiny-GGUF:Q8_0", pull=True,
               on_progress=lambda s, _d, _t: seen.append(s)) as llm:
        assert llm.name == "hf.co/acme/tiny-GGUF:Q8_0"
        assert llm.ask("hi") == "Hello there!"
    assert seen == ["pulling manifest", "downloading", "success"]


def test_pull_of_unknown_model_explains(tmp_path: Path, fake: FakeOllama) -> None:
    with pytest.raises(air_sdk.ModelNotFoundError, match="Could not download"):
        _load(tmp_path, "nope:1b", pull=True)


def test_embedding_model_is_rejected_clearly(tmp_path: Path, fake: FakeOllama) -> None:
    with pytest.raises(air_sdk.SetupError, match="cannot chat"):
        _load(tmp_path, "embed:latest")


def test_local_gguf_is_imported(tmp_path: Path, fake: FakeOllama) -> None:
    src = write_gguf(tmp_path / "My Model Q8.gguf", SyntheticModel(file_type=7))
    with _load(tmp_path, src, speed_test=False) as llm:
        assert llm.name == "my-model-q8:air-local"
    created = fake.created["my-model-q8:air-local"]
    assert list(created["files"]) == ["My_Model_Q8.gguf"]  # type: ignore[call-overload]


def test_reserved_ollama_tag_rejected(tmp_path: Path, fake: FakeOllama) -> None:
    src = write_gguf(tmp_path / "m.gguf", SyntheticModel(file_type=7))
    with air_sdk.AdaptiveRuntime(backend="ollama", data_dir=tmp_path / "d") as rt, \
            pytest.raises(air_sdk.ValidationError, match="cannot be used"):
        rt.import_gguf(src, name="mine:local")


def test_ollama_down_explains_how_to_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    def client(url: str = "http://127.0.0.1:11434", **_kw: object) -> OllamaClient:
        return OllamaClient(url, transport=httpx.MockTransport(refuse))

    monkeypatch.setattr(runtime, "OllamaClient", client)
    monkeypatch.setattr(easy, "OllamaClient", client)
    with pytest.raises(air_sdk.SetupError) as info:
        _load(tmp_path)
    assert "ollama.com" in str(info.value)
    report = air_sdk.doctor()
    assert not report.ok
    assert "[FAIL] Ollama" in str(report)


def test_bad_arguments_are_rejected(tmp_path: Path, fake: FakeOllama) -> None:
    with pytest.raises(air_sdk.ValidationError):
        _load(tmp_path, goal="fastest")
    with pytest.raises(air_sdk.ValidationError):
        _load(tmp_path, device="tpu")
    with pytest.raises(air_sdk.ValidationError):
        _load(tmp_path, max_context="4096")


# ---- discovery & CLI --------------------------------------------------------------------------


def test_doctor_and_list_models(fake: FakeOllama) -> None:
    report = air_sdk.doctor()
    assert report.ok
    assert "[ OK ] Ollama" in str(report)
    models = {m["name"]: m for m in air_sdk.list_models()}
    assert models["tiny:latest"]["can_chat"] is True
    assert models["embed:latest"]["can_chat"] is False


def test_cli_parses_every_command() -> None:
    p = build_parser()
    for argv in (["doctor"], ["models"], ["ask", "m", "hi", "--goal", "speed"],
                 ["chat", "m", "--system", "x"], ["pull", "m"], ["import", "f.gguf"],
                 ["faster", "m"], ["quality", "m"], ["web", "--port", "9000"]):
        assert p.parse_args(argv).command == argv[0]


def test_cli_models_and_ask(fake: FakeOllama, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert main(["models"]) == 0
    assert "tiny:latest" in capsys.readouterr().out
    assert main(["ask", "tiny:latest", "hi", "--reserve-ram-gb", "0.25"]) == 0
    assert "Hello there!" in capsys.readouterr().out
    assert main(["ask", "missing:1b", "hi"]) == 2  # friendly error, non-zero exit
    assert "How to fix" in capsys.readouterr().err


# ---- robustness across architectures ---------------------------------------------------------


def test_attention_free_and_partial_metadata(tmp_path: Path) -> None:
    a = "llama"
    path = write_gguf(tmp_path / "ssm.gguf", SyntheticModel(), omit_keys=(
        f"{a}.attention.head_count", f"{a}.attention.head_count_kv", f"{a}.block_count",
        f"{a}.embedding_length"))
    p = analyze_model(path)
    assert p.attention_free
    assert p.n_layers == SyntheticModel().n_layers  # inferred from blk.N tensors
    assert p.n_embd == SyntheticModel().n_embd  # inferred from token_embd
    from air_sdk.analysis.quant import KVCacheType  # noqa: PLC0415

    assert p.kv_bytes_per_token(KVCacheType.F16) == 0
