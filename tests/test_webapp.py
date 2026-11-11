"""Web app tests: security middleware and schemas (always), live chat (if Ollama is up)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from webapp.app import create_app

BASE = "http://127.0.0.1"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    # Context manager => app lifespan runs, so the service unloads its Ollama model on exit.
    with TestClient(create_app(data_dir=tmp_path / "data"), base_url=BASE) as c:
        yield c


def test_security_headers_on_index(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"


def test_dns_rebinding_host_rejected(tmp_path: Path) -> None:
    c = TestClient(create_app(data_dir=tmp_path / "d"), base_url="http://evil.example")
    assert c.get("/").status_code == 400


def test_post_requires_json(client: TestClient) -> None:
    r = client.post("/api/session", content=b"model=x", headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_cross_origin_post_rejected(client: TestClient) -> None:
    r = client.post("/api/session", json={"model": "x"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


@pytest.mark.parametrize("body", [
    {"model": "../../etc/passwd"},
    {"model": "qwen2.5:0.5b", "device": "tpu"},
    {"model": "qwen2.5:0.5b", "n_ctx": "4096"},  # strict: no str->int coercion
    {"model": "qwen2.5:0.5b", "rl_enabled": 1},  # strict: no int->bool coercion
    {"model": "qwen2.5:0.5b", "extra": True},  # extra fields forbidden
])
def test_session_schema_is_strict(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/api/session", json=body).status_code == 422


def test_pull_endpoint_validates_model_name(client: TestClient) -> None:
    assert client.post("/api/models/pull", json={"model": "../../etc"}).status_code == 422
    assert client.post("/api/models/pull", json={"model": "a b"}).status_code == 422
    assert client.post("/api/models/pull", json={"model": "x", "url": "y"}).status_code == 422


def test_quantize_endpoints_validate_model_name(client: TestClient) -> None:
    assert client.post("/api/quantize", json={"model": "../x"}).status_code == 422
    assert client.post("/api/quantize", json={"model": "a", "x": 1}).status_code == 422
    assert client.get("/api/quantize/plan", params={"model": "a b"}).status_code == 422


def test_chat_schema_is_strict(client: TestClient) -> None:
    bad = {"messages": [{"role": "root", "content": "x"}]}
    assert client.post("/api/chat", json=bad).status_code == 422
    unknown_skill = {"messages": [{"role": "user", "content": "x"}], "skill": "hacking"}
    assert client.post("/api/chat", json=unknown_skill).status_code == 422
    huge_ctx = {"messages": [{"role": "user", "content": "x"}], "skill": "sql",
                "context": "x" * 20_001}
    assert client.post("/api/chat", json=huge_ctx).status_code == 422
    too_long = {"messages": [{"role": "user", "content": "x" * 40_000}]}
    assert client.post("/api/chat", json=too_long).status_code == 422
    bad_id = {"messages": [{"role": "user", "content": "x"}], "conversation_id": "../x"}
    assert client.post("/api/chat", json=bad_id).status_code == 422


def test_conversation_memory_endpoints(client: TestClient) -> None:
    assert client.get("/api/conversations/short").status_code == 422
    r = client.get("/api/conversations/abcdef0123456789")
    assert r.status_code == 200
    assert r.json()["sections"] == []
    assert client.delete("/api/conversations/abcdef0123456789").json() == {"forgotten": True}


def _ollama_has(model: str) -> bool:
    try:
        tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(m.get("name") == model for m in tags.get("models", []))


LIVE_MODEL = "qwen2.5:0.5b"


@pytest.mark.live
@pytest.mark.skipif(not _ollama_has(LIVE_MODEL), reason="local Ollama with qwen2.5:0.5b needed")
def test_live_session_and_streaming_chat(client: TestClient) -> None:
    models = client.get("/api/models").json()["models"]
    assert LIVE_MODEL in [m["name"] for m in models]
    assert all("completion" in m["capabilities"] for m in models)

    s = client.post("/api/session", json={"model": LIVE_MODEL, "device": "cpu", "n_ctx": 2048})
    if s.status_code == 422 and "no configuration fits" in s.text:
        pytest.skip(f"machine currently lacks free RAM: {s.json()['error']}")
    assert s.status_code == 200, s.text
    assert s.json()["plan"]["config"]["n_gpu_layers"] == 0

    events: list[tuple[str, object]] = []
    with client.stream("POST", "/api/chat", json={
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8, "temperature": 0.0,
    }) as r:
        assert r.status_code == 200
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                name = raw.split("\n")[0].removeprefix("event: ")
                data = json.loads(raw.split("\n")[1].removeprefix("data: "))
                events.append((name, data))
    names = [e[0] for e in events]
    assert names[-1] == "done", events[-1]
    assert "token" in names
    done = events[-1][1]
    assert isinstance(done, dict)
    assert done["metrics"]["tokens_per_second"] > 0

    rep = client.get("/api/reports").json()
    assert rep["performance"]["requests"] == 1
    pol = client.get("/api/policy").json()
    assert len(pol["actions"]) == 11
    assert any(a["allowed"] for a in pol["actions"])
