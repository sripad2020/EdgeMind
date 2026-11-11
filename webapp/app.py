
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Query, Request, Response
from fastapi import Path as FastApiPath
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from air_sdk import AirError, InsufficientResourcesError, ValidationError

from .service import BusyError, ChatService, NoSessionError, SessionSettings

STATIC_DIR = Path(__file__).parent / "static"
_MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._\-/]{0,150}(:[A-Za-z0-9._\-]{1,80})?$"
_CONVERSATION_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


# ---- request schemas -------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SessionIn(_Strict):
    model: Annotated[str, Field(max_length=232, pattern=_MODEL_PATTERN)]
    device: Literal["auto", "cpu", "gpu"] = "auto"
    objective: Literal["balanced", "throughput", "latency", "memory"] = "balanced"
    rl_enabled: bool = True
    auto_adapt: bool = True
    n_ctx: Annotated[int, Field(ge=256, le=32768)] = 4096
    adapt_every: Annotated[int, Field(ge=1, le=100)] = 3
    probe: bool = True
    auto_context: bool = True


class MessageIn(_Strict):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(max_length=32_000)]


class ChatIn(_Strict):
    messages: Annotated[list[MessageIn], Field(min_length=1, max_length=200)]
    max_tokens: Annotated[int, Field(ge=1, le=4096)] = 512
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] | None = None
    skill: Literal["auto", "general", "sql", "python", "json", "math", "summarise"] = "auto"
    context: Annotated[str, Field(max_length=20_000)] | None = None
    # Client-chosen id: the server keeps this conversation's sectioned memory.
    conversation_id: Annotated[str, Field(pattern=_CONVERSATION_PATTERN)] | None = None


class QuantizeIn(_Strict):
    model: Annotated[str, Field(max_length=232, pattern=_MODEL_PATTERN)]


class LearningIn(_Strict):
    rl_enabled: bool
    auto_adapt: bool


# ---- app factory ---------------------------------------------------------------------------


def create_app(
    *, data_dir: Path | None = None, ollama_url: str = "http://127.0.0.1:11434",
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
) -> FastAPI:
    service = ChatService(data_dir or Path(".air_data"), ollama_url)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        service.close()

    app = FastAPI(title="AIR-SDK Chat", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    _install_security(app, allowed_hosts)
    _install_error_handler(app)
    _install_pages(app)
    _install_api(app, service)
    return app


def _install_security(app: FastAPI, allowed_hosts: tuple[str, ...]) -> None:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.middleware("http")
    async def security(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "POST":
            ctype = request.headers.get("content-type", "")
            if not ctype.startswith("application/json"):
                return JSONResponse({"error": "content-type must be application/json"}, 415)
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            if origin is not None and origin not in (f"http://{host}", f"https://{host}"):
                return JSONResponse({"error": "cross-origin request rejected"}, 403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


def _install_error_handler(app: FastAPI) -> None:
    @app.exception_handler(AirError)
    async def air_error(_req: Request, exc: AirError) -> JSONResponse:
        status = 400
        if isinstance(exc, (BusyError, NoSessionError)):
            status = 409
        elif isinstance(exc, InsufficientResourcesError):
            status = 422
        elif not isinstance(exc, ValidationError):
            status = 502  # backend / Ollama failures
        return JSONResponse({"error": str(exc)}, status_code=status)


def _install_pages(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _install_api(app: FastAPI, service: ChatService) -> None:
    @app.get("/api/hardware")
    async def hardware() -> dict[str, object]:
        return await asyncio.to_thread(service.hardware)

    @app.get("/api/models")
    async def models(refresh: bool = False) -> dict[str, object]:
        infos = await asyncio.to_thread(service.models, refresh=refresh)
        return {"models": [
            {"name": m.name, "family": m.family, "parameters": m.parameter_size,
             "quantization": m.quantization, "capabilities": list(m.capabilities)}
            for m in infos
        ]}

    @app.post("/api/session")
    async def start_session(body: SessionIn) -> dict[str, object]:
        settings = SessionSettings(**body.model_dump())
        return await asyncio.to_thread(service.start_session, settings)

    @app.post("/api/learning")
    async def learning(body: LearningIn) -> dict[str, object]:
        await asyncio.to_thread(service.update_learning, rl_enabled=body.rl_enabled,
                                auto_adapt=body.auto_adapt)
        return {"ok": True}

    @app.get("/api/reports")
    async def reports() -> dict[str, object]:
        return await asyncio.to_thread(service.reports)

    @app.get("/api/policy")
    async def policy() -> dict[str, object]:
        return await asyncio.to_thread(service.policy)

    @app.get("/api/quantize/plan")
    async def quantize_plan(
        model: Annotated[str, Query(max_length=232, pattern=_MODEL_PATTERN)],
    ) -> dict[str, object]:
        return await asyncio.to_thread(service.quantization_plan, model)

    @app.post("/api/quantize")
    async def quantize(body: QuantizeIn) -> dict[str, object]:
        return await asyncio.to_thread(service.quantize, body.model)

    @app.post("/api/models/pull")
    async def pull(body: QuantizeIn) -> StreamingResponse:
        """Download a model; Server-Sent Events: ``progress`` {status, done, total},
        ``done`` {model}, ``error`` {message}."""
        return _event_stream(lambda emit: service.pull(body.model, lambda s, d, t: emit(
            "progress", {"status": s, "done": d, "total": t})))

    @app.post("/api/quality")
    async def quality(body: QuantizeIn) -> dict[str, object]:
        return await asyncio.to_thread(service.evaluate_quality, body.model)

    @app.get("/api/skills")
    async def skills() -> dict[str, object]:
        return {"skills": await asyncio.to_thread(service.skills)}

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return {"busy": service.is_busy()}

    @app.post("/api/chat")
    async def chat(body: ChatIn) -> StreamingResponse:
        """Server-Sent Events: ``token`` (text piece), ``reset`` (rollback: discard partial
        answer), ``escalate`` (check failed, retrying), ``done`` (final text, metrics, skill
        check and context: section, detected task, budget), ``error``."""
        return _event_stream(lambda emit: service.chat(
            [m.model_dump() for m in body.messages],
            max_tokens=body.max_tokens, temperature=body.temperature,
            skill=body.skill, context=body.context, conversation_id=body.conversation_id,
            on_token=lambda t: emit("token", t), on_reset=lambda: emit("reset", None),
            on_escalate=lambda a, b, why: emit("escalate", {"from": a, "to": b, "why": why}),
        ))

    @app.get("/api/conversations/{conversation_id}")
    async def conversation_memory(
        conversation_id: Annotated[str, FastApiPath(pattern=_CONVERSATION_PATTERN)],
    ) -> dict[str, object]:
        """What the server remembers for a conversation: preferences and sections."""
        return service.conversation_memory(conversation_id)

    @app.delete("/api/conversations/{conversation_id}")
    async def forget_conversation(
        conversation_id: Annotated[str, FastApiPath(pattern=_CONVERSATION_PATTERN)],
    ) -> dict[str, object]:
        return {"forgotten": service.forget_conversation(conversation_id)}


Emit = Callable[[str, object], None]


def _event_stream(job: Callable[[Emit], dict[str, object]]) -> StreamingResponse:
    """Run a blocking ``job`` in a worker thread and stream its events as Server-Sent
    Events. The job reports progress through ``emit``; its return value is sent as ``done``,
    and SDK errors as ``error`` (internal details are never leaked)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    def emit(event: str, data: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def work() -> None:
        try:
            emit("done", job(emit))
        except AirError as exc:
            emit("error", {"message": str(exc), "busy": isinstance(exc, BusyError)})
        except Exception:  # noqa: BLE001 - never leak internals to the client
            emit("error", {"message": "internal error", "busy": False})

    async def stream() -> AsyncIterator[str]:
        task = loop.run_in_executor(None, work)
        while True:
            event, data = await queue.get()
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
            if event in ("done", "error"):
                break
        await task

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store"})


app = create_app(
    data_dir=Path(os.environ.get("AIR_DATA_DIR", ".air_data")),
    ollama_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
)
