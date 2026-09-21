"""FastAPI 会话服务（M0：WS echo 骨架 + /healthz）。

echo 语义：上行二进制音频帧原样回发，用于验证「音频进/音频出」的 WS 链路与序列化开销。
真实链路（ASR→LLM→TTS）在 M1 接入。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice_pipeline.config import Settings

from .protocol import (
    ErrorEvent,
    Interrupt,
    ProtocolError,
    SessionBegin,
    SessionClosed,
    SessionEnd,
    SessionReady,
    dump_event,
    parse_client_frame,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="voice-pipeline", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id = uuid4().hex[:12]
        trace_id = uuid4().hex[:16]
        await websocket.send_text(
            dump_event(SessionReady(session_id=session_id, trace_id=trace_id))
        )

        frames_echoed = 0
        try:
            while True:
                message: MutableMapping[str, Any] = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                audio = message.get("bytes")
                if isinstance(audio, bytes):
                    frames_echoed += 1
                    await websocket.send_bytes(audio)
                    continue

                text = message.get("text")
                if not isinstance(text, str):
                    continue

                try:
                    frame = parse_client_frame(text)
                except ProtocolError as exc:
                    await websocket.send_text(
                        dump_event(ErrorEvent(code=exc.code, message=exc.message))
                    )
                    continue

                if isinstance(frame, SessionEnd):
                    await websocket.send_text(
                        dump_event(
                            SessionClosed(session_id=session_id, frames_echoed=frames_echoed)
                        )
                    )
                    break
                if isinstance(frame, Interrupt):
                    # M0 没有 turn 可打断：如实回错，不留静默失败
                    await websocket.send_text(
                        dump_event(ErrorEvent(code="no_active_turn", message="no turn in progress"))
                    )
                elif isinstance(frame, SessionBegin):
                    await websocket.send_text(
                        dump_event(SessionReady(session_id=session_id, trace_id=trace_id))
                    )
        except WebSocketDisconnect:
            return

    return app


app = create_app()
