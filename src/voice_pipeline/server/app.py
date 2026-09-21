"""FastAPI 会话服务：上行音频帧 + JSON 控制帧，下行音频分片 + JSON 事件帧。

两种模式（session.begin.mode）：
- `echo`：上行音频原样回发，用于量 WS 链路本身的序列化开销（M0 回归）
- `serial`：真实链路 ASR → LLM → 分句 → TTS → 200ms 分片（M1 起）
"""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice_pipeline.config import Settings
from voice_pipeline.core.scheduler.base import AudioFragment

from .protocol import (
    AudioChunk,
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
from .session import Session


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
        session = Session()
        await websocket.send_text(
            dump_event(SessionReady(session_id=session.session_id, trace_id=session.trace_id))
        )

        async def drain_turn() -> None:
            async for item in session.finish():
                if isinstance(item, AudioFragment):
                    await websocket.send_text(
                        dump_event(AudioChunk(seq=item.seq, bytes_len=len(item.data)))
                    )
                    await websocket.send_bytes(item.data)
                else:
                    await websocket.send_text(dump_event(item))
            await websocket.send_text(
                dump_event(
                    SessionClosed(
                        session_id=session.session_id,
                        frames_echoed=session.frames_echoed,
                    )
                )
            )

        try:
            while True:
                message: MutableMapping[str, Any] = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                audio = message.get("bytes")
                if isinstance(audio, bytes):
                    for fragment in await session.on_audio(audio):
                        await websocket.send_bytes(fragment.data)
                    # real 模式：EOU 判停器触发即自动收尾，不必等客户端 session.end
                    if session.pending_end():
                        await drain_turn()
                        break
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

                if isinstance(frame, SessionBegin):
                    session = Session(mode=frame.mode)
                    await websocket.send_text(
                        dump_event(
                            SessionReady(session_id=session.session_id, trace_id=session.trace_id)
                        )
                    )
                    continue

                if isinstance(frame, SessionEnd):
                    await drain_turn()
                    break

                if isinstance(frame, Interrupt):
                    # M1 尚无进行中的 turn 可打断：如实回错，不留静默失败
                    await websocket.send_text(
                        dump_event(ErrorEvent(code="no_active_turn", message="no turn in progress"))
                    )
        except WebSocketDisconnect:
            return

    return app


app = create_app()
