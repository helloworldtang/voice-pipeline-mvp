"""WS 协议 v1 的 wire 契约：JSON 控制帧 / 事件帧 + 二进制音频帧。

约定（技术方案 §3）：
- 上行二进制帧 = PCM16 mono 16kHz 音频；上行文本帧 = JSON 控制帧
- 下行二进制帧 = PCM16 mono 24kHz 音频分片；下行文本帧 = JSON 事件帧
- 控制帧走优先通道，永不被音频积压阻塞（见 queues.py）
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class ProtocolError(Exception):
    """帧解析失败。code 会原样下发为 error 事件。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SessionBegin(BaseModel):
    type: Literal["session.begin"] = "session.begin"
    sample_rate: int = 16000
    client: str = "unknown"
    mode: Literal["echo", "serial", "real"] = "serial"


class Interrupt(BaseModel):
    type: Literal["interrupt"] = "interrupt"
    reason: str | None = None


class SessionEnd(BaseModel):
    type: Literal["session.end"] = "session.end"


class AsrPartial(BaseModel):
    type: Literal["asr.partial"] = "asr.partial"
    text: str


class AsrFinal(BaseModel):
    type: Literal["asr.final"] = "asr.final"
    text: str


class TurnStart(BaseModel):
    type: Literal["turn.start"] = "turn.start"


class TtsSentence(BaseModel):
    type: Literal["tts.sentence"] = "tts.sentence"
    seq: int
    text: str


class AudioChunk(BaseModel):
    type: Literal["audio.chunk"] = "audio.chunk"
    seq: int
    bytes_len: int


class TurnEnd(BaseModel):
    type: Literal["turn.end"] = "turn.end"
    metrics: dict[str, float]


class SessionReady(BaseModel):
    type: Literal["session.ready"] = "session.ready"
    session_id: str
    trace_id: str


class SessionClosed(BaseModel):
    type: Literal["session.closed"] = "session.closed"
    session_id: str
    frames_echoed: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


ClientFrame = Annotated[SessionBegin | Interrupt | SessionEnd, Field(discriminator="type")]
ServerEvent = Annotated[
    SessionReady
    | SessionClosed
    | ErrorEvent
    | AsrPartial
    | AsrFinal
    | TurnStart
    | TtsSentence
    | AudioChunk
    | TurnEnd,
    Field(discriminator="type"),
]

_CLIENT_FRAME_ADAPTER: TypeAdapter[ClientFrame] = TypeAdapter(ClientFrame)


def parse_client_frame(raw: str) -> SessionBegin | Interrupt | SessionEnd:
    """解析上行 JSON 控制帧。失败抛 ProtocolError(code="bad_frame")。"""
    try:
        return _CLIENT_FRAME_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise ProtocolError(code="bad_frame", message=str(exc)) from exc


def dump_event(event: BaseModel) -> str:
    """序列化下行事件帧。"""
    return event.model_dump_json()
