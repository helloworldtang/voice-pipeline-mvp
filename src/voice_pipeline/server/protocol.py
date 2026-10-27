"""WS 协议 v1 的 wire 契约：JSON 控制帧 / 事件帧 + 二进制音频帧。

约定（技术方案 §3）：
- 上行二进制帧 = PCM16 mono 16kHz 音频；上行文本帧 = JSON 控制帧
- 下行二进制帧 = PCM16 mono 24kHz 音频分片；下行文本帧 = JSON 事件帧
- 控制帧走优先通道，永不被音频积压阻塞（见 queues.py）
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from voice_pipeline.core.eou import GatingMode

# wire 上可点名的模式。pipelined 必须在这里——它是唯一能逼近 300ms 的调度器，
# 选不到就等于「实验组只能脚本跑、WS 服务里没有」，与 §4.5「A/B 打在真实服务链路上」相违。
MODE = Literal[
    "echo",
    "serial",
    "pipelined",
    "real",
    "real-serial",
    "real-pipelined",
]


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
    mode: MODE = "serial"
    # 判停档位。默认 neural（D2 实测后从 conservative 切过来，2026-09-22）：
    #   官方测试集受控实验（中文 n=335）：静音 100ms + 神经判停 → 句末检出 70%、
    #   句中抢话 4.6%；conservative 要等 500ms 才判停，而句中停顿一旦到 500ms 就 100% 抢。
    # 缺权重时 build_eou 会退回 conservative 并告警，那时账单里的 eou_mode 是 conservative。
    # 默认档第四次定档（2026-10-04，真人验收 D5）：neural 的 100ms 确认窗在真人
    # 半秒停顿下抢话 3/3（"我想"/"帮我查一下"/"我等会儿"全部在停顿处抢答），
    # 退回 conservative（判停 100→500ms，换 0% 抢话）。neural 仍是可选项：
    # 说话不停顿的场景它快 400ms。
    gating: GatingMode = GatingMode.CONSERVATIVE
    # pipelined 专用：EOU 后最多再等多少毫秒让投机落地（0 = 不等，没好就走串行）。
    # 本机实测 grace=0 时 6/6 全未命中，grace=300 才吃到那唯一一次 179ms。
    grace_ms: float = 0.0
    # WS 鉴权（可选）：服务端设了 VOICE_WS_TOKEN 才校验，compare_digest 防时序侧信道。
    # 注意 pydantic 对多余字段是忽略——字段名拼错会被静默吞掉，wire 契约测试钉住它。
    token: str | None = None


class Interrupt(BaseModel):
    type: Literal["interrupt"] = "interrupt"
    reason: str | None = None


class PlaybackReport(BaseModel):
    """客户端回传播放侧的两段测量（毫秒）。**全部是时长，不是时刻**。

    为什么必须是时长：客户端与服务端的时钟没有同步（也不必同步），拿客户端
    的绝对时刻去和服务端打点相减必然漂；而一段纯本地的时长不受时钟影响。
    服务端把它挂在「首片落网线」那一刻之后——**位置是推算的，数值是实测的**。

    为什么需要它：服务端的追踪止于落网线。之后的网络、抖动缓冲、播放排程、
    声卡那一段全程是盲区，而用户感知到的延迟恰恰包含它。`recv_to_play_ms`
    就是这一段的实测上界（含 `AudioContext.outputLatency`），
    `tail_ms` 是打断停播时播放队列里被丢弃的排程音频（判据 3 的尾音量化）。

    上报失败/浏览器不支持时客户端直接不发——服务端那一格保持 -1（缺失），不填 0。
    """

    type: Literal["playback.report"] = "playback.report"
    recv_to_play_ms: float = Field(ge=0)
    # 打断时才会有：停播那一刻队列里尚未出声的音频时长
    tail_ms: float | None = Field(default=None, ge=0)


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


class AudioStop(BaseModel):
    """播放端应立即丢弃已排队、尚未出声的音频分片。

    barge-in 的 L2：服务端 cancel 了在飞计算，但已经 send 出去的音频分片追不回来，
    只能由客户端在收到本事件后停止播放队列里剩下的块。没有它就「AI 嘴上停了、
    耳朵里还在自己回声」。
    """

    type: Literal["audio.stop"] = "audio.stop"


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


ClientFrame = Annotated[
    SessionBegin | Interrupt | PlaybackReport | SessionEnd, Field(discriminator="type")
]
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


def parse_client_frame(raw: str) -> SessionBegin | Interrupt | PlaybackReport | SessionEnd:
    """解析上行 JSON 控制帧。失败抛 ProtocolError(code="bad_frame")。"""
    try:
        return _CLIENT_FRAME_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        raise ProtocolError(code="bad_frame", message=str(exc)) from exc


def dump_event(event: BaseModel) -> str:
    """序列化下行事件帧。"""
    return event.model_dump_json()
