"""调度器契约：serial 与 pipelined 同接口，A/B 实验只换实现。

产出统一为 TurnItem：要么是 JSON 事件帧，要么是下行音频分片。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

from voice_pipeline.server.protocol import ServerEvent


@dataclass(slots=True)
class AudioFragment:
    """下行音频分片（PCM16 24k，固定 200ms）。"""

    seq: int
    data: bytes


TurnItem = Union[ServerEvent, AudioFragment]  # noqa: UP007 - 运行时别名


@runtime_checkable
class Scheduler(Protocol):
    async def on_audio(self, chunk: bytes) -> None: ...

    def on_end_of_utterance(self) -> AsyncIterator[TurnItem]: ...
