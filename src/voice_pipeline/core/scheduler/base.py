"""调度器契约：serial 与 pipelined 同接口，A/B 实验只换实现。

产出统一为 TurnItem：要么是 JSON 事件帧，要么是下行音频分片。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.providers.base import TtsProvider
from voice_pipeline.server.protocol import ServerEvent


@dataclass(slots=True)
class AudioFragment:
    """下行音频分片（PCM16 24k，固定 200ms）。"""

    seq: int
    data: bytes


TurnItem = Union[ServerEvent, AudioFragment]  # noqa: UP007 - 运行时别名


async def synth_to_fragments(
    tts: TtsProvider, sentence: str, chunker: Chunker
) -> AsyncIterator[bytes]:
    """按 TTS 能力分流合成，产出固定尺寸音频分片。

    - 流式 provider（`tts_streams = True`，如 DashScope CosyVoice）：边收边切，
      首 200ms 分片在整句合成完之前就能出——这是它存在的全部意义
      （本地 melo 在 2C x86 上 12 字要 11.9s，见 docs/ecs-baseline.md）。
    - 整句 provider（sherpa OfflineTts）：卸到线程池合成完整句再切片，原路径。

    两条路径对调用方的分片顺序与尺寸完全一致，指标打点由调用方按需处理。
    """
    if getattr(tts, "tts_streams", False):
        async for piece in tts.synth(sentence):
            for block in chunker.feed(piece):
                yield block
        for block in chunker.flush():
            yield block
    else:
        pcm = await asyncio.to_thread(tts.synth_blocking, sentence)
        for block in chunker.feed_and_flush(pcm):
            yield block


@runtime_checkable
class Scheduler(Protocol):
    async def on_audio(self, chunk: bytes) -> None: ...

    def text(self) -> str:
        """当前 ASR partial（EOU 语义判停用）。"""
        ...

    def on_end_of_utterance(self) -> AsyncIterator[TurnItem]: ...

    def cancel(self) -> None:
        """barge-in：丢弃在飞的计算。没有后台工作的实现可以提供空实现。"""
        ...
