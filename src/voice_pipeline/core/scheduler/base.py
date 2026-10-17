"""调度器契约：serial 与 pipelined 同接口，A/B 实验只换实现。

产出统一为 TurnItem：要么是 JSON 事件帧，要么是下行音频分片。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.filler import FillerBank
from voice_pipeline.metrics.events import TurnMetrics
from voice_pipeline.providers.base import TtsProvider
from voice_pipeline.server.protocol import ServerEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioFragment:
    """下行音频分片（PCM16 24k，固定 200ms）。"""

    seq: int
    data: bytes


TurnItem = Union[ServerEvent, AudioFragment]  # noqa: UP007 - 运行时别名


async def play_filler(
    filler: FillerBank | None, chunker: Chunker, metrics: TurnMetrics
) -> AsyncIterator[AudioFragment]:
    """把一段预合成 filler 切成下行分片先播出去（首片即出声，不等 TTS）。

    计量纪律（见 `core/filler.py`）：这里只打 `first_sound`，**不碰 first_audio_out**——
    口径B 必须继续只算「真实回答的首片」。filler 没配/库为空时什么都不做（传 None 即关闭）。

    分片序号全部为**负数**（−1, −2, −3…）：filler 不在调度器的分片序号空间里（调度器从 1 起）。
    这一条不只是命名——`Session.run_turn` 用 `seq >= 0` 识别「真实回答的首片」来记
    `tx.first_audio` 那根 span；filler 的首片若用 0，就会被误认成真实首片（2026-10-17 修正：
    原先写的是「0 与负数」，首个 filler 分片恰好是 0，正好撞上那个条件）。
    """
    if filler is None:
        return
    clip = filler.next_clip()
    if clip is None:
        return
    metrics.filler_used = True
    for i, block in enumerate(chunker.feed_and_flush(clip.pcm)):
        if i == 0:
            metrics.mark("first_sound")
        metrics.audio_bytes += len(block)
        yield AudioFragment(seq=-(i + 1), data=block)


async def synth_to_fragments(
    tts: TtsProvider, sentence: str, chunker: Chunker
) -> AsyncIterator[bytes]:
    """按 TTS 能力分流合成，产出固定尺寸音频分片。

    - 流式 provider（`tts_streams = True`，如 DashScope CosyVoice）：边收边切，
      首 200ms 分片在整句合成完之前就能出——这是它存在的全部意义
      （本地 melo 在 2C x86 上 12 字要 11.9s，见 docs/ecs-baseline.md）。
    - 整句 provider（sherpa OfflineTts）：卸到线程池合成完整句再切片，原路径。

    两条路径对调用方的分片顺序与尺寸完全一致，指标打点由调用方按需处理。

    **单句失败不炸整轮**（2026-10-17 真人验收 turn 12 的教训）：provider 抛
    RuntimeError（如 DashScope 判文本 InvalidParameter）时跳过这一句、其余照播。
    一句念不出来，好过整轮静音 + "Task exception was never retrieved"。
    CancelledError 不是 Exception，天然穿透——打断语义不受影响。
    """
    if getattr(tts, "tts_streams", False):
        try:
            async for piece in tts.synth(sentence):
                for block in chunker.feed(piece):
                    yield block
        except RuntimeError:
            logger.warning("TTS 合成失败，跳过该句（其余照播）: %r", sentence[:60], exc_info=True)
            return
        for block in chunker.flush():
            yield block
    else:
        try:
            pcm = await asyncio.to_thread(tts.synth_blocking, sentence)
        except RuntimeError:
            logger.warning("TTS 合成失败，跳过该句（其余照播）: %r", sentence[:60], exc_info=True)
            return
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
