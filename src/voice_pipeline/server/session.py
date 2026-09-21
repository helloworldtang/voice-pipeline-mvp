"""会话编排：把上行音频帧喂给调度器，把 ToneItem 流吐给传输层。

Session 不认识 WebSocket，也不认识具体 provider —— 这样同一份逻辑既能跑 WS，也能跑
scripted 回放（CI 确定性回归）与 A/B 实验。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.scheduler.base import AudioFragment, TurnItem
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts


class Session:
    """一个 WS 连接一个 Session，trace_id 贯穿日志与账单。"""

    def __init__(
        self,
        mode: str = "serial",
        *,
        asr: AsrProvider | None = None,
        llm: LlmProvider | None = None,
        tts: TtsProvider | None = None,
        clock: SessionClock | None = None,
    ) -> None:
        self.session_id = uuid4().hex[:12]
        self.trace_id = uuid4().hex[:16]
        self.mode = mode
        self.frames_echoed = 0
        self._clock = clock or SessionClock()
        self.metrics = TurnMetrics(clock=self._clock)

        if mode == "echo":
            self._scheduler: SerialScheduler | None = None
        else:
            self._scheduler = SerialScheduler(
                asr=asr or SimAsr(),
                llm=llm or SimLlm(),
                tts=tts or SimTts(),
                metrics=self.metrics,
                splitter=SentenceSplitter(),
                chunker=Chunker(),
            )

    async def on_audio(self, chunk: bytes) -> list[AudioFragment]:
        """上行音频帧。echo 模式立即回发；链路模式攒给 ASR。"""
        if self._scheduler is None:
            self.frames_echoed += 1
            return [AudioFragment(seq=self.frames_echoed, data=chunk)]
        await self._scheduler.on_audio(chunk)
        return []

    async def finish(self) -> AsyncIterator[TurnItem]:
        """收到 session.end：跑完一轮，产出事件与音频分片。"""
        if self._scheduler is None:
            return
        async for item in self._scheduler.on_end_of_utterance():
            yield item
