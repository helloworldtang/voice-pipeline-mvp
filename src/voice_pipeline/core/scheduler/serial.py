"""serial 调度：忠实复现质疑方的时序模型——EOU 之后才启动 LLM。

它是 A/B 实验的对照组：实测若落在 1.2~1.9s 区间，就证明了「在串行前提下质疑方的算术完全成立」，
分歧不在数字而在时序模型（见背景与基线 §二）。

注：即便是对照组也要把 TTS 卸到线程池（`asyncio.to_thread`）——否则一次合成就会堵死
整个事件循环，对照组测出来就不再是「慢」而是「坏」，A/B 失去意义。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import TurnMetrics
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.server.protocol import AsrFinal, TtsSentence, TurnEnd, TurnStart

from .base import AudioFragment, TurnItem, synth_to_fragments


class SerialScheduler:
    def __init__(
        self,
        asr: AsrProvider,
        llm: LlmProvider,
        tts: TtsProvider,
        metrics: TurnMetrics,
        splitter: SentenceSplitter | None = None,
        chunker: Chunker | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> None:
        """history：Session 持有的多轮上下文（可变引用）。调用 LLM 时取快照传入，
        传引用的列表本身绝不在这里被改写。"""
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._metrics = metrics
        self._splitter = splitter or SentenceSplitter()
        self._chunker = chunker or Chunker()
        self._history = history

    async def on_audio(self, chunk: bytes) -> None:
        await self._asr.push_audio(chunk)

    def text(self) -> str:
        return self._asr.text()

    def cancel(self) -> None:
        """barge-in：本类没有后台 speculative 工作，仅需复位由 EOU 驱动的状态。"""
        self._splitter = SentenceSplitter()

    async def on_end_of_utterance(self) -> AsyncIterator[TurnItem]:
        self._metrics.mark("utterance_end")
        yield TurnStart()

        transcript = await self._asr.finalize()
        # finalize 已终结识别流；立即开新流，EOU 之后到达的音频（用户抢在 AI
        # 开口前继续说、或 barge-in 窗口里的语音）才不会落进死流被丢掉。
        self._asr.reset()
        yield AsrFinal(text=transcript)

        if not transcript:
            # 空转写不起轮（回声自激回环的第二道门）：对空问题作答只会
            # 生成更多声音喂回麦克风。账单照出（证据留痕），但 LLM/TTS 不跑。
            self._metrics.mark("turn_end")
            yield TurnEnd(metrics=self._metrics.bill())
            return

        sentence_seq = 0
        saw_token = False
        async for piece in self._llm.astream(
            transcript, history=list(self._history) if self._history else None
        ):
            if not saw_token:
                self._metrics.mark("llm_first_token")
                saw_token = True
            for sentence in self._splitter.feed(piece):
                sentence_seq += 1
                if sentence_seq == 1:
                    self._metrics.mark("first_sentence_done")
                async for item in self._synthesize(sentence, sentence_seq):
                    yield item

        tail = self._splitter.flush()
        if tail:
            sentence_seq += 1
            async for item in self._synthesize(tail, sentence_seq):
                yield item

        self._metrics.mark("turn_end")
        yield TurnEnd(metrics=self._metrics.bill())

    async def _synthesize(self, sentence: str, seq: int) -> AsyncIterator[TurnItem]:
        self._metrics.sentences += 1
        if seq == 1:
            self._metrics.tts_first_chars = len(sentence)
        yield TtsSentence(seq=seq, text=sentence)
        first_chunk = True
        # 流式 provider（tts_streams）在整句合成完之前就开始出片；整句 provider
        # 合成完再切——分流逻辑与指标语义见 synth_to_fragments。
        async for chunk in synth_to_fragments(self._tts, sentence, self._chunker):
            if first_chunk:
                first_chunk = False
                self._metrics.mark("tts_first_chunk")
            self._metrics.mark("first_audio_out")
            self._metrics.audio_bytes += len(chunk)
            yield AudioFragment(seq=self._chunker.next_seq(), data=chunk)
