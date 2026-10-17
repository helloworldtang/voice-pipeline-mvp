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
from voice_pipeline.metrics.trace import Span, phase_span, phases_of
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
        prewarm_tts: bool = True,
    ) -> None:
        """history：Session 持有的多轮上下文（可变引用）。调用 LLM 时取快照传入，
        传引用的列表本身绝不在这里被改写。

        prewarm_tts：判停时是否触发 TTS 连接预热（P0.5）。探针的对照臂关掉它。
        """
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._metrics = metrics
        self._splitter = splitter or SentenceSplitter()
        self._chunker = chunker or Chunker()
        self._history = history
        self._prewarm_tts_on = prewarm_tts

    async def on_audio(self, chunk: bytes) -> None:
        await self._asr.push_audio(chunk)

    def text(self) -> str:
        return self._asr.text()

    def cancel(self) -> None:
        """barge-in：本类没有后台 speculative 工作，仅需复位由 EOU 驱动的状态。"""
        self._splitter = SentenceSplitter()

    def _prewarm_tts(self) -> None:
        """TTS 连接预热（P0.5）：判停一刻就在后台连上，非阻塞。

        建连 ~260ms 必须发生在**用户话音刚落之后、LLM 回来之前**的窗口里
        （headers 一个往返 ~230ms 就是它的一生），等它再跑 LLM 等于把账加进
        口径B。provider 没有 prewarm()（sim/本机 TTS）时静默跳过。
        """
        if not self._prewarm_tts_on:
            return
        prewarm = getattr(self._tts, "prewarm", None)
        if callable(prewarm):
            prewarm()

    async def on_end_of_utterance(self) -> AsyncIterator[TurnItem]:
        self._prewarm_tts()
        self._metrics.mark("utterance_end")
        yield TurnStart()

        tracer = self._metrics.tracer
        root = (tracer.window or tracer.root) if tracer is not None else None

        # ASR finalize 单独一格：过去它混在 "utterance_end → llm_first_token"
        # 那一格里，被打点差算成 LLM 的账——看不出来就没法定它的优化优先级。
        fin = tracer.start("asr.finalize", root) if tracer else None
        transcript = await self._asr.finalize()
        # finalize 已终结识别流；立即开新流，EOU 之后到达的音频（用户抢在 AI
        # 开口前继续说、或 barge-in 窗口里的语音）才不会落进死流被丢掉。
        self._asr.reset()
        if tracer and fin:
            tracer.finish(fin, {"chars": len(transcript)})
        yield AsrFinal(text=transcript)

        if not transcript:
            # 空转写不起轮（回声自激回环的第二道门）：对空问题作答只会
            # 生成更多声音喂回麦克风。账单照出（证据留痕），但 LLM/TTS 不跑。
            self._metrics.mark("turn_end")
            yield TurnEnd(metrics=self._metrics.bill())
            return

        sentence_seq = 0
        saw_token = False
        # LLM 一格拆两层：`llm.request`（整段生成）里嵌 `llm.ttft`（首 token）与
        # `llm.first_sentence`（首 token → 首句成）。嵌套是因为 ttft 是 request 的
        # 一部分——平铺会把同一段时间记两次，而自有时间（self time）算得出来才不准ाओ
        req = tracer.start("llm.request", root, {"chars": len(transcript)}) if tracer else None
        ttft = tracer.start("llm.ttft", req) if tracer else None
        sent: Span | None = None
        try:
            async for piece in self._llm.astream(
                transcript, history=list(self._history) if self._history else None
            ):
                if not saw_token:
                    self._metrics.mark("llm_first_token")
                    saw_token = True
                    if tracer:
                        # 深拆 llm.ttft：`headers`（发包→响应头 = 网络+路由）与
                        # `prefill`（响应头→首个 content = 服务端排队+prefill）。
                        # 时刻是 provider 在 astream 里记的 perf 原值，经会话钟换算。
                        phases = phases_of(self._llm)
                        phase_span(tracer, "llm.headers", phases, "start", "headers", ttft)
                        phase_span(tracer, "llm.prefill", phases, "headers", "first_content", ttft)
                    if tracer and ttft:
                        tracer.finish(ttft)  # 首 token：这一格是模型自己的 RTT+prefill
                    if tracer and req is not None:
                        sent = tracer.start("llm.first_sentence", req)
                for sentence in self._splitter.feed(piece):
                    sentence_seq += 1
                    if sentence_seq == 1:
                        self._metrics.mark("first_sentence_done")
                        if tracer and sent:
                            tracer.finish(sent, {"chars": len(sentence)})  # 攒到句末标点
                    async for item in self._synthesize(sentence, sentence_seq):
                        yield item
        finally:
            if tracer:
                if sent is not None and sent.open:
                    tracer.finish(sent)
                if ttft is not None and ttft.open:
                    tracer.finish(ttft)
                if req is not None:
                    tracer.finish(req, {"sentences": sentence_seq})

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
        warm = False
        tr = self._metrics.tracer
        root = (tr.window or tr.root) if tr is not None else None
        # TTS 单独一格 **且记字数**：本机最强的相关性就在这一格，`chars` 是解释它的唯一自变量
        synth = tr.start("tts.synth", root, {"seq": seq, "chars": len(sentence)}) if tr else None
        head = tr.start("tts.first_chunk", synth) if tr and seq == 1 else None
        async for chunk in synth_to_fragments(self._tts, sentence, self._chunker):
            if first_chunk:
                first_chunk = False
                warm = bool(getattr(self._tts, "last_warm", False))
                self._metrics.mark("tts_first_chunk")
                if tr is not None:
                    # 深拆 tts.first_chunk：`connect`（建连+认证）与 `first_byte`
                    # （服务端合成首片）。on_open 没 fire 时 connect 不记、first_byte
                    # 从 submit 起算——缺证据的格子宁空不填 0（phase_span 的纪律）。
                    phases = phases_of(self._tts)
                    phase_span(tr, "tts.connect", phases, "submit", "open", synth)
                    open_key = "open" if "open" in phases else "submit"
                    phase_span(tr, "tts.first_byte", phases, open_key, "first_data", synth)
                if tr is not None and head is not None:
                    tr.finish(head)
            self._metrics.mark("first_audio_out")
            if tr is not None and tr.window is not None and tr.window.open:
                # 口径B 窗口在这里闭合（**不含**传输层那一跳，另记 tx.*）
                tr.finish_window()
            self._metrics.audio_bytes += len(chunk)
            yield AudioFragment(seq=self._chunker.next_seq(), data=chunk)
        if tr is not None and synth is not None:
            # warm：这一句借到的是不是预热连接（P0.5 生效的直接证据）
            tr.finish(synth, {"warm": 1 if warm else 0})
