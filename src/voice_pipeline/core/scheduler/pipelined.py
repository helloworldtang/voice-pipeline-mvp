"""投机调度：在用户**还在说话时**就把能算的算完，EOU 一到立刻播音。

它回答的是 §13.4 那个结论——串行前提下 300ms 数学上不可达，因为 TTS 地板就有
156ms（melo）/ 660ms（kokoro），而这段必须发生在用户闭嘴之后。投机把这段搬到用户说话期间去跑。

三条纪律（缺一条就退化成抢话或答非所问）：

1. **只提前算，不提前播。** 投机产物全程缓存，一个字节都不在 EOU 之前下发。
   「激进」的含义是把已经算好的东西立刻播出来，不是早于用户说完就开口。
2. **必须校验。** EOU 后拿 final 与投机时的 partial 比对，相等才用。
   不等就作废重来——宁可慢一次，也不能对着错问题给出流利的答案。
3. **可被取消。** 用户打断（barge-in）或判定作废时立即 cancel 后台任务，
   不留悬挂任务白吃 CPU。

「快」与「不抢话」在这里不是取舍关系：正因为要在用户说完之后 300ms 内出声，
才必须在他说完之前把活干完。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import TurnMetrics
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.server.protocol import AsrFinal, TtsSentence, TurnEnd, TurnStart

from .base import AudioFragment, TurnItem


@dataclass(slots=True)
class SpecRun:
    """一次投机尝试的状态与产物。head / tail 分开存，是为了「第一句先走」——
    首音只看第一句，没必要等整段答案合成完。
    """

    text: str
    head: list[TurnItem] = field(default_factory=list)
    tail: list[TurnItem] = field(default_factory=list)
    sentences: int = 0
    ready: asyncio.Event = field(default_factory=asyncio.Event)  # 第一句可播
    valid: bool = True  # ASR 修正过 partial 后置 False
    ready_ms: float | None = None

    @property
    def items(self) -> list[TurnItem]:
        return [*self.head, *self.tail]


class PipelinedScheduler:
    def __init__(
        self,
        asr: AsrProvider,
        llm: LlmProvider,
        tts: TtsProvider,
        metrics: TurnMetrics,
        splitter: SentenceSplitter | None = None,
        chunker: Chunker | None = None,
        *,
        speculate: bool = True,
        min_partial_chars: int = 4,
        grace_ms: float = 0.0,
        tail_timeout_ms: float = 5000.0,
        debounce_ms: float = 0.0,
    ) -> None:
        """grace_ms：EOU 后最多再等多少毫秒让投机落地（0 = 不等，没好就走串行）。

        这是个诚实参数——给正值就是「用系统性多一点等待换命中率」，
        必须在账单里量出来，不能藏在实现里偷偷加。实测：grace=300 命中时把
        口径B 从 736ms 压到 179ms，未命中时则要付上接近 grace 本身的罚金。

        debounce_ms：partial 稳定多久才启动投机。0 = 一变就重启。
        非零值用来避免「用户每吐一个字就砍掉重来」——那是最贵的浪费：
        CPU 全花在注定作废的合成上，而真正能用的那一版反而没时间跑。
        """
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._metrics = metrics
        self._splitter = splitter or SentenceSplitter()
        self._chunker = chunker or Chunker()
        self._speculate = speculate
        self._min_partial_chars = min_partial_chars
        self._grace_ms = grace_ms
        self._tail_timeout_ms = tail_timeout_ms
        self._debounce_ms = debounce_ms

        self._task: asyncio.Task[None] | None = None
        self._run: SpecRun | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._last_partial = ""
        self.spec_hits = 0
        self.spec_misses = 0
        self.spec_aborts = 0
        self.spec_last_reason = "n/a"  # 上一次为什么没命中 —— 不可解释就不许调优

    # ---- 上行：边听边投机 ------------------------------------------------

    async def on_audio(self, chunk: bytes) -> None:
        await self._asr.push_audio(chunk)
        self._maybe_speculate()

    def _maybe_speculate(self) -> None:
        if not self._speculate:
            return
        partial = self._asr.text()
        if not partial or partial == self._last_partial:
            return
        self._last_partial = partial
        if len(partial) < self._min_partial_chars:
            return

        # 每次 partial 变化都把还没到点的投机往后推一格：用户明显还在说，
        # 现在开跑注定作废。debounce=0 时等价于「一变就重启」，留作对照组。
        if self._timer is not None:
            self._timer.cancel()
            self.spec_aborts += 1
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self._debounce_ms / 1000.0, self._start_speculation, partial)

    def _start_speculation(self, text: str) -> None:
        self._timer = None
        task = self._task
        if task is not None and not task.done():
            # 在飞的投机答的是上一版问题，留着只是白烧 CPU
            task.cancel()
            self.spec_aborts += 1
        self._run = SpecRun(text=text)
        self._task = asyncio.create_task(self._run_speculation(self._run))

    async def _run_speculation(self, run: SpecRun) -> None:
        """后台跑 LLM → 分句 → TTS，产物攒进 run，**一个字节都不外发**。"""
        splitter = SentenceSplitter()
        chunker = Chunker()

        async for piece in self._llm.astream(run.text):
            for sentence in splitter.feed(piece):
                await self._synth_into(run, sentence, chunker)
                if run.sentences == 1:
                    run.ready.set()
            self._invalidate_if_stale(run)

        tail_sentence = splitter.flush()
        if tail_sentence:
            await self._synth_into(run, tail_sentence, chunker)
        self._invalidate_if_stale(run)
        run.ready.set()

    def _invalidate_if_stale(self, run: SpecRun) -> None:
        """投机所依据的 partial 还在不在当前 partial 里。

        判据是**前缀**而不是相等：用户在继续说话时 partial 只会变长，
        那是追加不是推翻，此时作废等于把每一次投机都砍掉（本机实测命中率会掉到 0）。
        真正该作废的是「已经说出去的字被 ASR 改写」——那才是前提塌了。
        """
        if not self._asr.text().startswith(run.text):
            run.valid = False

    async def _synth_into(self, run: SpecRun, sentence: str, chunker: Chunker) -> None:
        run.sentences += 1
        pcm = await asyncio.to_thread(self._tts.synth_blocking, sentence)
        if run.sentences == 1:
            self._metrics.mark("tts_first_chunk")
            run.ready_ms = self._metrics.clock.elapsed_ms()
        items: list[TurnItem] = [TtsSentence(seq=run.sentences, text=sentence)]
        items += [AudioFragment(seq=chunker.next_seq(), data=block) for block in chunker.feed(pcm)]
        if run.sentences == 1:
            run.head = items
        else:
            run.tail += items

    # ---- EOU：校验 → 命中即刻播 / 未命中走串行 ---------------------------

    async def on_end_of_utterance(self) -> AsyncIterator[TurnItem]:
        self._metrics.mark("utterance_end")
        yield TurnStart()

        transcript = await self._asr.finalize()
        yield AsrFinal(text=transcript)

        run = await self._settle(transcript)
        if run is None:
            self.spec_misses += 1
            async for item in self._fallback(transcript):
                yield item
        else:
            self.spec_hits += 1
            async for item in self._replay(run):
                yield item

        self._metrics.mark("turn_end")
        yield TurnEnd(
            metrics={
                **self._metrics.bill(),
                "spec_hit": float(self.spec_hits),
                "spec_miss": float(self.spec_misses),
                "spec_ready_ms": round(run.ready_ms or -1.0, 3) if run else -1.0,
            }
        )

    async def _settle(self, transcript: str) -> SpecRun | None:
        """等到能判定「用还是不用」为止：第一句就绪 + 文本没被修正。"""
        run = self._run
        if run is None:
            self.spec_last_reason = "no_speculation"
            return None
        if not run.ready.is_set():
            # 先同步看一眼：wait_for(timeout=0) 对「已经就绪」的 Event 也会抛 TimeoutError，
            # 因为 ensure_future 出来的协程还没被调度过。这里的 if 不是为了优化，是正确性。
            if self._grace_ms <= 0:
                self.spec_last_reason = f"not_ready(sentences={run.sentences})"
                return None
            try:
                await asyncio.wait_for(run.ready.wait(), timeout=self._grace_ms / 1000.0)
            except TimeoutError:
                self.spec_last_reason = f"grace_expired(sentences={run.sentences})"
                return None

        if not run.valid:
            reason = "stale_partial"
        elif run.text != transcript:
            reason = f"text_differs({run.text!r}!={transcript!r})"
        elif run.sentences == 0:
            reason = "no_sentence"
        else:
            return run
        self.spec_last_reason = reason
        return None

    async def _replay(self, run: SpecRun) -> AsyncIterator[TurnItem]:
        """把投机产物吐出去。此刻用户已经说完，发声不早于 EOU。"""
        # 投机开跑时这几个点其实早已越过，如实补记，waterfall 才看得出「藏掉了多少」
        self._metrics.mark("llm_first_token")
        self._metrics.mark("first_sentence_done")
        self._metrics.sentences += run.sentences

        for item in run.head:
            async for out in self._pass_through(item):
                yield out

        # 头部播音的同时后台还在算尾巴：给它一个上限，超时就砍掉
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._tail_timeout_ms / 1000)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()

        for item in run.tail:
            async for out in self._pass_through(item):
                yield out

    async def _pass_through(self, item: TurnItem) -> AsyncIterator[TurnItem]:
        if isinstance(item, AudioFragment):
            self._metrics.mark("first_audio_out")
            self._metrics.audio_bytes += len(item.data)
        yield item

    async def _fallback(self, transcript: str) -> AsyncIterator[TurnItem]:
        """未命中：退化成串行路径，整句重新生成并合成一遍。"""
        self._metrics.mark("llm_first_token")
        splitter = SentenceSplitter()
        seq = 0
        async for piece in self._llm.astream(transcript):
            for sentence in splitter.feed(piece):
                seq += 1
                if seq == 1:
                    self._metrics.mark("first_sentence_done")
                async for item in self._emit(sentence, seq):
                    yield item
        tail = splitter.flush()
        if tail:
            seq += 1
            async for item in self._emit(tail, seq):
                yield item

    async def _emit(self, sentence: str, seq: int) -> AsyncIterator[TurnItem]:
        self._metrics.sentences += 1
        yield TtsSentence(seq=seq, text=sentence)
        pcm = await asyncio.to_thread(self._tts.synth_blocking, sentence)
        self._metrics.mark("tts_first_chunk")
        for block in self._chunker.feed(pcm):
            self._metrics.mark("first_audio_out")
            self._metrics.audio_bytes += len(block)
            yield AudioFragment(seq=self._chunker.next_seq(), data=block)

    # ---- barge-in --------------------------------------------------------

    def cancel(self) -> None:
        """用户打断：丢弃在飞的投机，绝不让它继续算一个没人要的答案。"""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._run = None
        self._last_partial = ""
        self._splitter = SentenceSplitter()
