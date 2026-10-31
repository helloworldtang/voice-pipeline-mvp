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
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import TurnMetrics
from voice_pipeline.metrics.trace import phase_span, phases_of
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.server.protocol import AsrFinal, TtsSentence, TurnEnd, TurnStart

from .base import AudioFragment, TurnItem, synth_to_fragments


def _prefix_tail_limit() -> int:
    """前缀命中允许的缺失尾词字数（0 = 关闭）。env 可调，供 A/B。"""
    raw = os.environ.get("VOICE_SPEC_PREFIX_TAIL", "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


@dataclass(slots=True)
class SpecRun:
    """一次投机尝试的状态与产物。head / tail 分开存，是为了「第一句先走」——
    首音只看第一句，没必要等整段答案合成完。

    head 是**增量追加**的：ready 在第一句首片到手就置位（不等整句合成完），
    回放（_replay）开始时 head 可能还在长——progress 每追加一片就置位一次，
    消费端靠它跟播。
    """

    text: str
    head: list[TurnItem] = field(default_factory=list)
    tail: list[TurnItem] = field(default_factory=list)
    sentences: int = 0
    ready: asyncio.Event = field(default_factory=asyncio.Event)  # 第一句可播
    progress: asyncio.Event = field(default_factory=asyncio.Event)  # head/tail 有新片
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
        history: list[dict[str, str]] | None = None,
        prewarm_tts: bool = True,
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
        # debounce 在途时待投机的文本：EOU 入口要用它立即补跑最后一次投机（见下）
        self._pending_spec_text = ""
        self._last_partial = ""
        self._history = history
        # EOU 已经触发后，partial 前缀校验失效（final 才是权威）——见 _invalidate_if_stale。
        self._eou_done = False
        self._prewarm_tts_on = prewarm_tts
        self.spec_hits = 0
        self.spec_misses = 0
        self.spec_aborts = 0
        self.spec_prefix_hits = 0  # 靠前缀校验（尾词 ≤2 字）命中的次数
        self.spec_last_reason = "n/a"  # 上一次为什么没命中 —— 不可解释就不许调优

    # ---- 上行：边听边投机 ------------------------------------------------

    async def on_audio(self, chunk: bytes) -> None:
        await self._asr.push_audio(chunk)
        self._maybe_speculate()

    def text(self) -> str:
        return self._asr.text()

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
        self._pending_spec_text = partial
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self._debounce_ms / 1000.0, self._start_speculation, partial)

    def _start_speculation(self, text: str) -> None:
        self._timer = None
        self._pending_spec_text = ""
        # 用户还在说：此刻预热 TTS 连接，建连那 ~260ms 藏在说话窗口里，
        # 不进口径B（与判停时那次预热互为兜底，幂等）。
        self._prewarm_tts()
        task = self._task
        if task is not None and not task.done():
            # 在飞的投机答的是上一版问题，留着只是白烧 CPU
            task.cancel()
            self.spec_aborts += 1
        self._run = SpecRun(text=text)
        self._task = asyncio.create_task(self._run_speculation(self._run))

    def _prewarm_tts(self) -> None:
        """TTS 连接预热（P0.5）：非阻塞，后台补预连。语义同 SerialScheduler。"""
        if not self._prewarm_tts_on:
            return
        prewarm = getattr(self._tts, "prewarm", None)
        if callable(prewarm):
            prewarm()

    async def _run_speculation(self, run: SpecRun) -> None:
        """后台跑 LLM → 分句 → TTS，产物攒进 run，**一个字节都不外发**。"""
        splitter = SentenceSplitter()
        chunker = Chunker()

        # 投机的 LLM 段也开 span（审查第 5 项）：过去只有 fallback 包 llm.request，
        # 投机那一次生成在账单上不可见——命中率低的时候看不出「CPU 花在注定作废的
        # 生成上有多少」。与 tts.synth(speculative=1) 同一标记法。
        tr = self._metrics.tracer
        root = (tr.window or tr.root) if tr is not None else None
        req = (
            tr.start("llm.request", root, {"speculative": 1, "chars": len(run.text)})
            if tr
            else None
        )
        ttft = tr.start("llm.ttft", req, {"speculative": 1}) if tr else None
        saw_token = False
        try:
            async for piece in self._llm.astream(
                run.text, history=list(self._history) if self._history else None
            ):
                if not saw_token:
                    saw_token = True
                    if tr is not None and ttft is not None:
                        phases = phases_of(self._llm)
                        phase_span(tr, "llm.headers", phases, "start", "headers", ttft)
                        phase_span(tr, "llm.prefill", phases, "headers", "first_content", ttft)
                        phase_span(
                            tr,
                            "llm.worker_ttft",
                            phases,
                            "headers",
                            "worker_first_token",
                            ttft,
                        )
                        phase_span(
                            tr, "llm.wakeup", phases, "worker_first_token", "first_content", ttft
                        )
                        tr.finish(ttft)
                for sentence in splitter.feed(piece):
                    await self._synth_into(run, sentence, chunker)
                    if run.sentences == 1:
                        run.ready.set()
                self._invalidate_if_stale(run)
        finally:
            if tr:
                if ttft is not None and ttft.open:
                    tr.finish(ttft)
                if req is not None:
                    tr.finish(req, {"sentences": run.sentences})

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

        EOU 之后本校验失效：finalize→reset 会让 text() 归零重来，拿新流的
        partial 去比对旧流的投机前提必然误判作废。EOU 后的权威判据是
        `_settle` 里的 `run.text != transcript` 比对。
        """
        if self._eou_done:
            return
        if not self._asr.text().startswith(run.text):
            run.valid = False

    async def _synth_into(self, run: SpecRun, sentence: str, chunker: Chunker) -> None:
        run.sentences += 1
        if run.sentences == 1:
            self._metrics.tts_first_chars = len(sentence)
        first_chunk = True
        warm = False
        tr = self._metrics.tracer
        # 投机合成与正品合成同名（tts.synth），靠 speculative 属性区分：
        # 命中率低的时候要能一眼看出"CPU 花在注定作废的东西上"有多少。
        synth = (
            tr.start(
                "tts.synth",
                tr.root,
                {"seq": run.sentences, "chars": len(sentence), "speculative": 1},
            )
            if tr
            else None
        )
        head = tr.start("tts.first_chunk", synth) if tr and run.sentences == 1 else None
        # 增量追加（不再是攒完一整句再赋值）：首片到手即 ready，回放开始时
        # head 可能还在长，_replay 靠 run.progress 跟播（2026-10-27）。
        target = run.head if run.sentences == 1 else run.tail
        target.append(TtsSentence(seq=run.sentences, text=sentence))
        run.progress.set()
        async for block in synth_to_fragments(
            self._tts, sentence, chunker, on_skip=self._metrics.note_tts_skip
        ):
            if first_chunk and run.sentences == 1:
                first_chunk = False
                warm = bool(getattr(self._tts, "last_warm", False))
                self._metrics.mark("tts_first_chunk")
                run.ready_ms = self._metrics.clock.elapsed_ms()
                # 首片到手即可判定「用得上」：真人实测等整句合成完，EOU 永远先到
                # （2026-10-27 真人 pipelined 0/10 命中，全是 not_ready(sentences=1)）。
                # 整句合成完的那次 ready.set() 留在 _run_speculation 里兜底
                # （首片永远不来的极端情形，如 TTS 静默失败零分片）。
                run.ready.set()
                if tr:
                    # 深拆（同 serial）：建连 vs 服务端合成首片。speculative 路径也要拆——
                    # 命中率低时这些合成注定作废，拆开才知道废掉的是哪一段。
                    phases = phases_of(self._tts)
                    phase_span(tr, "tts.connect", phases, "submit", "open", synth)
                    open_key = "open" if "open" in phases else "submit"
                    phase_span(tr, "tts.first_byte", phases, open_key, "first_data", synth)
                if tr and head:
                    tr.finish(head)
            target.append(AudioFragment(seq=chunker.next_seq(), data=block))
            run.progress.set()
        if tr and synth:
            tr.finish(synth, {"warm": 1 if warm else 0})

    # ---- EOU：校验 → 命中即刻播 / 未命中走串行 ---------------------------

    async def on_end_of_utterance(self) -> AsyncIterator[TurnItem]:
        # 判停即预热：没走过投机路径（partial 太短/被取消）时这是唯一一次预热机会
        self._prewarm_tts()
        # 收掉在途的 debounce 定时器（2026-11-01 审查缺陷 7）：EOU 已成立，定时器
        # 若还在飞，会在 settle/abort/fallback 之后的某个 await 点补跑
        # _start_speculation——僵尸投机与 fallback 抢同一个模型（MLX 子进程锁串行，
        # fallback 首 token 排队变慢）。修法：EOU 入口立即照常启动最后一次投机
        # （命中机会不丢），定时器与待投文本一并清零。
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
            pending, self._pending_spec_text = self._pending_spec_text, ""
            if pending:
                self._start_speculation(pending)
        self._metrics.mark("utterance_end")
        yield TurnStart()

        tr = self._metrics.tracer
        root = (tr.window or tr.root) if tr is not None else None
        fin = tr.start("asr.finalize", root) if tr else None
        transcript = await self._asr.finalize()
        # finalize 已终结识别流，立即开新流收 EOU 之后到达的音频（同 serial）。
        # reset 会让 text() 归零，所以必须先立 _eou_done 旗标，否则仍在算尾巴的
        # 投机任务拿新流的空 partial 做前缀校验，会把本已命中的投机误判作废。
        self._asr.reset()
        self._eou_done = True
        if tr and fin:
            tr.finish(fin, {"chars": len(transcript)})
        yield AsrFinal(text=transcript)

        if not transcript:
            # 空转写不起轮：与 serial 同一条纪律（回声自激回环第二道门）
            self._metrics.mark("turn_end")
            self._eou_done = False
            yield TurnEnd(metrics=self._metrics.bill())
            return

        run = await self._settle(transcript)
        if run is None:
            self.spec_misses += 1
            self._abort_inflight_spec()
            if tr and root is not None:
                # 哨兵与合法零值不许混用（2026-11-01 审查缺陷 8）：elapsed 恰为 0.0
                # 时 `or -1.0` 会把真值吞成缺失。显式判 None。
                elapsed = self._metrics.elapsed("utterance_end")
                tr.record(
                    "spec.miss",
                    -1.0 if elapsed is None else elapsed,
                    self._metrics.clock.elapsed_ms(),
                    parent=root,
                    attrs={"reason": self.spec_last_reason},
                )
            async for item in self._fallback(transcript):
                yield item
        else:
            self.spec_hits += 1
            async for item in self._replay(run):
                yield item

        self._metrics.mark("turn_end")
        # 本轮收尾：放行下一轮说话期间的 partial 前缀校验。
        self._eou_done = False
        # 哨兵与合法零值不许混用（2026-11-01 审查缺陷 8）：ready_ms 恰为 0.0 时
        # `or -1.0` 会把真值吞成缺失。显式判 None。
        ready_ms = run.ready_ms if run is not None else None
        yield TurnEnd(
            metrics={
                **self._metrics.bill(),
                "spec_hit": float(self.spec_hits),
                "spec_miss": float(self.spec_misses),
                "spec_prefix_hit": float(self.spec_prefix_hits),
                "spec_ready_ms": round(ready_ms, 3) if ready_ms is not None else -1.0,
            }
        )

    def _abort_inflight_spec(self) -> None:
        """miss 时砍掉还在飞的投机任务。

        它答的已经是错问题（text_differs / stale_partial / grace 没赶上），
        留着只会：一、白烧 CPU/GPU 直到 max_tokens；二、跟 fallback 的重新生成
        抢同一个模型——进程内是两条 stream_generate 线程在 Metal 队列上交错，
        子进程模式（`mlx_proc_llm._lock` 串行）更是让 fallback 的首 token
        排队等它生成完。这是 2026-10-17 复盘发现的缺口：cancel 只有三处调用
        （新 partial 顶旧、replay 尾部超时、barge-in），唯独 miss 路径漏了。
        """
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            self.spec_aborts += 1

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
            # 前缀命中（2026-10-27 晚间真人数据拍板）：sherpa 的尾词（"吗""一下"）
            # 直到 finalize 才出现、从不进 partial 流，等值校验把 4/9 的轮次挡在
            # 门外，而尾词几乎都是语气助词——前缀已完整表达问题主体，答前缀≈答全句。
            # 保护：缺失尾巴超过 MAX_SPEC_TAIL_CHARS 字不算命中（"……别/不要"这类
            # 会翻转语义的真尾词通常更长，也更少见）。设 VOICE_SPEC_PREFIX_TAIL=0 关闭。
            tail = len(transcript) - len(run.text)
            if 0 < tail <= _prefix_tail_limit() and transcript.startswith(run.text):
                self.spec_prefix_hits += 1
                self.spec_last_reason = f"prefix_hit(tail={tail})"
                return run
            reason = f"text_differs({run.text!r}!={transcript!r})"
        elif run.sentences == 0:
            reason = "no_sentence"
        else:
            return run
        self.spec_last_reason = reason
        return None

    async def _replay(self, run: SpecRun) -> AsyncIterator[TurnItem]:
        """把投机产物吐出去。此刻用户已经说完，发声不早于 EOU。

        ready 在第一句**首片**到手就置位（见 _synth_into），所以 head 在回放途中
        还会增长——跟播：追平后等 `run.progress`，直到投机任务结束；迟到的片
        **按序补播，一片不丢**（清旗标后复查再等，append 与 clear 的竞争不会丢唤醒）。
        """
        # 投机开跑时这几个点其实早已越过，如实补记，waterfall 才看得出「藏掉了多少」
        self._metrics.mark("llm_first_token")
        self._metrics.mark("first_sentence_done")
        # tts_first_chars 在投机期设过，但 run_turn 开头的 reset_turn 会清掉它
        # （投机跑在 reset 之前、EOU 之前）——命中轮的账单因此恒 -1（2026-10-27
        # 真人 pipelined 实锤），这里按第一句原文补记。
        if run.head and isinstance(run.head[0], TtsSentence):
            self._metrics.tts_first_chars = len(run.head[0].text)

        task = self._task
        i = 0
        while True:
            while i < len(run.head):
                item = run.head[i]
                i += 1
                async for out in self._pass_through(item):
                    yield out
            if task is None or task.done():
                break
            run.progress.clear()
            if i < len(run.head) or task.done():  # 清完旗标再复查：刚追的这片不算丢
                continue
            try:
                await asyncio.wait_for(
                    asyncio.shield(run.progress.wait()),
                    timeout=self._tail_timeout_ms / 1000,
                )
            except TimeoutError:
                task.cancel()
                break

        # sentences 计数挪到任务收尾后取终值：早播开始时尾巴还没算完，
        # 进 replay 那一刻的 run.sentences 是欠账（旧实现在这里 += 的是中间值）
        self._metrics.sentences += run.sentences

        for item in run.tail:
            async for out in self._pass_through(item):
                yield out

    async def _pass_through(self, item: TurnItem) -> AsyncIterator[TurnItem]:
        if isinstance(item, AudioFragment):
            self._mark_first_audio()
            self._metrics.audio_bytes += len(item.data)
        yield item

    async def _fallback(self, transcript: str) -> AsyncIterator[TurnItem]:
        """未命中：退化成串行路径，整句重新生成并合成一遍。"""
        self._metrics.mark("llm_first_token")
        splitter = SentenceSplitter()
        seq = 0
        # 与 serial 同构的 LLM 分格：过去这里一个 llm span 都没有，未命中轮的 LLM
        # 时间全落进「真·未归因」——而它偏偏是未命中轮最大的一块。补上恒等式才闭合。
        tr = self._metrics.tracer
        root = (tr.window or tr.root) if tr is not None else None
        req = tr.start("llm.request", root, {"chars": len(transcript)}) if tr else None
        ttft = tr.start("llm.ttft", req) if tr else None
        saw_token = False
        try:
            async for piece in self._llm.astream(
                transcript, history=list(self._history) if self._history else None
            ):
                if not saw_token:
                    saw_token = True
                    if tr is not None and ttft is not None:
                        phases = phases_of(self._llm)
                        phase_span(tr, "llm.headers", phases, "start", "headers", ttft)
                        phase_span(tr, "llm.prefill", phases, "headers", "first_content", ttft)
                        # 深拆（同 serial）：生成本身 vs 线程→事件循环的唤醒
                        phase_span(
                            tr, "llm.worker_ttft", phases, "headers", "worker_first_token", ttft
                        )
                        phase_span(
                            tr, "llm.wakeup", phases, "worker_first_token", "first_content", ttft
                        )
                        tr.finish(ttft)
                for sentence in splitter.feed(piece):
                    seq += 1
                    if seq == 1:
                        self._metrics.mark("first_sentence_done")
                    async for item in self._emit(sentence, seq):
                        yield item
        finally:
            if tr:
                if ttft is not None and ttft.open:
                    tr.finish(ttft)
                if req is not None:
                    tr.finish(req, {"sentences": seq})
        tail = splitter.flush()
        if tail:
            seq += 1
            async for item in self._emit(tail, seq):
                yield item

    async def _emit(self, sentence: str, seq: int) -> AsyncIterator[TurnItem]:
        self._metrics.sentences += 1
        if seq == 1:
            # 未命中轮走这条出口（_fallback → _emit），而 _synth_into 里的同类赋值
            # 只在投机期执行过、被 run_turn 开头的 reset_turn 清掉 ⇒ **慢的那一半
            # 轮次**首句字数恒 -1（2026-10-27 真人 pipelined 5/10 实锤）。
            # 与 _replay 的补记对齐：不补，则「首句字数↔口径B」的相关性恰好在
            # 最该分析的那条路径上不可算。
            self._metrics.tts_first_chars = len(sentence)
        yield TtsSentence(seq=seq, text=sentence)
        first_chunk = True
        warm = False
        tr = self._metrics.tracer
        synth = (
            tr.start("tts.synth", (tr.window or tr.root), {"seq": seq, "chars": len(sentence)})
            if tr
            else None
        )
        head = tr.start("tts.first_chunk", synth) if tr and seq == 1 else None
        async for block in synth_to_fragments(
            self._tts, sentence, self._chunker, on_skip=self._metrics.note_tts_skip
        ):
            if first_chunk:
                first_chunk = False
                warm = bool(getattr(self._tts, "last_warm", False))
                self._metrics.mark("tts_first_chunk")
                if tr:
                    # 深拆（同 serial）：建连 vs 服务端合成首片
                    phases = phases_of(self._tts)
                    phase_span(tr, "tts.connect", phases, "submit", "open", synth)
                    open_key = "open" if "open" in phases else "submit"
                    phase_span(tr, "tts.first_byte", phases, open_key, "first_data", synth)
                if tr and head:
                    tr.finish(head)
            self._mark_first_audio()
            self._metrics.audio_bytes += len(block)
            yield AudioFragment(seq=self._chunker.next_seq(), data=block)
        if tr and synth:
            tr.finish(synth, {"warm": 1 if warm else 0})

    def _mark_first_audio(self) -> None:
        """首片出炉：**这里是口径B 的终点**，根 span 在此闭合（两条出境路径都走它）。"""
        self._metrics.mark("first_audio_out")
        tr = self._metrics.tracer
        if tr is not None and tr.window is not None and tr.window.open:
            tr.finish_window()

    # ---- barge-in --------------------------------------------------------

    def cancel(self) -> None:
        """用户打断：丢弃在飞的投机，绝不让它继续算一个没人要的答案。"""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending_spec_text = ""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._run = None
        self._last_partial = ""
        # 打断可能发生在 EOU 之后（replay 途中），旗标必须一并复位，
        # 否则下一轮说话期间的投机永远不做前缀校验。
        self._eou_done = False
        self._splitter = SentenceSplitter()
