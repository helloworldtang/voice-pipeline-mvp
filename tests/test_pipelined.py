"""投机调度：只提前算，不提前播；作废了必须答对的问题。

这三条是这个调度器的全部道德：
1. 命中时首音压到 EOU 之后几毫秒内
2. 作废时结果仍然正确（不能对着错问题流利作答）
3. 打断时不留悬挂任务
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from voice_pipeline.core.scheduler.base import AudioFragment, TurnItem
from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import TurnEnd

TRANSCRIPT = "今天天气怎么样"
FRAME = b"\x00" * 960


class EchoLlm:
    """把收到的 prompt 原样吐出来，用来验证「答的是不是同一个问题」。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        self.prompts.append(prompt)

        async def producer() -> AsyncIterator[str]:
            yield prompt + "。"

        return producer()


class GatedLlm(EchoLlm):
    """第一次调用（投机）卡在门口，直到放行——制造「EOU 已到、投机还在飞」。"""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.cancelled_in_flight = False

    def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        first = not self.prompts  # 只有第一次调用（投机）卡门，fallback 直通
        self.prompts.append(prompt)

        async def producer() -> AsyncIterator[str]:
            try:
                if first:
                    await self.gate.wait()  # 投机：永远等不到（除非测试放行）
                yield prompt + "。"
            except asyncio.CancelledError:
                self.cancelled_in_flight = True
                raise

        return producer()


def _tts(cpu_ms: float = 150.0) -> SimTts:
    return SimTts(cpu_ms=cpu_ms)


def _converged_asr() -> SimAsr:
    """partial 一开始就收敛成最终文本。

    模拟真实链路里最值钱的那一幕：用户已经说完，但 VAD 还在等静音窗确认
    （本机实测至少 250ms）。投机要抢的就是这段时间。
    """
    return SimAsr(TRANSCRIPT, partials=[TRANSCRIPT] * 5)


async def _drain(
    scheduler: PipelinedScheduler | SerialScheduler,
) -> tuple[list[TurnItem], dict[str, float]]:
    """跑完一轮，返回全部产出与账单。"""
    items: list[TurnItem] = []
    async for item in scheduler.on_end_of_utterance():
        items.append(item)
    for item in items:
        if isinstance(item, TurnEnd):
            return items, dict(item.metrics)
    msg = "no TurnEnd emitted"
    raise AssertionError(msg)


async def test_speculation_hit_cuts_out_the_wait():
    """partial 收敛 + EOU 前投机完成 → 首音紧跟 EOU（合成时间被藏掉了）。"""
    sched = PipelinedScheduler(
        asr=_converged_asr(),
        llm=EchoLlm(),
        tts=_tts(cpu_ms=300.0),
        metrics=TurnMetrics(clock=SessionClock()),
    )

    await sched.on_audio(FRAME)
    await asyncio.sleep(0.6)  # 让投机在 EOU 之前跑完

    items, bill = await _drain(sched)

    assert bill["spec_hit"] == 1.0, bill
    assert sched.spec_hits == 1
    # TTS 实际耗时 300ms，但它在投机期就跑完了 —— 口径B 必须远小于它
    assert bill["caliber_b_ms"] < 100, bill
    # 藏掉的量要能在账单里看出来：第一句就绪早于用户闭嘴
    assert 0 < bill["spec_ready_ms"] < bill["utterance_end"], bill
    assert [i for i in items if isinstance(i, AudioFragment)]
    # 首句字数必须在命中轮里有值：投机跑在 run_turn 的 reset_turn **之前**，
    # 那时设的 tts_first_chars 会被清账抹掉（与 eou_latency_ms 恒 0 同一个失效
    # 模式），命中轮曾因此恒 -1（2026-10-27 真人 pipelined 10 轮实锤）。
    assert bill["tts_first_chars"] > 0, bill


async def test_stale_speculation_is_discarded_and_answer_is_correct():
    """ASR 在流式期间给的是错误/未收敛的前缀，最终 transcript 才修正它。

    投机基于错误前缀跑完，但 settle 时 run.text != transcript → 作废，
    退化为串行、用最终全文重算正确答案。绝不能对着过时的 partial 流利作答。
    """
    llm = EchoLlm()
    # 流式期间卡在错误前缀，finalize 返回正确全文 —— 真实 ASR 的「先错后改」
    asr = SimAsr(TRANSCRIPT, partials=["今天天色", "今天天色", "今天天色"])
    sched = PipelinedScheduler(
        asr=asr,
        llm=llm,
        tts=_tts(cpu_ms=50.0),
        metrics=TurnMetrics(clock=SessionClock()),
        debounce_ms=0.0,  # 立即投机，确保 EOU 之前投机已跑完
    )

    await sched.on_audio(FRAME)  # 触发投机，基于错误前缀「今天天色」
    await asyncio.sleep(0.1)  # 让投机在 EOU 之前完成
    assert sched._run is not None and sched._run.text == "今天天色"

    items, bill = await _drain(sched)
    said = [getattr(i, "text", "") for i in items]
    said = [s for s in said if s]

    assert bill["spec_hit"] == 0.0, bill
    assert sched.spec_misses == 1
    # 关键：没有对着过时的 partial 作答，最终答案基于正确全文
    assert llm.prompts[-1] == TRANSCRIPT
    assert said[-1] == TRANSCRIPT + "。"


async def test_debounce_skips_speculating_on_every_intermediate_word():
    """partial 每变一次就重启 = CPU 全烧在注定作废的合成上。

    debounce 把「用户边说边吐字」合并成一次真正的投机：只有最后一个前缀真正跑合成，
    前面每个前缀只是把还没触发的定时器取消掉（记进 aborts），不会起一次合成。
    """
    # 用逐步变长、且每个都 >= min_partial_chars 的前缀，模拟 ASR 边听边修正
    partials = [TRANSCRIPT[:k] for k in range(4, len(TRANSCRIPT) + 1)]
    asr = SimAsr(TRANSCRIPT, partials=partials)
    tts = _tts(cpu_ms=0.0)
    sched = PipelinedScheduler(
        asr=asr,
        llm=EchoLlm(),
        tts=tts,
        metrics=TurnMetrics(clock=SessionClock()),
        debounce_ms=20.0,  # 够短：最后一个前缀的去抖窗会在测试末尾触发一次投机
    )
    for _ in range(len(partials)):
        await sched.on_audio(FRAME)
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)  # 等最后一个去抖窗触发并跑完投机

    # 每次 partial 变化都取消了上一个未触发的定时器 → 记进 aborts；
    # 但真正跑起来的合成只有 1 次（最后一个前缀）
    assert sched.spec_aborts == len(partials) - 1, sched.spec_aborts
    assert len(tts.calls) == 1, tts.calls
    assert tts.calls[0].startswith(TRANSCRIPT), tts.calls


async def test_nothing_is_emitted_before_end_of_utterance():
    """纪律一：EOU 之前一个字节都不许出去 —— 投机只算不播。"""
    sched = PipelinedScheduler(
        asr=_converged_asr(),
        llm=EchoLlm(),
        tts=_tts(cpu_ms=50.0),
        metrics=TurnMetrics(clock=SessionClock()),
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.2)

    run = sched._run
    assert run is not None  # 投机确实发生了
    assert run.head  # 产物确实攒在手里
    await sched.on_audio(FRAME)  # 上行帧不吐回任何音频


async def test_interrupt_cancels_inflight_speculation():
    """纪律三：打断时在飞的任务必须被取消，不能继续算没人要的答案。"""
    sched = PipelinedScheduler(
        asr=_converged_asr(),
        llm=EchoLlm(),
        tts=_tts(cpu_ms=400.0),
        metrics=TurnMetrics(clock=SessionClock()),
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.01)  # 去抖在下一个 tick 才起投机
    task = sched._task
    assert task is not None and not task.done()

    sched.cancel()
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    assert sched._task is None


async def test_miss_cancels_inflight_speculation():
    """miss 时在飞的投机任务必须被取消，不能继续烧 CPU/占住模型。

    2026-10-17 复盘发现的缺口：cancel 只有「新 partial 顶旧 / replay 尾部超时 /
    barge-in」三处调用，miss → fallback 路径漏了。后果是作废的生成继续跑到
    max_tokens，并和 fallback 的重新生成抢同一个模型（子进程模式下 _lock 串行，
    fallback 首 token 直接排队等它）。用「投机永远不产出」的 LLM 制造在飞场景。
    """
    llm = GatedLlm()
    asr = SimAsr(TRANSCRIPT, partials=["今天天色", "今天天色", "今天天色"])
    sched = PipelinedScheduler(
        asr=asr,
        llm=llm,
        tts=_tts(cpu_ms=50.0),
        metrics=TurnMetrics(clock=SessionClock()),
        grace_ms=0.0,  # 没就绪立刻判 miss，不等
    )
    await sched.on_audio(FRAME)  # 起投机（基于错误前缀），卡在 gate 上
    await asyncio.sleep(0.05)
    task = sched._task
    assert task is not None and not task.done()

    items, bill = await _drain(sched)
    await asyncio.sleep(0)  # 让 CancelledError 送达投机任务
    said = [s for s in (getattr(i, "text", "") for i in items) if s]

    assert bill["spec_hit"] == 0.0, bill
    # 在飞的投机被砍掉了（而不是留它烧到 max_tokens）
    assert task.cancelled() or llm.cancelled_in_flight
    assert sched.spec_aborts == 1, sched.spec_aborts
    # fallback 仍然用正确全文作答
    assert llm.prompts[-1] == TRANSCRIPT
    assert said[-1] == TRANSCRIPT + "。"


async def test_grace_window_trades_wait_for_hit_rate():
    """grace_ms 是「多等一会儿换命中率」，这个取舍必须能量出来。"""
    for grace_ms, expect_hit in ((0.0, False), (900.0, True)):
        sched = PipelinedScheduler(
            asr=_converged_asr(),
            llm=EchoLlm(),
            tts=_tts(cpu_ms=400.0),
            metrics=TurnMetrics(clock=SessionClock()),
            grace_ms=grace_ms,
        )
        await sched.on_audio(FRAME)
        await asyncio.sleep(0.01)  # 投机在去抖到点后才起
        _, bill = await _drain(sched)
        assert (bill["spec_hit"] == 1.0) is expect_hit, grace_ms
        if expect_hit:
            # 代价写进账单：命中确实多等了
            assert bill["caliber_b_ms"] > 1.0


async def test_pipelined_beats_serial_when_tts_is_slow():
    """同一组 provider：串行在 EOU 后干等 TTS，投机把它挪到了 EOU 之前。"""
    llm = EchoLlm()
    tts = _tts(cpu_ms=300.0)

    serial = SerialScheduler(
        asr=_converged_asr(), llm=llm, tts=tts, metrics=TurnMetrics(clock=SessionClock())
    )
    await asyncio.gather(*[serial.on_audio(FRAME) for _ in range(2)])

    piped = PipelinedScheduler(
        asr=_converged_asr(), llm=llm, tts=tts, metrics=TurnMetrics(clock=SessionClock())
    )
    await piped.on_audio(FRAME)
    await asyncio.sleep(0.6)  # 用户还在沉默等 VAD，这段时间里投机已跑完

    _, serial_bill = await _drain(serial)
    piped_items, piped_bill = await _drain(piped)

    assert piped_bill["caliber_b_ms"] < serial_bill["caliber_b_ms"]
    # 内容必须一样 —— 快不能靠少说话
    for key in ("sentences", "audio_bytes"):
        assert piped_bill[key] == serial_bill[key], key
    assert [i for i in piped_items if isinstance(i, AudioFragment)]


@pytest.mark.parametrize("speculate", [True, False])
async def test_speculation_can_be_switched_off(speculate: bool) -> None:
    """关掉投机必须退化成串行语义，而不是报错 —— 开关是用来做对照实验的。"""
    sched = PipelinedScheduler(
        asr=_converged_asr(),
        llm=EchoLlm(),
        tts=_tts(cpu_ms=0.0),
        metrics=TurnMetrics(clock=SessionClock()),
        speculate=speculate,
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.02)
    items, bill = await _drain(sched)

    assert [i for i in items if isinstance(i, AudioFragment)]
    assert bill["spec_hit"] == float(speculate)


async def test_speculation_llm_span_is_visible() -> None:
    """投机的 LLM 段也要进 span 树（speculative=1，审查第 5 项）——过去只有
    fallback 包 llm.request，命中率低时看不出「CPU 花在注定作废的生成上有多少」。"""
    from voice_pipeline.metrics.trace import Tracer

    metrics = TurnMetrics(clock=SessionClock())
    metrics.tracer = Tracer("spec-spans")
    metrics.tracer.start_root("turn")
    sched = PipelinedScheduler(
        asr=_converged_asr(),
        llm=EchoLlm(),
        tts=_tts(cpu_ms=0.0),
        metrics=metrics,
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.1)

    spans = metrics.tracer.to_list()
    names = [(s["name"], s.get("a.speculative")) for s in spans]
    assert ("llm.request", 1) in names, names
    assert any(n == "llm.ttft" and spec == 1 for n, spec in names), names


async def test_hit_replays_head_while_first_sentence_still_streaming() -> None:
    """ready 的正确语义：第一句**首片**到手即命中，不等整句合成完。

    2026-10-27 真人 pipelined 0/10 命中、全是 not_ready(sentences=1)——等整句
    合成完，EOU（+500ms 静音窗）永远先到。这个测试钉住两件事：
    ① 首片到了 ready 就置位（第二片还卡着也照样命中）；
    ② 回放跟播增长中的 head，迟到的片按序补播，一片不丢。
    """
    release = asyncio.Event()

    class SlowStreamTts(SimTts):
        """首片立即可出，第二片卡到测试放行——模拟整句还没合成完。"""

        tts_streams = True

        async def synth(self, sentence: str) -> AsyncIterator[bytes]:
            yield b"\x01" * 9600
            await release.wait()
            yield b"\x02" * 9600

    sched = PipelinedScheduler(
        asr=_converged_asr(),
        llm=EchoLlm(),
        tts=SlowStreamTts(),
        metrics=TurnMetrics(clock=SessionClock()),
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.1)  # 首片到手；第二片仍卡着（整句未合成完）

    run = sched._run
    assert run is not None
    assert run.ready.is_set(), "首片已到手，ready 必须已置位（旧实现等整句，恒 False）"
    assert len(run.head) == 2, "此刻 head = TtsSentence + 首片"  # noqa: PLR2004

    gen = sched.on_end_of_utterance()
    item = await gen.__anext__()
    assert getattr(item, "type", None) == "turn.start"
    # 推进到第一个音频片：不等整句合成完就必须能出声
    while not isinstance(item, AudioFragment):
        item = await gen.__anext__()
    assert item.data[0:1] == b"\x01"

    release.set()  # 放行第二片
    rest = [x async for x in gen]
    frags = [x for x in rest if isinstance(x, AudioFragment)]
    assert [f.data[0:1] for f in frags] == [b"\x02"], "卡住后的迟到片必须补播，不丢"
    assert sched.spec_hits == 1


async def test_prefix_hit_accepts_short_trailing_particle() -> None:
    """前缀命中：final 只比投机前缀多 ≤2 字（"吗""一下"这类尾词）时算命中。

    sherpa 尾词直到 finalize 才出现、从不进 partial 流——等值校验把这类轮次
    全挡在门外（2026-10-27 晚间真人 4/9 miss 里 3 个是纯尾词延伸）。
    """
    llm = EchoLlm()
    # partial 收敛成缺尾词的前缀，finalize 补出"吗"
    asr = SimAsr("明天要下雨吗", partials=["明天要下雨"] * 5)
    sched = PipelinedScheduler(
        asr=asr, llm=llm, tts=_tts(cpu_ms=50.0), metrics=TurnMetrics(clock=SessionClock())
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.15)
    items, bill = await _drain(sched)
    assert bill["spec_hit"] == 1.0, bill
    assert bill["spec_prefix_hit"] == 1.0, "前缀命中要能在账单里数出来"
    # 答案基于前缀（投机产物原样播），不重算
    assert llm.prompts == ["明天要下雨"], llm.prompts


async def test_prefix_hit_rejects_long_tail() -> None:
    """缺失尾巴超过 2 字不算前缀命中——"……别/不要"这类翻转语义的真尾词要挡住。"""
    asr = SimAsr("明天去北京别迟到", partials=["明天去北京"] * 5)
    sched = PipelinedScheduler(
        asr=asr, llm=EchoLlm(), tts=_tts(cpu_ms=50.0), metrics=TurnMetrics(clock=SessionClock())
    )
    await sched.on_audio(FRAME)
    await asyncio.sleep(0.15)
    items, bill = await _drain(sched)
    assert bill["spec_hit"] == 0.0, bill
    assert bill["spec_miss"] == 1.0
    assert "text_differs" in sched.spec_last_reason
