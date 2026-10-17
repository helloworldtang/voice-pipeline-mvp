"""预热接线：判停/建会话那一刻，TTS 连接必须在后台连上（P0.5 的调用侧纪律）。

池本身的行为在 tests/test_tts_pool.py 白盒钉住；这里只钉「谁在什么时候调用
prewarm()」——调用晚于合成启动，建连那 ~260ms 就进口径B 了，等于没优化。
断言一律用**事件顺序**（prewarm 与 synth 的先后），不比跨时钟的绝对时间。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.metrics.events import TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.tts.sim_tts import SimTts

TRANSCRIPT = "今天天气怎么样"


class RecordingTts(SimTts):
    """带 prewarm() 的 TTS 替身：按发生顺序记下 prewarm/synth 事件。

    记在 synth_blocking 上——SimTts 没有 tts_streams，调度器走整句路径。
    """

    def __init__(self) -> None:
        super().__init__(cpu_ms=1.0)
        self.events: list[str] = []

    def prewarm(self) -> None:
        self.events.append("prewarm")

    def synth_blocking(self, sentence: str) -> bytes:
        self.events.append(f"synth:{sentence}")
        return super().synth_blocking(sentence)


class EchoLlm:
    def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        async def producer() -> AsyncIterator[str]:
            yield prompt + "。"

        return producer()


async def test_serial_prewarms_before_any_synth() -> None:
    """serial：判停一刻就预热——发生在第一次合成之前，建连窗口覆盖整个 LLM 往返。"""
    tts = RecordingTts()
    sched = SerialScheduler(asr=SimAsr(TRANSCRIPT), llm=EchoLlm(), tts=tts, metrics=TurnMetrics())
    async for _item in sched.on_end_of_utterance():
        pass

    assert "prewarm" in tts.events
    assert tts.events.index("prewarm") < tts.events.index(f"synth:{TRANSCRIPT}。")
    assert tts.events.count("prewarm") == 1  # 一次就够（幂等）


async def test_pipelined_prewarms_at_speculation_start_before_first_synth() -> None:
    """pipelined：用户还在说（partial 收敛）就预热，且早于投机合成的第一句。"""
    tts = RecordingTts()
    sched = PipelinedScheduler(
        asr=SimAsr(TRANSCRIPT, partials=[TRANSCRIPT] * 5),
        llm=EchoLlm(),
        tts=tts,
        metrics=TurnMetrics(),
    )
    await sched.on_audio(b"\x00" * 960)  # 触发 partial → 启动投机（debounce=0）
    await asyncio.sleep(0.05)  # 让投机启动回调跑掉

    assert tts.events, "说话期间一次预热都没有——建连成本全落进口径B"
    assert tts.events[0] == "prewarm"
    assert tts.events.index("prewarm") < tts.events.index(f"synth:{TRANSCRIPT}。")

    async for _item in sched.on_end_of_utterance():
        pass
    # 判停再兜底一次（幂等；覆盖没走投机路径的轮次）
    assert tts.events.count("prewarm") >= 2


async def test_session_prewarms_on_construction() -> None:
    """Session 建立即预热：首轮第一句也不必付建连费（provider 没有 prewarm 时静默跳过）。"""
    from voice_pipeline.server.session import Session

    tts = RecordingTts()
    session = Session(mode="serial", asr=SimAsr(TRANSCRIPT), llm=EchoLlm(), tts=tts)
    assert tts.events == ["prewarm"]  # 建会话即预热
    assert session._scheduler is not None

    # echo 模式（无调度器/无 TTS 参与）也不能炸
    echo = Session(mode="echo")
    assert echo._scheduler is None
