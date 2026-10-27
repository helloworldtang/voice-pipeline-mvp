"""M1 serial 全链：印证「在串行前提下质疑方的算术完全成立」。

这条链路是 A/B 实验的对照组。它故意不投机——EOU 之后才启动 LLM，
所以实测出来的数字应当落在评论区预言的 1.2~1.9s 量级（real 模式下验证）。
"""

from __future__ import annotations

from voice_pipeline.core.scheduler.base import AudioFragment
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import AsrFinal, TtsSentence, TurnEnd, TurnStart
from voice_pipeline.server.session import Session

TRANSCRIPT = "上海今天多少度"
REPLY = "今天晴，气温 22 度。适合出门。"


def build() -> Session:
    return Session(
        mode="serial",
        asr=SimAsr(transcript=TRANSCRIPT),
        llm=SimLlm(reply=REPLY),
        tts=SimTts(chunk_per_sentence=2),
    )


async def serial_run() -> tuple[list[str], dict[str, float], list[int]]:
    """跑一轮，返回（事件类型序列, 终账单, 音频分片大小）。"""
    session = build()
    await session.on_audio(b"\x00" * 3200)

    types: list[str] = []
    bill: dict[str, float] = {}
    sizes: list[int] = []
    async for item in session.finish():
        if isinstance(item, AudioFragment):
            sizes.append(len(item.data))
            continue
        types.append(item.type)
        if isinstance(item, TurnEnd):
            bill = item.metrics
    return types, bill, sizes


async def test_event_order_matches_design() -> None:
    types, _, _ = await serial_run()
    assert types[0] == TurnStart().type
    assert types[1] == AsrFinal(text="").type
    assert types[-1] == TurnEnd(metrics={}).type
    assert types.count(TtsSentence(seq=1, text="").type) == 2  # 两句


async def test_transcript_is_scripted() -> None:
    session = build()
    await session.on_audio(b"\x00" * 3200)
    async for item in session.finish():
        if isinstance(item, AsrFinal):
            assert item.text == TRANSCRIPT


async def test_sentences_are_split_before_tts() -> None:
    session = build()
    await session.on_audio(b"\x00" * 3200)
    sentences: list[str] = []
    async for item in session.finish():
        if isinstance(item, TtsSentence):
            sentences.append(item.text)
    assert sentences == ["今天晴，气温 22 度。", "适合出门。"]


async def test_audio_fragments_are_fixed_size() -> None:
    _, _, sizes = await serial_run()
    assert sizes
    assert set(sizes) == {9600}  # 24k × 200ms × PCM16


async def test_bill_carries_caliber_b_and_waterfall() -> None:
    _, bill, _ = await serial_run()
    assert bill["sentences"] == 2.0
    assert bill["caliber_b_ms"] >= 0
    for point in ("utterance_end", "llm_first_token", "first_sentence_done", "tts_first_chunk"):
        assert point in bill
    # first_audio_out 必须晚于 utterance_end（口径B 的定义）
    assert bill["first_audio_out"] >= bill["utterance_end"]


async def test_short_sentence_tail_is_not_dropped() -> None:
    """回归：句长不足一片时，残余音频曾永久滞留 buffer——首音打不上，口径B 记 -1。

    用 8k/200ms 的 TTS 造出 3200B/句（< 9600B）的音频：修之前这一轮一个音频片都不会出。
    """
    session = Session(
        mode="serial",
        asr=SimAsr(transcript=TRANSCRIPT),
        llm=SimLlm(reply=REPLY),
        tts=SimTts(sample_rate=8000, chunk_ms=200, chunk_per_sentence=1),
    )
    await session.on_audio(b"\x00" * 3200)

    sizes: list[int] = []
    bill: dict[str, float] = {}
    async for item in session.finish():
        if isinstance(item, AudioFragment):
            sizes.append(len(item.data))
        elif isinstance(item, TurnEnd):
            bill = item.metrics
    assert sizes == [3200, 3200]  # 每句音频在合成完的当拍就全部外发
    assert bill["first_audio_out"] >= bill["utterance_end"]
    assert bill["caliber_b_ms"] >= 0


async def test_tts_failure_on_one_sentence_does_not_kill_turn() -> None:
    """单句 TTS 失败只跳该句，不炸整轮（2026-10-17 验收 turn 12：裸标点句被
    DashScope 拒 → RuntimeError 一路上炸 → "Task exception was never retrieved"，
    用户一个字听不到）。第二句必须照播、轮次要正常收账。"""
    from collections.abc import AsyncIterator

    class PickyTts(SimTts):
        tts_streams = True

        def __init__(self) -> None:
            super().__init__()
            self.failed: list[str] = []

        async def synth(self, sentence: str) -> AsyncIterator[bytes]:
            if "坏句" in sentence:
                self.failed.append(sentence)
                msg = f'DashScope TTS 合成失败: InvalidParameter ("{sentence}")'
                raise RuntimeError(msg)
            async for chunk in super().synth(sentence):
                yield chunk

    tts = PickyTts()
    session = Session(
        mode="serial",
        asr=SimAsr(transcript=TRANSCRIPT),
        llm=SimLlm(reply="坏句。好句，念得出。"),
        tts=tts,
    )
    await session.on_audio(b"\x00" * 3200)
    sizes: list[int] = []
    bill: dict[str, float] = {}
    async for item in session.finish():
        if isinstance(item, AudioFragment):
            sizes.append(len(item.data))
        elif isinstance(item, TurnEnd):
            bill = item.metrics

    assert tts.failed == ["坏句。"], tts.failed  # 失败句确实试过（而非没送到）
    assert bill["tts_skipped_sentences"] == 1.0, "跳句必须进账单（审查第 8 项）"
    assert sizes, "好句必须照播，不能整轮静音"
    assert bill["audio_bytes"] > 0, bill
    assert bill["caliber_b_ms"] > 0, "首句坏但第二句出了声，口径B 仍要记账"


async def test_echo_mode_still_echoes() -> None:
    """echo 是 M0 的回归基线：立即回发，不走链路。"""
    session = Session(mode="echo")
    fragments = await session.on_audio(b"\x01\x02")
    assert [f.data for f in fragments] == [b"\x01\x02"]


async def test_echo_mode_finish_produces_nothing() -> None:
    session = Session(mode="echo")
    assert [item async for item in session.finish()] == []


def test_missing_vad_onset_leaves_caliber_a_absent() -> None:
    """sim 链路没有 VAD，口径A 如实缺失，不猜不填。"""
    m = TurnMetrics(clock=SessionClock())
    m.mark("utterance_end")
    m.mark("first_audio_out")
    bill = m.bill()
    assert "caliber_a_ms" not in bill
    assert bill["caliber_b_ms"] >= 0


def test_bill_missing_uturance_reports_negative() -> None:
    m = TurnMetrics(clock=SessionClock())
    m.mark("first_audio_out")
    assert m.bill()["caliber_b_ms"] == -1.0
