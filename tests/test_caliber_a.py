"""口径A（用户开口 → AI 首音）的两个锚点：`vad_onset` / `asr_first_char`。

为什么单独一个文件：这两个点是 2026-10-27 才埋上的，而**语义偏差比代码更需要钉住**——
`vad_onset` 用的是 barge-in L1 的「连续 3 帧超阈」能量判定，不是 VAD onset，100ms 帧节奏
下比真实开口晚 ~300ms。所以口径A 只作参考、不与口径B 横比；这里钉的是**打点时机正确**
（判停前的首个人声帧、ASR 首次出字），不是「数值准确」。
"""

from __future__ import annotations

import asyncio

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import TurnEnd
from voice_pipeline.server.session import Session

SILENCE = b"\x00" * 960  # 静音帧：RMS≈0，不触发 L1
LOUD = b"\xff\x7f" * 480  # 满幅帧：RMS≈1，连续 3 帧即判开口（去抖窗口）


def _session(partials: list[str] | None = None) -> Session:
    return Session(
        mode="serial",
        asr=SimAsr("今天天气怎么样", partials=partials),
        llm=SimLlm(reply="今天晴。"),
        tts=SimTts(chunk_per_sentence=2),
    )


async def _run_turn_to_end(sess: Session) -> TurnEnd:
    task = asyncio.create_task(sess.run_turn())
    while True:
        item = await asyncio.wait_for(sess.outbox.get(), timeout=5)
        if isinstance(item, TurnEnd):
            await task
            return item


async def test_vad_onset_marks_first_speech_frame() -> None:
    sess = _session()
    await sess.on_audio(SILENCE)
    assert sess.metrics.elapsed("vad_onset") is None, "静音帧不该记开口"
    # 去抖：第 3 帧才算开口，前两帧不许抢先打点（早打点 = 口径A 凭空变大）
    await sess.on_audio(LOUD)
    assert sess.metrics.elapsed("vad_onset") is None
    await sess.on_audio(LOUD)
    assert sess.metrics.elapsed("vad_onset") is None
    await sess.on_audio(LOUD)
    assert sess.metrics.elapsed("vad_onset") is not None


async def test_asr_first_char_marks_when_partial_appears() -> None:
    """sim 链路（没挂 EOU）走 `_note_first_char`：未出字不记，出字那一帧才记。"""
    sess = _session(partials=["", "", "今天"])
    await sess.on_audio(SILENCE)
    assert sess.metrics.elapsed("asr_first_char") is None
    await sess.on_audio(SILENCE)
    assert sess.metrics.elapsed("asr_first_char") is None
    await sess.on_audio(SILENCE)
    assert sess.metrics.elapsed("asr_first_char") is not None


async def test_caliber_a_appears_in_bill() -> None:
    """埋点之后账单里必须出现口径A：`first_audio_out − vad_onset`，缺失则写 -1。"""
    sess = _session()
    for _ in range(3):
        await sess.on_audio(LOUD)
    bill = (await _run_turn_to_end(sess)).metrics
    assert bill["vad_onset"] > 0
    assert bill["caliber_a_ms"] >= 0
    # 容差 0.01ms：bill 里的点是各自 round(3) 后的值，caliber_a 是未 round 之差
    assert abs(bill["caliber_a_ms"] - (bill["first_audio_out"] - bill["vad_onset"])) < 0.01


async def test_vad_onset_not_rewritten_by_post_eou_energy() -> None:
    """判停之后的上行能量（回声残余 / 用户插话）不是「这一句的开口」。

    mark() 本身只写首次到达，但 utterance_end 之后若没有这层守门，AI 播放期间的能量
    会把下一轮还没 reset 的账单起点改写——口径A 就不再是用户那一句。
    """
    sess = _session()
    for _ in range(3):
        await sess.on_audio(LOUD)
    await _run_turn_to_end(sess)
    before = sess.metrics.elapsed("vad_onset")
    assert before is not None
    for _ in range(3):
        await sess.on_audio(LOUD)  # 判停已成立：utterance_end 在账上
    assert sess.metrics.elapsed("vad_onset") == before
