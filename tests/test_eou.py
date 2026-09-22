"""EOU 判停：四路融合 + 三档门控 + 抢话/漏判指标，全零模型零网络。

合成信号约定：16k / PCM16；语音帧用 220Hz 正弦（amp 0.3 → 归一化 RMS≈0.21），
静音帧全 0。检测器的能量门限（0.01）远低于语音、高于静音，分离干净。
"""

from __future__ import annotations

import asyncio
import math

from voice_pipeline.core.eou import (
    FusedEou,
    GatingMode,
    ProsodyEou,
    SemanticHeuristicEou,
    VadSilenceEou,
)
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.session import Session

SR = 16000


def _frame(ms: float, amp: float = 0.3, sr: int = SR) -> bytes:
    n = int(ms / 1000 * sr)
    out = bytearray()
    for i in range(n):
        s = int(amp * 32767 * math.sin(2 * math.pi * 220 * i / sr))
        out += s.to_bytes(2, "little", signed=True)
    return bytes(out)


def _silence(ms: float, sr: int = SR) -> bytes:
    return _frame(ms, amp=0.0, sr=sr)


def _speech(ms: float, amp: float = 0.3) -> bytes:
    return _frame(ms, amp=amp)


def test_l1_silence_triggers_after_hangover():
    """L0+L1：有人声 + 静音达到 min_silence 才判停。"""
    det = VadSilenceEou(sample_rate=SR, min_silence_ms=300.0, min_speech_ms=100.0)
    for _ in range(3):  # 300ms 语音
        det.push_audio(_speech(100))
    assert not det.should_end(), "说话中不应判停"
    det.push_audio(_silence(200))
    assert not det.should_end(), "静音未到窗"
    det.push_audio(_silence(120))
    assert det.should_end(), "静音越过窗应判停"


def test_semantic_ends_earlier_than_conservative():
    """SEMANTIC 档靠韵律/语义确认，比 CONSERVATIVE（等完整静音窗）更早判停。"""
    # 语音尾部能量下行 → 触发 L2 终顿
    speech = [_speech(100, 0.30), _speech(100, 0.15), _speech(100, 0.05)]
    short_silence = 200.0  # 短于 CONSERVATIVE 的 500ms 窗

    semantic = FusedEou(gating=GatingMode.SEMANTIC, min_silence_short_ms=150.0)
    conservative = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=500.0)

    for f in speech:
        semantic.push_audio(f)
        conservative.push_audio(f)
    for _ in range(int(short_silence // 50)):
        semantic.push_audio(_silence(50))
        conservative.push_audio(_silence(50))

    assert semantic.should_end(), "SEMANTIC 应在短静音+终顿即判停"
    assert not conservative.should_end(), "CONSERVATIVE 应继续等完整静音窗"


def test_prosody_distinguishes_terminal_from_flat():
    """L2：能量下行（终顿）判完，平稳能量（未完）不判完。"""
    terminal = ProsodyEou(sample_rate=SR)
    for amp in (0.30, 0.15, 0.05):  # 明显下行
        terminal.push_audio(_speech(100, amp))
    assert terminal.is_terminal(), "能量下行应判为句末终顿"

    flat = ProsodyEou(sample_rate=SR)
    for _ in range(3):  # 平稳能量
        flat.push_audio(_speech(100, 0.30))
    assert not flat.is_terminal(), "平稳能量不应判为终顿（用户还在说）"


def test_semantic_heuristic_completeness():
    """L3：partial 稳定且够长 → 判为完整；仍在校正 → 不判完整。"""
    done = SemanticHeuristicEou(sample_rate=SR, stable_ms=200.0)
    done.update_text("今天天气怎么样")  # 长度 7，无标点但够长
    for _ in range(3):  # 稳定 300ms
        done.push_audio(_silence(100))
    assert done.is_complete(), "稳定的较长 partial 应判完整"

    editing = SemanticHeuristicEou(sample_rate=SR, stable_ms=200.0)
    for i in range(1, 5):  # 持续变化
        editing.update_text("今天天气怎么"[:i])
        editing.push_audio(_silence(60))
    assert not editing.is_complete(), "仍在校正的 partial 不应判完整"


def test_premature_eou_flagged_when_speech_resumes():
    """抢话指标：判停决定后、窗口内又开口 → premature_eou=True。"""
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=300.0)
    for _ in range(3):
        det.push_audio(_speech(100))
    for _ in range(3):  # 300ms 静音 → 判停
        det.push_audio(_silence(100))
    assert det.should_end()
    assert not det.premature_eou

    det.push_audio(_speech(100))  # 窗口内又开口
    assert det.premature_eou, "判停后窗口内又开口应记为抢话"


def test_missed_eou_watchdog():
    """漏判看门狗：长时间无有效人声也未判停 → missed_eou=True。"""
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=500.0, max_wait_ms=400.0)
    for _ in range(6):  # 只喂静音，越过 max_wait
        det.push_audio(_silence(100))
    assert not det.should_end(), "无人声不应判停"
    assert det.missed_eou, "越过看门狗应记漏判"


def test_session_auto_end_with_eou():
    """Session 接入 EOU：real 风格下 pending_end 在短静音后为真，finish 账单含门控档。"""
    asr: AsrProvider = SimAsr("今天天气怎么样", partials=["今天天气怎么样"])
    sess = Session(
        mode="serial",
        asr=asr,
        llm=SimLlm(),
        tts=SimTts(),
        gating=GatingMode.SEMANTIC,
        eou=FusedEou(gating=GatingMode.SEMANTIC, min_silence_short_ms=150.0),
    )
    for _ in range(3):
        asyncio.run(sess.on_audio(_speech(100)))  # 语音；partial 立即收敛为完整句
    for _ in range(4):
        asyncio.run(sess.on_audio(_silence(100)))  # 静音 + 文本稳定 → SEMANTIC 判停

    assert sess.pending_end(), "EOU 应已判定用户说完"

    async def _drain() -> dict[str, float]:
        from voice_pipeline.server.protocol import TurnEnd

        async for item in sess.finish():
            if isinstance(item, TurnEnd):
                return dict(item.metrics)
        return {}

    bill = asyncio.run(_drain())
    assert bill.get("eou_mode") == 1.0, bill  # 1 = semantic
    assert bill.get("premature_eou") == 0.0, bill


def test_semantic_latches_terminal_instead_of_falling_back_to_watchdog():
    """回归：SEMANTIC 档曾因两个 250ms 互斥而**永远不成立**。

    `ProsodyEou.coda_ms=250` 会在静音超过 250ms 后主动清掉终顿，
    而 SEMANTIC 档要求静音 ≥250ms 才去读终顿 —— 帧粒度 100ms 下这两个条件
    永远采不到交集，于是退化成「等满 2000ms 看门狗」（实测判停延迟 1804ms）。

    终顿是**事件**不是状态，必须锁存。这条断言防的是有人把锁存改回瞬时读取。
    """
    eou = FusedEou(gating=GatingMode.SEMANTIC)
    for _ in range(4):  # 稳态语音，建立能量峰值
        eou.push_audio(_speech(100, 0.30))
    for amp in (0.15, 0.10, 0.05):  # 尾部能量下行 → L2 终顿
        eou.push_audio(_speech(100, amp))
    for _ in range(3):  # 静音 300ms
        eou.push_audio(_silence(100))

    assert eou.should_end(), "终顿后静音 300ms 应判停"
    assert eou.eou_latency_ms() < 500, (
        f"判停延迟 {eou.eou_latency_ms()}ms —— 退化到 2000ms 看门狗了"
    )


def test_semantic_without_terminal_does_not_fire_on_silence_alone():
    """锁存不能退化成「只要静音够久就判停」——那样 SEMANTIC 就失去了安全来源。

    没有终顿、也没有语义完整时，300ms 静音不该判停。
    """
    eou = FusedEou(gating=GatingMode.SEMANTIC)
    for _ in range(6):
        eou.push_audio(_speech(100, 0.30))  # 恒定能量，无下行 → 不产生终顿
    for _ in range(3):
        eou.push_audio(_silence(100))

    assert not eou.should_end(), "仅凭 300ms 静音（无终顿/无完整语义）不应判停"
