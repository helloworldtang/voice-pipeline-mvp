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


def test_missed_eou_on_fallback_cap():
    """漏判的唯一信号：静音拖过兜底上限（档位阈值没拦住）。

    `min_silence_ms` 故意设成 5s：档位阈值永远够不到，静音涨到
    `max_silence_ms` 兜底才判停 → missed_eou=True。
    （2026-10-28 复盘 83cc545：原看门狗「elapsed>max_wait 只打标记」含整个
    说话时长，用户说得慢就被误标——8/15 个 500ms 正常判停全被标漏判。）
    """
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=5000.0)
    for _ in range(2):
        det.push_audio(_speech(100))  # 开过口——漏判以此为门槛
    for _ in range(25):  # 2500ms 静音：越过 2000ms 兜底，够不到 5000ms 档位
        det.push_audio(_silence(100))
    assert det.should_end(), "兜底上限必须能收尾（否则链路卡死）"
    assert det.missed_eou, "说过话却只能靠兜底收尾 = 漏判"


def test_slow_speech_with_timely_stop_is_not_missed_eou():
    """说得慢不等于漏判（2026-10-28 真人验收 8/15 假漏判的根因）：
    整窗 elapsed 远超 max_wait，但静音一到档位阈值就正常判停 → 不记漏判。
    """
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=500.0, max_wait_ms=1000.0)
    # 说 6s（含 <500ms 的小停顿——静音计数被话音重置，档位阈值够不到）
    for _ in range(10):
        det.push_audio(_speech(300))
        det.push_audio(_silence(300))
    for _ in range(5):  # 500ms 静音 → 档位阈值正常判停
        det.push_audio(_silence(100))
    assert det.should_end()
    assert not det.missed_eou, "判停按档位阈值及时给出，elapsed 大不是漏判"


def test_waiting_for_speech_is_not_missed_eou():
    """反向自测（真人验收 7/10 假漏判的根因）：**等人开口不是漏判**。

    旧语义「一直没等到人声也算漏判」把用户照剧本读数秒的等待全记成漏判，
    而每一轮的转写其实都拿到了——一个恒定虚高的指标比缺指标更危险。
    """
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=500.0, max_wait_ms=400.0)
    for _ in range(6):  # 只有静音：用户还没开口
        det.push_audio(_silence(100))
    assert not det.should_end(), "无人声不应判停"
    assert not det.missed_eou, "等用户开口的时间不该记成判停器的漏判"


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


def test_conservative_survives_intra_sentence_pause() -> None:
    """conservative 档在 350ms 句内停顿不得判停 —— 这是静音窗档的保底性质。

    依据 `scripts/probe_premature.py`（句内停顿 + 真 ASR partial，n=5 句）：
      conservative 停顿 250/350ms → 抢话 0/5、1/5（延迟 495ms）
      semantic     停顿 250/350ms → 抢话 1/5、2/5
    语义档会在这里立刻判停（静音 250ms + 终顿锁存），换成它就红。

    注意默认档现在是 neural（D2），所以这里显式构造 conservative ——「默认档是哪一档」
    由 test_protocol.py 钉，这里钉的是「这一档该有什么性质」。
    """
    eou = FusedEou(gating=GatingMode.CONSERVATIVE)
    for amp in (0.30, 0.15, 0.05):  # 尾部能量下行 → L2 终顿会成立
        eou.push_audio(_speech(100, amp))
    for _ in range(3):  # 350ms 句内停顿
        eou.push_audio(_silence(100))
    assert not eou.should_end(), "350ms 句内停顿就判停 = 抢话"
    eou.push_audio(_silence(200))  # 累计 550ms ≥ 500ms 阈值
    assert eou.should_end(), "真说完（静音 ≥500ms）应判停"


def test_premature_window_survives_reset():
    """抢话观察窗横跨 reset：run_turn 采完指标立即 reset，判停后的人声大多落在
    reset 之后——reset 把 _decided 清掉曾经让 premature_eou 恒为 0（系统性漏报，
    真链路验收 2026-09-27 §5 抓到）。"""
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=300.0)
    for _ in range(3):
        det.push_audio(_speech(100))
    for _ in range(3):
        det.push_audio(_silence(100))
    assert det.should_end()
    det.reset()  # Session.run_turn 在判停后立即做这件事

    det.push_audio(_silence(100))  # 静音继续走窗口（帧时长计）
    det.push_audio(_speech(100))  # 判停后 ~200ms 又开口 → 在 300ms 窗内
    assert det.premature_eou, "reset 后窗口内的人声必须仍能记为抢话"


def test_premature_window_expires_across_reset():
    """窗口过期后就不再记：判停半秒后才开口是新一轮说话，不是抢话。"""
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=300.0)
    for _ in range(3):
        det.push_audio(_speech(100))
    for _ in range(3):
        det.push_audio(_silence(100))
    assert det.should_end()
    det.reset()
    for _ in range(4):  # 400ms > 300ms 窗口，耗尽残余
        det.push_audio(_silence(100))
    det.push_audio(_speech(100))
    assert not det.premature_eou


def test_take_premature_reads_once_and_clears():
    """读后即清：上一轮的幽灵标志不许污染下一轮的账。"""
    det = FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=300.0)
    for _ in range(3):
        det.push_audio(_speech(100))
    for _ in range(3):
        det.push_audio(_silence(100))
    assert det.should_end()
    det.push_audio(_speech(100))
    assert det.take_premature() is True
    assert det.take_premature() is False  # 已清


# ─────────────────────────────────────────────────────────────────────────────
# §三③ 判停尾部最差体感：尾音噪声毛刺清零静音窗 → 拖到 max_silence 看门狗
# ─────────────────────────────────────────────────────────────────────────────


def test_intermittent_noise_used_to_trigger_watchdog():
    """§三③ 根因复现（旧阈值 0.01）：尾音噪声毛刺每帧清零静音 → 500ms 攒不到 →
    拖到 max_silence 看门狗（真人账 2000/2100ms）。用 silence_reset_threshold=0.01
    模拟修复前，防回退。"""
    old = FusedEou(
        gating=GatingMode.CONSERVATIVE,
        silence=VadSilenceEou(sample_rate=SR, min_silence_ms=500.0, silence_reset_threshold=0.01),
        max_silence_ms=2000.0,
    )
    for _ in range(3):
        old.push_audio(_speech(100))
    for _ in range(5):  # 仅 1s：旧行为下噪声毛刺每帧重置 → 静音永远 <100ms
        old.push_audio(_frame(100, amp=0.021))  # 噪声毛刺 RMS≈0.015
        old.push_audio(_silence(100))
    assert not old.should_end(), "旧阈值下噪声毛刺应使 500ms 窗永远攒不到"


def test_intermittent_noise_fixed():
    """§三③ 修复后（默认 0.02）：噪声毛刺是中性帧，不清零静音 → 500ms 保守阈值正常触发，
    不拖到看门狗。"""
    new = FusedEou(gating=GatingMode.CONSERVATIVE, max_silence_ms=2000.0)
    for _ in range(3):
        new.push_audio(_speech(100))
    for _ in range(5):  # 1s：修复后静音稳定累积 → 500ms 即判停
        new.push_audio(_frame(100, amp=0.021))
        new.push_audio(_silence(100))
    assert new.should_end(), "修复后噪声毛刺不应拖住静音窗"


def test_silence_reset_threshold_default_respects_invariant():
    """不动点：清零静音的阈值（0.02）必须 ≥ 标记有人声的阈值（0.01），否则比 L1 barge-in
    还灵敏，会与打断层判定打架。0.015 的帧是中性帧——标记有人声、但不清零静音；
    清晰语音（RMS 0.21）仍清零。"""
    det = VadSilenceEou(sample_rate=SR)
    assert det.silence_reset_threshold >= det.energy_threshold
    det.push_audio(_speech(100))  # 真实语音 100ms，清零静音
    det.push_audio(_frame(100, amp=0.021))  # 噪声毛刺 RMS≈0.015
    det.push_audio(_silence(100))  # 静音 +100ms
    assert det.silence_ms() >= 100.0, "噪声毛刺不应清零静音计时"
    det.push_audio(_silence(100))  # 再 +100 → 200
    det.push_audio(_speech(100))  # 清晰语音应清零
    assert det.silence_ms() == 0.0, "清晰语音必须清零静音计时"
