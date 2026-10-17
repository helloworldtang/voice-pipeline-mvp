"""ASR 前置能量门（P1）：静音帧不推 ASR，省口径B 窗口里的循环竞争。

两层验证：
- **门本身**（纯逻辑，零依赖）：滞后语义——进静音要慢（连续 N 帧），出静音要快
  （一帧人声立即恢复）；计数器诚实。
- **接线**（Session 白盒）：real 模式静音帧不进 ASR 但**全进 EOU**（静音是判停的
  原料，门只拦 ASR 一条路）；sim 模式门不装（SimAsr 的 text() 按 push 前进，
  装门会把确定性回放变成非确定性，且没有可省的 CPU）；--no-energy-gate 对照臂。

阈值方向的不变式（追踪文档 §六 P1 行「别互相抵消」）：门限 ≤ L1 门限——
L1 认定的人声必须能进 ASR。反过来不成立也行不通：直接用 L1 的 0.02 会把
RMS≈0.018 的轻声整段拦掉，识别直接失败。
"""

from __future__ import annotations

import asyncio

import pytest

from voice_pipeline.core.barge_in import BargeInDetector
from voice_pipeline.core.energy_gate import AsrEnergyGate
from voice_pipeline.core.eou import ENERGY_THRESHOLD
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.session import Session


def _frame(amp: int, n: int = 320) -> bytes:
    """定幅 PCM16 小端帧。amp/32768 ≈ RMS：2000→0.061（正常说话）、590→0.018（轻声）、0→静音。"""
    return (amp.to_bytes(2, "little", signed=True)) * n


SPEECH = _frame(2000)  # RMS≈0.061：任意门限都算人声
QUIET = _frame(590)  # RMS≈0.018：过 EOU 人声阈(0.01)、不过 L1 阈(0.02)
SILENCE = _frame(0)


def _drain(session: Session) -> list[object]:
    """测试里没有 pump 协程消费 outbox，直接排空检查内容（同 test_bargein）。"""
    items: list[object] = []
    while not session.outbox.empty():
        items.append(session.outbox.get_nowait())
    return items


# ---- 门本身 ------------------------------------------------------------------


async def test_silence_blocked_only_after_debounce() -> None:
    """进静音要慢：前 2 帧静音照常推（句中停顿/换气不误伤），第 3 帧起才拦。"""
    gate = AsrEnergyGate(frames=3)
    assert [gate.allows(SILENCE) for _ in range(5)] == [True, True, False, False, False]
    assert gate.is_silent is True
    assert (gate.passed, gate.blocked) == (2, 3)


async def test_speech_exits_silence_immediately() -> None:
    """出静音要快：确认静音后任意一帧人声立即恢复推流（丢语音比多推一帧贵）。"""
    gate = AsrEnergyGate(frames=3)
    for _ in range(3):
        gate.allows(SILENCE)
    assert gate.allows(SPEECH) is True
    assert gate.is_silent is False
    # 恢复后重新计数：又得连续 3 帧静音才再拦
    assert [gate.allows(SILENCE) for _ in range(3)] == [True, True, False]


async def test_mid_utterance_pause_keeps_audio_continuous() -> None:
    """说-停-说：两帧停顿不足以确认静音，三段语音全部进 ASR（无音频空洞）。"""
    gate = AsrEnergyGate(frames=3)
    fed = [gate.allows(c) for c in (SPEECH, SPEECH, SILENCE, SILENCE, SPEECH, SPEECH)]
    assert fed == [True] * 6
    assert gate.blocked == 0


async def test_reset_clears_silence_state() -> None:
    """reset 清静音确认状态（计数是会话级诊断量，不清）。

    用计数器证明状态真被清了：没清的话 reset 后第一帧就该被拦（blocked 直接 +1），
    而不是重新攒够 3 帧。
    """
    gate = AsrEnergyGate(frames=3)
    for _ in range(3):
        gate.allows(SILENCE)
    assert gate.blocked == 1  # 已确认静音并拦过 1 帧
    gate.reset()
    assert [gate.allows(SILENCE) for _ in range(3)] == [True, True, False]
    assert (gate.passed, gate.blocked) == (4, 2)  # 计数保留：2过1拦 + 2过1拦


async def test_quiet_speech_never_blocked() -> None:
    """轻声（RMS≈0.018）必须能进 ASR——这是门限不取 L1 的 0.02 的全部理由。"""
    gate = AsrEnergyGate()  # 默认阈 0.01
    assert [gate.allows(QUIET) for _ in range(10)] == [True] * 10
    assert gate.blocked == 0


async def test_threshold_never_above_l1() -> None:
    """不变式：门限 ≤ L1 门限——L1 认定的人声一定要能进 ASR（不互相抵消）。

    同时钉住默认值就是 EOU 人声阈：本仓对「有人声」的基准定义只有一个，
    不发明第三个数字。
    """
    gate = AsrEnergyGate()
    l1 = BargeInDetector()
    assert gate.threshold <= l1.threshold
    assert gate.threshold == ENERGY_THRESHOLD == 0.01
    # L1 级别的帧（满幅附近）必然过门
    assert gate.allows(b"\xff\x7f" * 480) is True


# ---- 接线（Session 白盒）-----------------------------------------------------


class CountingAsr(SimAsr):
    """数 push 次数的 ASR 替身。text/finalize 沿用 SimAsr，门只影响 push 一路。"""

    def __init__(self) -> None:
        super().__init__("今天天气怎么样")
        self.pushes = 0

    async def push_audio(self, chunk: bytes) -> None:
        self.pushes += 1
        await super().push_audio(chunk)


class CountingEou:
    """记收帧数的判停器替身：验证门只拦 ASR、不拦 EOU。"""

    def __init__(self) -> None:
        self.frames = 0

    def push_audio(self, chunk: bytes) -> None:
        self.frames += 1

    def update_text(self, partial: str) -> None:
        pass

    def should_end(self) -> bool:
        return False

    def unlatch(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def silence_ms(self) -> float:
        return 0.0

    def speech_ms(self) -> float:
        return 0.0


async def test_real_mode_gate_blocks_silence_from_asr_but_not_eou() -> None:
    """real 模式：静音帧不进 ASR（攒满 3 帧才拦），但每一帧都进 EOU。"""
    asr, eou = CountingAsr(), CountingEou()
    session = Session(mode="real-serial", asr=asr, llm=SimLlm(), tts=SimTts(), eou=eou)
    assert session._energy_gate is not None, "real 模式必须装门"

    for _ in range(3):
        await session.on_audio(SPEECH)
    for _ in range(5):
        await session.on_audio(SILENCE)

    # ASR：3 帧语音 + 前 2 帧静音（去抖未满）= 5；后 3 帧被拦
    assert asr.pushes == 5
    # EOU：8 帧全收——静音正是它判「说完没有」的原料
    assert eou.frames == 8
    gate = session._energy_gate
    assert gate is not None and (gate.passed, gate.blocked) == (5, 3)


async def test_real_mode_quiet_speech_still_reaches_asr() -> None:
    """real 模式 + 轻声（0.018）：一帧不拦——直接用 L1 的 0.02 会整段丢语音。"""
    asr = CountingAsr()
    session = Session(mode="real-serial", asr=asr, llm=SimLlm(), tts=SimTts())
    for _ in range(10):
        await session.on_audio(QUIET)
    assert asr.pushes == 10


async def test_sim_mode_has_no_gate() -> None:
    """sim 模式不装门：SimAsr.text() 按 push 前进，装门会破坏确定性回放。"""
    asr = CountingAsr()
    session = Session(mode="serial", asr=asr, llm=SimLlm(), tts=SimTts())
    assert session._energy_gate is None
    for _ in range(5):
        await session.on_audio(SILENCE)
    assert asr.pushes == 5  # 一帧不拦


async def test_gate_off_flag_is_control_arm() -> None:
    """energy_gate=False（探针对照臂）：real 模式也不装门，行为回到加门前。"""
    asr = CountingAsr()
    session = Session(mode="real-serial", asr=asr, llm=SimLlm(), tts=SimTts(), energy_gate=False)
    assert session._energy_gate is None
    for _ in range(5):
        await session.on_audio(SILENCE)
    assert asr.pushes == 5


async def test_gated_frames_are_visible_in_trace() -> None:
    """被拦的帧在 span 上打 gated=1——A/B 与线上账单都数得出来（不靠口头声明）。

    断言走 `to_list()` 的**序列化形态**：那正是落进账单 JSONL、`report_turns
    --spans` 读得到的样子。
    """
    session = Session(mode="real-serial", asr=CountingAsr(), llm=SimLlm(), tts=SimTts())
    for _ in range(5):
        await session.on_audio(SILENCE)
    tracer = session.metrics.tracer
    assert tracer is not None
    frames = [s for s in tracer.to_list() if s["name"] == "uplink.frame"]
    assert len(frames) == 5
    gated = [s for s in frames if s.get("a.gated") == 1]
    assert len(gated) == 3  # 前 2 帧去抖未满，后 3 帧被拦
    assert all(s.get("a.bytes") for s in frames)  # 原属性不被覆盖


async def test_l1_debounce_is_three_frames_not_two() -> None:
    """L1 去抖回归：一帧只判一次人声——修双调用前，2 帧满幅就触发打断。"""
    from voice_pipeline.core.turn_fsm import TurnState

    session = Session(
        mode="real-serial",
        asr=SimAsr("今天天气怎么样", partials=["今天天气怎么样"] * 5),
        llm=SimLlm(),
        tts=SimTts(),
    )
    session._fsm.state = TurnState.SPEAKING  # 白盒推到播放中，避开真起一轮
    loud = b"\xff\x7f" * 480  # 满幅：RMS≈1

    # 逐帧喂到打断为止，记录第几帧触发：恰好 3 帧（文档语义）。修双调用前
    # 是 2 帧——is_speech 一帧被调两次，_above 双倍计数。
    fired_after = -1
    for i in range(1, 4):
        await session.on_audio(loud)
        if session.interrupted:
            fired_after = i
            break
    assert fired_after == 3, f"第 {fired_after} 帧就触发打断——去抖失效（应为 3 帧）"
    assert session.metrics.barge_in_count == 1


async def test_echo_mode_unaffected() -> None:
    """echo 模式（无 ASR 链路）加门不炸，回发语义不变。"""
    session = Session(mode="echo")
    assert session._energy_gate is None
    out = await session.on_audio(SILENCE)
    assert out and out[0].data == SILENCE


@pytest.mark.parametrize("amp", [0, 590, 2000])
async def test_gate_and_l1_never_contradict(amp: int) -> None:
    """扫描三个能量级：L1 判定与门的判定不打架——L1 说人声的帧门一定放行。"""
    gate = AsrEnergyGate()
    l1 = BargeInDetector()
    chunk = _frame(amp)
    l1_speech = l1.is_speech(chunk)
    gate_allows = gate.allows(chunk)
    if l1_speech:
        assert gate_allows, f"amp={amp}：L1 认定人声却被门拦下——互相抵消"
    # 反向不要求（门比 L1 敏感是设计使然：丢语音比误打断贵）


async def test_run_turn_still_works_with_gate() -> None:
    """门开着跑完整一轮：判停→ASR finalize→LLM→TTS 全链路不塌（回归网）。"""
    from voice_pipeline.server.protocol import AsrFinal

    asr = SimAsr("今天天气怎么样", partials=["今天天气怎么样"] * 5)
    session = Session(mode="real-serial", asr=asr, llm=SimLlm(), tts=SimTts(cpu_ms=0.0))
    for _ in range(3):
        await session.on_audio(SPEECH)
    for _ in range(4):
        await session.on_audio(SILENCE)
    # 手动驱动收尾（real 模式由传输层轮询 pending_end，这里直接起轮）
    task = asyncio.create_task(session.run_turn())
    await asyncio.wait_for(task, timeout=5.0)
    finals = [item for item in _drain(session) if isinstance(item, AsrFinal)]
    assert finals and finals[0].text == "今天天气怎么样"
    bill = session.metrics.bill()
    assert bill["sentences"] >= 1.0
    assert bill["audio_bytes"] > 0
    assert bill["caliber_b_ms"] >= 0
