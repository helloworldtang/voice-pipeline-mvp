"""barge-in 三层 + 状态机：零模型零网络的竞态专项。

这些测试验证的不是「能出声」，而是「被打断时系统的每个角落都干净」——这是语音对话
系统最经典的竞态来源，也是「能真正交流」里「不抢话、被打断就停」那一端。

三层：
- L1 检测   ：on_audio 在 SPEAKING/THINKING 时检测到用户又开口（能量去抖）
- L2 播放层 ：cancel 在飞计算 + 往 outbox 投 AudioStop，让播放端停播
- L3 语义层 ：本轮作废，状态机回到 LISTENING，等用户插话说完再答（不重开一轮）

注意 SPEAKING 是个会随播放推进的状态：用多句回复 + 慢合成让它在测试里持续数百毫秒，
否则只回一句时首音即整轮结束，SPEAKING 是微秒级瞬态，轮询不可能命中。
"""

from __future__ import annotations

import asyncio

import pytest

from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.turn_fsm import IllegalTransition, TurnFsm, TurnState
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import AudioStop
from voice_pipeline.server.session import Session

TRANSCRIPT = "今天天气怎么样"
FRAME = b"\x00" * 960  # 静音帧：RMS≈0，不触发 barge-in
LOUD = b"\xff\x7f" * 480  # 满幅帧（小端 0x7fff=32767）：RMS≈1，3 帧连续即判开口


def _tts(cpu_ms: float) -> SimTts:
    return SimTts(cpu_ms=cpu_ms)


def _converged_asr() -> SimAsr:
    return SimAsr(TRANSCRIPT, partials=[TRANSCRIPT] * 5)


async def _await_state(session: Session, state: TurnState, timeout: float = 2.0) -> bool:
    step = 0.005
    elapsed = 0.0
    while elapsed < timeout:
        if session._fsm.state == state:
            return True
        await asyncio.sleep(step)
        elapsed += step
    return False


def _drain_outbox(sess: Session) -> list[object]:
    """测试里没有 pump 协程消费 outbox，直接排空检查内容。"""
    items: list[object] = []
    while not sess.outbox.empty():
        items.append(sess.outbox.get_nowait())
    return items


# ---- FSM 本身 ----------------------------------------------------------------


def test_fsm_illegal_transition_raises_in_strict() -> None:
    fsm = TurnFsm(strict=True)
    with pytest.raises(IllegalTransition):
        fsm.move(TurnState.SPEAKING)  # IDLE 不能直接到 SPEAKING


def test_fsm_legal_path_listening_thinking_speaking() -> None:
    fsm = TurnFsm(strict=True)
    assert fsm.move(TurnState.LISTENING)
    assert fsm.move(TurnState.THINKING)
    assert fsm.move(TurnState.SPEAKING)
    assert fsm.move(TurnState.LISTENING)  # 播完回听


# ---- 三层 barge-in（白盒驱动到 SPEAKING 再打断）-----------------------------


async def test_barge_in_requires_speaking_or_thinking() -> None:
    """L1 前置：用户在「听」或「已打断」时不应触发打断。"""
    sess = Session(mode="pipelined", asr=_converged_asr(), llm=SimLlm(), tts=_tts(0.0), eou=None)
    # 刚建好：IDLE / LISTENING，没在说
    assert sess._fsm.state in (TurnState.IDLE, TurnState.LISTENING)
    assert sess.barge_in() is False
    assert sess.interrupted is False


async def test_barge_in_during_speaking_emits_audio_stop_and_cancels() -> None:
    """SPEAKING 中被打断：L2 取消在飞 + AudioStop 进 outbox；L3 回 LISTENING。"""
    sess = Session(
        mode="pipelined",
        asr=_converged_asr(),
        llm=SimLlm(),  # 多句回复：让 SPEAKING 持续足够长，轮询才命中
        tts=_tts(300.0),  # 慢合成：确保打断发生在首句还在算/播时
        eou=None,
    )
    await sess.on_audio(FRAME)  # 触发投机
    task = asyncio.create_task(sess.run_turn())
    assert await _await_state(sess, TurnState.SPEAKING)

    assert sess.barge_in() is True
    assert sess.interrupted is True
    assert sess.metrics.barge_in_count == 1
    # L2：在飞任务被取消，不允许悬挂（pipelined 专有 _task）
    sched = sess._scheduler
    assert isinstance(sched, PipelinedScheduler)
    assert sched._task is None or sched._task.done() or sched._task.cancelled()
    # L3：回到听
    assert sess._fsm.state is TurnState.LISTENING

    await asyncio.wait_for(task, timeout=2.0)
    # outbox 里一定投过 AudioStop（播放端据此停播）
    stops = [it for it in _drain_outbox(sess) if isinstance(it, AudioStop)]
    assert stops, "打断必须让播放端收到 AudioStop"


async def test_barge_in_during_thinking_cancels_inflight() -> None:
    """THINKING（AI 在算还没出声）被打断：同样作废，不能把旧答案吐出。"""
    sess = Session(mode="pipelined", asr=_converged_asr(), llm=SimLlm(), tts=_tts(0.0), eou=None)
    # 直接把状态机推到 THINKING（模拟 run_turn 已进入一轮、首音之前）
    sess._fsm.state = TurnState.THINKING
    assert sess.barge_in() is True
    assert sess.interrupted is True
    assert sess._fsm.state is TurnState.LISTENING
    assert sess.metrics.barge_in_count == 1


async def test_on_audio_l1_detects_speech_and_barge_in() -> None:
    """L1：AI 播放期间用户又开口（连续高能量帧）→ on_audio 自动触发 barge-in。"""
    sess = Session(
        mode="pipelined",
        asr=_converged_asr(),
        llm=SimLlm(),
        tts=_tts(300.0),
        eou=None,
    )
    await sess.on_audio(FRAME)
    task = asyncio.create_task(sess.run_turn())
    assert await _await_state(sess, TurnState.SPEAKING)

    # 用户插话：连续 3 帧满幅（去抖窗口），模拟「哎你等下」
    for _ in range(3):
        await sess.on_audio(LOUD)

    assert sess.interrupted is True
    assert sess.metrics.barge_in_count == 1
    assert sess._fsm.state is TurnState.LISTENING
    await asyncio.wait_for(task, timeout=2.0)


async def test_no_in_flight_task_after_barge_in() -> None:
    """打断后 run_turn 必须干净收尾，不能留下悬挂协程把 CPU 烧给没人要的答案。"""
    sess = Session(
        mode="pipelined",
        asr=_converged_asr(),
        llm=SimLlm(),
        tts=_tts(300.0),
        eou=None,
    )
    await sess.on_audio(FRAME)
    task = asyncio.create_task(sess.run_turn())
    assert await _await_state(sess, TurnState.SPEAKING)
    sess.barge_in()
    # run_turn 应迅速结束（被打断立即停止产出）
    await asyncio.wait_for(task, timeout=2.0)
    assert task.done()
    assert sess._turn_running is False
