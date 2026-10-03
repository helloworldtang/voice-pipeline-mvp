"""多轮闭环：ASR reset 与 history 的联合回归网。

第 2 轮能不能正常工作，取决于两件互锁的事：

1. **finalize 之后 ASR 被 reset**——sherpa 的 `input_finished()` 会终结识别流，
   不 reset 的话第 2 轮音频全部进死流，转写为空。这个 bug 在单轮测试里永远
   看不出来（单轮恰好不需要第二条流）。
2. **第 1 轮的 user/assistant 真的进了第 2 轮的 LLM messages**——history 是
   Session 攒的、调度器传的，任何一环断了都表现为第 2 轮失忆。

所以这里用一个「不 reset 就说不出话」的假 ASR + 记录调用的 spy LLM，
把两件事钉在同一条测试里。
"""

from __future__ import annotations

import struct
from collections.abc import AsyncIterator

from voice_pipeline.core.eou import FusedEou, GatingMode
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import TurnEnd
from voice_pipeline.server.session import MAX_HISTORY_MESSAGES, Session


class DeadWithoutResetAsr:
    """模拟 sherpa 流的生命周期：finalize 终结流，reset 才开下一条。

    死流语义刻意从严：finalize 后再 finalize 返回空串（而不是重复上一轮的
    转写）——如果调度器忘了 reset，测试会以「第 2 轮 prompt 为空」爆出来，
    而不是拿旧转写蒙混过关。
    """

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self._turn = 0
        self._alive = True

    async def push_audio(self, chunk: bytes) -> None:
        pass

    def text(self) -> str:
        return self._transcripts[self._turn] if self._alive else ""

    async def finalize(self) -> str:
        if not self._alive:
            return ""
        self._alive = False
        return self._transcripts[self._turn]

    def reset(self) -> None:
        self._turn += 1
        self._alive = True


class SpyLlm:
    """记录每次 astream 收到的 prompt 与 history 快照。"""

    def __init__(self, reply: str = "这是回答。") -> None:
        self._reply = reply
        self.calls: list[tuple[str, list[dict[str, str]] | None]] = []

    async def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        self.calls.append((prompt, list(history) if history else None))
        yield self._reply


def _session(asr: DeadWithoutResetAsr, llm: SpyLlm) -> Session:
    return Session(mode="serial", asr=asr, llm=llm, tts=SimTts())


async def test_second_turn_gets_fresh_transcript_and_history() -> None:
    asr = DeadWithoutResetAsr(["第一句问话", "第二句问话"])
    llm = SpyLlm()
    session = _session(asr, llm)

    await session.on_audio(b"\x00" * 320)
    await session.run_turn()
    assert llm.calls[0][0] == "第一句问话"
    assert llm.calls[0][1] is None  # 首轮无历史

    await session.on_audio(b"\x00" * 320)
    await session.run_turn()
    prompt, history = llm.calls[1]
    # reset 没接上的话，这里会是 ""（音频进了 finalize 过的死流）
    assert prompt == "第二句问话"
    assert history == [
        {"role": "user", "content": "第一句问话"},
        {"role": "assistant", "content": "这是回答。"},
    ]


async def test_history_is_capped_to_last_ten_turns() -> None:
    """超过 10 轮后从头整对裁——语音对话时效性强，更早的轮次只会白付 token。"""
    turns = 12
    asr = DeadWithoutResetAsr([f"第{i}句问话" for i in range(turns)])
    llm = SpyLlm()
    session = _session(asr, llm)

    for _ in range(turns):
        await session.on_audio(b"\x00" * 320)
        await session.run_turn()

    prompt, history = llm.calls[-1]
    assert prompt == "第11句问话"
    assert history is not None
    assert len(history) == MAX_HISTORY_MESSAGES
    # LLM 调用发生在本轮入史之前：第 12 轮看到的是第 1~11 轮（22 条）裁完的
    # 最近 20 条，头部是第 2 轮的 user——整对裁，从 user 开头。
    assert history[0] == {"role": "user", "content": "第1句问话"}
    assert history[-1] == {"role": "assistant", "content": "这是回答。"}


async def test_each_turn_gets_a_fresh_bill() -> None:
    """每轮一本新账：mark 只记首次到达，跨轮不清账的话第 2 轮复读第 1 轮。

    真链路两轮实测抓到 caliber_b_ms 一字不差（probe_multi_turn 可复现）。
    判据：会话钟单调，第 2 轮的 utterance_end 打点必须严格晚于第 1 轮。
    """
    asr = DeadWithoutResetAsr(["第一句问话", "第二句问话"])
    llm = SpyLlm()
    session = _session(asr, llm)

    bills: list[dict[str, float]] = []
    for _ in range(2):
        await session.on_audio(b"\x00" * 320)
        await session.run_turn()
        while not session.outbox.empty():
            item = session.outbox.get_nowait()
            if isinstance(item, TurnEnd):
                bills.append(dict(item.metrics))

    assert len(bills) == 2
    assert bills[0]["caliber_b_ms"] >= 0
    assert bills[1]["caliber_b_ms"] >= 0
    assert bills[1]["utterance_end"] > bills[0]["utterance_end"]


class EmptyTextAsr:
    """回声形态的 ASR：VAD 听见人声（音频能量够），却认不出任何字（text() 恒空）。"""

    def __init__(self) -> None:
        self.frames = 0

    async def push_audio(self, chunk: bytes) -> None:
        self.frames += 1

    def text(self) -> str:
        return ""

    async def finalize(self) -> str:
        return ""

    def reset(self) -> None:
        self.frames = 0


async def test_echo_loop_gated_by_empty_transcript() -> None:
    """回声自激回环的根治门：真人验收实测一句"我想睡"引来 6 连答，账单
    后续轮 user_text 全空——EOU 在「有能量、无文字」的回声音频上判停，
    AI 作答的声音又回灌麦克风。判停必须被空转写门拦下并重置。
    """
    llm = SpyLlm()
    session = Session(
        mode="real-serial",
        asr=EmptyTextAsr(),
        llm=llm,
        tts=SimTts(),
        eou=FusedEou(gating=GatingMode.CONSERVATIVE, min_silence_ms=100.0),
    )
    speech = _speech_frame(600)  # RMS≈0.018：过 EOU 人声阈(0.01)、不过打断阈(0.02)
    silence = _speech_frame(0)

    # 「AI 的回声」：有人声能量的音频 + 静音尾巴 → EOU 事实上会判停
    for _ in range(3):
        await session.on_audio(speech)
    for _ in range(2):
        await session.on_audio(silence)
    assert session._eou is not None and session._eou.should_end(), "前置：EOU 确已判停"
    # 门生效：pending_end 必须拦下并重置判停器，轮次不起
    assert session.pending_end() is False
    assert isinstance(session._eou, FusedEou)
    assert not session._eou._decided, "判停器应已解闩（保留 VAD 状态，见 unlatch）"
    await session.run_turn()  # 即使被外部驱动，空转写也不进 LLM
    assert llm.calls == [], "空转写绝不该触发 LLM——那只会生成更多声音喂回麦克风"

    # 放行上限：ASR 永远不出字（极端回声）时，门在 ~EMPTY_GATE_MAX_BLOCKS 次
    # 后放行——链路不许卡死在"永远判不了停"；放行的轮由第二道门兜底（不出音频）。
    session._empty_blocks = 0
    fired = 0
    for _ in range(30):
        await session.on_audio(speech)
        await session.on_audio(silence)
        fired += 1
        if session.pending_end():
            break
    assert fired <= 15, f"门拦了 {fired} 次还没放行——上限失效会卡死链路"
    llm.calls.clear()
    await session.run_turn()
    assert llm.calls == [], "放行的空转写轮同样不进 LLM（回环仍是断的）"


async def test_empty_transcript_skips_llm_but_bills() -> None:
    """第二道门：finalize 空转写 → 账单照出（证据留痕），LLM/TTS 不跑。"""
    from voice_pipeline.server.protocol import TurnEnd

    llm = SpyLlm()
    asr = DeadWithoutResetAsr([""])  # 合法空转写
    session = _session(asr, llm)
    await session.on_audio(b"\x00" * 320)
    await session.run_turn()

    assert llm.calls == []
    items = []
    while not session.outbox.empty():
        items.append(session.outbox.get_nowait())
    types = [getattr(i, "type", "audio") for i in items]
    assert "asr.final" in types
    assert types[-1] == TurnEnd(metrics={}).type, "空转写轮也要有账单（证据留痕）"


def _speech_frame(amp: int, ms: int = 100, sr: int = 16000) -> bytes:
    return struct.pack("<" + "h" * (sr * ms // 1000), *([amp] * (sr * ms // 1000)))
