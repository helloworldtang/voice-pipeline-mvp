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

from collections.abc import AsyncIterator

from voice_pipeline.providers.tts.sim_tts import SimTts
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
