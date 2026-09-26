"""provider 协议接口：ASR / LLM / TTS / VAD 可插拔的契约。

real 与 sim 两套实现共用这些协议，A/B 实验只换实现不换调用方。

关于 `*_blocking` 的说明（M4 引入）：
    sherpa-onnx 的推理是**同步 CPU 密集**调用，直接在 asyncio 事件循环里跑会阻塞
    整个 loop——单连接时看不出来，一旦有第二个会话、或要在合成期间继续推音频就废了。
    因此协议显式区分：协程版本给调用方，`*_blocking` 版本给 `asyncio.to_thread`。
    两者都必须是契约的一部分，否则某个实现偷偷写回阻塞形式也没有机制拦住。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class AsrProvider(Protocol):
    async def push_audio(self, chunk: bytes) -> None: ...
    async def finalize(self) -> str: ...

    def text(self) -> str:
        """当前 partial 结果（未 finalize）。投机调度靠它判断何时能提前开跑。"""
        ...

    def reset(self) -> None:
        """开一条新识别流，供下一轮使用。

        finalize() 会终结当前流（sherpa 的 input_finished），之后到达的音频若不
        reset 就进了死流——EOU 后用户马上再开口的那几个字会整个丢掉。调度器
        必须在 finalize 之后调用本方法。
        """
        ...


@runtime_checkable
class LlmProvider(Protocol):
    def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        """history 是**之前轮次**的 [user/assistant] 消息，由 Session 持有并传入。

        None 或空表示首轮。实现可以忽略它（SimLlm），但不许改写调用方传入的列表。
        """
        ...


@runtime_checkable
class TtsProvider(Protocol):
    def synth(self, sentence: str) -> AsyncIterator[bytes]: ...

    def synth_blocking(self, sentence: str) -> bytes:
        """同步合成整句，返回完整 PCM 字节。

        由调用方决定是否 `asyncio.to_thread(...)` 卸载；本方法本身**不得**是协程，
        否则「卸到线程池」这条路就断了——而那是 sherpa-onnx 唯一能不阻塞 loop 的走法。
        """
        ...
