"""确定性 LLM 替身：按标点切块吐字，模拟流式首 token 与分句节奏。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from voice_pipeline.providers.base import LlmProvider

_SPLITTERS = "。！？；\n"


class SimLlm:
    def __init__(self, reply: str = "今天晴，气温 22 度。适合出门。") -> None:
        self._reply = reply

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        buffer = ""
        for char in self._reply:
            buffer += char
            if char in _SPLITTERS:
                yield buffer
                buffer = ""
        if buffer:
            yield buffer


def build(reply: str = "今天晴，气温 22 度。适合出门。") -> LlmProvider:
    return SimLlm(reply)
