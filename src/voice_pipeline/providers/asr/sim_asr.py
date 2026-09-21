"""确定性 ASR 替身：不看音频内容，finalize 返回脚本化转写。

用途：CI 里零模型零网络跑通全链路逻辑，并可注入纠偏序列触发作废路径（M4）。
"""

from __future__ import annotations

from voice_pipeline.providers.base import AsrProvider


class SimAsr:
    def __init__(self, transcript: str = "今天天气怎么样") -> None:
        self._transcript = transcript
        self.frames = 0

    async def push_audio(self, chunk: bytes) -> None:
        self.frames += 1

    async def finalize(self) -> str:
        return self._transcript


def build(transcript: str = "今天天气怎么样") -> AsrProvider:
    return SimAsr(transcript)
