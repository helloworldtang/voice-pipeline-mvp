"""确定性 ASR 替身：不看音频内容，按注入序列吐出 partial。

用途：CI 里零模型零网络跑通全链路逻辑，并可注入**纠错序列**来触发投机作废路径（M4）——
真实语音之所以难做投机，正是因为 partial 会在后一帧被修正。
"""

from __future__ import annotations

from voice_pipeline.providers.base import AsrProvider


class SimAsr:
    def __init__(
        self, transcript: str = "今天天气怎么样", partials: list[str] | None = None
    ) -> None:
        """transcript 是 finalize 的结果；partials 决定每次 push 后 `text()` 返回什么。

        不传 partials 时默认逐步吐出 transcript 的前缀，模拟「边说边出词」。
        """
        self._transcript = transcript
        self._partials = partials if partials is not None else list(transcript)
        self.frames = 0

    async def push_audio(self, chunk: bytes) -> None:
        self.frames += 1

    def text(self) -> str:
        """每调用一次前进一格，模拟 ASR 逐步吐词。"""
        if not self._partials:
            return ""
        idx = min(self.frames, len(self._partials)) - 1
        return self._partials[max(idx, 0)] if self.frames else ""

    async def finalize(self) -> str:
        return self._transcript

    def reset(self) -> None:
        """开新轮：partial 回放从头开始（frames 清零），transcript 保持不变。"""
        self.frames = 0


def build(transcript: str = "今天天气怎么样") -> AsrProvider:
    return SimAsr(transcript)
