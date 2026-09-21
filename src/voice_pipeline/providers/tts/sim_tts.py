"""确定性 TTS 替身：产出静音 PCM16 分片，字节数与时长可预期（用于时序断言）。"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from voice_pipeline.providers.base import TtsProvider


class SimTts:
    """默认 `cpu_ms=0`：CI 里时间零成本，时序断言只看结构不看耗时。

    要模拟真实合成开销时传 cpu_ms——它会在 synth_blocking 里同步 sleep，
    因此**必须**经线程池卸载，否则跟真实实现的行为不一致（会堵住事件循环）。
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_ms: int = 200,
        chunk_per_sentence: int = 3,
        cpu_ms: float = 0.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._chunk_per_sentence = chunk_per_sentence
        self._cpu_ms = cpu_ms
        self.calls: list[str] = []

    def chunk_bytes(self) -> bytes:
        samples = self._sample_rate * self._chunk_ms // 1000
        return b"\x00" * (samples * 2)  # PCM16 = 2 bytes/sample

    def synth_blocking(self, sentence: str) -> bytes:
        self.calls.append(sentence)
        if self._cpu_ms:
            time.sleep(self._cpu_ms / 1000.0)
        return self.chunk_bytes() * self._chunk_per_sentence

    async def synth(self, sentence: str) -> AsyncIterator[bytes]:
        block = self.synth_blocking(sentence)
        step = len(block) // self._chunk_per_sentence
        for off in range(0, len(block), step):
            yield block[off : off + step]


def build(sample_rate: int = 24000, chunk_ms: int = 200) -> TtsProvider:
    return SimTts(sample_rate=sample_rate, chunk_ms=chunk_ms)
