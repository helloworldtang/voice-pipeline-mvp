"""确定性 TTS 替身：产出静音 PCM16 分片，字节数与时长可预期（用于时序断言）。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from voice_pipeline.providers.base import TtsProvider


class SimTts:
    def __init__(
        self, sample_rate: int = 24000, chunk_ms: int = 200, chunk_per_sentence: int = 3
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._chunk_per_sentence = chunk_per_sentence

    def chunk_bytes(self) -> bytes:
        samples = self._sample_rate * self._chunk_ms // 1000
        return b"\x00" * (samples * 2)  # PCM16 = 2 bytes/sample

    async def synth(self, sentence: str) -> AsyncIterator[bytes]:
        chunk = self.chunk_bytes()
        for _ in range(self._chunk_per_sentence):
            yield chunk


def build(sample_rate: int = 24000, chunk_ms: int = 200) -> TtsProvider:
    return SimTts(sample_rate=sample_rate, chunk_ms=chunk_ms)
