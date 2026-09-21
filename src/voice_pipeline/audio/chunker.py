"""音频参数与分片：文章「第三板」的落点——所有参数可配置，字节数可算。

下行固定 24kHz / PCM16 / mono / 200ms 分片（kokoro 原生采样率 24k，省掉重采样）。
每片字节数 = sample_rate × ms / 1000 × 2（PCM16）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioParams:
    sample_rate: int = 24000
    chunk_ms: int = 200
    sample_width: int = 2  # PCM16

    @property
    def chunk_bytes(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000 * self.sample_width


class Chunker:
    """把任意长度的 PCM 字节流切成等长分片，尾部不足一片的由 flush 补齐。

    打断语义依赖分片：清空未下发的分片即可截断语音（评论区「明明砸」的意见），
    因此「已下发 / 未下发」必须以分片为界，已播出不可撤回。
    """

    def __init__(self, params: AudioParams | None = None) -> None:
        self._params = params or AudioParams()
        self._size = self._params.chunk_bytes
        self._buffer = bytearray()
        self._seq = 0

    @property
    def chunk_bytes(self) -> int:
        return self._size

    def feed(self, data: bytes) -> list[bytes]:
        if not data:
            return []
        self._buffer.extend(data)
        chunks: list[bytes] = []
        while len(self._buffer) >= self._size:
            chunks.append(bytes(self._buffer[: self._size]))
            del self._buffer[: self._size]
        return chunks

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def flush(self) -> list[bytes]:
        if not self._buffer:
            return []
        rest = bytes(self._buffer)
        self._buffer.clear()
        return [rest]
