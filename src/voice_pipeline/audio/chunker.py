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

    def __init__(self, params: AudioParams | None = None, *, first_min_ms: int = 0) -> None:
        """`first_min_ms`：**首片**只要求这么多音频即可下发（0 = 与常规片同长）。

        为什么要有这个：分片尺寸固定 200ms，而流式 TTS 的首包往往只有 40~90ms
        （讯飞实测）。等凑满 200ms 才发首片，等于把 TTS 省下的时间又原样吐回去——
        首片 89ms 被拖成 ~290ms。首片短一点，客户端按到达顺序播，听感无差别。
        """
        self._params = params or AudioParams()
        self._size = self._params.chunk_bytes
        self._first_size = (
            max(1, self._params.sample_rate * first_min_ms // 1000 * self._params.sample_width)
            if first_min_ms > 0
            else self._size
        )
        self._first_pending = True
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
        while True:
            take = self._first_size if self._first_pending else self._size
            if len(self._buffer) < take:
                return chunks
            chunks.append(bytes(self._buffer[:take]))
            del self._buffer[:take]
            self._first_pending = False

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def feed_and_flush(self, data: bytes) -> list[bytes]:
        """喂入一段**语义完整**的音频（一句 TTS 的结果）并立刻补齐尾部残余。

        为什么不能只 feed：句中不足一片的残余会滞留到下一句才凑齐，于是
        ① 若首句音频 < 200ms，`first_audio_out` 永远打不上，口径B 记成 -1；
        ② 最后一句的残余在下一句到来前不会下发，句尾被静默截断。
        补齐后每句的音频在该句合成完的瞬间就全部外发，barge-in 也能在句末干净截断。
        """
        return self.feed(data) + self.flush()

    def flush(self) -> list[bytes]:
        if not self._buffer:
            return []
        rest = bytes(self._buffer)
        self._buffer.clear()
        return [rest]
