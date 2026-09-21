"""分片：24kHz / PCM16 / 200ms = 9600 bytes，一字节都不能差。

打断语义以分片为界（未下发的可清，已下发的不可撤回），所以分片必须严格等长。
"""

from __future__ import annotations

from voice_pipeline.audio.chunker import AudioParams, Chunker


def test_chunk_size_matches_design() -> None:
    params = AudioParams()
    assert params.chunk_bytes == 9600
    assert Chunker(params).chunk_bytes == 9600


def test_feed_splits_only_complete_chunks() -> None:
    c = Chunker()
    assert c.feed(b"\x00" * 9599) == []
    chunks = c.feed(b"\x00" * 2)
    assert [len(x) for x in chunks] == [9600]


def test_multiple_chunks_at_once() -> None:
    c = Chunker()
    chunks = c.feed(b"\x07" * (9600 * 3))
    assert [len(x) for x in chunks] == [9600, 9600, 9600]


def test_flush_returns_partial_tail() -> None:
    c = Chunker()
    c.feed(b"\x01" * 100)
    rest = c.flush()
    assert [len(x) for x in rest] == [100]
    assert c.flush() == []


def test_seq_increments_monotonically() -> None:
    c = Chunker()
    assert [c.next_seq() for _ in range(3)] == [1, 2, 3]


def test_params_are_configurable() -> None:
    """转移到\" 参数抠字节 \"：16k/100ms 也要能算对。"""
    params = AudioParams(sample_rate=16000, chunk_ms=100)
    assert params.chunk_bytes == 3200
