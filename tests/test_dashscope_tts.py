"""DashScopeTts：零网络测试 SDK 桥接纪律。

真 SDK 是同步回调式（合成跑在自己的线程），这里的假件模拟「另一个线程上的
回调」——重点钉住三件事：
1. 流式路径：on_data 的 chunk 逐个透出、None 哨兵终止、on_error 抛错；
2. 线程→loop 必须走 call_soon_threadsafe（假件直接在 worker 线程里回调）；
3. 缺 key 且未注入工厂时响亮失败。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from voice_pipeline.providers.tts.dashscope_tts import API_KEY_ENV, DashScopeTts


class FakeSynthesizer:
    """模拟 SDK：.call(text) 在被调用线程上依次触发回调。chunks 控制分几片。

    `open_delay`：on_open 之前的休眠——真实 SDK 也是先建连（on_open）再出数据，
    phase 拆解（建连 vs 服务端合成）要靠这个顺序才能被测出来。
    """

    def __init__(
        self, callback: object, chunks: list[bytes], delay: float = 0.0, open_delay: float = 0.0
    ) -> None:
        self._cb = callback
        self._chunks = chunks
        self._delay = delay
        self._open_delay = open_delay
        self.calls: list[str] = []

    def call(self, text: str) -> bytes:
        import time

        self.calls.append(text)
        time.sleep(self._open_delay)
        self._cb.on_open()  # type: ignore[attr-defined]
        for chunk in self._chunks:
            time.sleep(self._delay)
            self._cb.on_data(chunk)  # type: ignore[attr-defined]
        self._cb.on_complete()  # type: ignore[attr-defined]
        return b"".join(self._chunks)


def _factory(
    chunks: list[bytes], delay: float = 0.0, open_delay: float = 0.0
) -> tuple[Callable[[object], FakeSynthesizer], list[FakeSynthesizer]]:
    made: list[FakeSynthesizer] = []

    def factory(callback: object) -> FakeSynthesizer:
        synth = FakeSynthesizer(callback, chunks, delay, open_delay)
        made.append(synth)
        return synth

    return factory, made


async def test_synth_yields_chunks_in_order() -> None:
    chunks = [b"\x01" * 4800, b"\x02" * 4800, b"\x03" * 100]
    factory, made = _factory(chunks)
    tts = DashScopeTts(synth_factory=factory)

    got = [c async for c in tts.synth("一句话。")]
    assert got == chunks
    assert made[0].calls == ["一句话。"]


async def test_phase_marks_split_connect_from_server_synthesis() -> None:
    """深拆首片的两段都要有证据：`submit→open`（建连）与 `open→first_data`（合成）。

    假件按真实 SDK 的顺序回调（先 on_open 再出数据）。这两段是 2026-10-17
    深拆时新加的格子：云 TTS 首片 ~600ms 到底花在建连还是服务端，之前没人知道。
    """
    factory, _ = _factory([b"a" * 100], delay=0.05, open_delay=0.03)
    tts = DashScopeTts(synth_factory=factory)

    got = [c async for c in tts.synth("句")]
    assert got
    marks = tts.last_phases
    assert {"construct", "submit", "open", "first_data", "complete"} <= set(marks)
    assert marks["construct"] <= marks["submit"] <= marks["open"] <= marks["first_data"]
    # 假件的休眠就是下界：建连 ≥30ms、出数据前再 ≥50ms
    assert (marks["open"] - marks["submit"]) * 1000 >= 25
    assert (marks["first_data"] - marks["open"]) * 1000 >= 45


def test_synth_blocking_also_records_phases() -> None:
    """同步路径同样留 phase 证据（探针走它，调度器走流式那条）。"""
    factory, _ = _factory([b"ab", b"cd"], open_delay=0.01)
    tts = DashScopeTts(synth_factory=factory)
    assert tts.synth_blocking("hi") == b"abcd"
    assert {"submit", "open", "first_data"} <= set(tts.last_phases)


async def test_synth_propagates_sdk_error() -> None:
    class ErrorSynth:
        def __init__(self, cb: object) -> None:
            self._cb = cb

        def call(self, text: str) -> bytes:
            self._cb.on_error("boom")  # type: ignore[attr-defined]
            return b""

    tts = DashScopeTts(synth_factory=lambda cb: ErrorSynth(cb))
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in tts.synth("x"):
            pass


def test_synth_blocking_concatenates() -> None:
    chunks = [b"ab", b"cd"]
    factory, _ = _factory(chunks)
    tts = DashScopeTts(synth_factory=factory)
    assert tts.synth_blocking("hi") == b"abcd"


def test_missing_key_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        DashScopeTts()


async def test_streaming_tts_feeds_scheduler_incrementally() -> None:
    """调度器分流：tts_streams=True 的 provider 边合成边出固定尺寸分片。"""
    from voice_pipeline.audio.chunker import Chunker
    from voice_pipeline.core.scheduler.base import synth_to_fragments
    from voice_pipeline.providers.tts.sim_tts import SimTts

    # 3 片 × 9600B（200ms@24k PCM16）+ 尾巴 100B —— 流式路径应出 3 整片 + 尾片
    chunks = [bytes([i]) * 9600 for i in (1, 2, 3)] + [b"\xff" * 100]
    factory, _ = _factory(chunks)
    streaming_tts = DashScopeTts(synth_factory=factory)

    got = [b async for b in synth_to_fragments(streaming_tts, "句", Chunker())]
    assert [len(b) for b in got] == [9600, 9600, 9600, 100]

    # 整句路径（SimTts 未声明 tts_streams）行为不变：整句合成再切片
    blocking = SimTts(chunk_per_sentence=1)
    got2 = [b async for b in synth_to_fragments(blocking, "句", Chunker())]
    assert all(len(b) == 9600 for b in got2)


def test_thread_callback_crosses_into_loop_safely() -> None:
    """回调来自 worker 线程：桥必须经 call_soon_threadsafe 投递（跨线程直接
    put_nowait 是未定义行为）。用真线程跑一轮验证不炸。"""
    import asyncio

    chunks = [b"x" * 100]

    def factory(callback: object) -> FakeSynthesizer:
        return FakeSynthesizer(callback, chunks)

    tts = DashScopeTts(synth_factory=factory)

    async def go() -> list[bytes]:
        return [c async for c in tts.synth("t")]

    got = asyncio.run(go())
    assert got == chunks
