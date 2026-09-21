"""双队列铁律：控制帧永不被音频积压阻塞 + 有界队列背压可见。"""

from __future__ import annotations

from voice_pipeline.server.queues import PRIO_AUDIO, PRIO_CONTROL, DualPriorityQueue


async def test_control_beats_backlogged_audio() -> None:
    q = DualPriorityQueue(max_audio_frames=100)
    for i in range(50):
        q.put_audio(bytes([i]))
    q.put_control("interrupt")

    first = await q.get()
    assert first.priority == PRIO_CONTROL
    assert first.payload == "interrupt"

    second = await q.get()
    assert second.priority == PRIO_AUDIO


async def test_audio_is_fifo() -> None:
    q = DualPriorityQueue()
    q.put_audio(b"a")
    q.put_audio(b"b")
    assert (await q.get()).payload == b"a"
    assert (await q.get()).payload == b"b"


async def test_audio_bound_drops_oldest_and_counts() -> None:
    q = DualPriorityQueue(max_audio_frames=2)
    for i in range(5):
        q.put_audio(bytes([i]))

    assert q.dropped_audio == 3  # 5 进 2 容量 → 丢 3
    assert q.qsize() == (0, 2)
    assert (await q.get()).payload == b"\x03"  # 最旧的被丢，留下最后两个
    assert (await q.get()).payload == b"\x04"


async def test_control_queue_is_unbounded() -> None:
    q = DualPriorityQueue(max_audio_frames=1)
    for _ in range(100):
        q.put_control("interrupt")
    assert q.qsize()[0] == 100
    assert q.dropped_audio == 0
