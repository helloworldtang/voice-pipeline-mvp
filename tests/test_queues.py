"""下行队列铁律：杀停帧跳队 + 其余保序 + 音频有界丢旧且成对丢弃。"""

from __future__ import annotations

from voice_pipeline.server.queues import OutboundQueue


async def test_kill_frame_beats_backlogged_audio() -> None:
    """audio.stop 晚到也要先到——这是「叫不停」问题的全部解法。"""
    q = OutboundQueue(max_audio_frames=100)
    for i in range(50):
        q.put_audio(f'{{"seq":{i}}}', bytes([i]))
    q.put_event('{"type":"turn.end"}')
    q.put_kill('{"type":"audio.stop"}')

    first = await q.get()
    assert first.text == '{"type":"audio.stop"}'
    # 杀停之后回到主道原顺序
    second = await q.get()
    assert second.data == b"\x00"


async def test_events_and_audio_stay_strictly_ordered() -> None:
    """turn.end / session.closed 不跳队：账单先于音频到达会打断收流方。"""
    q = OutboundQueue()
    q.put_audio('{"seq":1}', b"a")
    q.put_event('{"type":"turn.end"}')
    q.put_event('{"type":"session.closed"}')

    assert (await q.get()).data == b"a"
    assert (await q.get()).text == '{"type":"turn.end"}'
    assert (await q.get()).text == '{"type":"session.closed"}'


async def test_audio_bound_drops_oldest_pair_and_counts() -> None:
    """积压超限丢最旧**音频对**（越过事件单元），JSON 通知与二进制同进同出。"""
    q = OutboundQueue(max_audio_frames=2)
    q.put_audio('{"seq":0}', b"\x00")
    q.put_event('{"type":"asr.final"}')  # 事件不占音频额度，也不被丢
    for i in range(1, 5):
        q.put_audio(f'{{"seq":{i}}}', bytes([i]))

    assert q.dropped_audio == 3  # 5 对音频进 2 容量 → 丢 3
    entries = []
    while True:
        kill, main = q.qsize()
        if kill == 0 and main == 0:
            break
        entries.append(await q.get())
    # 最旧的音频对没了；夹在中间的事件保住了；留下最后两对音频
    assert [(e.text, e.data) for e in entries] == [
        ('{"type":"asr.final"}', None),
        ('{"seq":3}', b"\x03"),
        ('{"seq":4}', b"\x04"),
    ]


async def test_join_waits_until_drained() -> None:
    """join 与 get 配合排空（SessionEnd 路径靠它保证 closed 不抢在音频前面）。"""
    import asyncio

    q = OutboundQueue()
    q.put_event("a")
    q.put_event("b")
    drained: list[str] = []

    async def sender() -> None:
        while True:
            e = await q.get()
            drained.append(e.text)
            q.task_done()

    task = asyncio.create_task(sender())
    await q.join()
    assert drained == ["a", "b"]
    task.cancel()
