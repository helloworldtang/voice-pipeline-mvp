"""TTS 连接预热池（P0.5）：零网络钉住「建连不在口径B 里付」的机制。

背景（docs/链路追踪-根因分析.md §四 4.1）：云流式首片 604ms 里 259ms 是
建连+认证——SDK 默认 close_ws_after_use=True，每句新建合成器。本文件的假件
对齐真 SDK 的行为契约（2026-10-17 读源码钉住）：

  1. call() 是**异步提交**——提交文本后立即返回，音频事后从 on_* 回调出；
  2. 同一合成器复用前必须 __reset()（清 complete_event）+ __update_params()
     （换回调 + 新 Request/task_id），否则上一任务的 complete 立即返回、
     task_id 重复；
  3. close_ws_after_use=False 时任务结束后 ws 不关（可复用）；
  4. __is_connected() 说连接死没死——服务端空闲超时会掐线，而 socket 层面
     未必立刻发现（call 才炸），这条 race 由「温连接已死且未出片 → 冷路径
     重试一次」兜住。

测的不是 SDK，是**我们这层池的纪律**：借还、健康判定、失败处置、重试边界。
少数断言直接看池内部状态（_idle/_warm_ok）——白盒钉纪律，计数一律走 pool_stats。
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from voice_pipeline.providers.tts.dashscope_tts import DashScopeTts


class FakeFormat:
    """audio_format 替身：__update_params 只认 .format/.sample_rate/.bit_rate。"""

    format = "pcm"
    sample_rate = 24000
    bit_rate = 64000


class FakeWarmSynthesizer:
    """带 SDK 改名私有方法的假合成器（预热/复用路径专用）。

    `stale=True` 模拟「is_connected 还说活着、call 才炸」的空闲掐线 race。
    """

    def __init__(
        self,
        callback: object,
        chunks: list[bytes],
        *,
        connect_delay: float = 0.0,
        open_delay: float = 0.0,
        chunk_delay: float = 0.0,
    ) -> None:
        self._cb: Any = callback
        self._chunks = chunks
        self._connect_delay = connect_delay
        self._open_delay = open_delay
        self._chunk_delay = chunk_delay
        self.connected = False
        self.stale = False
        self.fail_connect = False
        self.connect_count = 0
        self.reset_count = 0
        self.update_count = 0
        self.calls: list[str] = []
        self.task_ids: list[str] = []
        self.last_close_ws: bool | None = None
        self._task_id = uuid.uuid4().hex
        self._complete_pending = False  # 上一任务的 complete_event 还没清
        self._started = False

    # ---- SDK 改名私有方法（池协议的四件套）----

    def _SpeechSynthesizer__connect(self, timeout_seconds: float = 5) -> None:  # noqa: N802
        if self.fail_connect:
            raise TimeoutError("websocket could not be established within 5s")
        time.sleep(self._connect_delay)
        self.connected = True
        self.connect_count += 1

    def _SpeechSynthesizer__is_connected(self) -> bool:  # noqa: N802
        return self.connected

    def _SpeechSynthesizer__reset(self) -> None:  # noqa: N802
        self.reset_count += 1
        self._complete_pending = False
        self._started = False

    def _SpeechSynthesizer__update_params(  # noqa: N802 - 对齐 SDK 改名私有方法
        self,
        model: str,
        voice: str,
        format: object,
        callback: object,
        close_ws_after_use: bool = True,
        **kwargs: Any,
    ) -> None:
        self.update_count += 1
        self._cb = callback
        self.last_close_ws = close_ws_after_use
        self._task_id = uuid.uuid4().hex  # 新 Request → 新 task_id

    # ---- 合成 ----

    def call(self, text: str) -> None:
        if self._started:
            raise RuntimeError("InvalidTask: task has already started.")
        if self._complete_pending:
            # 真 SDK 这里不会炸，但 __waiting_for_complete 会立即返回——
            # 表现是「合成没完就以为完了」。池必须靠 __reset 避免，故假件炸给你看。
            raise RuntimeError("complete_event 未清理：上一任务的 complete 会立即返回")
        if self.stale:
            raise ConnectionError("WebSocket connection is not established or has been closed.")
        if not self.connected:
            self._SpeechSynthesizer__connect(5)  # 懒连（真 SDK __start_stream 行为）
        self._started = True
        self.calls.append(text)
        self.task_ids.append(self._task_id)
        threading.Thread(target=self._stream, daemon=True).start()
        return None  # 异步提交：立即返回，音频在路上

    def _stream(self) -> None:
        time.sleep(self._open_delay)
        if self._cb is not None:
            self._cb.on_open()
        for chunk in self._chunks:
            time.sleep(self._chunk_delay)
            if self._cb is not None:
                self._cb.on_data(chunk)
        # 真 SDK on_message 的 FINISHED 分支（tts_v2/speech_synthesizer.py:803-805）：
        # complete_event.set() 在 callback.on_complete() **之前**——消费者收到完成
        # 信号时 complete_event 已置位。曾把这两行倒过：倒出来的假 race 让旧线程的
        # 尾声落进下一次 __reset 之后，call() 误炸「complete_event 未清理」，
        # test_pool_stats_counts_are_honest 偶发冷重试（cold=1/dropped=2，2026-10-17 抓到）。
        self._complete_pending = True
        if self._cb is not None:
            self._cb.on_complete()
        # 真 SDK __cleanup_task（:594-599）：只归位 _is_started/_stopped，
        # **不碰 complete_event**——所以它晚于我们的 __reset 落地也无妨
        # （_stopped 的残留窗口由 call() 内部 __start_stream 自己清 + 池的
        # 「温连接已死且未出片→冷重试一次」兜底，见 synth() docstring）。
        self._started = False


def _warm_factory(
    chunks: list[bytes], **kw: Any
) -> tuple[Callable[[object], FakeWarmSynthesizer], list[FakeWarmSynthesizer]]:
    made: list[FakeWarmSynthesizer] = []

    def factory(callback: object) -> FakeWarmSynthesizer:
        synth = FakeWarmSynthesizer(callback, chunks, **kw)
        made.append(synth)
        return synth

    return factory, made


def _tts(
    chunks: list[bytes] | None = None, *, pool_size: int = 1, **kw: Any
) -> tuple[DashScopeTts, list[FakeWarmSynthesizer]]:
    factory, made = _warm_factory(chunks or [b"a" * 4800, b"b" * 4800], **kw)
    tts = DashScopeTts(synth_factory=factory, audio_format=FakeFormat(), pool_size=pool_size)
    return tts, made


async def _settle(tts: DashScopeTts, timeout_s: float = 2.0) -> None:
    """等后台预热落地：要么池满，要么预热失败落旗（两种情况都算 settle 完成）。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        stats = tts.pool_stats()
        if stats["prewarming"] == 0 and (
            stats["idle"] >= stats["size"] or stats["prewarm_fail"] > 0
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("预热未在超时内落地")


async def test_cold_then_warm_reuses_connection() -> None:
    """第一句冷（付一次建连），第二句借同一条热连接：不再建连、reset+update_params。"""
    tts, made = _tts()

    got1 = [c async for c in tts.synth("甲")]
    got2 = [c async for c in tts.synth("乙")]

    assert got1 == got2
    assert len(made) == 1  # 冷路径造的连接进池循环，不无限增长
    assert made[0].calls == ["甲", "乙"]
    assert made[0].connect_count == 1  # 只有第一句建连
    assert made[0].reset_count == 1 and made[0].update_count == 1  # 第二次借用前的重置
    assert made[0].last_close_ws is False  # 复用连接绝不再 per-use 关闭
    assert tts.warm_acquires == 1
    assert tts.cold_acquires == 1
    assert tts.last_warm is True


async def test_prewarm_moves_connect_off_the_critical_path() -> None:
    """判停即预热：synth 之前连接已连上，第一句就是热的（connect 停在预热那一次）。"""
    tts, made = _tts(pool_size=1)
    tts.prewarm()
    await _settle(tts)
    assert len(made) == 1 and made[0].connect_count == 1

    got = [c async for c in tts.synth("甲")]
    assert got
    assert tts.warm_acquires == 1 and tts.cold_acquires == 0
    assert made[0].connect_count == 1  # 一句都没再建连
    assert made[0].calls == ["甲"]


async def test_attach_refreshes_task_id_and_callback() -> None:
    """同一连接上第二个任务必须换 task_id（服务端看得见 run-task），回调换成本次的。"""
    tts, made = _tts(pool_size=1)
    tts.prewarm()
    await _settle(tts)
    first_id = made[0]._task_id

    async for _ in tts.synth("甲"):
        pass
    assert made[0].task_ids[0] != first_id  # __update_params 换了新 Request
    assert made[0].reset_count == 1

    async for _ in tts.synth("乙"):
        pass
    assert made[0].task_ids[0] != made[0].task_ids[1]  # 两次任务不同 id
    assert made[0].reset_count == 2


async def test_stale_warm_conn_retries_cold_once() -> None:
    """温连接被服务端掐线（is_connected 还说活着、call 才炸）且未出片 → 冷路径重试一次。"""
    tts, made = _tts(pool_size=1)
    tts.prewarm()
    await _settle(tts)

    async for _ in tts.synth("甲"):
        pass
    made[0].stale = True  # 空闲超时后的 race：健康检查骗过了池

    got = [c async for c in tts.synth("乙")]
    assert got  # 重试成功，不是把一句失败丢给上层
    # warm_acquires=2：「甲」成功那次 + 「乙」借用后才发现已死那次（借了就是借了）
    assert tts.warm_acquires == 2
    assert tts.cold_acquires == 1  # 兜住这一句的是冷路径
    assert tts.dropped_conns >= 1
    assert tts.last_warm is False  # 成功的那次是冷路径


async def test_no_retry_after_first_chunk_went_out() -> None:
    """出过片就绝不重试——音频已经出门，重放会造成重复播音。"""

    class MidFail(FakeWarmSynthesizer):
        def _stream(self) -> None:
            time.sleep(self._open_delay)
            self._cb.on_open()
            self._cb.on_data(self._chunks[0])
            self._cb.on_error("mid-stream boom")

    made: list[MidFail] = []

    def factory(callback: object) -> MidFail:
        synth = MidFail(callback, [b"x" * 4800])
        made.append(synth)
        return synth

    tts = DashScopeTts(synth_factory=factory, audio_format=FakeFormat(), pool_size=1)
    tts.prewarm()
    await _settle(tts)

    with pytest.raises(RuntimeError, match="mid-stream boom"):
        async for _ in tts.synth("甲"):
            pass
    assert len(made) == 1  # 没有第二次尝试（没新建合成器）
    assert made[0].calls == ["甲"]
    assert tts.pool_stats()["idle"] == 0  # 出过片的病连接不留池


async def test_error_drops_connection_not_returned_to_pool() -> None:
    """合成出错（on_error）→ 连接销毁，不还池（病连接不留给下一句）。"""

    class ErrorSynth(FakeWarmSynthesizer):
        def call(self, text: str) -> None:
            self._started = True
            self.calls.append(text)
            threading.Thread(target=self._fail, daemon=True).start()
            return None

        def _fail(self) -> None:
            self._cb.on_error("boom")

    made: list[ErrorSynth] = []

    def factory(callback: object) -> ErrorSynth:
        synth = ErrorSynth(callback, [b"x" * 4800])
        made.append(synth)
        return synth

    tts = DashScopeTts(synth_factory=factory, audio_format=FakeFormat(), pool_size=1)
    tts.prewarm()
    await _settle(tts)

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in tts.synth("甲"):
            pass
    # 温连接出错 → 丢 + 冷路径重试一次 → 又出错 → 抛给上层；
    # 两条病连接都销毁（池里最多剩 _drop 触发背景补位的新连接，且绝不是这两条）
    assert len(made) >= 2
    assert tts.dropped_conns >= 2
    assert made[0] not in tts._idle
    assert made[1] not in tts._idle


async def test_dead_idle_conn_is_not_reused() -> None:
    """池里空闲期间被掐线的连接：acquire 时就地识破，走冷路径并背景补位。"""
    tts, made = _tts(pool_size=1)
    tts.prewarm()
    await _settle(tts)
    made[0].connected = False  # 服务端掐了，is_connected 如实说

    got = [c async for c in tts.synth("甲")]
    assert got
    assert tts.warm_acquires == 0
    assert tts.cold_acquires == 1
    assert tts.dropped_conns >= 1


async def test_abandoned_consumer_recycles_after_synthesis_ends() -> None:
    """barge-in 丢弃在飞合成：连接不当场还池（防 InvalidTask），等合成真正结束再回收。"""
    tts, made = _tts(pool_size=1, chunk_delay=0.05)
    tts.prewarm()
    await _settle(tts)
    synth = made[0]

    gen = tts.synth("甲")
    await gen.__anext__()  # 出了第一片就打断（barge-in）
    await gen.aclose()

    # 打断当场：合成还在跑，连接不在池里（还了就是下一句 InvalidTask 地雷）
    assert tts.pool_stats()["idle"] == 0
    assert synth.calls == ["甲"]

    await asyncio.sleep(0.5)  # 等合成跑完 + 回收协程醒来
    assert tts._idle == [synth]  # 事后回收：还是那条连接，没有泄漏（白盒：认对象）

    # 回收后的连接可正常再用（同一对象复用，不新建）
    got = [c async for c in tts.synth("乙")]
    assert got
    assert made == [synth]
    assert synth.calls == ["甲", "乙"]


async def test_prewarm_is_idempotent_and_fills_to_pool_size() -> None:
    """预热幂等：连调多次只建 pool_size 条；没有 loop 时静默跳过不炸。"""
    tts, made = _tts(pool_size=2)
    tts.prewarm()
    tts.prewarm()
    await _settle(tts)
    assert len(made) == 2
    assert tts.pool_stats()["idle"] == 2

    tts.prewarm()  # 池满：no-op
    await asyncio.sleep(0.05)
    assert len(made) == 2

    # 同步上下文（无运行循环）：不炸，只是不预热
    result: list[str] = []

    def in_thread() -> None:
        tts.prewarm()
        result.append("ok")

    threading.Thread(target=in_thread).start()
    await asyncio.sleep(0.05)
    assert result == ["ok"]


async def test_prewarm_failure_does_not_raise() -> None:
    """预热建连失败：响亮记账但绝不炸主链路（冷路径照常）。"""

    def broken_factory(callback: object) -> FakeWarmSynthesizer:
        synth = FakeWarmSynthesizer(callback, [b"x" * 4800])
        if callback is None:
            # 只坏预热件（预连必炸，模拟网络不通）；冷路径的懒连不受影响
            synth.fail_connect = True
        return synth

    tts = DashScopeTts(synth_factory=broken_factory, audio_format=FakeFormat(), pool_size=1)
    tts.prewarm()  # 不抛
    await _settle(tts)  # 失败也落旗（finally 里清 _prewarming，池空即视为 settle 完成）
    assert tts.prewarm_failures >= 1

    got = [c async for c in tts.synth("甲")]
    assert got  # 冷路径照常工作


async def test_sdk_without_private_methods_disables_warm_entirely() -> None:
    """SDK 版本变了（取不到改名私有方法）：整条预热路径关闭，退回 2026-10-17 前的冷路径。"""
    made: list[Any] = []

    class PlainSynth:
        """没有四个私有方法的假件（老 SDK 形态）。"""

        def __init__(self, callback: object) -> None:
            self._cb: Any = callback
            self.calls: list[str] = []

        def call(self, text: str) -> None:
            self.calls.append(text)
            self._cb.on_open()
            self._cb.on_data(b"z" * 100)
            self._cb.on_complete()

    def factory(callback: object) -> PlainSynth:
        synth = PlainSynth(callback)
        made.append(synth)
        return synth

    tts = DashScopeTts(synth_factory=factory, audio_format=FakeFormat(), pool_size=1)
    tts.prewarm()  # 构造一次就识破、钉死不支持
    await asyncio.sleep(0.05)
    assert tts._warm_ok is False

    got = [c async for c in tts.synth("甲")]
    assert got == [b"z" * 100]
    assert len(made) == 2  # 预热识破那次 + 冷路径这次；之后不再尝试预热
    tts.prewarm()  # 已钉死：直接 no-op
    await asyncio.sleep(0.05)
    assert len(made) == 2


async def test_missing_audio_format_disables_warm() -> None:
    """注入工厂没给 audio_format：没有 format 就没法回填 __update_params，同样不预热。"""
    factory, made = _warm_factory([b"x" * 4800])
    tts = DashScopeTts(synth_factory=factory, pool_size=1)  # 不传 audio_format
    tts.prewarm()
    await asyncio.sleep(0.05)
    assert len(made) == 0  # 入口就拦下，连构造都没发生

    got = [c async for c in tts.synth("甲")]
    assert got
    assert len(made) == 1  # 只有冷路径那一个
    assert tts._warm_ok is False  # 冷路径构造时识破并钉死


async def test_pool_stats_counts_are_honest() -> None:
    """pool_stats：省几次建连、毁几条连接，探针/账单直接读。"""
    tts, _ = _tts(pool_size=1)
    tts.prewarm()
    await _settle(tts)
    async for _ in tts.synth("甲"):
        pass
    async for _ in tts.synth("乙"):
        pass
    stats = tts.pool_stats()
    assert stats == {
        "warm": 2,
        "cold": 0,
        "dropped": 0,
        "prewarm_fail": 0,
        "idle": 1,
        "size": 1,
        "prewarming": 0,
    }
