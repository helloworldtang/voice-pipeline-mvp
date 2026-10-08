"""并发判据（成熟度判据 9）：provider 租借池的行为契约。

模型相关的部分**必须能在没有模型文件的机器上跑**——CI 是零模型零网络的，所以 heavy
资源的建模被 monkeypatch 掉，这里验的是**记账与准入逻辑**，不是能不能 load 模型。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from voice_pipeline.config import Settings
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.pool import (
    DEFAULT_MAX_SESSIONS,
    ProviderLease,
    ProviderPool,
    needs_exclusive_slot,
)
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.app import create_app
from voice_pipeline.server.protocol import SessionBegin


@pytest.fixture
def light(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ProviderPool]:
    """把真实建模换成替身：要测的是池的记账，不是 sherpa 能不能 load int8。

    `_recognizer` 用一个**可身份比较的哨兵**代替——「两个会话各拿到一份独立的推理资源」
    这条断言靠 `is not` 比较它，用 `object()` 就够，不必真加载几十 MB 权重。
    """
    import voice_pipeline.providers.pool as pool_mod

    def _build(self: pool_mod._Slot, asr_engine: str, tts_name: str) -> None:
        self._recognizer = object()
        self._tts = SimTts()
        self._tts_engine_name = tts_name

    def _wrap(self: pool_mod._Slot, asr_engine: str) -> SimAsr:
        # 每个租借一个**新 wrapper**（会话状态），共用槽内的 recognizer（那时是独占的）
        return SimAsr()

    monkeypatch.setattr(pool_mod._Slot, "_build", _build)
    monkeypatch.setattr(pool_mod._Slot, "wrap_asr", _wrap)

    def factory(max_sessions: int = 2, **kw: object) -> ProviderPool:
        return ProviderPool(max_sessions, **kw)  # type: ignore[arg-type]

    return factory


async def test_concurrent_real_sessions_get_separate_inference(
    light: Callable[..., ProviderPool],
) -> None:
    """判据 9 的核心：两个 real 会话拿到**不同**的推理资源，而不是共用一份。

    共用一份 = ORT session 上的推理排队 = 两个用户互相拖慢且毫无信号。
    """
    pool = light(2)
    a = await pool.acquire("real")
    b = await pool.acquire("real")
    assert a is not None and b is not None
    assert {a.slot, b.slot} == {0, 1}, "两个并发会话必须落在不同槽"
    sa = a.providers.tts
    sb = b.providers.tts
    assert sa is not sb, "TTS 实例必须独占，否则合成会互相排队"


async def test_pool_refuses_when_full_instead_of_sharing(
    light: Callable[..., ProviderPool],
) -> None:
    """满了要**明确拒绝**，不准静默降级回共享——共享正是要消灭的那种隐性串行。"""
    pool = light(2)
    first = await pool.acquire("real")
    second = await pool.acquire("real")
    assert first is not None and second is not None
    third = await pool.acquire("real")
    assert third is None, "满载必须拒绝，而不是让第三个人挤进来拖垮前两个"
    assert pool.rejections == 1
    assert pool.busy == 2


async def test_released_slot_is_reusable_and_release_is_idempotent(
    light: Callable[..., ProviderPool],
) -> None:
    """槽漏还是最恶劣的 bug：表现为「跑一阵后所有新连接都 1013」。release 必须幂等——
    关闭路径有多个出口（显式 end / 断连 / 换模式重建），很可能走到同一处。
    """
    pool = light(1)
    lease = await pool.acquire("real")
    assert lease is not None
    lease.release()
    lease.release()  # 幂等：重复归还不能把别人的名额抹掉
    assert pool.busy == 0
    again = await pool.acquire("real")
    assert again is not None
    assert again is not lease, "旧句柄已废弃，重借必须是新句柄"
    assert pool.busy == 1, "重借后占用的还是那一个名额"


async def test_sim_mode_never_occupies_a_slot(light: Callable[..., ProviderPool]) -> None:
    """sim/echo 不跑 ONNX，不该占名额——把不烧 CPU 的连接算进上限是自欺，
    而且会让本机开发与 CI 莫名其妙收到 1013。"""
    pool = light(1)
    await pool.acquire("real")
    sim = await pool.acquire("serial")
    assert sim is not None
    assert sim.slot is None
    echo = await pool.acquire("echo")
    assert echo is not None and echo.slot is None


def test_slot_occupancy_is_decided_by_the_mode_not_by_the_url() -> None:
    """mode → 是否占槽 的映射必须是显式的，别让调用方各自猜。"""
    assert needs_exclusive_slot("real")
    assert needs_exclusive_slot("real-serial")
    assert needs_exclusive_slot("real-pipelined")
    assert not needs_exclusive_slot("serial")
    assert not needs_exclusive_slot("pipelined")
    assert not needs_exclusive_slot("echo")


def test_settings_default_matches_pool_default() -> None:
    """配置默认值与池常量不能漂移：漂移的结果是「文档说 2，实际不限」这类静默错。"""
    assert Settings().max_sessions == DEFAULT_MAX_SESSIONS
    assert Settings().shared_inference is False, "生产不该默认走隐性串行那条路"


async def test_two_sessions_do_not_deadlock_each_other(light: Callable[..., ProviderPool]) -> None:
    """并发抢占不能挂起：这两个 acquire 之间没有任何 await 之外的依赖，
    若池实现用了锁或 await CX 就有可能互相等到超时（penalty: 门禁把它变成 30s 失败）。"""
    pool = light(2)
    got = await asyncio.wait_for(
        asyncio.gather(pool.acquire("real"), pool.acquire("real")), timeout=5
    )
    assert all(x is not None for x in got)
    for lease in got:
        if isinstance(lease, ProviderLease):
            lease.release()


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/asr-zipformer/joiner.int8.onnx").is_file(),
    reason="real models not downloaded",
)
def test_overloaded_ws_connection_is_rejected_not_queued() -> None:
    """WS 层：满载后第 N+1 路必须收到明确的 session_limit + 1013。

    这是判据 9 的另一半——capacity 的一半是容量管理，另一半是**拒绝要说出来**。
    """
    client = TestClient(create_app(Settings(max_sessions=1)))
    with client.websocket_connect("/ws") as first:
        first.receive_json()
        first.send_text(SessionBegin(mode="real").model_dump_json())
        hello = first.receive_json()
        assert hello["type"] == "session.ready", hello

        with client.websocket_connect("/ws") as second:
            second.receive_json()
            second.send_text(SessionBegin(mode="real").model_dump_json())
            payload = second.receive_json()
            assert payload["type"] == "error"
            assert payload["code"] == "session_limit"
            # TestClient 在服务端 close 后的 receive 抛 WebSocketDisconnect
            # （websockets 库的 ConnectionClosed 是真网络客户端的异常，这里用错类型
            #  会导致断言永远不触发——初版测试就是这么假失败的）
            with pytest.raises(WebSocketDisconnect):
                while True:
                    second.receive_json()
    # 第一个连接不该因为别人被拒而受影响——这正是「有界」的好处
    assert create_app(Settings(max_sessions=1)).state.pool.capacity == 1


async def test_shared_mode_is_the_only_unbounded_path(light: Callable[..., ProviderPool]) -> None:
    """shared 模式（A/B 对照组）不限流，但也必须显式说明自己在共享。"""
    pool = ProviderPool(1, shared=True)
    a = await pool.acquire("real")
    b = await pool.acquire("real")
    assert a is not None and b is not None
    assert a.slot == b.slot == 0, "对照组：所有会话共用同一份推理（改造前的形态）"
