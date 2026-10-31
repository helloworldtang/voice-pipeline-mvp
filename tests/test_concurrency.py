"""并发判据（成熟度判据 9）：provider 租借池的行为契约。

模型相关的部分**必须能在没有模型文件的机器上跑**——CI 是零模型零网络的，所以 heavy
资源的建模被 monkeypatch 掉，这里验的是**记账与准入逻辑**，不是能不能 load 模型。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voice_pipeline.providers.pool as pool_mod
from voice_pipeline.config import Settings
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.pool import (
    DEFAULT_MAX_SESSIONS,
    ProviderLease,
    ProviderPool,
    needs_exclusive_slot,
)
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.app import create_app
from voice_pipeline.server.protocol import SessionBegin, SessionEnd


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


class _ClosableLlm(SimLlm):
    """带 close 计数的替身：MlxProcLlm 的子进程不 close 就孤儿（审查缺陷 4）。"""

    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _HangingLlm(SimLlm):
    """先吐半句再永久挂住的替身：把一轮吊在 LLM 上，供「收掉旧轮」验证。"""

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.cancelled = False

    async def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        self.started = True
        try:
            yield "半句"
            await asyncio.Event().wait()  # 挂住：模拟一次生成很久的轮次，直到被取消
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_mode_switch_cancels_running_turn(
    light: Callable[..., ProviderPool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """换模式必须先收掉在跑的旧轮（2026-11-01 审查缺陷 1）。

    修前：SessionBegin 只 cancel pump_task，旧轮任务句柄被覆盖——它拿着已归还/
    将被复用的槽资源继续推理，与新会话抢算力，finally 也收不掉它。
    """
    stub = _HangingLlm()
    monkeypatch.setattr(pool_mod, "build_llm", lambda *a, **k: stub)

    class _InstantEou:
        """should_end 恒真：第一帧音频就判停，立刻起一轮——把轮次吊在 LLM 上。"""

        def push_audio(self, chunk: bytes) -> None: ...

        def update_text(self, text: str) -> None: ...

        def should_end(self) -> bool:
            return True

        def reset(self) -> None: ...

        def unlatch(self) -> None: ...

    import voice_pipeline.server.session as session_mod

    monkeypatch.setattr(session_mod, "build_eou", lambda gating: _InstantEou())

    client = TestClient(create_app(Settings(max_sessions=1)))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # 首帧 ready（默认 serial）
        ws.send_text(SessionBegin(mode="real").model_dump_json())
        assert ws.receive_json()["type"] == "session.ready"
        # 一帧有声音频：能量门放行 → ASR 出字 → _InstantEou 即刻判停 → 起一轮，吊在 LLM
        ws.send_bytes(b"\x00\x40")
        deadline = time.monotonic() + 5.0
        while not stub.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stub.started, "轮次没进 LLM，测试前提不成立"

        # 换模式：旧轮必须被取消，且新会话照常建好
        ws.send_text(SessionBegin(mode="echo").model_dump_json())
        while True:
            msg = ws.receive_json()
            if msg["type"] == "session.ready":
                break
        assert stub.cancelled, "换模式后旧轮必须被取消，不许拿着槽资源继续算"


def test_turn_failure_reaches_client_as_error_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """一轮生成异常必须让客户端知道（2026-11-01 审查缺陷 3）。

    修前：run_turn 的任务异常无人取——客户端收不到 turn.end 也不知道为什么。
    修后：守卫接住异常、下发 error(turn_failed)，会话照常收尾（closed 照发）。
    """

    class _RaisingLlm(SimLlm):
        async def astream(
            self, prompt: str, history: list[dict[str, str]] | None = None
        ) -> AsyncIterator[str]:
            raise RuntimeError("llm boom")
            yield ""  # type: ignore[unreachable]  # pragma: no cover —— 让它成为异步生成器

    monkeypatch.setattr(pool_mod, "SimLlm", _RaisingLlm)  # sim 租借从这里取 LLM
    client = TestClient(create_app())
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # 首帧 ready（连接初始会话，LLM 来自 factory）
        # 显式 begin：租借路径的 LLM 从 pool_mod.SimLlm 取（被 patch 的那个）
        ws.send_text(SessionBegin(mode="serial").model_dump_json())
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_text(SessionEnd().model_dump_json())
        # error 帧由守卫直发（不经过下行队列），可能抢在 turn.start/asr.final 之前到，
        # 所以断言不依赖顺序：error(turn_failed) 与 session.closed 都要出现。
        seen: list[dict[str, object]] = []
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg["type"] == "session.closed" or len(seen) > 20:
                break
        assert any(m.get("code") == "turn_failed" for m in seen), f"缺少 error 帧：{seen}"
        assert seen[-1]["type"] == "session.closed", f"异常后会话生命周期仍要闭环：{seen}"


async def test_release_closes_lease_level_llm(
    light: Callable[..., ProviderPool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """租借归还时必须关掉租借级 LLM：`VOICE_LLM_PROCESS=1` 下不 close 的
    MlxProcLlm 子进程常驻吃 Metal+内存（2026-11-01 审查缺陷 4）。
    QwenLlm/SimLlm/MlxLlm 没有 close——getattr 落空跳过，权重进程内共享不随租借生死。"""
    closable = _ClosableLlm()
    monkeypatch.setattr(pool_mod, "build_llm", lambda *a, **k: closable)
    pool = light(1)
    lease = await pool.acquire("real")
    assert lease is not None
    assert closable.close_calls == 0, "租着的时候不该被关"
    lease.release()
    assert closable.close_calls == 1
    lease.release()  # 幂等：重复归还不能重复关
    assert closable.close_calls == 1


async def test_aclose_closes_live_lease_and_slot_tts(
    light: Callable[..., ProviderPool], monkeypatch: pytest.MonkeyPatch
) -> None:
    """停机收尾：未归还租借的 LLM 与槽内 TTS（aclose/close）都要被关，且幂等。"""
    closable = _ClosableLlm()
    monkeypatch.setattr(pool_mod, "build_llm", lambda *a, **k: closable)
    pool = light(1)
    lease = await pool.acquire("real")
    assert lease is not None and lease.slot is not None

    class _TtsWithAclose:
        def __init__(self) -> None:
            self.calls = 0

        async def aclose(self) -> None:
            self.calls += 1

    tts = _TtsWithAclose()
    pool._slots[lease.slot]._tts = tts  # type: ignore[assignment]  # noqa: SLF001

    await pool.aclose()
    assert closable.close_calls == 1, "停机时仍连着的会话，其 LLM 也要收"
    assert tts.calls == 1, "槽内 TTS 的 aclose 必须被调"
    await pool.aclose()  # 幂等：重复收尾不重复关
    assert closable.close_calls == 1
    assert tts.calls == 1


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/asr-zipformer/joiner.int8.onnx").is_file(),
    reason="real models not downloaded",
)
def test_overloaded_ws_connection_is_rejected_not_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    """WS 层：满载后第 N+1 路必须收到明确的 session_limit + 1013。

    这是判据 9 的另一半——capacity 的一半是容量管理，另一半是**拒绝要说出来**。

    ⚠️ LLM 引擎钉在 sim：本测与 LLM 无关，而 auto 会因缺 key 降级 MlxLlm——
    修前它 preload 时对未缓存模型发无超时的 Hub 网络请求，断网机器上
    executor 线程永不退出，TestClient 收尾 join 不到 → 30s 超时（2026-11-01 审查）。
    """
    monkeypatch.setenv("VOICE_LLM_ENGINE", "sim")
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
