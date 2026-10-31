"""讯飞 TTS 的预建连语义：一次性、能用则用、过期丢弃。

不测网络细节（那是 `scripts/probe_vendor_tts.py` 的事），只钉三件会静默错的事：
① 预建连被用掉后**不能**再被第二条句子复用（讯飞一个连接只做一个任务）；
② 过期的预建连要丢（否则发出旧签名，401 满天飞）；
③ 预建连失败不能影响合成（必须自己重连）。
"""

from __future__ import annotations

import asyncio
import base64
import time

import numpy as np
import pytest

from voice_pipeline.providers.tts import xfyun_tts as mod
from voice_pipeline.providers.tts.xfyun_tts import PREWARM_TTL_S, XfyunTts


class FakeWS:
    """够用的假连接：只记关闭状态。"""

    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None
        self.close_sent = False

    async def close(self) -> None:
        self.closed = True
        self.close_sent = True


@pytest.fixture()
def tts(monkeypatch: pytest.MonkeyPatch) -> XfyunTts:
    for env, value in (
        (mod.APP_ID_ENV, "app"),
        (mod.KEY_ENV, "k"),
        (mod.SECRET_ENV, "s"),
    ):
        monkeypatch.setenv(env, value)
    monkeypatch.delenv("VOICE_XFYUN_PREWARM", raising=False)
    return XfyunTts()


def _patch_connect(monkeypatch: pytest.MonkeyPatch, made: list[FakeWS], fail: bool = False) -> None:
    async def fake_connect(*args: object, **kwargs: object) -> FakeWS:
        if fail:
            msg = "boom"
            raise OSError(msg)
        ws = FakeWS()
        made.append(ws)
        return ws

    monkeypatch.setattr("voice_pipeline.providers.tts.xfyun_tts.websockets.connect", fake_connect)


async def _wait_pending(tts: XfyunTts) -> None:
    for _ in range(100):
        if getattr(tts, "_pending", None) is not None:
            return
        await asyncio.sleep(0.005)


def test_prewarm_is_used_once(tts: XfyunTts, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeWS] = []
    _patch_connect(monkeypatch, made)

    async def go() -> tuple[object, object]:
        tts.prewarm()
        await _wait_pending(tts)
        first = await tts._take()  # noqa: SLF001
        second = await tts._take()  # noqa: SLF001
        return first, second

    first, second = asyncio.run(go())
    assert first is made[0]  # 用的就是预建连那条
    assert second is not first  # 第二个任务必须新建（讯飞不能复用）
    assert len(made) == 2


def test_stale_prewarm_is_dropped(tts: XfyunTts, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeWS] = []
    _patch_connect(monkeypatch, made)
    stale = FakeWS()
    tts._pending = (time.monotonic() - PREWARM_TTL_S - 1, stale)  # type: ignore[assignment]  # noqa: SLF001

    async def go() -> object:
        return await tts._take()  # noqa: SLF001

    fresh = asyncio.run(go())
    assert stale.closed  # 过期连接被关掉
    assert fresh is not stale


def test_prewarm_failure_is_harmless(tts: XfyunTts, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch, [], fail=True)

    async def go() -> None:
        tts.prewarm()
        await asyncio.sleep(0.02)  # 预建连失败已在后台吞掉

    asyncio.run(go())
    assert getattr(tts, "_pending", None) is None


def test_pcm_helpers_roundtrip() -> None:
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    decoded = mod._decode(base64.b64encode(mod._encode(samples)).decode())
    assert np.allclose(decoded, samples, atol=1e-4)


# ---- 首片前失败冷重试（2026-11-01 审查缺陷 8） --------------------------------


class _ScriptedWS:
    """recv 按剧本走的假连接：事件用尽/遇异常即抛。"""

    def __init__(self, events: list[object]) -> None:
        self._events = list(events)
        self.closed = False

    async def send(self, payload: object) -> None: ...

    async def recv(self) -> str:
        if not self._events:
            raise TimeoutError("recv timeout")
        ev = self._events.pop(0)
        if isinstance(ev, Exception):
            raise ev
        assert isinstance(ev, str)
        return ev

    async def close(self) -> None:
        self.closed = True


def _audio_event(status: int) -> str:
    # 1600 样本（100ms）：soxr 流式重采样的滤波器 warmup 会吃掉整个首块——
    # 10ms 的事件产不出任何 chunk（out.size==0），剧本会失真。
    pcm16 = (np.ones(1600, dtype=np.float32) * 0.25).astype("<i2").tobytes()
    import json as _json

    return _json.dumps(
        {"code": 0, "data": {"audio": base64.b64encode(pcm16).decode(), "status": status}}
    )


async def test_retry_once_before_first_chunk(
    tts: XfyunTts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首连在首片前失败（recv 超时）→ 冷重试一次，整句不再无声。"""
    conns = [_ScriptedWS([]), _ScriptedWS([_audio_event(status=2)])]

    async def fake_take() -> _ScriptedWS:
        return conns.pop(0)

    monkeypatch.setattr(tts, "_take", fake_take)  # noqa: SLF001
    chunks = [chunk async for chunk in tts.synth("今天天气怎么样")]
    assert chunks, "首连失败后重试必须出片"
    assert len(conns) == 0


async def test_no_retry_after_first_chunk(tts: XfyunTts, monkeypatch: pytest.MonkeyPatch) -> None:
    """已经出过片的失败不重试：调用方已消费的音频不能重放第二遍。"""
    conns = [
        _ScriptedWS([_audio_event(status=0), TimeoutError("mid-stream death")]),
        _ScriptedWS([_audio_event(status=2)]),  # 若误重试会消费这条
    ]

    async def fake_take() -> _ScriptedWS:
        return conns.pop(0)

    monkeypatch.setattr(tts, "_take", fake_take)  # noqa: SLF001
    got = 0
    with pytest.raises(TimeoutError):
        async for _ in tts.synth("今天天气怎么样"):
            got += 1
    assert got == 1
    assert len(conns) == 1, "出片后的失败不许重试（第二条连接不该被碰）"
