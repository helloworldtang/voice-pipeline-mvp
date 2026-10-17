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
