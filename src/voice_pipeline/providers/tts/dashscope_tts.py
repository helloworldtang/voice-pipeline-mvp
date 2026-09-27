"""DashScope CosyVoice 流式 TTS —— CPU 弱机的 TTS 出路。

为什么需要它：sherpa OfflineTTS（melo/kokoro）没有增量流（首片到手=整句合成
完成），且在 2C x86 上 RTF≈14.5——12 字要 11.9s（docs/ecs-baseline.md）。
CosyVoice 走 WebSocket 边合成边推 PCM，首片延迟由网络与服务端决定，与本地
CPU 无关；输出 24k PCM16 mono，与链路下行格式零转换。

SDK 是同步回调式的：合成跑在自己的线程里，on_data 回调经 call_soon_threadsafe
投进 asyncio.Queue——线程→loop 的方向必须走线程安全入口，直接 put_nowait
跨线程操作 asyncio 原语是未定义行为。

测试零网络纪律：构造器可注入 synth_factory（收到回调对象、返回带 .call(text)
的合成器），测试用假件模拟「另一个线程上的回调」。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable
from typing import Protocol

logger = logging.getLogger(__name__)

API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_MODEL = "cosyvoice-v2"
DEFAULT_VOICE = "longxiaochun_v2"  # cosyvoice-v2 的音色带 _v2 后缀，不带会 task-failed


class _Callback(Protocol):
    """SDK 回调接口（dashscope.audio.tts_v2.ResultCallback 的鸭子类型）。"""

    def on_data(self, data: bytes) -> None: ...

    def on_complete(self) -> None: ...


class _Synthesizer(Protocol):
    def call(self, text: str) -> bytes: ...


class _AsyncQueueCallback:
    """SDK 线程 → asyncio.Queue 的桥。结束/出错都投 None 哨兵终止迭代。"""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[bytes | None]) -> None:
        self._loop = loop
        self._queue = queue
        self.error: str | None = None

    def on_data(self, data: bytes) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

    def on_complete(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def on_error(self, message: str) -> None:
        self.error = str(message)
        logger.error("DashScope TTS on_error: %s", message)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    # SDK 生命周期回调：桥不关心，但接口得齐（缺了会 AttributeError）
    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def on_event(self, message: str) -> None:
        pass


class _ListCallback:
    """同步路径（synth_blocking）用的收集器。

    ⚠️ SDK 的 call() 是**异步提交**——立即返回 None，音频与 complete 事件
    事后从 SDK 内部线程回调。所以这里带一个 Event，synth_blocking 等它。
    """

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.error: str | None = None
        self.done = threading.Event()

    def on_data(self, data: bytes) -> None:
        self.chunks.append(data)

    def on_complete(self) -> None:
        self.done.set()

    def on_error(self, message: str) -> None:
        self.error = str(message)
        self.done.set()

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def on_event(self, message: str) -> None:
        pass


class DashScopeTts:
    """CosyVoice 流式合成。`tts_streams=True`：调度器走增量路径。"""

    tts_streams = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        synth_factory: Callable[[_Callback], _Synthesizer] | None = None,
    ) -> None:
        """synth_factory：收到回调对象、返回合成器（.call(text) 阻塞直到合成完）。
        不注入时用真 SDK；缺 API key 必须响亮失败，不能静默退回本地 TTS——
        否则 2C 机器会假装自己在跑流式。"""
        self.model = model
        self.voice = voice
        self.timeout_s = timeout_s
        self._synth_factory = synth_factory or self._make_default_factory(api_key)

    def _make_default_factory(self, api_key: str | None) -> Callable[[_Callback], _Synthesizer]:
        resolved = api_key or os.environ.get(API_KEY_ENV, "")
        if not resolved:
            msg = f"缺少 {API_KEY_ENV}：DashScopeTts 需要 it，否则 2C 机器没有可用的 TTS"
            raise RuntimeError(msg)

        def factory(callback: _Callback) -> _Synthesizer:
            from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

            return SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                callback=callback,  # type: ignore[arg-type]
            )

        return factory

    async def synth(self, sentence: str) -> AsyncIterator[bytes]:
        """流式合成：首片到达即 yield，不等整句。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        callback = _AsyncQueueCallback(loop, queue)
        synthesizer = self._synth_factory(callback)
        # 合成是阻塞调用（含 WS 收流），放线程里跑；回调持续往 queue 投片，
        # 消费端因此能在合成完成前就开始出片。
        task = asyncio.create_task(asyncio.to_thread(synthesizer.call, sentence))
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
        await task
        if callback.error is not None:
            msg = f"DashScope TTS 合成失败: {callback.error}"
            raise RuntimeError(msg)

    def synth_blocking(self, sentence: str) -> bytes:
        """整句合成（协议兼容）：同一 SDK 路径，同步收集。

        call() 是异步提交（见 _ListCallback 的警告），必须等 done 事件。
        """
        callback = _ListCallback()
        synthesizer = self._synth_factory(callback)
        synthesizer.call(sentence)
        if not callback.done.wait(timeout=self.timeout_s):
            msg = f"DashScope TTS {self.timeout_s}s 内未完成（无 complete 也无 error）"
            raise RuntimeError(msg)
        if callback.error is not None:
            msg = f"DashScope TTS 合成失败: {callback.error}"
            raise RuntimeError(msg)
        return b"".join(callback.chunks)


def build(**kwargs: object) -> DashScopeTts:
    """工厂入口，与 sim_tts.build 同形。"""
    return DashScopeTts(**kwargs)  # type: ignore[arg-type]
