"""DashScope CosyVoice 流式 TTS —— CPU 弱机的 TTS 出路。

为什么需要它：sherpa OfflineTTS（melo/kokoro）没有增量流（首片到手=整句合成
完成），且在 2C x86 上 RTF≈14.5——12 字要 11.9s（docs/ecs-baseline.md）。
CosyVoice 走 WebSocket 边合成边推 PCM，首片延迟由网络与服务端决定，与本地
CPU 无关；输出 24k PCM16 mono，与链路下行格式零转换。

SDK 是同步回调式的：合成跑在自己的线程里，on_data 回调经 call_soon_threadsafe
投进 asyncio.Queue——线程→loop 的方向必须走线程安全入口，直接 put_nowait
跨线程操作 asyncio 原语是未定义行为。

**连接预热（P0.5，2026-10-17）**：SDK 默认 `close_ws_after_use=True`——每句
新建一个 SpeechSynthesizer，`submit→open` 那 ~260ms 建连+认证每句都白付
（隔离探针实测：冷首片 604ms = 建连 259 + 服务端 338）。本类因此持有一个
小型连接池：预热好的合成器（WS 已连、认证已过）在借出时只换回调与 task_id，
首句不再付建连费。预热走 SDK 改名私有方法（`__connect/__reset/__update_params/
__is_connected`，与 SDK 自带 SpeechSynthesizerObjectPool 同一套路），取不到
这些方法（SDK 版本变了）时整条预热路径关闭，退化成 2026-10-17 之前的冷路径——
宁可不快，不可半温不热地炸。

测试零网络纪律：构造器可注入 synth_factory（收到回调对象、返回带 .call(text)
的合成器），测试用假件模拟「另一个线程上的回调」。要测预热路径，假件还需带
四个改名私有方法 + 通过 audio_format 传回 format 信息（见 _warm_capable）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from typing import Any, Protocol

logger = logging.getLogger(__name__)

API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_MODEL = "cosyvoice-v2"
DEFAULT_VOICE = "longxiaochun_v2"  # cosyvoice-v2 的音色带 _v2 后缀，不带会 task-failed
DEFAULT_POOL_SIZE = 2  # 与 VOICE_MAX_SESSIONS 默认值对齐：两个会话各占一条热连接
# 预热建连失败后的重试间隔：重试只让「到位」晚 150ms（用户还在说话），
# 而一次失败会让整轮走冷路径（实测 +213ms，口径B 775→1204，见根因文档 §4.6）。
WARM_RETRY_DELAY_S = 0.15


class _Callback(Protocol):
    """SDK 回调接口（dashscope.audio.tts_v2.ResultCallback 的鸭子类型）。"""

    def on_data(self, data: bytes) -> None: ...

    def on_complete(self) -> None: ...


class _Synthesizer(Protocol):
    def call(self, text: str) -> object: ...

    """回调模式下 call() 返回 None（音频走 on_data），返回值没人消费——
    协议因此只要求「能调」，不钉返回类型（钉了 bytes 就等于逼真件说谎）。"""


class _WarmConnDead(Exception):
    """温连接已死且尚未出片：synth() 应冷路径重试一次（内部信号，不外泄）。"""


class _AsyncQueueCallback:
    """SDK 线程 → asyncio.Queue 的桥。结束/出错都投 None 哨兵终止迭代。

    `marks`：phase 时刻表（perf_counter 原值）。**由 SDK 线程写、事件循环读**：
    只 `setdefault` 单个 float（GIL 下原子），调度器在首片到达那一刻就能拆
    「建连 vs 服务端合成」两段，不必等整句合成完。不传就是普通桥（测试替身）。

    `finished`：合成真正结束的信号（on_complete/on_error 置位）。**不能拿
    call() 返当结束**——SDK 的 call() 在提交文本后立即返回，音频还在路上；
    消费端提前退出（barge-in）后靠它判断连接什么时候能回收。
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[bytes | None],
        marks: dict[str, float] | None = None,
    ) -> None:
        self._loop = loop
        self._queue = queue
        self.marks = marks if marks is not None else {}
        self.error: str | None = None
        self.finished = threading.Event()

    def on_data(self, data: bytes) -> None:
        self.marks.setdefault("first_data", time.perf_counter())
        self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

    def on_complete(self) -> None:
        self.marks.setdefault("complete", time.perf_counter())
        self.finished.set()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def on_error(self, message: str) -> None:
        self.error = str(message)
        logger.error("DashScope TTS on_error: %s", message)
        self.finished.set()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    # SDK 生命周期回调：桥不关心，但接口得齐（缺了会 AttributeError）
    def on_open(self) -> None:
        self.marks.setdefault("open", time.perf_counter())

    def on_close(self) -> None:
        pass

    def on_event(self, message: str) -> None:
        pass


class _ListCallback:
    """同步路径（synth_blocking）用的收集器。

    ⚠️ SDK 的 call() 是**异步提交**——立即返回 None，音频与 complete 事件
    事后从 SDK 内部线程回调。所以这里带一个 Event，synth_blocking 等它。
    """

    def __init__(self, marks: dict[str, float] | None = None) -> None:
        self.chunks: list[bytes] = []
        self.marks = marks if marks is not None else {}
        self.error: str | None = None
        self.done = threading.Event()

    def on_data(self, data: bytes) -> None:
        self.marks.setdefault("first_data", time.perf_counter())
        self.chunks.append(data)

    def on_complete(self) -> None:
        self.marks.setdefault("complete", time.perf_counter())
        self.done.set()

    def on_error(self, message: str) -> None:
        self.error = str(message)
        self.done.set()

    def on_open(self) -> None:
        self.marks.setdefault("open", time.perf_counter())

    def on_close(self) -> None:
        pass

    def on_event(self, message: str) -> None:
        pass


def _healthy(call_task: asyncio.Task[Any], callback: _AsyncQueueCallback) -> bool:
    """这条连接还能不能留给下一句：call 没炸 + 合成正常走完（有结束信号、无 error）。"""
    if call_task.cancelled():
        return False
    if call_task.done() and call_task.exception() is not None:
        return False
    return callback.finished.is_set() and callback.error is None


def _spawn(coro: Coroutine[Any, Any, None]) -> None:
    """把事后回收/关连接挂到事件循环上；没有 loop（进程退出中）就直接收掉。"""
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()


class DashScopeTts:
    """CosyVoice 流式合成。`tts_streams=True`：调度器走增量路径。

    连接池语义（全部只在事件循环里被触碰，因此不需要锁）：
      - `prewarm()` 非阻塞预热——建会话/判停时调用，池不满就在后台补预连；
      - `synth()` 优先借池里的热连接，没有才现造（冷路径，构造后也留在池里循环）；
      - 温连接已死且尚未出片 → 冷路径重试一次（空闲超时被服务端掐线的自愈）；
      - 消费端提前退出（barge-in）→ `_recycle` 等合成真正结束再回收连接。
    """

    tts_streams = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        *,
        key: str | None = None,
        timeout_s: float = 30.0,
        synth_factory: Callable[[_Callback | None], _Synthesizer] | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        connect_timeout_s: float = 5.0,
        audio_format: object | None = None,
    ) -> None:
        """synth_factory：收到回调对象（预热时收 None）、返回合成器（.call(text)）。
        不注入时用真 SDK；缺 API key 必须响亮失败，不能静默退回本地 TTS——
        否则 2C 机器会假装自己在跑流式。

        pool_size：池目标大小（默认 2，与 VOICE_MAX_SESSIONS 对齐）。两条以上
        并发合成时多出来的走冷路径，还池时池满则丢弃，不会无界增长。

        connect_timeout_s：预热建连的超时（SDK `__connect` 的阻塞上限）。

        audio_format：预热/复用连接时回填 SDK 参数用（`__update_params` 需要
        format 枚举）。默认工厂会自动填上 PCM 24k；注入工厂的测试若想测预热
        路径，传一个同形替身（有 .format/.sample_rate/.bit_rate 即可）。
        """
        self.model = model
        self.voice = voice
        self.timeout_s = timeout_s
        self._pool_size = max(1, pool_size)
        self._connect_timeout_s = connect_timeout_s
        self._audio_format = audio_format
        self._synth_factory = synth_factory or self._make_default_factory(key)
        # phase 时刻表（perf_counter 原值）：`construct`（借/造合成器的起点）/
        # `submit`（call 提交前）/ `open`（WS 连上并认证完，任务已起）/ `first_data`
        # （首个音频分片）/ `complete`。**实时可读**——synth() 一进来就挂上，
        # 调度器在首片到达那一刻即可拆子 span，不必等整句。约定属性名见
        # metrics/trace.py phases_of。
        self.last_phases: dict[str, float] = {}
        # 本次合成借到的是不是热连接（span attrs 用；账单/探针另看 pool_stats）
        self.last_warm = False

        # ---- 连接池状态（只在事件循环里动）----
        self._idle: list[Any] = []
        self._prewarming = False
        self._prewarm_task: asyncio.Task[None] | None = None
        # SDK 是否支持预热协议；None = 还没构造过合成器、待判定（见 _warm_capable）
        self._warm_ok: bool | None = None
        # 是否已经烧过「连接上第一个任务」的那笔开销（任务级预备，见 _warm_one_task）
        self._task_warm = False
        # 任务级预热（默认开，2026-10-17 起）：只烧一次极短合成，对体验零影响，
        # 换来冷启动首轮 tts.connect 840→37ms。关掉：VOICE_TTS_TASK_WARM=0。
        _task_warm_env = os.environ.get("VOICE_TTS_TASK_WARM", "1").strip().lower()
        self._task_warm_enabled = _task_warm_env not in ("0", "false", "no", "off")
        # 计数：探针/账单要能算出「省了几次建连、毁了几条连接」
        self.warm_acquires = 0
        self.cold_acquires = 0
        self.dropped_conns = 0
        self.prewarm_failures = 0

    def _make_default_factory(self, key: str | None) -> Callable[[_Callback | None], _Synthesizer]:
        resolved = key or os.environ.get(API_KEY_ENV, "")
        if not resolved:
            msg = f"缺少 {API_KEY_ENV}：DashScopeTts 需要它，否则 2C 机器没有可用的 TTS"
            raise RuntimeError(msg)

        from dashscope.audio.tts_v2 import AudioFormat

        # 预连/复用时要回填的 format（__update_params 收枚举不收字符串）
        self._audio_format = AudioFormat.PCM_24000HZ_MONO_16BIT

        def factory(callback: _Callback | None) -> _Synthesizer:
            from dashscope.audio.tts_v2 import SpeechSynthesizer

            return SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                callback=callback,  # type: ignore[arg-type]
            )

        return factory

    # ---- 连接池 ------------------------------------------------------------

    def pool_stats(self) -> dict[str, int]:
        """池计数：探针/账单要能算出「省了几次建连、毁了几条连接」。"""
        return {
            "warm": self.warm_acquires,
            "cold": self.cold_acquires,
            "dropped": self.dropped_conns,
            "prewarm_fail": self.prewarm_failures,
            "idle": len(self._idle),
            "size": self._pool_size,
            "prewarming": int(self._prewarming),
        }

    def prewarm(self) -> None:
        """非阻塞预热：池不满就在后台补预连（建会话/判停/投机开跑时调用）。

        为什么不 await：建连 ~260ms 必须发生在**用户还在说/还在想**的窗口里，
        等它再跑 LLM 等于把账加进口径B。调用方因此都不等——LLM 那几个往返
        （headers ~230ms）就是预热的窗口。没有运行中的事件循环（同步脚本/
        线程）时静默跳过：冷路径照常工作。
        """
        if self._warm_ok is False:
            return
        if self._warm_ok is None and self._audio_format is None:
            return  # 没有 format 就没法回填 __update_params，预热无从谈起
        if self._prewarming:
            return
        if len(self._idle) >= self._pool_size:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._prewarming = True
        self._prewarm_task = loop.create_task(self._prewarm_async())

    async def _prewarm_async(self) -> None:
        try:
            missing = self._pool_size - len(self._idle)
            if missing > 0:
                # 并发连：别把 2×260ms 串成 520ms——首轮第一句等不起
                results = await asyncio.gather(*(self._connect_one() for _ in range(missing)))
                for synth in results:
                    if synth is not None:
                        self._idle.append(synth)
            # 任务级预热：只连上 = 只烧掉 TCP 握手。**连接上第一个任务**还要再付一次
            # 服务端开销（2026-10-17 实测 840ms，把冷启动首轮直接抬到 1743ms）；
            # 同一连接跑过一次之后，connect 降到 43~50ms。用一小段合成把它提前烧掉
            # ——就是文章 WarmPool 里「预合成静音帧」那半边（我们之前只做了预连）。
            await self._warm_one_task()
        except Exception:
            logger.warning("TTS 连接预热失败（冷路径照常）", exc_info=True)
        finally:
            self._prewarming = False

    async def _warm_one_task(self) -> None:
        """烧掉「连接上第一个任务」的服务端开销（只做一次）。

        为什么需要：预连只解决 TCP/TLS；真正的任务是 `call(text)`——服务端要建会话、
        加载音色、跑第一个 chunk。实测同一连接：首任务 `tts.connect` 840ms，之后 43~50ms。
        做法：在后台用一小段合成（「好。」）跑一遍并丢弃音频，让池里的连接变成"跑过任务"的。
        代价：一次极短合成的配额 + 后台 ~0.6s（用户还在说话）。
        """
        if (
            not self._task_warm_enabled
            or self._task_warm
            or self._warm_ok is False
            or not self._idle
        ):
            return
        self._task_warm = True
        try:
            async for _ in self.synth("好。"):
                pass
        except Exception:  # noqa: BLE001 - 预热失败不能拖垮会话
            self._task_warm = False
            logger.warning("TTS 任务级预热失败（首轮会多付一次任务开销）", exc_info=True)

    async def _connect_one(self, attempts: int = 2) -> Any | None:
        """建一条预连。**失败重试一次**。

        为什么重试：预热是后台的，重试只让到位晚 ~150ms（用户还在说话）；而一次失败会让
        整个会话走冷路径——2026-10-17 晚实测到一轮 `tts.connect` 263ms：口径B 775→1204ms。
        仍失败才计数并退回冷路径（不抛，不拖垮会话）。
        """
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.to_thread(self._make_warm_synth)
            except Exception as exc:  # noqa: BLE001 - 建连失败不允许冒泡
                if attempt >= attempts:
                    self.prewarm_failures += 1
                    logger.warning(
                        "TTS 预热建连失败（已重试 %d 次，本次退回冷路径）: %s", attempts - 1, exc
                    )
                    return None
                logger.info("TTS 预热建连失败，%.0fms 后重试: %s", WARM_RETRY_DELAY_S * 1000, exc)
                await asyncio.sleep(WARM_RETRY_DELAY_S)
        return None

    def _make_warm_synth(self) -> Any | None:
        """线程里跑：构造合成器 + 预连 WS（`__connect` 阻塞最多 connect_timeout_s）。

        预连件**不带回调**（callback=None，async_call=False）——借出时 `_attach`
        才把本次的回装上。取不到预热协议（SDK 版本变了）返回 None，缓存标志
        随之钉死，以后不再尝试。
        """
        synth: Any = self._synth_factory(None)
        if not self._warm_capable(synth):
            return None
        synth._SpeechSynthesizer__connect(self._connect_timeout_s)
        return synth

    def _warm_capable(self, synth: Any) -> bool:
        """SDK 是否支持「预连 + 复用」协议：四个改名私有方法 + format 回填信息。

        判定过一次就缓存。缺 format（注入工厂没传 audio_format）或取不到私有
        方法，都算不支持——**整条预热路径关闭**，退回冷路径。
        """
        if self._warm_ok is None:
            self._warm_ok = self._audio_format is not None and all(
                callable(getattr(synth, f"_SpeechSynthesizer{suffix}", None))
                for suffix in ("__connect", "__reset", "__update_params", "__is_connected")
            )
        return self._warm_ok

    def _acquire(self, callback: _Callback) -> tuple[Any, bool]:
        """借一个合成器：优先预连好的（warm），没有就现造（cold）。

        只在事件循环里调用（synth/prewarm 都是协程路径）——池状态因此不需要锁。
        """
        while self._idle:
            synth: Any = self._idle.pop()
            if self._is_alive(synth):
                self._attach(synth, callback)
                self.warm_acquires += 1
                return synth, True
            # 空闲期间被服务端掐了（空闲超时）：就地销毁，顺手背景补位
            self._drop(synth)
        synth = self._synth_factory(callback)
        if self._warm_capable(synth):
            # 冷构造的这一句白付一次建连，但把连接留在池里：后面每句都省
            synth._close_ws_after_use = False
        self.cold_acquires += 1
        return synth, False

    def _attach(self, synth: Any, callback: _Callback) -> None:
        """把本次合成的回调装到（预连好的）合成器上，并重置上一次任务的状态。

        走 SDK 改名私有方法（与 SDK 自带对象池同一套路）：`__reset` 清事件与
        任务状态；`__update_params` 换回调并**换一个新 Request/task_id**——
        同一连接上跑第二个任务必须换 id，否则服务端看到重复的 run-task。
        format 用构造时记下的枚举回填。
        """
        synth._SpeechSynthesizer__reset()
        synth._SpeechSynthesizer__update_params(
            model=self.model,
            voice=self.voice,
            format=self._audio_format,
            callback=callback,
            close_ws_after_use=False,
        )

    def _is_alive(self, synth: Any) -> bool:
        hook = getattr(synth, "_SpeechSynthesizer__is_connected", None)
        if callable(hook):
            return bool(hook())
        # 取不到私有方法（不该发生——进池的都判定过）时看 socket 状态
        sock = getattr(getattr(synth, "ws", None), "sock", None)
        return bool(getattr(sock, "connected", False))

    def _release(self, synth: Any, healthy: bool) -> None:
        """用完处置：健康且支持复用且池没满 → 留下次用；否则销毁。"""
        if healthy and self._warm_ok and len(self._idle) < self._pool_size:
            self._idle.append(synth)
            return
        self._drop(synth)

    def _drop(self, synth: Any) -> None:
        """销毁一条连接：计数 + best-effort 关闭 + 后台补位（下次又是热的）。"""
        self.dropped_conns += 1
        self.prewarm()
        close = getattr(synth, "close", None)
        if callable(close):
            _spawn(self._close_quietly(close))

    async def _close_quietly(self, close: Callable[[], object]) -> None:
        try:
            await asyncio.to_thread(close)
        except Exception:
            logger.debug("TTS 连接关闭失败（反正要扔，无关紧要）", exc_info=True)

    async def _recycle(
        self, synth: Any, callback: _AsyncQueueCallback, call_task: asyncio.Task[Any]
    ) -> None:
        """消费端提前退出（barge-in/取消）后的事后回收。

        为什么必须等合成真正结束：SDK 的 call() 在提交文本后立即返回，音频还在
        路上；现在就还池，下一个任务会在同一合成器上撞 InvalidTask（task has
        already started）。结束信号只能是 on_complete/on_error（callback.finished）；
        call_task 自己炸了时不会有回调——那条 escape 防死等。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_s
        while not callback.finished.is_set():
            if call_task.done() and not call_task.cancelled() and call_task.exception() is not None:
                break
            if loop.time() >= deadline:
                break
            await asyncio.sleep(0.05)
        self._release(synth, _healthy(call_task, callback))

    # ---- 合成 --------------------------------------------------------------

    async def synth(self, sentence: str) -> AsyncGenerator[bytes, None]:
        """流式合成：首片到达即 yield，不等整句。

        连接池（P0.5）：优先借预热好的连接，首句不付 ~260ms 建连费；温连接被
        服务端掐线（空闲超时）且尚未出过片时，冷路径重试一次——重试要付一次
        完整冷建连，但比把一句失败丢给上层好。出过片就绝不重试（音频已经出门，
        重放会造成重复播音）。
        """
        for attempt in (0, 1):
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            marks: dict[str, float] = {}
            callback = _AsyncQueueCallback(loop, queue, marks)
            t_construct = time.perf_counter()
            synthesizer, warm = self._acquire(callback)
            marks["construct"] = t_construct
            # 先把表挂上再 submit：调度器可能在首片到达前的任何一刻来读。
            # SDK 线程往里 setdefault 写，事件循环只读——跨线程只走这一条路。
            self.last_phases = marks
            self.last_warm = warm
            try:
                async for chunk in self._pump(
                    synthesizer,
                    sentence,
                    callback,
                    queue,
                    marks,
                    warm=warm,
                    allow_retry=attempt == 0,
                ):
                    yield chunk
                return
            except _WarmConnDead:
                logger.warning("TTS 温连接已死且未出片，冷路径重试一次")
                continue

    async def _pump(
        self,
        synthesizer: Any,
        sentence: str,
        callback: _AsyncQueueCallback,
        queue: asyncio.Queue[bytes | None],
        marks: dict[str, float],
        *,
        warm: bool,
        allow_retry: bool,
    ) -> AsyncIterator[bytes]:
        """一次合成尝试：借来的/现造的合成器 + 队列桥，首片到达即外抛。

        连接的三种出口都被处置：正常结束还池；出错判健康度再还或毁；消费端
        提前退出（barge-in）时合成可能还在跑——交给 _recycle 等它真正结束
        （on_complete/on_error）再回收，否则同一合成器会被两个任务同时用。
        """
        marks["submit"] = time.perf_counter()
        call_task = asyncio.create_task(asyncio.to_thread(synthesizer.call, sentence))
        released = False

        def fail_sentinel(task: asyncio.Task[Any]) -> None:
            """call() 炸了（连接已死/提交失败）：音频不会再来，投个哨兵把消费者
            从 queue.get() 上放下来——否则这一等就是永远（2026-10-17 探针里
            「偶发卡死」的根因就是没有这个哨兵）。正常提交也立即返回，但那时
            **不能**投：音频还在路上。"""
            if not task.cancelled() and task.exception() is not None:
                queue.put_nowait(None)

        call_task.add_done_callback(fail_sentinel)

        def release(healthy: bool) -> None:
            nonlocal released
            if not released:
                released = True
                self._release(synthesizer, healthy)

        got_any = False
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                got_any = True
                yield chunk
            await call_task
            if callback.error is not None:
                msg = f"DashScope TTS 合成失败: {callback.error}"
                raise RuntimeError(msg)
        except Exception as exc:
            release(_healthy(call_task, callback))
            if warm and allow_retry and not got_any:
                raise _WarmConnDead(str(exc)) from exc
            raise
        else:
            release(True)
        finally:
            if not released:
                _spawn(self._recycle(synthesizer, callback, call_task))

    def synth_blocking(self, sentence: str) -> bytes:
        """整句合成（协议兼容）：同一 SDK 路径，同步收集。

        **有意走冷路径**（不复用连接）：调用方是协议兼容/探针旁路，流水线本身
        走 synth() 的池化路径。call() 是异步提交（见 _ListCallback 的警告），
        必须等 done 事件。
        """
        marks: dict[str, float] = {}
        callback = _ListCallback(marks)
        self.last_phases = marks
        self.last_warm = False
        t_construct = time.perf_counter()
        synthesizer = self._synth_factory(callback)
        marks["construct"] = t_construct
        marks["submit"] = time.perf_counter()
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
