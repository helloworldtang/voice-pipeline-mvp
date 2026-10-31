"""并发租借池：把「跨连接共享 ORT session」换成「会话独占 + 有界容量」。

为什么需要它
------------

成熟度判据 9（并发）此前是唯一 ❌，根因写在 `factory.py` 的缓存注释里：
本地 ASR recognizer 与本地 TTS 引擎实例**跨连接共享**，而 ort session 上的推理是
**串行**的——两个用户同时说话，两边互相拖慢，而链路里没有任何信号告诉你「现在挤了」。
隐性串行比明确拒绝更糟：它把事故变成体感，且事后无从归因。

本模块把它换成显式的容量管理：

- 每个会话借一个**槽**，槽内的 recognizer / 本地 TTS 实例归这一次租借独占（推理真并行）；
- 槽**有界**（`VOICE_MAX_SESSIONS`），借不到就是**明确拒绝**：`error(session_limit)` +
  1013 关连接。不悄悄降级回共享——「静默少做」是最坏的失败形态。

哪些资源**不**占槽（把它们算进上限是自欺）：
- LLM：`QwenLlm` 共用一份 httpx.AsyncClient，网络 IO 天然并发；
- 云端 TTS：`DashScopeTts` 不跑 ONNX，不占本地推理槽——它自持一个小型**连接池**
  （P0.5，2026-10-17；预热一条 WS 供后续句子复用），池大小与并发上限无关；
- sim 三替身：没有重量资源，不设上限（CI 与结构压测要开任意多路）。

计的是 ONNX 推理，不是连接数：一个 WebSocket 连接很便宜，真正被独占的是 ort session。
按错的资源计费，上限就没有意义。

泄漏纪律
--------

`acquire()` 成功就必须有人 `release()`——槽是**计数资源**，漏还一个就永久少一个名额，
表现为「跑了一阵之后所有新连接都 1013」。因此 release 做成**幂等**（关闭路径有多个出口：
显式 session.end / WebSocketDisconnect / 换模式重建，可能走到同一处），并由 WS 层的
`finally` 兜底。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, TtsProvider
from voice_pipeline.providers.factory import (
    DEFAULT_ASR_ENGINE,
    DEFAULT_LLM_MODEL,
    ProviderSet,
    build_llm,
    resolve_tts_engine,
)
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts

logger = logging.getLogger(__name__)

# 默认 2 路：每路独占一份 zipformer(24MB) + melo(58MB)，2 路约 164MB 常驻。
# 这不是拍脑袋的值，是「每槽内存开销 × 目标部署形态（2C/2G ECS）」的折中；真实上限
# 取决于机器内存与核数，用 `scripts/bench_concurrency.py` 实测过载时的退化形态再定。
DEFAULT_MAX_SESSIONS = 2

# 本地 TTS 线程数默认 4（本机实测最优点：**单路**场景下 2 线程 665ms → 4 线程 624ms，
# 8 线程反而退步）。并发下这个最优点会移动——总线程数超过可用核就退化成抢核，
# 见 docs/实验报告.md 并发一节与 bench/concurrency-*.json。
LOCAL_TTS_THREADS = 4


def _tts_threads() -> int:
    return int(os.environ.get("VOICE_LOCAL_TTS_THREADS", LOCAL_TTS_THREADS))


# ASR 线程数默认 1（2026-10-17 实测改默认，原为 2）。
# 依据：ASR RTF 只有 0.008~0.016，1 线程就远超实时；而多出来的那个线程会和
# 同进程的 LLM 生成（尤其本地 MLX）抢核——实测把 prefill 140→53~116ms、
# 首句 120→73~76ms、口径B 741→619ms（本地 LLM 档）。识别结果不受线程数影响。
def _asr_threads() -> int:
    return int(os.environ.get("VOICE_ASR_THREADS", 1))


def needs_exclusive_slot(mode: str) -> bool:
    """只有 real 系列才占槽——它们才有本地 ORT 推理。

    `echo` / `serial` / `pipelined`（sim 替身）没有重量资源，不占名额：
    把不烧 CPU 的连接算进上限，只会让本机开发与 CI 莫名其妙收到 1013。
    """
    return mode == "real" or mode.startswith("real-")


class _Slot:
    """一个并发槽：一套会话独占的重资产（ASR recognizer + TTS 引擎实例）。

    懒建立：只有真的借到这个槽时才加载模型。冷槽不占内存——并发不常见，
    为它预付两份模型常驻内存不划算。
    """

    def __init__(self, index: int) -> None:
        self.index = index
        self._recognizer: object | None = None
        self._tts: TtsProvider | None = None
        self._tts_engine_name: str | None = None

    def heavy_ready(self, tts_name: str) -> bool:
        """这套重量资源是否已建好且引擎名没变（切引擎必须重建）。"""
        return self._recognizer is not None and self._tts_engine_name == tts_name

    def _build(self, asr_engine: str, tts_name: str) -> None:
        """在**线程**里调用：模型加载是秒级的，卡住事件循环会让其它连接一起停。"""
        from voice_pipeline.providers import sherpa as real

        asr_cls: Callable[..., AsrProvider] = {
            "paraformer": real.ParaformerStreamingAsr,
            "transducer": real.TransducerStreamingAsr,
        }[asr_engine]
        asr = asr_cls(num_threads=_asr_threads())
        self._recognizer = asr._recognizer  # type: ignore[attr-defined]
        if tts_name == "dashscope":
            from voice_pipeline.providers.tts.dashscope_tts import DashScopeTts

            self._tts = DashScopeTts()
        elif tts_name == "xfyun":
            from voice_pipeline.providers.tts.xfyun_tts import XfyunTts

            # 讯飞 v2：不跑 ONNX（不占推理核），但仍走这个槽，保持「一份 provider 一套会话」
            self._tts = XfyunTts()
        else:
            tts_cls = real.TTS_ENGINES.get(tts_name)
            if tts_cls is None:
                msg = f"unknown tts engine: {tts_name}"
                raise ValueError(msg)
            self._tts = tts_cls(num_threads=_tts_threads())
        self._tts_engine_name = tts_name

    def wrap_asr(self, asr_engine: str) -> AsrProvider:
        """给本次租借新建一个 ASR wrapper（会话状态），复用槽内 recognizer（模型权重）。"""
        from voice_pipeline.providers import sherpa as real

        cls: Callable[..., AsrProvider] = {
            "paraformer": real.ParaformerStreamingAsr,
            "transducer": real.TransducerStreamingAsr,
        }[asr_engine]
        return cls(recognizer=self._recognizer)


class ProviderLease:
    """一次会话借到的资源句柄。**用完必须 release()**，release 幂等。"""

    def __init__(self, providers: ProviderSet, *, slot: int | None, pool: ProviderPool) -> None:
        self.providers = providers
        self.slot = slot
        self._pool = pool
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool._free(self)  # noqa: SLF001  # 池与租借是同一个封装单元

    def __enter__(self) -> ProviderSet:
        return self.providers

    def __exit__(self, *_: object) -> None:
        self.release()


class ProviderPool:
    """有界的 provider 租借池。

    并发前提：公开方法都在事件循环里被调用，`acquire` 在把槽标成 busy **之前**没有
    await 点，因此不需要锁。
    """

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        asr_engine: str = DEFAULT_ASR_ENGINE,
        llm_engine: str = "auto",
        llm_model: str = DEFAULT_LLM_MODEL,
        *,
        shared: bool = False,
    ) -> None:
        self._max = max_sessions
        self._asr_engine = asr_engine
        self._llm_engine = llm_engine
        self._llm_model = llm_model
        # `shared` 只为**测量**存在：它复现改造前的形态（所有会话共用一份本地推理，
        # 并发即隐性串行），让 `bench_concurrency.py` 能在同一台机器上 A/B 出
        # 「改了到底快多少」。生产不要开。
        self._shared = shared
        self._busy: set[int] = set()
        self._slots: dict[int, _Slot] = {}
        # 仍在租约中的 real 租借：停机时 aclose 要能摸到它们（LLM 子进程等）
        self._live_leases: set[ProviderLease] = set()
        self._closed = False
        self.rejections = 0

    @property
    def capacity(self) -> int:
        return self._max

    @property
    def busy(self) -> int:
        return len(self._busy)

    def available(self) -> bool:
        """还有名额吗。shared 模式恒为真——它就是「不设限地挤」的那个对照组。"""
        return self._shared or self.busy < self.capacity

    async def acquire(self, mode: str, tts_engine: str | None = None) -> ProviderLease | None:
        """借一套 provider。**返回 None = 满了**，这不是异常，是背压信号。

        调用方必须把它翻译成给客户端的明确错误（见 app.py 的 session_limit），
        不要静默降级回共享资源——共享正是要消灭的那个隐性串行。
        """
        if not needs_exclusive_slot(mode):
            return ProviderLease(
                ProviderSet(asr=SimAsr(), llm=SimLlm(), tts=SimTts()), slot=None, pool=self
            )
        index = self._take_free_index()
        if index is None:
            self.rejections += 1
            logger.warning(
                "并发会话已达上限（%s/%s），拒绝新连接：宁可明确 1013，"
                "也不让新会话静默挤占已有会话的推理",
                self.busy,
                self._max,
            )
            return None
        slot = self._slots.setdefault(index, _Slot(index))
        tts_name = resolve_tts_engine(tts_engine)
        if not slot.heavy_ready(tts_name):
            # 建模在线程里跑：秒级的加载不能堵住其它连接的接收循环。
            # shared 模式下所有 lease 拿到的是同一个 0 号槽——复用即串行，这是有意的。
            try:
                await asyncio.to_thread(slot._build, self._asr_engine, tts_name)  # noqa: SLF001
            except Exception:
                # 建模失败必须还槽：槽在 _take_free_index 时就标成 busy 了，
                # 异常一冒泡这个名额就永久消失——「跑一阵后所有新连接都 1013」
                # 的另一种成因。异常照抛（调用方翻译成 5xx/断连），名额不陪葬。
                self._busy.discard(index)
                raise
        lease = ProviderLease(
            ProviderSet(
                asr=slot.wrap_asr(self._asr_engine),
                llm=build_llm(self._llm_engine, self._llm_model),
                tts=slot._tts or SimTts(),  # noqa: SLF001
            ),
            slot=index,
            pool=self,
        )
        self._live_leases.add(lease)
        return lease

    def _take_free_index(self) -> int | None:
        if self._shared:
            # 恒返回 0 号槽：所有租借共用同一份重量资源（改造前的形态，仅用于 A/B）
            self._busy.add(0)
            return 0
        for i in range(self._max):
            if i not in self._busy:
                self._busy.add(i)
                return i
        return None

    def _free(self, lease: ProviderLease) -> None:
        if lease.slot is not None:
            self._busy.discard(lease.slot)
            self._live_leases.discard(lease)
        # 租借级 LLM 的生命周期到归还为止：MlxProcLlm 不 close 就子进程孤儿，
        # 常驻吃 Metal + 内存（2026-11-01 审查缺陷 4）。QwenLlm/SimLlm/MlxLlm
        # 没有 close（权重进程内共享，不随租借生死），getattr 落空即跳过。
        close = getattr(lease.providers.llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning("LLM provider close 失败（忽略，不影响槽位归还）", exc_info=True)

    async def aclose(self) -> None:
        """停机收尾（幂等）：把仍带重资源的 provider 关干净。

        两层：①未归还租借的租借级 LLM（连接取消可能晚于 lifespan；正常路径
        release 已关，这里兜底）；②槽内重资产（讯飞 TTS 的 WS、未来其他
        带 close/aclose 的槽内资源）。不收的代价：`VOICE_LLM_PROCESS=1` 时
        MLX 子进程成孤儿常驻吃 Metal+内存（2026-11-01 审查缺陷 4）。
        """
        if self._closed:
            return
        self._closed = True
        for lease in list(self._live_leases):
            close = getattr(lease.providers.llm, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._live_leases.clear()
        for slot in self._slots.values():
            tts = slot._tts  # noqa: SLF001  # 池与槽是同一个封装单元
            if tts is None:
                continue
            aclose = getattr(tts, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()
                continue
            close = getattr(tts, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
