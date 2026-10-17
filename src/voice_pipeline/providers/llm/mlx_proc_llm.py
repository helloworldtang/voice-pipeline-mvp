"""把 MLX 推理放进独立进程（`VOICE_LLM_PROCESS=1`）。

动机是实测出来的：同一份代码，MLX 线程同时跑时
- TTS 的 `tts.first_chunk` 从 216ms → 411ms；
- LLM 自己的 TTFT 从微基准 45ms → 链路内 66~141ms（prefill span）。

都是 GIL/线程调度的代价。拆进程后父进程只做管道转发，事件循环不再被推理线程拖住。

契约与 `MlxLlm` 完全一致（`LlmProvider.astream`），调度器一行不用改。
失败策略：子进程没起来/中途死掉 → **抛 RuntimeError**，绝不静默退回 SimLlm
（那会让账单看起来正常、其实没模型，本仓踩过一次）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess  # noqa: S404 - 只起自己仓库里的 worker 模块
import sys
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from .mlx_llm import DEFAULT_SYSTEM
from .qwen_llm import build_messages

logger = logging.getLogger(__name__)

PROCESS_ENV = "VOICE_LLM_PROCESS"
READY_TIMEOUT_S = 180.0  # 首次要加载模型 + Metal 编译
TOKEN_TIMEOUT_S = 30.0


class MlxProcLlm:
    """MLX 子进程 provider：加载一次、常驻、按请求流式回 token。"""

    def __init__(
        self,
        *,
        system: str | None = None,
        max_tokens: int = 160,
    ) -> None:
        self._system = system or os.environ.get("VOICE_LLM_SYSTEM") or DEFAULT_SYSTEM
        self.max_tokens = max_tokens
        self._proc: subprocess.Popen[str] | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = None  # type: ignore[assignment]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()
        self.last_phases: dict[str, float] = {}

    # ---- 子进程生命周期 -------------------------------------------------------

    def _spawn(self) -> subprocess.Popen[str]:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(  # noqa: S603 - 固定模块名，无 shell
            [sys.executable, "-u", "-m", "voice_pipeline.providers.llm.mlx_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # 子进程日志直接进父进程 stderr，便于排查
            text=True,
            bufsize=1,
            env=env,
        )
        self._queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._loop = loop

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:  # 协议外的输出（如第三方 print）
                    logger.debug("[mlx-proc] 非协议输出: %s", line[:200])
                    continue
                loop.call_soon_threadsafe(self._queue.put_nowait, event)
            loop.call_soon_threadsafe(self._queue.put_nowait, {"event": "eof"})

        threading.Thread(target=_reader, name="mlx-proc-reader", daemon=True).start()
        return proc

    async def _ensure(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = self._spawn()
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                msg = f"MLX 子进程 {READY_TIMEOUT_S:.0f}s 内未 ready（模型加载失败？）"
                raise RuntimeError(msg)
            event = await asyncio.wait_for(self._queue.get(), timeout=left)
            if event.get("event") == "ready":
                logger.info(
                    "[mlx-proc] ready model=%s load=%sms",
                    event.get("model"),
                    event.get("load_ms"),
                )
                return
            if event.get("event") == "error":
                raise RuntimeError(str(event.get("message", "MLX 子进程报错")))
            if event.get("event") == "eof":
                msg = "MLX 子进程在 ready 之前退出"
                raise RuntimeError(msg)

    def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            msg = "MLX 子进程未启动"
            raise RuntimeError(msg)
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    # ---- 生成 ----------------------------------------------------------------

    async def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        self.last_phases = {"start": time.perf_counter()}
        async with self._lock:  # 我们的链路是串行的：一条请求跑完再下一条
            await self._ensure()
            self.last_phases["headers"] = time.perf_counter()
            self._next_id += 1
            rid = self._next_id
            self._send(
                {
                    "id": rid,
                    "messages": build_messages(self._system, history, prompt),
                    "max_tokens": self.max_tokens,
                }
            )
            try:
                while True:
                    event = await asyncio.wait_for(self._queue.get(), timeout=TOKEN_TIMEOUT_S)
                    if event.get("id") not in (None, rid):
                        continue
                    if "text" in event:
                        if "t_first" in event:
                            # worker 产出首 token 的时刻（子进程 perf 原值，同机同源可直接减）：
                            # 与父侧 first_content 相减 = 唤醒/传输，与 headers 相减 = 生成本身
                            self.last_phases.setdefault(
                                "worker_first_token", float(event["t_first"])
                            )
                        self.last_phases.setdefault("first_content", time.perf_counter())
                        yield str(event["text"])
                        continue
                    if event.get("event") in ("done", "cancelled"):
                        if os.environ.get("VOICE_LLM_PROC_DEBUG") and "first_token_ms" in event:
                            logger.warning(
                                "[mlx-proc] 子进程内部首 token=%sms", event["first_token_ms"]
                            )
                        return
                    if event.get("event") == "eof":
                        msg = "MLX 子进程中途退出"
                        raise RuntimeError(msg)
            finally:
                # 被打断（barge-in）时通知子进程停手，别白烧算力
                try:
                    self._send({"op": "cancel", "id": rid})
                except Exception:  # 进程已死就没什么可取消的
                    logger.debug("[mlx-proc] 取消失败（进程可能已退出）", exc_info=True)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
