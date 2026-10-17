"""本地 LLM（MLX）——「极致延迟」档，也是云端挂掉时的降级。

实测（M5 / 32GB，`Qwen2.5-1.5B-Instruct-4bit`，热态）：**TTFT 45ms、首句 119ms、~120 chars/s**；
云端 `qwen-flash` 是 250 + 57~120ms ⇒ 口径B 从 ~700~780 降到 ≈575（2026-10-17）。

三条工程约束（都是踩过的）：

1. **模型进程内常驻**：`load` 一次（1~3s），**首次生成还要 Metal 编译**（实测 +900ms）。
   所以第一轮一定慢——要么在会话建立时预热，要么如实接受第一轮慢。
2. **同步生成器必须进线程**：`stream_generate` 是 sync 迭代器，直接在事件循环里跑会堵死
   整条链路（本仓并发实验的结论：`asyncio` 里跑 CPU 密集推理，独占资源也救不了）。
3. **可选依赖，缺了要响亮失败**：`mlx-lm` 只有 macOS/arm64 有，不进主依赖；
   缺依赖时抛 RuntimeError，**不许静默退回 SimLlm**（那会让账单看起来正常、其实没模型）。

开关：`VOICE_LLM_ENGINE=mlx`（`auto` 时：有云 key 走云，缺 key 且有 mlx 就走本地）。
模型：`VOICE_LLM_MLX_MODEL`，默认 `mlx-community/Qwen2.5-1.5B-Instruct-4bit`。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from .qwen_llm import build_messages

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_ENV = "VOICE_LLM_MLX_MODEL"
SYSTEM_ENV = "VOICE_LLM_SYSTEM"
DEFAULT_SYSTEM = "你是语音助手。回答要短，一句话说完，不要 markdown，不要用列表。直接给结论。"

# 进程内模型缓存：load 一次（1~3s），之后每轮共用
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def _apply_template(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    """套聊天模板，并**关掉思考链**。

    Qwen3.5 这一代默认先输出一段 "Thinking Process:" 再作答（实测同一句问话，
    60 token 内一句人话都没有）。语音助手的首句延迟直接翻倍到不可用，
    所以必须传 `enable_thinking=False`；老模型（Qwen2.5）的模板不认这个参数，
    会抛 TypeError，这时退回默认行为。
    """
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def _resolve_system(system: str | None) -> str:
    return system or os.environ.get(SYSTEM_ENV) or DEFAULT_SYSTEM


class MlxLlm:
    """MLX 本地 LLM。与 `QwenLlm` 同一个 `LlmProvider.astream` 契约，调度器一行不用改。"""

    def __init__(
        self,
        model: str | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 160,
    ) -> None:
        self.model = _resolve_model(model)
        self.max_tokens = max_tokens
        self._system = _resolve_system(system)

    # 与 QwenLlm 同一套 phase 约定（调度器按 start/headers/first_content 取）：
    # 本地没有网络那一段，所以 headers == start ⇒ `llm.headers` 记 ~0ms，
    # 整段 TTFT（含首次 Metal 编译）落在 `llm.prefill` 这一格。
    last_phases: dict[str, float]

    def preload(self) -> None:
        """加载模型 **并烧掉首次 Metal 编译**（一次极短生成）。

        为什么生成一次：`load` 只把权重读进来；第一 `stream_generate` 还要编译 Metal 图
        （实测 +900ms），不烧掉的话首轮照样慢——和 TTS 「只预连、不跑任务」是同一个坑。
        """
        model, tokenizer = self._ensure()
        try:
            from mlx_lm import stream_generate

            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "好"}], add_generation_prompt=True
            )
            for _ in stream_generate(model, tokenizer, chat, max_tokens=1):
                break
        except Exception:  # noqa: BLE001 - 预热失败只降级，不拖垮会话
            logger.warning("MLX 首次生成预热失败（首轮会多付一次编译）", exc_info=True)

    def _ensure(self) -> tuple[Any, Any]:
        cached = _MODEL_CACHE.get(self.model)
        if cached is None:
            from mlx_lm import load  # 延迟导入：缺依赖只在真正用本地时炸

            cached = load(self.model)
            _MODEL_CACHE[self.model] = cached
        return cached

    async def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        """流式生成：同步 `stream_generate` 跑在线程里，经队列桥回事件循环。"""
        try:
            from mlx_lm import stream_generate
        except ImportError as exc:  # pragma: no cover - 只有没装 mlx-lm 的机器会走到
            msg = (
                "本地 LLM 需要 mlx-lm（仅 macOS/arm64）：`uv sync --extra local` 或 "
                "`uv run --with mlx-lm ...`。缺依赖时不退回 SimLlm——那会让账单看起来正常。"
            )
            raise RuntimeError(msg) from exc

        self.last_phases = {"start": time.perf_counter()}
        model, tokenizer = await asyncio.to_thread(self._ensure)
        self.last_phases["headers"] = time.perf_counter()
        chat = _apply_template(tokenizer, build_messages(self._system, history, prompt))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        errors: list[BaseException] = []
        # 消费端取消（投机作废/barge-in）要能传进线程：cancel to_thread 的任务
        # 并不会停掉线程本身，没有这个旗标，注定作废的生成会继续占住模型到
        # max_tokens——与紧接着的重新生成抢 Metal 队列（2026-10-17 miss 路径复盘）。
        stop = threading.Event()

        def _run() -> None:
            try:
                for resp in stream_generate(model, tokenizer, chat, max_tokens=self.max_tokens):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, resp.text)
            except BaseException as exc:  # noqa: BLE001 - 线程里的异常要带回事件循环
                errors.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(_run))
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                self.last_phases.setdefault("first_content", time.perf_counter())
                yield chunk
            await task
        finally:
            stop.set()
        if errors:
            raise errors[0]


def build(model: str | None = None, **kwargs: object) -> MlxLlm:
    """工厂入口，与 qwen_llm.build 同形。"""
    return MlxLlm(model, **kwargs)  # type: ignore[arg-type]
