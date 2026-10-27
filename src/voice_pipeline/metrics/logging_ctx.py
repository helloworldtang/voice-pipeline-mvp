"""日志与 trace_id 的接缝——让「这一行日志属于哪个会话」可查。

为什么要有它：`Session` 的 docstring 一直写着「trace_id 贯穿日志与账单」，
但账单里有、日志里没有——排查线上问题时只能看着一堆无主的日志行猜。它是
`背景与基线.md` §4.5 服务化架构判据里明文要求的一项（结构化日志 + 每会话 trace_id）。

为什么用 ContextVar 而不是 `LoggerAdapter`：一条 WS 连接一个会话，而事件循环里
同时跑着多条连接，logger 是模块级全局——只有上下文变量能把「这一行属于哪个
会话」带进去。`asyncio.create_task` 会复制当前上下文，所以在 `ws_endpoint` 里
绑定一次，之后派生的一切协程（pump / sender / run_turn）自动带上。

为什么改 `LogRecordFactory` 而不是加 `logging.Filter`：Filter 只有在 formatter
的格式串里写了 `%(trace_id)s` 才有用（而且挂在 logger 上的 filter 对子 logger
的传播记录不生效，必须挂 handler，等于要接管别人的 handler 配置）。工厂是
**所有** LogRecord 的唯一入口，注入一次，任何 formatter 都能用，别人的配置不动。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar

_TRACE_ID: ContextVar[str | None] = ContextVar("voice_trace_id", default=None)

# 会话外（启动期、脚本、无连接的请求）打成一个显眼的占位符，而不是空串——
# 空串会被当成"有 trace_id 但为空"，那就又是一个能潜伏的假值。
NO_TRACE = "-"

LOG_FORMAT = "%(asctime)s %(levelname)-7s [trace=%(trace_id)s] %(name)s: %(message)s"

_old_factory: Callable[..., logging.LogRecord] | None = None


def bind_trace_id(trace_id: str) -> None:
    """把当前上下文（一条连接 / 一个会话）绑到 trace_id 上。"""
    _TRACE_ID.set(trace_id)


def current_trace_id() -> str:
    return _TRACE_ID.get() or NO_TRACE


def _factory(*args: object, **kwargs: object) -> logging.LogRecord:
    assert _old_factory is not None  # 装工厂时才赋值，断言给 mypy 看
    record = _old_factory(*args, **kwargs)
    record.trace_id = current_trace_id()
    return record


def install_trace_logging(level: int = logging.INFO, *, fmt: str = LOG_FORMAT) -> None:
    """装好 trace_id 注入 + 兜底的日志格式。**幂等，且只在没人配过时接管**。

    为什么「只在没人配过时接管」：进程可能已经被 uvicorn / pytest 配好了日志，
    那时强行改 formatter 是在拆别人的配置。我们的最低要求是 trace_id **能取到**
    （工厂保证），格式只是本机直跑时的兜底。
    """
    global _old_factory
    if _old_factory is None:
        _old_factory = logging.getLogRecordFactory()
        logging.setLogRecordFactory(_factory)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt)
