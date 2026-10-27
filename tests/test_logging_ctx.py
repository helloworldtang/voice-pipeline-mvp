"""trace_id 进日志：绑定后所有协程派生日志可归属到会话（审查第 7 项的验收）。"""

from __future__ import annotations

import logging

import pytest

from voice_pipeline.metrics.logging_ctx import (
    bind_trace_id,
    current_trace_id,
    install_trace_logging,
    trace_id_of,
)


def test_unbound_logs_as_no_trace_not_empty() -> None:
    """会话外是显式占位符 `-`，不是空串——空串是能潜伏的假值。"""
    bind_trace_id("")
    assert current_trace_id() == "-", repr(current_trace_id())


def test_factory_injects_trace_id_into_every_record(caplog: pytest.LogCaptureFixture) -> None:
    """注入点在 LogRecord 工厂：任何 logger、任何已配置的 handler 都能取到
    record.trace_id（格式是不是我们兜底的那个无关紧要——pytest 就接管了格式）。"""
    install_trace_logging()
    with caplog.at_level(logging.WARNING):
        bind_trace_id("abc123")
        logging.getLogger("voice_pipeline.test").warning("hello")
        assert trace_id_of(caplog.records[-1]) == "abc123"

        # 上下文切换后（换会话）同一工厂读新值——ContextVar 的意义就在这
        bind_trace_id("def456")
        logging.getLogger("voice_pipeline.test").warning("again")
        assert trace_id_of(caplog.records[-1]) == "def456"


def test_install_is_idempotent() -> None:
    """装两次不包两层工厂（重复包装会让每条日志多一次无谓调用，且难排查）。"""
    install_trace_logging()
    factory_after_first = logging.getLogRecordFactory()
    install_trace_logging()
    assert logging.getLogRecordFactory() is factory_after_first
