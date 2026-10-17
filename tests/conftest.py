"""测试默认关掉「任务级预热」。

`DashScopeTts` 默认会在 prewarm 时额外合成一小段（`VOICE_TTS_TASK_WARM=1`，
换来冷启动首轮 tts.connect 840→37ms）。它是**背景副作用**：会多借一次连接、
多喊一句「好。」，把池用例原本要断言的计数口径全打乱。

所以测试里统一关掉（需要验它的用例自己 setenv 打开）——这比给 8 条用例逐个
加偏移量更不容易掩盖真回归。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_tts_task_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_TTS_TASK_WARM", "0")
