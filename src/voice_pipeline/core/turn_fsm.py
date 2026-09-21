"""barge-in 状态机：一轮对话在「听-想-说」之间的合法流转。

为什么要有状态机：barge-in（用户插话打断 AI）不是「再发一个事件」这么简单，
它牵涉到调度器在飞任务的丢弃、播放端的停播、以及打断后该回到哪个状态继续听。
没有显式的状态，这些并发事件会互相踩脚——这是语音对话系统最经典的竞态来源。

状态：
- IDLE        会话建立，尚未开始听
- LISTENING   听用户说（含 EOU 判停等待中的静音段）
- THINKING    已判停，LLM+TTS 在算（pipelined 下其实早就开算了，这里只是语义起点）
- SPEAKING    AI 正在播放音频
- INTERRUPTED 用户打断，正在清场（取消在飞 + 通知播放端停播）

barge-in 的合法落点只有「AI 在想或在说时用户又开口」：
- THINKING  → INTERRUPTED → LISTENING（还没出声就被打断：之前那次判停/投机作废，重新听）
- SPEAKING  → INTERRUPTED → LISTENING（出声中被打断：停播 + 丢掉剩余，重新听）
"""

from __future__ import annotations

from enum import Enum


class TurnState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class IllegalTransition(ValueError):
    """FSM 走到了设计之外的转换。

    strict 模式下抛，用于 CI 抓回归；生产路径把 strict=False，未知转换静默忽略
    而不是崩掉一整轮对话。
    """


# 允许到达的目标状态集合。不在集合里的转换即非法。
_TRANSITIONS: dict[TurnState, frozenset[TurnState]] = {
    TurnState.IDLE: frozenset({TurnState.LISTENING}),
    TurnState.LISTENING: frozenset({TurnState.THINKING}),
    TurnState.THINKING: frozenset(
        {TurnState.SPEAKING, TurnState.INTERRUPTED, TurnState.LISTENING}
    ),
    TurnState.SPEAKING: frozenset({TurnState.LISTENING, TurnState.INTERRUPTED}),
    TurnState.INTERRUPTED: frozenset({TurnState.LISTENING, TurnState.IDLE}),
}


class TurnFsm:
    """会话轮次状态机。move 返回是否成功（strict=False 时非法转换返回 False）。"""

    def __init__(self, strict: bool = False) -> None:
        self.state = TurnState.IDLE
        self.strict = strict
        self.transitions = 0
        self.last_error: str | None = None

    def move(self, to: TurnState) -> bool:
        allowed = _TRANSITIONS.get(self.state, frozenset())
        if to not in allowed:
            msg = f"illegal FSM transition: {self.state.value} -> {to.value}"
            self.last_error = msg
            if self.strict:
                raise IllegalTransition(msg)
            return False
        self.state = to
        self.transitions += 1
        return True

    def can(self, to: TurnState) -> bool:
        """当前状态下 to 是否合法（不实际移动）。"""
        return to in _TRANSITIONS.get(self.state, frozenset())
