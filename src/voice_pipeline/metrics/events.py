"""每轮延迟账单 + waterfall 打点：回应「300ms 怎么得到的」的证据就在这里。

所有时间戳相对「会话钟」（session start = 0），单位 ms。双口径：
- 口径A（文章口径）  = first_audio_out − vad_onset      —— 用户开口 → AI 首音
- 口径B（质疑方口径）= first_audio_out − utterance_end  —— 用户说完 → AI 首音

M1 只有 sim 链路，vad_onset 缺位时口径A 如实留空，不做填补。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

WATERFALL_POINTS: tuple[str, ...] = (
    "vad_onset",
    "asr_first_char",
    "fragment_commit",
    "eou",
    "utterance_end",
    "llm_first_token",
    "first_sentence_done",
    "tts_first_chunk",
    "first_audio_out",
    "turn_end",
)


class SessionClock:
    """会话钟：所有时延都相对它计算。"""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0


@dataclass(slots=True)
class TurnMetrics:
    """单轮账单。mark 只写第一次到达的时间（ repeats 不覆盖）。"""

    clock: SessionClock = field(default_factory=SessionClock)
    marks: dict[str, float] = field(default_factory=dict)
    sentences: int = 0
    audio_bytes: int = 0

    def mark(self, name: str) -> None:
        if name not in self.marks:
            self.marks[name] = self.clock.elapsed_ms()

    def elapsed(self, name: str) -> float | None:
        return self.marks.get(name)

    def bill(self) -> dict[str, float]:
        """对外报告的口径：只有实测到的数字，缺失就缺失，不做推测填补。"""
        first_audio = self.marks.get("first_audio_out")
        utterance_end = self.marks.get("utterance_end")
        vad_onset = self.marks.get("vad_onset")

        bill: dict[str, float] = {
            "caliber_b_ms": round(first_audio - utterance_end, 3)
            if first_audio is not None and utterance_end is not None
            else -1.0,
            "sentences": float(self.sentences),
            "audio_bytes": float(self.audio_bytes),
        }
        if first_audio is not None and vad_onset is not None:
            bill["caliber_a_ms"] = round(first_audio - vad_onset, 3)
        for point in WATERFALL_POINTS:
            value = self.marks.get(point)
            if value is not None:
                bill[point] = round(value, 3)
        return bill

    def waterfall(self) -> list[tuple[str, float]]:
        """按标准顺序输出已有打点，用于时间线与报告。"""
        return [(p, round(self.marks[p], 3)) for p in WATERFALL_POINTS if p in self.marks]
