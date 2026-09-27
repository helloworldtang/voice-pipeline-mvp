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
    # 送进 TTS 的第一句有多少字。TTS 是端到端最大的一格，而它的耗时几乎线性于
    # 字数（本机实测 melo +37ms/字、kokoro +47ms/字）—— 不知道字数就无法解释
    # 这一格的波动，也无法验证「让 LLM 说短一点」到底值多少 ms。
    tts_first_chars: int = -1
    # EOU 判停指标（§3 / §12.4）：抢话与漏判是一等验收指标
    premature_eou: bool = False  # FP：判停后窗口内用户又开口 = 抢话
    missed_eou: bool = False  # FN：该判未判，用户闭嘴后仍在干等
    eou_latency_ms: float = -1.0  # 判停时刻的静音时长（残余间隙里判停占的一段）
    eou_mode: str = ""  # 本次判停用的门控档
    # barge-in 指标（M3）：用户插话打断 AI 的次数与时刻
    barge_in_count: int = 0  # 本轮被打断次数
    barge_in_latency_ms: float = -1.0  # 最近一次打断发生的会话钟时刻（ms）

    def mark(self, name: str) -> None:
        if name not in self.marks:
            self.marks[name] = self.clock.elapsed_ms()

    def reset_turn(self) -> None:
        """开新轮：清空上一轮的打点与计数。

        mark() 只写第一次到达——这是单轮 waterfall 的语义；Session 跨轮复用同一
        实例，不清的话第 2 轮起的账单会永远复读第 1 轮（真链路两轮实测抓到：
        caliber_b_ms 一字不差，probe_multi_turn 可复现）。调用方须在采完上一轮
        EOU 指标之后、本轮任何 mark 之前调用。
        """
        self.marks.clear()
        self.sentences = 0
        self.audio_bytes = 0
        self.tts_first_chars = -1
        self.premature_eou = False
        self.missed_eou = False
        self.eou_latency_ms = -1.0
        self.eou_mode = ""
        self.barge_in_count = 0
        self.barge_in_latency_ms = -1.0

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
            "tts_first_chars": float(self.tts_first_chars),
            "audio_bytes": float(self.audio_bytes),
        }
        # EOU 判停指标（§3 / §12.4）：抢话/漏判是一等验收指标
        # eou_mode 编码：0=conservative, 1=semantic, 2=aggressive, -1=未用判停器
        _mode_code = {"conservative": 0.0, "semantic": 1.0, "aggressive": 2.0}.get(
            self.eou_mode, -1.0
        )
        bill["eou_mode"] = _mode_code
        bill["premature_eou"] = 1.0 if self.premature_eou else 0.0
        bill["missed_eou"] = 1.0 if self.missed_eou else 0.0
        bill["eou_latency_ms"] = round(self.eou_latency_ms, 3)
        # barge-in 指标（M3）
        bill["barge_in_count"] = float(self.barge_in_count)
        bill["barge_in_latency_ms"] = round(self.barge_in_latency_ms, 3)
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
