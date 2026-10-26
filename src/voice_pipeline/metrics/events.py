"""每轮延迟账单 + waterfall 打点：回应「300ms 怎么得到的」的证据就在这里。

所有时间戳相对「会话钟」（session start = 0），单位 ms。双口径：
- 口径A（文章口径）  = first_audio_out − vad_onset      —— 用户开口 → AI 首音
- 口径B（质疑方口径）= first_audio_out − utterance_end  —— 用户说完 → AI 首音

M1 只有 sim 链路，vad_onset 缺位时口径A 如实留空，不做填补。
2026-10-27 补：vad_onset / asr_first_char 已在 `Session.on_audio` 埋上，口径A 现在
可测；但它带已知系统性偏差（见下方 WATERFALL_POINTS 的注释），只作参考，不与口径B 横比。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 避免与 metrics/trace.py 形成 import 环（那边要用 SessionClock）
    from voice_pipeline.metrics.trace import Tracer

# 口径A 的两个锚点（2026-10-27 补，都埋在 `Session.on_audio`）：
#   vad_onset      = 本轮**判停之前**第一次被 barge-in L1 判定「有人声」的那一帧。
#                    ⚠️ 已知系统性偏差：L1 是「连续 3 帧超阈」，100ms 帧节奏下它比
#                    真实开口晚 ~300ms；且是能量判定不是 VAD onset。所以 caliber_a_ms
#                    是**偏小**的估计——只作参考，不拿去对外宣称。
#   asr_first_char = ASR 第一次吐出非空 partial（real 链路复用 eou.update_text 那次读）。
# `fragment_commit` **仍未埋**：仓内没有与它对应的时刻，留在表里是预留位，账单里
# 不会出现。宁可空着，也不拿别的时刻冒领它（与「缺失写 -1 不写 0」同一条纪律）。
WATERFALL_POINTS: tuple[str, ...] = (
    "vad_onset",
    "asr_first_char",
    "fragment_commit",
    "eou",
    "utterance_end",
    # 预合成 filler 的首片（体感首音）。**口径B 不用它**——first_audio_out 仍只指
    # 真实回答的首片，否则 caliber_b_ms 会静默变义（见 core/filler.py 的计量纪律）。
    "first_sound",
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

    def ms_at(self, ts: float) -> float:
        """把 `time.perf_counter()` 的绝对值换算成会话 ms。

        provider 在**另一个线程**里记的时刻（DashScope SDK 的 on_open 之类）
        只有 perf_counter 原值。换算必须走时钟统一做：两边各自找原点相减的话，
        span 会整体漂移，而账面上看不出来（2026-10-17 深拆时加）。
        """
        return (ts - self._t0) * 1000.0


@dataclass(slots=True)
class TurnMetrics:
    """单轮账单。mark 只写第一次到达的时间（ repeats 不覆盖）。"""

    clock: SessionClock = field(default_factory=SessionClock)
    # 追踪器（metrics/trace.py）：Session 构造后注入，trace_id 与会话一致。
    # 为什么挂在 TurnMetrics 上：调度器手里已经有 metrics，不必为了埋点改它们的签名。
    tracer: Tracer | None = None
    marks: dict[str, float] = field(default_factory=dict)
    sentences: int = 0
    audio_bytes: int = 0
    # 预合成 filler 是否在本轮播过（2026-10-10，体感优化）。**不进口径B**：
    # caliber_b_ms 永远只算「真实回答的首片」，否则历史账与本轮的账不可比。
    filler_used: bool = False
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

    def reset_turn(self, keep: Sequence[str] = ()) -> None:
        """开新轮：清空上一轮的打点与计数。

        `keep`：跨 reset 保留的打点名。**为什么需要它**：口径A 的两个锚点
        （`vad_onset` / `asr_first_char`）打在「本轮第一帧上行、用户开口」那一侧，
        而每轮的清账发生在**判停之后**——不保留就会把用户开口那一刻连同上一轮的
        账一起清掉（2026-10-27 实测：`caliber_a_ms` 因此恒缺，与 `eou_latency_ms`
        曾是同一个失效模式）。这两个时刻属于**本轮**，不是上一轮的残留。

        mark() 只写第一次到达——这是单轮 waterfall 的语义；Session 跨轮复用同一
        实例，不清的话第 2 轮起的账单会永远复读第 1 轮（真链路两轮实测抓到：
        caliber_b_ms 一字不差，probe_multi_turn 可复现）。调用方须在采完上一轮
        EOU 指标之后、本轮任何 mark 之前调用。
        """
        carried = {name: self.marks[name] for name in keep if name in self.marks}
        self.marks.clear()
        self.marks.update(carried)
        self.sentences = 0
        self.audio_bytes = 0
        self.filler_used = False
        self.tts_first_chars = -1
        self.premature_eou = False
        self.missed_eou = False
        self.eou_latency_ms = -1.0
        self.eou_mode = ""
        self.barge_in_count = 0
        self.barge_in_latency_ms = -1.0
        # ⚠️ 这里**不清** span 树：树的生命周期归 Session 管（第一帧上行开根、
        # turn.end 闭根、下一轮第一帧 reset）。在 run_turn 开头清会把**判停前**的
        # 上行帧/EOU/partial 全部抹掉，trace 只剩"说完之后"那一段——2026-10-17
        # 实测就是这么丢的：树里 0 个 uplink.frame，而 mic 一直在收音。

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
        # eou_mode 编码：0=conservative, 1=semantic, 2=aggressive, 3=neural,
        # -1=未用判停器（缺失）。
        # ⚠️ 新增档位必须同步补这张表：`neural` 是**默认档**，漏掉它会让默认
        # 配置下的每一轮账单都写 -1.0（“未用判停器”）——而这一轮明明用了判停
        # （eou_latency_ms 有值），两个字段自相矛盾。与“缺失值别写 0”同一类错：
        # 探测不到的档位码看起来像个合法值，所以能潜伏。
        _mode_code = {
            "conservative": 0.0,
            "semantic": 1.0,
            "aggressive": 2.0,
            "neural": 3.0,
        }.get(self.eou_mode, -1.0)
        bill["eou_mode"] = _mode_code
        bill["premature_eou"] = 1.0 if self.premature_eou else 0.0
        bill["missed_eou"] = 1.0 if self.missed_eou else 0.0
        bill["eou_latency_ms"] = round(self.eou_latency_ms, 3)
        # barge-in 指标（M3）
        bill["barge_in_count"] = float(self.barge_in_count)
        bill["barge_in_latency_ms"] = round(self.barge_in_latency_ms, 3)
        if first_audio is not None and vad_onset is not None:
            bill["caliber_a_ms"] = round(first_audio - vad_onset, 3)
        # 体感首音（含 filler）：与口径B **分开报**——filler 省的是「用户以为在等」
        # 的那段，真实回答的首音一分不省。缺失如实 -1，不填 0。
        first_sound = self.marks.get("first_sound")
        bill["filler_used"] = 1.0 if self.filler_used else 0.0
        bill["first_sound_ms"] = (
            round(first_sound - utterance_end, 3)
            if first_sound is not None and utterance_end is not None
            else -1.0
        )
        for point in WATERFALL_POINTS:
            value = self.marks.get(point)
            if value is not None:
                bill[point] = round(value, 3)
        return bill

    def waterfall(self) -> list[tuple[str, float]]:
        """按标准顺序输出已有打点，用于时间线与报告。"""
        return [(p, round(self.marks[p], 3)) for p in WATERFALL_POINTS if p in self.marks]
