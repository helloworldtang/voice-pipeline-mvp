"""EOU 判停：四路融合 + 三档门控（M2b）。

为什么需要它（技术方案 §12）：「用户说完了没有」不是只靠静音时长。本仓原方案只写了
语义一路，漏了声学/韵律一路。本模块给出可插拔、零模型也能跑的四路融合：

    L0 能量门控（RMS + 滞回）        ← 本文件 VadSilenceEou 的底层
    L1 静音窗（VAD 静音 hangover）   ← VadSilenceEou（可换成 Silero VAD 的 is_speech）
    L2 声学/韵律判停                 ← ProsodyEou（模型无关的终顿启发式）
    L3 文本语义判停                  ← SemanticHeuristicEou（模型无关的完整度启发式）
    判停 = f(L2, L3)，静音窗退化为兜底上限

三档门控（§11 D3）：
    conservative —— 等完整静音窗（500ms），慢但 0% 抢话。**当前默认**
    semantic     —— 静音 + (韵律终顿 或 语义完整) 即判停，快
    aggressive   —— 极短静音即判停，实验开关
    neural (D2)  —— 静音 + 神经判停器（Smart Turn v3）说「说完了」即判停

⚠️ 2026-09-22 实测（`scripts/probe_premature.py`）：「semantic 快且不抢话」是**错的**。
句内停顿 250~350ms 下抢话率 **20~40%**，conservative 为 **0%**。原因是 L2 韵律终顿
是个黏性锁存（说话中任意一帧能量跌到峰值 40% 以下就永久置真），几乎恒为真，
semantic 因此退化成「静音 ≥250ms」。代价换不来：把静音阈值扫到 500ms 能回到 0% 抢话，
但延迟 520ms ≈ conservative 的 515ms —— semantic 在同等安全下**没有收益**。
语义档要可用，前提是换掉 L2 启发式（D2 神经判停器）或拿到真人测试集定标（D5）。

神经版（Smart Turn v3 / turn-detector INT8）是生产可插拔升级（D2）：它们实现同一个
`EouDetector` 接口，FusedEou 直接换上即可，不阻塞 M2b 的门禁与验收。
"""

from __future__ import annotations

import array
import collections
import enum
import math
from collections.abc import Callable
from typing import Protocol

# 能量门限（归一化到满幅 [0,1] 的 RMS）。真实语音 RMS 通常 0.05~0.3，静音=0。
ENERGY_FLOOR = 0.005  # 区分「有声/无声」的下限（韵律与抢话检测用）
ENERGY_THRESHOLD = 0.01  # L0 能量门控触发「有人声」的阈值


class GatingMode(enum.Enum):
    CONSERVATIVE = "conservative"
    SEMANTIC = "semantic"
    AGGRESSIVE = "aggressive"
    NEURAL = "neural"


class EouDetector(Protocol):
    """所有判停器的统一契约（含模型无关的启发式与神经版）。"""

    def push_audio(self, chunk: bytes) -> None: ...
    def update_text(self, partial: str) -> None: ...
    def should_end(self) -> bool: ...
    def reset(self) -> None: ...
    def silence_ms(self) -> float: ...
    def speech_ms(self) -> float: ...


class TerminalDetector(Protocol):
    """「终顿」判据的子契约（L2 的角色）：启发式韵律版与神经版都实现它。

    与 `EouDetector` 分开是因为 L1 静音窗不需要 is_terminal —— 协议按用得到的最小面切。
    """

    def push_audio(self, chunk: bytes) -> None: ...
    def update_text(self, partial: str) -> None: ...
    def reset(self) -> None: ...
    def is_terminal(self) -> bool: ...


def rms_of(chunk: bytes) -> float:
    """PCM16 小端的归一化 RMS（满幅=1），静音返回 0。"""
    if not chunk:
        return 0.0
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0
    total = sum(int(s) * int(s) for s in samples)
    rms = (total / len(samples)) ** 0.5
    return float(rms / 32768.0)


def _frame_ms(chunk: bytes, sample_rate: int) -> float:
    return len(chunk) // 2 / sample_rate * 1000.0


class VadSilenceEou:
    """L0+L1：能量门控 + 静音 hangover。模型无关，CI 可直接测。

    `speech_fn` 可覆盖能量门控——真实生产可传入 Silero VAD 的 `is_speech`，
    其它逻辑（静音窗、触发判定）不变。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_silence_ms: float = 500.0,
        min_speech_ms: float = 150.0,
        energy_threshold: float = ENERGY_THRESHOLD,
        speech_fn: Callable[[bytes], bool] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.energy_threshold = energy_threshold
        self._speech_fn = speech_fn
        self._elapsed = 0.0
        self._silence = 0.0
        self._speech = 0.0
        self._total_speech = 0.0
        self._has_speech = False

    def push_audio(self, chunk: bytes) -> None:
        self._elapsed += _frame_ms(chunk, self.sample_rate)
        if self._speech_fn is not None:
            speech = self._speech_fn(chunk)
        else:
            speech = rms_of(chunk) >= self.energy_threshold
        if speech:
            self._has_speech = True
            self._speech += _frame_ms(chunk, self.sample_rate)
            self._total_speech += _frame_ms(chunk, self.sample_rate)
            self._silence = 0.0
        else:
            self._speech = 0.0
            self._silence += _frame_ms(chunk, self.sample_rate)

    def update_text(self, partial: str) -> None:  # L0/L1 不依赖文本
        pass

    def should_end(self) -> bool:
        return (
            self._has_speech
            and self._total_speech >= self.min_speech_ms
            and (self._silence >= self.min_silence_ms)
        )

    def reset(self) -> None:
        self._elapsed = 0.0
        self._silence = 0.0
        self._speech = 0.0
        self._total_speech = 0.0
        self._has_speech = False

    @property
    def has_speech(self) -> bool:
        return self._has_speech

    def silence_ms(self) -> float:
        return self._silence

    def speech_ms(self) -> float:
        return self._speech


class ProsodyEou:
    """L2 声学/韵律判停（模型无关启发式）：检测「句末终顿」。

    真实判据（Ferrer/Shriberg/Stolcke 2003）：句末音高下降 + 末音节延长 = 结束。
    不引 librosa/torch 的前提下，用短时能量轮廓近似：语音段能量从峰值明显跌落且持续
    下行 → 记为终顿。能量归零（静音）后保留一个 coda 窗，超时则清掉（纯静音不提供信息）。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        coda_ms: float = 250.0,
        floor_ratio: float = 0.4,
        hist: int = 8,
    ) -> None:
        self.sample_rate = sample_rate
        self.coda_ms = coda_ms
        self.floor_ratio = floor_ratio
        self._hist: collections.deque[float] = collections.deque(maxlen=hist)
        self._silence = 0.0
        self._peak = 0.0
        self._terminal = False
        self._has_speech = False

    def push_audio(self, chunk: bytes) -> None:
        r = rms_of(chunk)
        if r >= ENERGY_FLOOR:
            self._has_speech = True
            self._silence = 0.0
            self._hist.append(r)
            self._peak = max(self._peak, r)
            if len(self._hist) >= 3:
                recent = list(self._hist)
                # 末帧能量明显低于峰值且仍在下行 → 终顿
                if recent[-1] < self.floor_ratio * self._peak and recent[-1] < recent[-3]:
                    self._terminal = True
        else:
            self._silence += _frame_ms(chunk, self.sample_rate)
            if self._silence > self.coda_ms:
                self._terminal = False  # 长静音：信息已耗尽，回到未判状态

    def update_text(self, partial: str) -> None:  # L2 不依赖文本
        pass

    def is_terminal(self) -> bool:
        """韵律判定「用户已说完」。"""
        return self._terminal and self._has_speech

    def should_end(self) -> bool:
        return self.is_terminal()

    def reset(self) -> None:
        self._hist.clear()
        self._silence = 0.0
        self._peak = 0.0
        self._terminal = False
        self._has_speech = False

    def silence_ms(self) -> float:
        return self._silence

    def speech_ms(self) -> float:
        return 0.0


class SemanticHeuristicEou:
    """L3 文本语义判停（模型无关启发式）：partial 稳定且看起来完整 → 判停。

    真实判据（LiveKit turn-detector）：文本在语法上完整才敢提前宣称已说完。流式 partial
    通常没有句末标点，故用「稳定未变 + 达到一定长度」近似「说完」——这与停顿瞬间转写相同、
    纯文本有天花板的结论（§12.2）一致，神经版 turn-detector 是生产升级。
    """

    def __init__(self, sample_rate: int = 16000, stable_ms: float = 400.0) -> None:
        self.sample_rate = sample_rate
        self.stable_ms = stable_ms
        self._current = ""
        self._stable = 0.0

    def push_audio(self, chunk: bytes) -> None:
        if self._current:
            self._stable += _frame_ms(chunk, self.sample_rate)

    def update_text(self, partial: str) -> None:
        if partial != self._current:
            self._current = partial
            self._stable = 0.0  # 文本变了 → 稳定性从头计

    def is_complete(self) -> bool:
        if not self._current:
            return False
        if self._stable < self.stable_ms:
            return False
        if any(c in self._current for c in "。！？；.!?;"):
            return True
        # 无标点但已稳定且够长：大概率是短句说完（"今天天气怎么样"）
        return len(self._current) >= 6

    def should_end(self) -> bool:
        return self.is_complete()

    def reset(self) -> None:
        self._current = ""
        self._stable = 0.0

    def silence_ms(self) -> float:
        return 0.0

    def speech_ms(self) -> float:
        return 0.0


class FusedEou:
    """四路融合判停器：接入 Session，统一对外。

    抢占式门控（aggressive）只作实验；semantic 是「快且不抢话」的达标档；
    conservative 是兜底上限。抢话（premature_eou）与漏判（missed_eou）指标由本类量化。
    """

    def __init__(
        self,
        # 默认档经 2026-09-22 抢话实测后定为 conservative：
        #   conservative(500ms)  句内停顿 250/350ms → 抢话 0%，判停延迟 515ms
        #   semantic(250ms)      同条件 → 抢话 20~40%，判停延迟 330ms
        # 用 185ms 的延迟换 20~40% 的抢话率不划算（真人只会更糟）。
        # 300ms 目标因此在本判据下与「不抢话」不可兼得——要兼得得先换掉 L2 启发式（D2）。
        gating: GatingMode = GatingMode.CONSERVATIVE,
        silence: EouDetector | None = None,
        prosody: ProsodyEou | None = None,
        semantic: SemanticHeuristicEou | None = None,
        neural: TerminalDetector | None = None,
        *,
        sample_rate: int = 16000,
        min_silence_ms: float = 500.0,
        min_silence_short_ms: float = 250.0,
        min_silence_aggressive_ms: float = 120.0,
        min_silence_neural_ms: float = 100.0,
        # 神经档的兜底：模型在句末有 ~30% 不吐 complete（实测），不能因此无限等。
        # 800ms 是「句中停顿几乎不会这么长」的量级 —— 兜底触发时已经牺牲了延迟，
        # 但换来的是不会卡死。扫参依据见 docs/实验报告.md §5。
        neural_fallback_ms: float = 800.0,
        max_silence_ms: float = 2000.0,
        premature_window_ms: float = 300.0,
        max_wait_ms: float = 5000.0,
        # 终顿的锁存窗。为什么必须有：ProsodyEou 在静音超过 coda_ms(250) 后会主动
        # 把终顿清零（纯静音不提供信息），而 semantic 档要求静音 ≥250ms 才去读终顿——
        # 两个条件用的是同一个 250，帧粒度 100ms 下永远采不到交集，semantic 档因此
        # 退化成「等满 2000ms 看门狗」（实测判停延迟 1804ms）。
        # 终顿是**事件**不是状态，所以这里锁存它最近的成立时刻，按窗口判。
        terminal_window_ms: float = 400.0,
    ) -> None:
        self.gating = gating
        self._silence = silence or VadSilenceEou(
            sample_rate=sample_rate, min_silence_ms=min_silence_ms
        )
        self._prosody = prosody or ProsodyEou(sample_rate=sample_rate)
        self._semantic = semantic or SemanticHeuristicEou(sample_rate=sample_rate)
        self._neural = neural
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_silence_short_ms = min_silence_short_ms
        self.min_silence_aggressive_ms = min_silence_aggressive_ms
        self.min_silence_neural_ms = min_silence_neural_ms
        self.neural_fallback_ms = neural_fallback_ms
        self.max_silence_ms = max_silence_ms
        self.premature_window_ms = premature_window_ms
        self.max_wait_ms = max_wait_ms
        self.terminal_window_ms = terminal_window_ms

        self._elapsed = 0.0
        self._last_terminal_ms = -math.inf  # 韵律终顿最近一次成立的时刻（锁存）
        self._decided = False
        self._decided_at_ms = 0.0
        # 抢话观察窗的「残余量」：reset 不抹掉证据（见 reset 的注释）
        self._premature_watch_ms = 0.0
        self.premature_eou = False
        self.missed_eou = False

    def push_audio(self, chunk: bytes) -> None:
        frame_ms = _frame_ms(chunk, self.sample_rate)
        self._elapsed += frame_ms
        self._silence.push_audio(chunk)
        if self._prosody is not None:
            self._prosody.push_audio(chunk)
            # 锁存终顿时刻：它一闪即逝（静音一累积就被 coda 清掉），不记下来就读不到
            if self._prosody.is_terminal():
                self._last_terminal_ms = self._elapsed
        if self._semantic is not None:
            self._semantic.push_audio(chunk)
        if self._neural is not None:
            self._neural.push_audio(chunk)
        # 抢话：判停决定之后、窗口内又出现人声 → 这次判停是错的。
        # 窗口有两条活路：本代表内（_decided 仍真，elapsed 计）；
        # 或跨过 reset 的残余窗（run_turn 采完指标立即 reset，判停后 300ms 内的
        # 人声大多落在 reset 之后——老实现在这里把 _decided 清掉，抢话从此恒 0）。
        if self._decided:
            in_window = (self._elapsed - self._decided_at_ms) <= self.premature_window_ms
        else:
            in_window = self._premature_watch_ms > 0.0
        if self._premature_watch_ms > 0.0:
            self._premature_watch_ms -= frame_ms
        if not self.premature_eou and rms_of(chunk) >= ENERGY_FLOOR and in_window:
            self.premature_eou = True

    def update_text(self, partial: str) -> None:
        if self._semantic is not None:
            self._semantic.update_text(partial)
        if self._neural is not None:
            self._neural.update_text(partial)

    def should_end(self) -> bool:
        if self._decided:
            return True
        if self._elapsed > self.max_wait_ms and not (
            isinstance(self._silence, VadSilenceEou) and self._silence.has_speech
        ):
            # 看门狗：长时间无人声也没判定 → 漏判（由上层决定是否强制收尾）
            self.missed_eou = True
        if not (isinstance(self._silence, VadSilenceEou) and self._silence.has_speech):
            return False
        sil = self._silence.silence_ms()
        if sil >= self.max_silence_ms:
            self._latch()
            return True
        if self.gating is GatingMode.CONSERVATIVE:
            if sil >= self.min_silence_ms:
                self._latch()
                return True
            return False
        if self.gating is GatingMode.AGGRESSIVE:
            if sil >= self.min_silence_aggressive_ms:
                self._latch()
                return True
            return False
        if self.gating is GatingMode.NEURAL:
            # D2：判停交给神经模型，静音窗退化成「给模型一个立足点」的短确认窗。
            #   - 短窗（默认 100ms）后才问模型：输入末尾没有静音时模型在分布外，
            #     概率是噪声（实测 0.01↔0.99 乱跳）——这是必须先等静音的原因。
            #   - 模型不说 complete 也不许无限等，到 neural_fallback_ms 强制判停。
            if self._neural is None:
                return False
            if sil >= self.min_silence_neural_ms and self._neural.is_terminal():
                self._latch()
                return True
            if sil >= self.neural_fallback_ms:
                self._latch()
                return True
            return False
        # SEMANTIC：静音 + (最近出现过韵律终顿 或 语义完整)。
        # 注意是「最近出现过」而不是「此刻成立」——此刻成立要求静音还没被 coda 清掉，
        # 与 sil>=250 互斥（见 terminal_window_ms 的注释）。
        if sil >= self.min_silence_short_ms:
            terminal_recent = self._elapsed - self._last_terminal_ms <= self.terminal_window_ms
            ok = terminal_recent or (self._semantic is not None and self._semantic.is_complete())
            if ok:
                self._latch()
                return True
        return False

    def _latch(self) -> None:
        if not self._decided:
            self._decided = True
            self._decided_at_ms = self._elapsed

    def reset(self) -> None:
        # 抢话观察窗**横跨 reset**：判停发生在静音里，人声回到窗口内往往在 reset
        # 之后（Session.run_turn 采完指标立即 reset）。这里把窗口的剩余量带过去，
        # 按**帧时长**而不是墙钟递减——测试与回放才确定。
        if self._decided:
            self._premature_watch_ms = max(
                0.0, self._decided_at_ms + self.premature_window_ms - self._elapsed
            )
        else:
            self._premature_watch_ms = 0.0
        self._silence.reset()
        if self._prosody is not None:
            self._prosody.reset()
        if self._semantic is not None:
            self._semantic.reset()
        if self._neural is not None:
            self._neural.reset()
        self._elapsed = 0.0
        self._last_terminal_ms = -math.inf
        self._decided = False
        self._decided_at_ms = 0.0
        # ⚠️ premature_eou **不在 reset 清**：它是「刚过去那次判停」的证据，而账单
        # 要到 turn.end（秒级之后）才定型——这里清掉等于取证窗口在定型前被抹
        # （判停后、reset 前开口的那种抢话就丢在缝里）。唯一的清点是 take_premature。
        self.missed_eou = False

    def take_premature(self) -> bool:
        """读走抢话标志（读后即清，连同残余观察窗）。

        为什么读后即清：标志的取证窗口在判停后 ~300ms，而账单定型在 turn.end
        （秒级）——Session 必须在**记账时**而不是判停时来读。不清的话，上一轮
        的幽灵标志会被下一轮 `_collect_eou_metrics` 当成本轮的抢话采走。
        """
        value = self.premature_eou
        self.premature_eou = False
        self._premature_watch_ms = 0.0
        return value

    def silence_ms(self) -> float:
        return self._silence.silence_ms()

    def speech_ms(self) -> float:
        return self._silence.speech_ms()

    @property
    def neural(self) -> TerminalDetector | None:
        """神经判停器（若挂载）——测试与调参要读它的概率。"""
        return self._neural

    def eou_latency_ms(self) -> float:
        """判停时刻的静音时长 —— 口径B 残余间隙里判停占的那一段。"""
        return self._silence.silence_ms()
