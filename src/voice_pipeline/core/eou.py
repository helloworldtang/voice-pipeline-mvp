"""EOU 判停：四路融合 + 三档门控（M2b）。

为什么需要它（技术方案 §12）：「用户说完了没有」不是只靠静音时长。本仓原方案只写了
语义一路，漏了声学/韵律一路。本模块给出可插拔、零模型也能跑的四路融合：

    L0 能量门控（RMS + 滞回）        ← 本文件 VadSilenceEou 的底层
    L1 静音窗（VAD 静音 hangover）   ← VadSilenceEou（可换成 Silero VAD 的 is_speech）
    L2 声学/韵律判停                 ← ProsodyEou（模型无关的终顿启发式）
    L3 文本语义判停                  ← SemanticHeuristicEou（模型无关的完整度启发式）
    判停 = f(L2, L3)，静音窗退化为兜底上限

三档门控（§11 D3）：
    conservative —— 等完整静音窗（兜底上限），慢但最稳，M2 接入前的默认
    semantic     —— 静音 + (韵律终顿 或 语义完整) 即判停，快且不抢话的达标档
    aggressive   —— 极短静音即判停，实验开关，用于证明「不是做不到更快，是选了不抢话」

神经版（Smart Turn v3 / turn-detector INT8）是生产可插拔升级（D2）：它们实现同一个
`EouDetector` 接口，FusedEou 直接换上即可，不阻塞 M2b 的门禁与验收。
"""

from __future__ import annotations

import array
import collections
import enum
from collections.abc import Callable
from typing import Protocol

# 能量门限（归一化到满幅 [0,1] 的 RMS）。真实语音 RMS 通常 0.05~0.3，静音=0。
ENERGY_FLOOR = 0.005  # 区分「有声/无声」的下限（韵律与抢话检测用）
ENERGY_THRESHOLD = 0.01  # L0 能量门控触发「有人声」的阈值


class GatingMode(enum.Enum):
    CONSERVATIVE = "conservative"
    SEMANTIC = "semantic"
    AGGRESSIVE = "aggressive"


class EouDetector(Protocol):
    """所有判停器的统一契约（含模型无关的启发式与神经版）。"""

    def push_audio(self, chunk: bytes) -> None: ...
    def update_text(self, partial: str) -> None: ...
    def should_end(self) -> bool: ...
    def reset(self) -> None: ...
    def silence_ms(self) -> float: ...
    def speech_ms(self) -> float: ...


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
        return self._has_speech and self._total_speech >= self.min_speech_ms and (
            self._silence >= self.min_silence_ms
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
        gating: GatingMode = GatingMode.CONSERVATIVE,
        silence: EouDetector | None = None,
        prosody: ProsodyEou | None = None,
        semantic: SemanticHeuristicEou | None = None,
        *,
        sample_rate: int = 16000,
        min_silence_ms: float = 500.0,
        min_silence_short_ms: float = 250.0,
        min_silence_aggressive_ms: float = 120.0,
        max_silence_ms: float = 2000.0,
        premature_window_ms: float = 300.0,
        max_wait_ms: float = 5000.0,
    ) -> None:
        self.gating = gating
        self._silence = silence or VadSilenceEou(
            sample_rate=sample_rate, min_silence_ms=min_silence_ms
        )
        self._prosody = prosody or ProsodyEou(sample_rate=sample_rate)
        self._semantic = semantic or SemanticHeuristicEou(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_silence_short_ms = min_silence_short_ms
        self.min_silence_aggressive_ms = min_silence_aggressive_ms
        self.max_silence_ms = max_silence_ms
        self.premature_window_ms = premature_window_ms
        self.max_wait_ms = max_wait_ms

        self._elapsed = 0.0
        self._decided = False
        self._decided_at_ms = 0.0
        self.premature_eou = False
        self.missed_eou = False

    def push_audio(self, chunk: bytes) -> None:
        self._elapsed += _frame_ms(chunk, self.sample_rate)
        self._silence.push_audio(chunk)
        if self._prosody is not None:
            self._prosody.push_audio(chunk)
        if self._semantic is not None:
            self._semantic.push_audio(chunk)
        # 抢话：判停决定之后、窗口内又出现人声 → 这次判停是错的
        if (
            self._decided
            and not self.premature_eou
            and rms_of(chunk) >= ENERGY_FLOOR
            and (self._elapsed - self._decided_at_ms) <= self.premature_window_ms
        ):
            self.premature_eou = True

    def update_text(self, partial: str) -> None:
        if self._semantic is not None:
            self._semantic.update_text(partial)

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
        # SEMANTIC：静音 + (韵律终顿 或 语义完整)
        if sil >= self.min_silence_short_ms:
            ok = (self._prosody is not None and self._prosody.is_terminal()) or (
                self._semantic is not None and self._semantic.is_complete()
            )
            if ok:
                self._latch()
                return True
        return False

    def _latch(self) -> None:
        if not self._decided:
            self._decided = True
            self._decided_at_ms = self._elapsed

    def reset(self) -> None:
        self._silence.reset()
        if self._prosody is not None:
            self._prosody.reset()
        if self._semantic is not None:
            self._semantic.reset()
        self._elapsed = 0.0
        self._decided = False
        self._decided_at_ms = 0.0
        self.premature_eou = False
        self.missed_eou = False

    def silence_ms(self) -> float:
        return self._silence.silence_ms()

    def speech_ms(self) -> float:
        return self._silence.speech_ms()

    def eou_latency_ms(self) -> float:
        """判停时刻的静音时长 —— 口径B 残余间隙里判停占的那一段。"""
        return self._silence.silence_ms()
