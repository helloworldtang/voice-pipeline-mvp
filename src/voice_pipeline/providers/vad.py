"""Silero VAD（sherpa-onnx）： utterance 的边界探测，EOU 判定 (L1) 铺垫。

现在有两条独立的信息源：
- VAD：有人声 / 没人声（能量+神经网络），决定「什么时候闭嘴」
- 语义判停：文本是不是语法完整，决定「敢不敢提前宣称已说完」

两条信息源的**组合**才是 M2 判停主线。本类只负责 VAD 一路，不做策略决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sherpa_onnx

from voice_pipeline.providers.sherpa import DEFAULT_MODEL_ROOT, _models, pcm16_to_f32


@dataclass(frozen=True, slots=True)
class VadConfig:
    threshold: float = 0.5
    min_silence_duration: float = 0.25  # 秒；短于此的静音不算说完（挡住句中停顿）
    min_speech_duration: float = 0.15  # 秒；短于此的语音不成为一段 utterance
    window_size: int = 512  # 16kHz 下 32ms


class SileroVad:
    """帧级 VAD：push_audio 喂 16k/PCM16，is_speech / segments 读状态。"""

    def __init__(
        self,
        config: VadConfig | None = None,
        model_root: Path | str | None = None,
    ) -> None:
        self.cfg = config or VadConfig()
        root = _models(model_root)
        model = root / "vad" / "silero_vad.onnx"
        if not model.is_file():
            msg = f"missing VAD model: {model}"
            raise FileNotFoundError(msg)
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model)
        cfg.silero_vad.threshold = self.cfg.threshold
        cfg.silero_vad.min_silence_duration = self.cfg.min_silence_duration
        cfg.silero_vad.min_speech_duration = self.cfg.min_speech_duration
        cfg.silero_vad.window_size = self.cfg.window_size
        cfg.sample_rate = 16000
        if not cfg.validate():
            msg = f"invalid VAD config: {model}"
            raise RuntimeError(msg)
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    @property
    def frame_ms(self) -> float:
        return self.cfg.window_size / 16000 * 1000.0

    def push_audio(self, chunk: bytes) -> list[tuple[float, float]]:
        """喂一段 16k/PCM16，返回本段里 VAD 判定结束的语音段 [(start_ms, dur_ms)]。

        ⚠️ 第二元是**时长**不是终点（sherpa 的 seg.samples 长度换算），
        接线时别当 end_ms 用（2026-11-01 审查纠偏：docstring 曾写错）。
        """
        self._vad.accept_waveform(pcm16_to_f32(chunk))
        out: list[tuple[float, float]] = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append((seg.start / 16000 * 1000.0, len(seg.samples) / 16000 * 1000.0))
            self._vad.pop()
        return out

    def is_speech(self) -> bool:
        return bool(self._vad.is_speech_detected())

    def flush(self) -> list[tuple[float, float]]:
        self._vad.flush()
        out: list[tuple[float, float]] = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append((seg.start / 16000 * 1000.0, len(seg.samples) / 16000 * 1000.0))
            self._vad.pop()
        return out

    @staticmethod
    def default_root() -> Path:
        return DEFAULT_MODEL_ROOT
