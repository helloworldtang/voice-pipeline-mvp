"""神经判停器：Pipecat Smart Turn v3.2（Whisper-Tiny encoder + 线性头，8MB INT8）。

它替换的是 L2 声学/韵律那一路（`ProsodyEou` 的启发式终顿）。为什么必须换：
2026-09-22 实测，启发式 L2 的「终顿」是个黏性锁存（说话中任意一帧能量跌到峰值
40% 以下就永久置真），几乎恒为真，导致 SEMANTIC 档退化成「静音 ≥250ms」，在句内
停顿 250~350ms 下抢话 20~40%。神经版看的是整段音频的韵律与语义收尾，不是单点能量。

模型：pipecat-ai/smart-turn-v3.2-cpu.onnx（BSD-2），输入 (1, 80, 800) log-mel，
输出 sigmoid 概率 —— **>0.5 = 用户说完了**。

依赖：onnxruntime + numpy（项目已有）。权重需单独下载到 models/eou/，没有就退回
启发式（由 `factory.build_eou` 决定，并告警）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from voice_pipeline.core.eou_mel import (
    SAMPLE_RATE,
    WINDOW_SECONDS,
    log_mel_spectrogram,
)

logger = logging.getLogger(__name__)

# 模型定位与 sherpa.py 同一套约定：模块相对定位 + VOICE_MODEL_ROOT 部署覆盖。
# 2026-11-01 审查缺陷 2：原先 DEFAULT_MODEL_PATH 是 CWD 相对（"models/eou/..."）——
# 非仓根启动（systemd 没设 WorkingDirectory、docker 外的任何方式）找不到权重，
# factory 静默降回 conservative，神经档的账单数字混成启发式数字。生产没踩中
# 纯属 deploy/voice-pipeline.service 设了 WorkingDirectory，是运气不是约定。
_DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[4] / "models"
MODEL_ROOT_ENV = "VOICE_MODEL_ROOT"
DEFAULT_MODEL_RELPATH = Path("eou/smart-turn-v3.2-cpu.onnx")
MODEL_PATH_ENV = "SMART_TURN_ONNX"


def default_model_path() -> Path:
    """权重默认落点：仓库模型根（或 VOICE_MODEL_ROOT 覆盖）下的 eou/ 子目录。"""
    root = os.environ.get(MODEL_ROOT_ENV)
    return (Path(root) if root else _DEFAULT_MODEL_ROOT) / DEFAULT_MODEL_RELPATH


DEFAULT_THRESHOLD = 0.5


def _pcm16_to_float(chunk: bytes) -> np.ndarray:
    return np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0


class SmartTurnEou:
    """神经 L2 判停器：喂 PCM16 音频，`probability()` 给出「用户已说完」的概率。

    推理按需进行（不是每帧都跑）：
      - 自上次推理以来没进新音频 → 直接返回缓存（静音期间的成本是 0）
      - 有新音频但距上次推理 < `min_interval_ms` → 也返回缓存
    这样一轮对话只跑几次推理，实测单次 mel+推理 ≈6ms（M 系列 CPU）。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = DEFAULT_THRESHOLD,
        min_interval_ms: float = 80.0,
        min_speech_ms: float = 150.0,
        energy_floor: float = 0.005,
        # 类型只能是 Any：onnxruntime 没有 py.typed，而 import 又要延迟到实例化时
        session: Any | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_interval_ms = min_interval_ms
        self.min_speech_ms = min_speech_ms
        self.energy_floor = energy_floor

        path = Path(model_path or os.environ.get(MODEL_PATH_ENV) or default_model_path())
        if not path.exists():
            msg = (
                f"Smart Turn 权重不存在: {path} —— 先下载 pipecat-ai/smart-turn-v3.2-cpu.onnx "
                f"（scripts/fetch_eou_model.py，或设 {MODEL_PATH_ENV} / {MODEL_ROOT_ENV}）"
            )
            raise FileNotFoundError(msg)
        self.model_path = path

        # 延迟加载：onnxruntime 初始化不该在 import 时付费
        if session is not None:
            # 评测要扫参数、建很多实例 —— 权重只加载一次，实例间共享 session
            self._session = session
        else:
            import onnxruntime as ort  # noqa: PLC0415

            opts = ort.SessionOptions()
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )

        self._buf = np.zeros(0, dtype=np.float32)
        self._speech_ms = 0.0
        self._total_speech_ms = 0.0
        self._silence_ms = 0.0
        self._has_speech = False
        self._dirty = True
        self._prob = 0.0
        self._prob_valid = False
        self._last_eval_ms = -1e9
        self._elapsed_ms = 0.0
        self.inference_count = 0
        self.inference_ms = 0.0

    # ---- EouDetector 协议 ----
    def push_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        pcm = _pcm16_to_float(chunk)
        frame_ms = len(pcm) / self.sample_rate * 1000.0
        self._elapsed_ms += frame_ms

        # 只保留最后 8 秒：模型窗口就这么大，多留没有信息还拖慢 mel
        keep = int(WINDOW_SECONDS * self.sample_rate)
        if len(self._buf) + len(pcm) > keep:
            self._buf = np.concatenate([self._buf, pcm])[-keep:]
        else:
            self._buf = np.concatenate([self._buf, pcm])
        self._dirty = True

        rms = float(np.sqrt((pcm**2).mean())) if len(pcm) else 0.0
        if rms >= self.energy_floor:
            self._has_speech = True
            self._speech_ms += frame_ms
            self._total_speech_ms += frame_ms
            self._silence_ms = 0.0
        else:
            self._speech_ms = 0.0
            self._silence_ms += frame_ms

    def update_text(self, partial: str) -> None:  # 纯音频模型，不吃文本
        return None

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech_ms = 0.0
        self._total_speech_ms = 0.0
        self._silence_ms = 0.0
        self._has_speech = False
        self._dirty = True
        self._prob = 0.0
        self._prob_valid = False
        self._last_eval_ms = -1e9
        self._elapsed_ms = 0.0
        self.inference_count = 0
        self.inference_ms = 0.0

    def silence_ms(self) -> float:
        return self._silence_ms

    def speech_ms(self) -> float:
        return self._speech_ms

    # ---- 神经判据 ----
    def probability(self) -> float:
        """「用户已说完」的概率。缓存命中时不跑推理。"""
        if not self._has_speech or self._total_speech_ms < self.min_speech_ms:
            return 0.0
        fresh_enough = self._prob_valid and (
            not self._dirty or (self._elapsed_ms - self._last_eval_ms) < self.min_interval_ms
        )
        if fresh_enough:
            return self._prob
        return self._run()

    def _run(self) -> float:
        import time  # noqa: PLC0415

        t0 = time.perf_counter()
        mel = log_mel_spectrogram(self._buf)
        out = self._session.run(None, {"input_features": mel[None]})
        self._prob = float(np.asarray(out[0]).reshape(-1)[0])
        self._prob_valid = True
        self._dirty = False
        self._last_eval_ms = self._elapsed_ms
        self.inference_count += 1
        self.inference_ms += (time.perf_counter() - t0) * 1000
        return self._prob

    def is_terminal(self) -> bool:
        """供 FusedEou 的 NEURAL 档读取：模型认为用户已说完。"""
        return self.probability() >= self.threshold

    def should_end(self) -> bool:  # 单独使用时的语义
        return self.is_terminal()

    @property
    def buffer_seconds(self) -> float:
        return len(self._buf) / self.sample_rate
