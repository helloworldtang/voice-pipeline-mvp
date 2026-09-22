"""Whisper 风格 log-mel 特征提取（Smart Turn v3 的输入）。

为什么自己写而不用 `transformers.WhisperFeatureExtractor`：后者要在运行时拖进
一个巨型库，而它做的事只是 STFT + 三角滤波 + log。**已逐值核对**：与 transformers
4.46 / 5.17 的输出最大绝对误差 ~6e-6（浮点级），两种口径各核对过真音频。

⚠️ 口径陷阱（2026-09-22 实测，改这里前先读）：
  `feature_extractor(..., do_normalize=True)` 里的 do_normalize **不是** mel 归一化，
  而是**波形级**零均值单位方差（ZMUV）。我一开始按「whisper 的 mel 归一化」
  理解它，结果两种口径差一个常数偏移（mel 域 +0.95），模型输出完全不同。
  实测核对：transformers **4.46 与 5.17 都是波形 ZMUV**，不是版本漂移。
  而 `log_spec = clamp(log10(mel), max-8)` 之后的 `(x+4)/4` 是**无条件**执行的。

  两个口径在官方测试集上的差别（v3.1-test 分片0，n=982）：
    ZMUV（官方口径）  中文 acc 88.1 / FP 5.3% / FN 19.2%   英文 acc 95.8
    不做 ZMUV        中文 acc 89.9 / FP 8.8% / FN 11.5%   英文 acc 94.8
  用官方口径，因为训练/推理代码写的就是 do_normalize=True。
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
WINDOW_SECONDS = 8
N_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS  # 128000 → 800 帧


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    mel = f / f_sp
    min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    t = f >= min_log_hz
    mel[t] = min_log_mel + np.log(f[t] / min_log_hz) / logstep
    return mel


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    t = m >= min_log_mel
    freqs[t] = min_log_hz * np.exp(logstep * (m[t] - min_log_mel))
    return freqs


def mel_filter_bank(
    n_mels: int = N_MELS, n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """slaney 三角滤波器组，返回 (n_fft//2+1, n_mels)，供 `power @ filters` 使用。"""
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    lo = float(_hz_to_mel(np.array([0.0]))[0])
    hi = float(_hz_to_mel(np.array([sample_rate / 2]))[0])
    edges = _mel_to_hz(np.linspace(lo, hi, n_mels + 2))
    fdiff = np.diff(edges)
    ramps = np.subtract.outer(edges, fftfreqs)  # (n_mels+2, n_bins)
    weights = np.zeros((n_mels, fftfreqs.size))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (edges[2 : n_mels + 2] - edges[:n_mels])
    weights *= enorm[:, None]
    return weights.T.astype(np.float32)


_MEL_FILTERS: np.ndarray | None = None


def filters() -> np.ndarray:
    """滤波器组只算一次（80×201 的构造有 python 循环，别每帧算）。"""
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = mel_filter_bank()
    return _MEL_FILTERS


def zero_mean_unit_var(audio: np.ndarray) -> np.ndarray:
    """官方口径 do_normalize=True：波形级 ZMUV（不是 mel 归一化，见文件头）。"""
    a = audio.astype(np.float64)
    std = float(a.std())
    if std < 1e-9:  # 纯静音：别除零
        return np.zeros_like(audio, dtype=np.float32)
    return ((a - a.mean()) / std).astype(np.float32)


def fit_window(audio: np.ndarray) -> np.ndarray:
    """裁到/补到 8 秒窗口：**左侧补零**（官方 `truncate_audio_to_last_n_seconds`）。"""
    a = np.asarray(audio, dtype=np.float32)
    if len(a) > N_SAMPLES:
        return a[-N_SAMPLES:]
    if len(a) < N_SAMPLES:
        return np.pad(a, (N_SAMPLES - len(a), 0), mode="constant")
    return a


def log_mel_spectrogram(audio: np.ndarray, *, do_normalize: bool = True) -> np.ndarray:
    """16k float32 波形 → (80, 800) log-mel。"""
    a = fit_window(audio)
    if do_normalize:
        a = zero_mean_unit_var(a)
    padded = np.pad(a, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH  # 801 → 末尾丢一帧，取 800
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    window = np.hanning(N_FFT + 1)[:N_FFT].astype(np.float32)
    spec = np.fft.rfft(padded[idx] * window, n=N_FFT)
    power = (spec.real**2 + spec.imag**2)[:-1]  # (800, 201)
    mel = (power @ filters()).T  # (80, 800)
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return np.asarray((log + 4.0) / 4.0, dtype=np.float32)
