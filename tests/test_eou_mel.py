"""Whisper log-mel 特征提取：口径必须钉死，错了模型就变随机数发生器。

每条断言都对应一个 2026-09-22 踩过的坑：
  1. 形状 (80, 800) —— 模型输入签名，错了直接跑不通
  2. ZMUV 口径 —— `do_normalize=True` 是**波形级**零均值单位方差，不是 mel 归一化。
     它的可观测后果是「波形整体缩放不改变 mel」（不 ZMUV 时会改变 0.5）。
     这条断言在 2026-09-22 抓到了口径漂移：两种口径差一个常数偏移，模型输出全变。
  3. 左侧补零 —— 官方 `truncate_audio_to_last_n_seconds` 是左 pad，pad 错边等于
     把语音挪到时间轴的另一头，位置编码全错。
"""

from __future__ import annotations

import numpy as np

from voice_pipeline.core.eou_mel import (
    N_SAMPLES,
    fit_window,
    log_mel_spectrogram,
    mel_filter_bank,
    zero_mean_unit_var,
)

SR = 16000


def _audio() -> np.ndarray:
    """确定性信号：440Hz 正弦 + 固定种子噪声，别用随机数（测试要可复现）。"""
    rng = np.random.default_rng(0)
    n = np.arange(SR)
    return (0.2 * np.sin(2 * np.pi * 440 * n / SR) + 0.05 * rng.standard_normal(SR)).astype(
        np.float32
    )


def test_mel_shape_and_reference_values():
    """形状 + 参考值：与 transformers 4.46/5.17 逐值核对过（误差 ~6e-6）。"""
    mel = log_mel_spectrogram(_audio())
    assert mel.shape == (80, 800), mel.shape
    assert mel.dtype == np.float32
    assert abs(float(mel.max()) - 1.888182) < 1e-4
    assert abs(float(mel.mean()) - 0.045509) < 1e-4
    assert abs(float(mel[0, 0]) - (-0.018803)) < 1e-4
    assert abs(float(mel[79, 799]) - 1.149820) < 1e-4


def test_zmuv_makes_mel_scale_invariant():
    """口径判别式：ZMUV 下波形整体缩放不改变 mel；不做 ZMUV 则必然改变。

    这是「do_normalize 到底是哪一个」的唯一可观测指纹 —— 2026-09-22 就是靠它
    确认 transformers 4.x 与 5.x 都是波形级 ZMUV（而不是 mel 归一化）。
    """
    a = _audio()
    base = log_mel_spectrogram(a)
    scaled = log_mel_spectrogram(a * 0.1)
    assert np.abs(scaled - base).max() < 1e-5, "默认口径应是波形级 ZMUV（缩放不变）"

    raw_a = log_mel_spectrogram(a, do_normalize=False)
    raw_b = log_mel_spectrogram(a * 0.1, do_normalize=False)
    assert np.abs(raw_b - raw_a).max() > 0.1, "不做 ZMUV 时缩放必须改变 mel（对照）"


def test_zmuv_is_zero_mean_unit_var():
    z = zero_mean_unit_var(_audio())
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std()) - 1.0) < 1e-4


def test_silence_does_not_blow_up():
    """纯静音：ZMUV 会除零，必须优雅退化成全 0 而不是 NaN。"""
    mel = log_mel_spectrogram(np.zeros(SR, dtype=np.float32))
    assert np.isfinite(mel).all()
    assert mel.shape == (80, 800)


def test_fit_window_pads_on_the_left():
    """左 pad：语音必须落在窗口**末尾**（官方语义），不是开头。"""
    a = np.ones(1000, dtype=np.float32)
    w = fit_window(a)
    assert len(w) == N_SAMPLES
    assert w[-1] == 1.0 and w[0] == 0.0, "补零应在左侧，语音贴在窗口末尾"


def test_fit_window_truncates_from_the_left():
    a = np.concatenate([np.zeros(100, dtype=np.float32), np.ones(200, dtype=np.float32)])
    w = fit_window(a)
    assert len(w) == N_SAMPLES
    assert w[-1] == 1.0


def test_filter_bank_is_slaney():
    f = mel_filter_bank()
    assert f.shape == (201, 80)
    assert (f >= 0).all(), "三角滤波器权重不能为负"


def test_long_audio_is_truncated_to_window():
    a = np.ones(SR * 20, dtype=np.float32)
    assert log_mel_spectrogram(a).shape == (80, 800)
