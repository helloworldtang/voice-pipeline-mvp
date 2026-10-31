"""pcm16_bytes 的 clip 纪律（2026-11-01 审查缺陷 8）。

float32→int16 的溢出是**静默回绕**：2.0×32767 溢出后变负数，听感为爆音。
重采样过冲（soxr 在陡峭波形上超 ±1）与高幅样本都会触发，必须 clip。
"""

from __future__ import annotations

import numpy as np

from voice_pipeline.providers.sherpa import pcm16_bytes


def test_pcm16_bytes_clips_instead_of_wrapping() -> None:
    overshoot = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    ints = np.frombuffer(pcm16_bytes(overshoot), dtype="<i2")
    assert ints[0] == 32767, "正过冲必须 clip 到满幅，不许回绕成负数（爆音）"
    assert ints[1] == -32767, "负过冲必须 clip 到满幅，不许回绕成正数（爆音）"
    assert ints[2] == int(0.5 * 32767), "正常样本照常线性缩放"


def test_pcm16_bytes_normal_range_untouched() -> None:
    ints = np.frombuffer(pcm16_bytes(np.array([0.0, 1.0, -1.0], dtype=np.float32)), dtype="<i2")
    assert ints.tolist() == [0, 32767, -32767]
