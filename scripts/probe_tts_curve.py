"""复核两个异常：
1. 40字与20字产出的样本数完全相同（41472）—— 是不是被静默截断？
2. §1 的 20字 798ms vs §4 的 1728ms —— 哪个是生产口径？
"""

from __future__ import annotations

import time

import numpy as np

from voice_pipeline.providers import sherpa as R

CASES = {
    "2字": "好的。",
    "7字": "今天天气不错。",
    "12字": "我马上就帮你查一下。",
    "20字": "这个问题我来帮你看看，稍等我查询一下最新的数据。",
    "40字": "这个问题我来帮你看看，稍等我查询一下最新的数据，"
    "然后给你一个比较详细的分析，包括最近的变化和可能的原因。",
}

print("=" * 84)
print("异常1：文本变长，音频还变长吗？（melo，44.1k 原生 → 24k 输出）")
print("=" * 84)
tts = R.build_tts_engine("melo", num_threads=4)
print(
    f"{'文本':<6}{'字数':>5}{'合成ms':>9}{'原生样本':>9}{'原生时长s':>10}{'24k样本':>9}{'24k时长s':>10}"
)
prev = None
for name, text in CASES.items():
    t0 = time.perf_counter()
    audio = tts._tts.generate(text, sid=0, speed=1.0)
    ms = (time.perf_counter() - t0) * 1000
    s = np.asarray(audio.samples, dtype=np.float32)
    sr = int(audio.sample_rate)
    dur = len(s) / sr
    out24 = int(len(s) / sr * 24000)
    flag = ""
    if prev is not None and len(s) <= prev:
        flag = "  ⚠️ 没有随文本变长"
    print(
        f"{name:<6}{len(text):>5}{ms:>8.0f}ms{len(s):>9}{dur:>9.2f}s"
        f"{out24:>9}{out24 / 24000:>9.2f}s{flag}"
    )
    prev = len(s)

print("\n" + "=" * 84)
print("异常2：生产口径（synth_blocking，含重采样）到底多少 ms —— n=5 取 min/中位")
print("=" * 84)
for name, text in CASES.items():
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        pcm = tts.synth_blocking(text)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(
        f"{name:<6} min={ts[0]:>7.0f}ms  p50={ts[2]:>7.0f}ms  max={ts[-1]:>7.0f}ms  "
        f"字节={len(pcm):>7}  ({len(pcm) / 48000:.2f}s @24k)"
    )

print("\n" + "=" * 84)
print("参考：重采样本身多少 ms（melo 44.1k → 24k）")
print("=" * 84)
import soxr  # noqa: E402

for n in (13312, 41472, 90000):
    x = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    t0 = time.perf_counter()
    y = soxr.resample(x, 44100, 24000)
    print(f"  {n:>7}样本 → {(time.perf_counter() - t0) * 1000:>6.2f}ms  (产出{len(y)})")
