"""TTS 逐格探针：量清楚「LLM 首句 → 可播首片」这一格到底由什么决定。

为什么要单独写这个探针：端到端分解显示 TTS 首 chunk 占 1302ms，比 LLM 首句
（379ms）和判停（100ms）加起来还大 —— 它是口径B 最大的一格。但仓库注释里
关于 TTS 引擎快慢的说法互相矛盾（工厂默认 kokoro，`MeloTts` 的 docstring 却说
自己快 2~4 倍）。**注释不是数据**，这一格必须自己量。

量三件事：
1. **固定开销**：单字合成的耗时。它是「缩短首句」这条路的天花板 ——
   如果单字就要 600ms，那再怎么切分句也降不下来。
2. **边际成本**：每多一个字加多少 ms。它决定「首句切多短」的收益斜率。
3. **增量能力**：引擎是合成完再一次性交付（首片延迟 = 整句耗时），
   还是边合成边吐（首片延迟 << 整句耗时）。`callback_profile` 看回调画像。

用法：
    uv run python scripts/probe_tts.py                      # 默认：两引擎 × 6 种长度 × 5 轮
    uv run python scripts/probe_tts.py --engines melo
    uv run python scripts/probe_tts.py --profile            # 额外打印回调画像
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from voice_pipeline.providers import sherpa as R  # noqa: E402

SR = 24000
CHUNK_MS = 200

# 长度梯度：从「一个字」到「一整句」。单字那一行是固定开销的直接读数。
CASES: list[tuple[str, str]] = [
    ("1字", "好"),
    ("3字", "好的。"),
    ("5字", "今天天气好"),
    ("7字", "今天天气怎么样"),
    ("12字", "我想问一下明天的天气怎么样"),
    ("20字", "我想问一下明天的天气怎么样，另外帮我看看后天要不要带伞"),
]


def measure(tts: R.BaseTts, text: str, chunk_ms: int) -> dict[str, float]:
    """量一句：整句合成耗时、首片（一个下行分片）到手耗时、产出音频时长。"""
    target_samples = int(SR * chunk_ms / 1000)
    state = {"samples": 0, "first_ms": -1.0}
    t0 = time.perf_counter()

    def callback(chunk: object, _progress: float) -> int:
        arr = np.asarray(chunk if not isinstance(chunk, tuple) else chunk[0])
        state["samples"] += int(arr.size)
        # 到手够一个下行分片就记时：这就是「能开始播」的真实时刻
        if state["first_ms"] < 0 and state["samples"] >= target_samples:
            state["first_ms"] = (time.perf_counter() - t0) * 1000.0
            return 1
        return 0

    tts._tts.generate(text, sid=tts.speaker_id, speed=tts.speed, callback=callback)
    total_ms = (time.perf_counter() - t0) * 1000.0

    audio = tts._tts.generate(text, sid=tts.speaker_id, speed=tts.speed)
    dur_ms = len(audio.samples) / int(audio.sample_rate) * 1000.0
    return {
        "total_ms": total_ms,
        "first_ms": state["first_ms"] if state["first_ms"] >= 0 else total_ms,
        "audio_ms": dur_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="kokoro,melo")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--chunk-ms", type=int, default=CHUNK_MS)
    ap.add_argument("--profile", action="store_true", help="打印回调画像（验证是否真的增量）")
    args = ap.parse_args()

    results: list[dict] = []
    for name in args.engines.split(","):
        name = name.strip()
        if not name:
            continue
        cls = R.TTS_ENGINES.get(name)
        if cls is None:
            print(f"未知引擎 {name}，可选: {list(R.TTS_ENGINES)}")
            continue
        print(f"\n=== {name} (原生 {cls.__name__}) ===")
        tts = cls(num_threads=4, chunk_ms=args.chunk_ms)
        print(f"  采样率 {tts.sample_rate}Hz  分块 {args.chunk_ms}ms")
        for label, text in CASES:
            runs = [measure(tts, text, args.chunk_ms) for _ in range(args.repeat)]
            total = [r["total_ms"] for r in runs]
            first = [r["first_ms"] for r in runs]
            dur = statistics.median([r["audio_ms"] for r in runs])
            row = {
                "engine": name,
                "case": label,
                "chars": len(text),
                "总耗时p50": statistics.median(total),
                "总耗时min": min(total),
                "首片p50": statistics.median(first),
                "首片min": min(first),
                "音频时长": dur,
            }
            results.append(row)
            print(
                f"  {label:<6} {len(text):>2}字  总耗时 p50 {row['总耗时p50']:>7.0f}ms "
                f"(min {row['总耗时min']:>6.0f})  首片 p50 {row['首片p50']:>7.0f}ms "
                f"(min {row['首片min']:>6.0f})  音频 {dur:>6.0f}ms"
            )

        if args.profile:
            calls = tts.callback_profile(CASES[4][1])
            print(f"  回调画像({CASES[4][0]}): {len(calls)} 次回调 → {calls[:6]}")

    if len({r["engine"] for r in results}) < 2:
        return 0

    print("\n=== 对比：同一文本的引擎差 ===")
    print(f"{'文本':<8}{'chars':>6}{'kokoro首片':>13}{'melo首片':>11}{'省下':>9}")
    by_case: dict[str, dict[str, float]] = {}
    for r in results:
        by_case.setdefault(r["case"], {})[r["engine"]] = r["首片p50"]
    for label, text in CASES:
        d = by_case[label]
        if "kokoro" in d and "melo" in d:
            print(
                f"{label:<8}{len(text):>6}{d['kokoro']:>12.0f}ms{d['melo']:>10.0f}ms"
                f"{d['kokoro'] - d['melo']:>8.0f}ms"
            )

    print("\n=== 判据 ===")
    k1 = next(r for r in results if r["engine"] == "kokoro" and r["case"] == "1字")
    m1 = next(r for r in results if r["engine"] == "melo" and r["case"] == "1字")
    print(f"固定开销（单字）: kokoro {k1['首片p50']:.0f}ms / melo {m1['首片p50']:.0f}ms")
    for eng in ("kokoro", "melo"):
        rs = [r for r in results if r["engine"] == eng]
        if len(rs) >= 2:
            slope = (rs[-1]["首片p50"] - rs[0]["首片p50"]) / (rs[-1]["chars"] - rs[0]["chars"])
            print(f"边际成本 {eng}: 每字 +{slope:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
