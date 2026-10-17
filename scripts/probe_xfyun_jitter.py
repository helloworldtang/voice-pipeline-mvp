"""讯飞收帧抖动探针：逐轮记 connect / 首帧 / 整句，落 JSONL，供分时段对比。

为什么单独一个脚本：链路内的 `tts.first_chunk` 会掺进 LLM 与事件循环的干扰，
没法回答"抖动是讯飞的还是我们自己的"。这里只留 provider + 预建连（与生产同构），
其余全剥掉。

用法：
    source ~/.voice-pilot/vendors.env
    uv run python scripts/probe_xfyun_jitter.py --n 20 --out /tmp/vp-ac/xfyun-jitter.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from voice_pipeline.providers.tts.xfyun_tts import XfyunTts

DEFAULT_TEXT = "今天上海天气怎么样"


async def run(n: int, out: Path, text: str) -> None:
    tts = XfyunTts()
    rows: list[dict[str, object]] = []
    for i in range(1, n + 1):
        tts.prewarm()
        # 等预建连就位（生产里这段被 finalize+LLM 首句挡住，这里显式等）
        warmed = False
        for _ in range(200):
            if getattr(tts, "_pending", None) is not None:  # noqa: SLF001
                warmed = True
                break
            await asyncio.sleep(0.005)

        t0 = time.perf_counter()
        first_ms = -1.0
        total_bytes = 0
        async for chunk in tts.synth(text):
            if first_ms < 0:
                first_ms = (time.perf_counter() - t0) * 1000
            total_bytes += len(chunk)
        total_ms = (time.perf_counter() - t0) * 1000

        row = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "epoch": round(time.time(), 3),
            "i": i,
            "connect_ms": 0.0 if warmed else -1.0,  # 0=预建连命中；-1=没预热上（这轮含握手）
            "first_frame_ms": round(first_ms, 1),
            "total_ms": round(total_ms, 1),
            "audio_s": round(total_bytes / 2 / 24000, 3),
        }
        rows.append(row)
        print(
            f"  {i:>2}/{n} 首帧 {first_ms:6.0f}ms  整句 {total_ms:6.0f}ms"
            f"  音频 {row['audio_s']:.2f}s"
        )

    with out.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    vals = sorted(float(r["first_frame_ms"]) for r in rows)  # type: ignore[arg-type]
    print(
        f"\n本组 n={len(vals)} min={vals[0]:.0f} p50={statistics.median(vals):.0f} "
        f"p90={vals[int(0.9 * (len(vals) - 1))]:.0f} max={vals[-1]:.0f}ms"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("/tmp/vp-ac/xfyun-jitter.jsonl"))
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.n, args.out, args.text))


if __name__ == "__main__":
    main()
