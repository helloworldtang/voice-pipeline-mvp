#!/usr/bin/env python3
"""增量文本 TTS 探针：把「攒整句再合成」换成「攒几个字就开喂」，首包能早多少？

背景：外部方案建议「Token 级流水线」——不等首句完整，缓冲 8~15 字就启动 TTS。
本仓的句子级流水线要为此改造（调度器要喂 partial 文本），但**先量收益再动架构**：
DashScope 的 `SpeechSynthesizer` 自带 `streaming_call(text)`（可多次追加）+ `streaming_complete()`，
所以这一刀现在就能测，不用换厂商。

测两臂（同一句话、同一音色、同一进程）：
- A 整句：`call(完整句子)` —— 现在流水线的行为（等句末标点）
- B 增量：`streaming_call(前半) → streaming_call(后半) → streaming_complete()`
  量的是**第一个音频包**的时刻（B 臂在第一段文字到达时就该出声）

用法：
    uv run python scripts/probe_tts_streaming.py --repeat 5
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from datetime import datetime
from typing import Any

from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

MODEL = "cosyvoice-v2"
VOICE = "longxiaochun_v2"
HEAD = "今天上海多云，"  # 7 字：与「首句护栏」的上限一致
TAIL = "气温18到25度。"


class _Probe:
    """收集回调：记下首个音频包的时刻（相对合成开始）。"""

    def __init__(self) -> None:
        self.t0 = 0.0
        self.first_ms = -1.0
        self.bytes = 0
        self.done = threading.Event()
        self.error: str | None = None

    def mark(self) -> None:
        self.t0 = time.perf_counter()

    def on_data(self, data: bytes) -> None:
        if self.first_ms < 0:
            self.first_ms = (time.perf_counter() - self.t0) * 1000.0
        self.bytes += len(data)

    def on_complete(self) -> None:
        self.done.set()

    def on_error(self, message: str) -> None:
        self.error = message
        self.done.set()

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def on_event(self, message: str) -> None:
        pass


def _synth(cb: Any) -> SpeechSynthesizer:
    return SpeechSynthesizer(
        model=MODEL, voice=VOICE, format=AudioFormat.PCM_24000HZ_MONO_16BIT, callback=cb
    )


def arm_whole(text: str, timeout_s: float = 20.0) -> dict[str, float]:
    cb = _Probe()
    synth = _synth(cb)
    cb.mark()
    t = threading.Thread(target=synth.call, args=(text,), daemon=True)
    t.start()
    if not cb.done.wait(timeout_s):
        return {"first_ms": -1.0, "total_ms": -1.0, "error": "timeout"}
    return {
        "first_ms": cb.first_ms,
        "total_ms": (time.perf_counter() - cb.t0) * 1000.0,
        "error": cb.error or "",
    }


def arm_incremental(
    head: str, tail: str, gap_ms: float, timeout_s: float = 20.0
) -> dict[str, float]:
    cb = _Probe()
    synth = _synth(cb)
    cb.mark()

    # streaming_call 是阻塞提交（内部建会话/发 run-task），所以放线程里跑
    def run() -> None:
        synth.streaming_call(head)
        time.sleep(gap_ms / 1000.0)  # 模拟「后半句还没生成出来」
        synth.streaming_call(tail)
        synth.streaming_complete()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    if not cb.done.wait(timeout_s):
        return {"first_ms": -1.0, "total_ms": -1.0, "error": "timeout"}
    return {
        "first_ms": cb.first_ms,
        "total_ms": (time.perf_counter() - cb.t0) * 1000.0,
        "error": cb.error or "",
    }


def summarize(label: str, runs: list[dict[str, float]]) -> None:
    firsts = [r["first_ms"] for r in runs if r["first_ms"] >= 0]
    totals = [r["total_ms"] for r in runs if r["total_ms"] >= 0]
    errs = [r["error"] for r in runs if r["error"]]
    if not firsts:
        print(f"{label}: 全部失败 {errs[:2]}")
        return
    print(
        f"{label}: 首包 p50={statistics.median(firsts):.0f}ms "
        f"(min {min(firsts):.0f} max {max(firsts):.0f})  "
        f"整句 total p50={statistics.median(totals):.0f}ms  失败 {len(errs)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="增量文本 TTS 探针")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--gap-ms", type=float, default=120.0, help="两段文字之间的间隔（模拟生成耗时）"
    )
    args = ap.parse_args()

    whole: list[dict[str, float]] = []
    incr: list[dict[str, float]] = []
    for i in range(args.repeat):
        whole.append(arm_whole(HEAD + TAIL))
        incr.append(arm_incremental(HEAD, TAIL, args.gap_ms))
        print(
            f"  run{i + 1}: 整句 {whole[-1]['first_ms']:.0f}ms | 增量 {incr[-1]['first_ms']:.0f}ms"
        )

    print(f"\n[{datetime.now():%H:%M:%S}] 同一句「{HEAD}{TAIL}」，重复 {args.repeat} 次")
    summarize("A 整句 call()      ", whole)
    summarize(f"B 增量 streaming_call（段间隔 {args.gap_ms:.0f}ms）", incr)


if __name__ == "__main__":
    main()
