"""投机可行性 v2：上行语音改用 kokoro 合成（melo 合成的话 ASR 识别为空，
这是 bench_ws.py 里已记录的坑）。

问题：pipelined 命中率 4%，校验条件是「投机时的 partial == 最后 final」。
  若 partial 早在说完前数百 ms 就稳定 ⇒ 命中率低是参数问题（有优化空间）
  若 partial 直到最后一刻还在变 ⇒ 原理性走不通
"""

from __future__ import annotations

import asyncio

import numpy as np
import soxr

from voice_pipeline.providers import sherpa as R

SR = 16000
FRAME = 1600  # 100ms


async def upspeak(text: str) -> np.ndarray:
    tts = R.build_tts_engine("kokoro", num_threads=4)
    samples, sr = await asyncio.to_thread(tts.generate, text)
    if sr != SR:
        samples = soxr.resample(samples, sr, SR)
    return np.asarray(samples, dtype=np.float32)


async def run(text: str) -> None:
    audio = await upspeak(text)
    dur_ms = len(audio) / SR * 1000

    asr = R.TransducerStreamingAsr()
    tl: list[tuple[float, str]] = []
    for i in range(0, len(audio), FRAME):
        await asr.push_audio(R.pcm16_bytes(audio[i : i + FRAME]))
        tl.append(((i + FRAME) / SR * 1000, asr.text()))
    # 说完后再喂 400ms 静音，看尾部是否还在修正
    for k in range(4):
        await asr.push_audio(R.pcm16_bytes(np.zeros(FRAME, dtype=np.float32)))
        tl.append((dur_ms + (k + 1) * 100, asr.text()))
    final = await asr.finalize()

    first_hit = next((t for t, p in tl if p == final and p != ""), None)
    stable = None
    for i, (t, p) in enumerate(tl):
        if p and all(q == p for _, q in tl[i:]):
            stable = t
            break

    print(f"\n『{text}』  语音 {dur_ms:.0f}ms")
    print(f"  final = {final!r}")
    print("  时间线:")
    prev = None
    for t, p in tl:
        if p != prev:
            tag = "  ✅=final" if p == final and p else ""
            tail = "  (说完后的静音期)" if t > dur_ms else ""
            print(f"    {t:>7.0f}ms  {p!r}{tag}{tail}")
            prev = p
    if first_hit is not None:
        print(
            f"  ⇒ partial 首次等于 final: {first_hit:.0f}ms，语音结束前 {dur_ms - first_hit:.0f}ms"
        )
    else:
        print("  ⇒ 语音期间 partial 从未等于 final")
    print(f"  ⇒ 从 {stable}ms 起不再变化" if stable else "  ⇒ 全程都在变")


async def main() -> None:
    print("=" * 80)
    print("投机可行性 v2：ASR partial 何时稳定成 final（上行语音 = kokoro）")
    print("=" * 80)
    for t in (
        "今天天气怎么样",
        "帮我查一下比亚迪的股价",
        "我想知道最近保利发展的走势如何",
    ):
        await run(t)


if __name__ == "__main__":
    asyncio.run(main())
