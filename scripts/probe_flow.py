"""流程优化探针：把「还能不能更快」的四个假设一次量清。

假设 1（最大杠杆）：TTS 有增量流 —— callback 若多次触发，首片就不必等整句合成完。
假设 2：判停后那 ~350ms 里，ASR finalize 占多少。
假设 3：Chunker 从不 flush，句尾不足一片的音频被永久丢弃（真 bug 还是无害）。
假设 4：melo 的线程数没被扫过（kokoro 扫过 4 线程最优）。

只读测量，不改任何生产代码。
"""

from __future__ import annotations

import time

import numpy as np

from voice_pipeline.providers import sherpa as R

SR = 16000


def sec1_callback_profile() -> None:
    print("=" * 78)
    print("§1  TTS 回调画像：到底有没有增量流？")
    print("=" * 78)
    cases = {
        "2字": "好的。",
        "7字": "今天天气不错。",
        "12字": "我马上就帮你查一下。",
        "20字": "这个问题我来帮你看看，稍等我查询一下最新的数据。",
        "40字": "这个问题我来帮你看看，稍等我查询一下最新的数据，"
        "然后给你一个比较详细的分析，包括最近的变化和可能的原因。",
    }
    for engine in ("melo", "kokoro"):
        print(f"\n--- {engine} ---")
        try:
            tts = R.build_tts_engine(engine, num_threads=4)
        except Exception as e:  # noqa: BLE001
            print(f"  加载失败: {e}")
            continue
        for name, text in cases.items():
            prof = tts.callback_profile(text)
            n = len(prof)
            first_t = prof[0][1] if prof else -1
            total = prof[-1][1] if prof else -1
            sizes = [s for s, _ in prof[:4]]
            print(
                f"  {name:<5} 回调{n:>3}次  首回调={first_t:>7.0f}ms  "
                f"末回调={total:>7.0f}ms  前几次样本数={sizes}"
            )
        # 首片（200ms @24k = 4800 样本）到手时刻 vs 整句完成时刻
        print(f"  {'':5} {'字数':<6}{'首片到手':>10}{'整句完成':>10}{'可省':>10}")
        for name, text in cases.items():
            lat, got, sr = tts.first_chunk_latency_ms(text)
            full = tts.callback_profile(text)[-1][1]
            save = full - lat if lat > 0 else 0
            print(f"  {'':5} {name:<6}{lat:>9.0f}ms{full:>9.0f}ms{save:>9.0f}ms")


def sec2_asr_finalize() -> None:
    print("\n" + "=" * 78)
    print("§2  判停后 → LLM 首 token 那 ~350ms 里，ASR finalize 占多少")
    print("=" * 78)
    asr = R.build_asr() if hasattr(R, "build_asr") else R.TransducerStreamingAsr()
    print(f"  引擎: {type(asr).__name__}")
    # 喂 3 秒语音（合成一段带能量的噪声 + 调频，模拟说话）
    rng = np.random.default_rng(0)
    n = np.arange(SR * 3)
    audio = 0.3 * np.sin(2 * np.pi * 220 * n / SR) * (1 + 0.5 * np.sin(2 * np.pi * 3 * n / SR))
    audio = (audio + 0.02 * rng.standard_normal(len(n))).astype(np.float32)
    for i in range(0, len(audio), 1600):  # 100ms 一帧
        import asyncio

        asyncio.run(asr.push_audio(R.pcm16_bytes(audio[i : i + 1600])))
    for i in range(5):
        import asyncio

        t0 = time.perf_counter()
        txt = asyncio.run(asr.finalize())
        el = (time.perf_counter() - t0) * 1000
        print(f"  第{i + 1}次 finalize: {el:>7.1f}ms  文本={txt[:30]!r}")
        asr.reset()


def sec3_chunker() -> None:
    print("\n" + "=" * 78)
    print("§3  Chunker 从不 flush：句尾残余会不会丢？")
    print("=" * 78)
    from voice_pipeline.audio.chunker import Chunker

    c = Chunker()
    # 24k / 200ms => 9600 字节一片
    print(f"  片大小 = {c.chunk_bytes} 字节 (24k×200ms×2)")
    for total in (500, 5000, 9600, 9601, 20000):
        c2 = Chunker()
        out = c2.feed(b"\x00" * total)
        flushed = c2.flush()
        got = sum(len(x) for x in out) + sum(len(x) for x in flushed)
        lost = total - got
        print(
            f"  输入{total:>6}B → feed出{len(out)}片({sum(len(x) for x in out)}B) "
            f"flush出{len(flushed)}片  丢弃={lost}B"
        )
    print("\n  生产代码里 _synthesize() 只调 feed、从不 flush（serial.py:87）")
    print("  ⇒ 每句尾部不足 200ms 的音频永久留在 buffer，且首句若 <200ms 则一片都不出")


def sec4_threads() -> None:
    print("\n" + "=" * 78)
    print("§4  melo 线程数扫描（kokoro 已扫过，melo 没有）")
    print("=" * 78)
    text12 = "我马上就帮你查一下。"
    text20 = "这个问题我来帮你看看，稍等我查询一下最新的数据。"
    for threads in (1, 2, 4, 8):
        try:
            tts = R.build_tts_engine("melo", num_threads=threads)
        except Exception as e:  # noqa: BLE001
            print(f"  {threads}线程 加载失败: {e}")
            continue
        row = []
        for t in (text12, text20):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                tts.synth_blocking(t)
                ts.append((time.perf_counter() - t0) * 1000)
            row.append(min(ts))
        print(f"  {threads}线程: 12字 min={row[0]:>7.0f}ms   20字 min={row[1]:>7.0f}ms")


if __name__ == "__main__":
    sec1_callback_profile()
    sec2_asr_finalize()
    sec3_chunker()
    sec4_threads()
