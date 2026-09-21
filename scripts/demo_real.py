"""端到端实测：把合成语音当成「上行麦克风输入」，跑真实 ASR→LLM→TTS 链路。

这不是 sim：模型真的在跑。用它回答两个问题：
  1. 真实链路的 waterfall 每段到底多少毫秒
  2. 口径B（用户说完 → AI 首音）落在哪

LLM 现在是真的：默认走 `build_llm("auto")`，有 DASHSCOPE_API_KEY 就用 qwen-flash，
没有就退回 SimLlm 并告警——**告警时账单里 LLM 那一格是 0ms，只是本机侧下界，
不要对外当真实延迟报**（`scripts/probe_llm.py` 量的是 LLM 单独那一格）。

用法：
  uv run python scripts/demo_real.py                 # 默认探针句
  uv run python scripts/demo_real.py --text "…"      # 指定「用户说了什么」
  uv run python scripts/demo_real.py --engine paraformer
  uv run python scripts/demo_real.py --llm sim       # 强制替身，用于和真 LLM 对照
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_pipeline.providers import sherpa as R  # noqa: E402
from voice_pipeline.providers.factory import build_llm  # noqa: E402
from voice_pipeline.providers.vad import SileroVad  # noqa: E402
from voice_pipeline.server.session import Session  # noqa: E402


def machine() -> str:
    return subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="我想问一下明天的天气怎么样")
    ap.add_argument("--engine", default="transducer", choices=["transducer", "paraformer"])
    ap.add_argument("--tts", default="kokoro", choices=["kokoro", "melo"])
    ap.add_argument("--llm", default="auto", choices=["auto", "qwen", "sim"])
    ap.add_argument("--silence-ms", type=int, default=300, help="EOU 前追加的静音")
    args = ap.parse_args()

    engine = R.TransducerStreamingAsr if args.engine == "transducer" else R.ParaformerStreamingAsr
    llm = build_llm(args.llm)
    print(f"机器: {machine()}   ASR={args.engine}   TTS={args.tts}   LLM={type(llm).__name__}")

    t0 = time.perf_counter()
    tts = R.build_tts_engine(args.tts, num_threads=4)
    asr = engine()
    vad = SileroVad()
    print(f"模型加载 {(time.perf_counter() - t0) * 1000:.0f}ms  TTS={args.tts}")

    samples, sr = tts.generate(args.text)
    user_pcm = soxr.resample(samples, sr, 16000)
    user_ms = len(user_pcm) / 16000 * 1000
    print(f"模拟用户说话 {user_ms:.0f}ms（合成语音上行）")

    session = Session(mode="real", asr=asr, llm=llm, tts=tts)
    frame = 3200  # 上行 100ms @16k
    stream = []
    for i in range(0, len(user_pcm), frame):
        stream.append(user_pcm[i : i + frame])
    silence = np.zeros(int(16000 * args.silence_ms / 1000), dtype=np.float32)
    stream.append(silence)

    t_start = time.perf_counter()
    for part in stream:
        await session.on_audio(R.pcm16_bytes(part))
        vad.push_audio(R.pcm16_bytes(part))
    # SessionClock 从模块导入起算，utterance_end 标记在这里打点在外部无法注入，
    # 因此用 Session 自己的 metrics：只打 utterance_end（闭嘴）即可。
    session.metrics.mark("utterance_end")
    feed_done = (time.perf_counter() - t_start) * 1000

    items = []
    async for item in session.finish():
        items.append(item)

    bill = session.metrics.bill()
    print(f"\n上行喂帧耗时 {feed_done:.0f}ms（含 {args.silence_ms}ms 静音）")
    print("\n=== waterfall（ms，相对会话钟）===")
    for k, v in sorted(bill.items(), key=lambda kv: (kv[1] is None, kv[1])):
        if v is None:
            print(f"  {k:<22} —（未测到，按 SD 不补）")
        else:
            print(f"  {k:<22} {v:>8.1f}")

    for key in ("utterance_end", "first_audio_out"):
        v = session.metrics.elapsed(key)
        print(f"\n[{key}] {v if v is not None else '未测到'}")
    if session.metrics.elapsed("utterance_end") and session.metrics.elapsed("first_audio_out"):
        b = session.metrics.elapsed("first_audio_out") - session.metrics.elapsed("utterance_end")
        print(f"口径B（说完→AI首音）: {b:.1f}ms")

    audio_items = [i for i in items if hasattr(i, "data")]
    texts = [i.text for i in items if hasattr(i, "text")]
    print(f"\nASR 转写: {texts[0] if texts else '(空)'}")
    print(f"下行音频: {len(audio_items)} 分片 / {sum(len(i.data) for i in audio_items)} bytes")
    print(f"LLM 出口句: {texts[1:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
