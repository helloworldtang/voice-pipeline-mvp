#!/usr/bin/env python3
"""EOU 判停探针：把「判不判停」拆成可观察的每一格。

跑法：
    uv run python scripts/probe_eou.py
    uv run python scripts/probe_eou.py --gating conservative,semantic,aggressive
    uv run python scripts/probe_eou.py --tail-silence-ms 3000

它回答的是：一次判停到底卡在哪一格。可能的卡点有四个，症状完全不同：

  1. VAD 根本没把这段音频当人声（has_speech=False）→ 永不判停，与档位无关
  2. 静音累计不够（silence_ms < 阈值）→ 喂帧停了就永远冻结在这一格
  3. semantic 档的 (韵律终顿 or 语义完整) 两个判据都不成立 → 退化到 2000ms 看门狗
  4. 判停了，但判晚了（口径B 被判停延迟吃掉）

⚠️ 与 bench_ws.py 的关键差别：本脚本发完语音后**继续喂静音帧**直到判停或超时。
真实麦克风在用户闭嘴后仍在送帧（静音），静音累计会持续增长；如果脚本发完就停，
静音累计冻结，判停就永远依赖启发式——那是测量方式造成的人为漏判，不是产品行为。
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import soxr

from voice_pipeline.core.eou import FusedEou, GatingMode
from voice_pipeline.providers import sherpa as R
from voice_pipeline.providers.factory import build_eou
from voice_pipeline.providers.sherpa import pcm16_bytes

BLOCK = 1600  # 100ms @ 16kHz

UTTERANCES = [
    "今天天气怎么样",
    "我想问一下明天的天气怎么样",
    "帮我把明天的日程念一遍，顺便提醒我下午三点开会",
]


async def synthesize(tts: object, text: str) -> np.ndarray:
    samples, sr = tts.generate(text)  # type: ignore[attr-defined]
    if sr != 16000:
        samples = soxr.resample(samples, sr, 16000)
    return np.asarray(samples, dtype=np.float32)


async def run_one(
    eou: FusedEou,
    pcm: np.ndarray,
    tail_ms: int,
    asr: object | None = None,
    frame_ms: float = 100.0,
) -> dict:
    """按 100ms/帧喂语音，然后持续喂静音直到判停。返回诊断字典。

    `asr` 传真 ASR 时用它的 partial 喂语义判据（与 Session.on_audio 的行为一致）；
    不传则**完全不喂文本**——语义判据收到的空串才是「ASR 没识别出东西」的真实表现。
    早期版本这里伪造 `"啊"*n`，会让语义判据凭空成立，得出「判停正常」的假结论。
    """
    sil_tail = np.zeros(int(16000 * tail_ms / 1000), dtype=np.float32)
    buf = np.concatenate([pcm, sil_tail])

    speech_end_ms = len(pcm) / 16000 * 1000
    decided_at_ms: float | None = None
    trace: list[tuple[float, float, bool, bool, bool]] = []
    partial_samples: list[str] = []
    last_partial = ""

    for i in range(0, len(buf), BLOCK):
        chunk = pcm16_bytes(buf[i : i + BLOCK])
        eou.push_audio(chunk)
        if asr is not None:
            await asr.push_audio(chunk)  # type: ignore[attr-defined]
            last_partial = asr.text()  # type: ignore[attr-defined]
            eou.update_text(last_partial)

        elapsed = (i + BLOCK) / 16000 * 1000
        sil = eou._silence.silence_ms()  # noqa: SLF001  # 探针就是要看内部这一格
        has_speech = getattr(eou._silence, "has_speech", None)  # noqa: SLF001
        pros = bool(eou._prosody is not None and eou._prosody.is_terminal())  # noqa: SLF001
        sem = bool(eou._semantic is not None and eou._semantic.is_complete())  # noqa: SLF001
        trace.append((elapsed, sil, bool(has_speech), pros, sem))
        if elapsed % 500 < frame_ms:
            partial_samples.append(last_partial)

        if eou.should_end() and decided_at_ms is None:
            decided_at_ms = elapsed
            break

    return {
        "decided_at_ms": decided_at_ms,
        "speech_end_ms": speech_end_ms,
        "eou_latency_ms": (decided_at_ms - speech_end_ms) if decided_at_ms else None,
        "missed": bool(eou.missed_eou),
        "has_speech_ever": any(t[2] for t in trace),
        "final_sil_ms": trace[-1][1],
        "prosody_terminal_ever": any(t[3] for t in trace),
        "semantic_complete_ever": any(t[4] for t in trace),
        "max_silence_ms": eou.max_silence_ms,
        "partial_samples": partial_samples,
        "trace": trace,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="EOU 判停逐格探针")
    ap.add_argument("--gating", default="semantic")
    ap.add_argument(
        "--tail-silence-ms",
        type=int,
        default=3000,
        help="语音后持续喂多久静音（真实麦克风会一直喂）",
    )
    ap.add_argument(
        "--real-asr",
        action="store_true",
        help="用真 ASR 的 partial 喂语义判据（与 Session.on_audio 一致）",
    )
    ap.add_argument("--trace", action="store_true", help="逐帧打印每一格（看判据是否同时成立）")
    args = ap.parse_args()

    tts = R.build_tts_engine("kokoro", num_threads=2)
    asr = R.TransducerStreamingAsr() if args.real_asr else None
    print(f"ASR partial: {'真 TransducerStreamingAsr' if asr else '不喂（模拟识别为空）'}")
    print(
        f"{'句':<16}{'档位':<14}{'has_speech':>11}{'静音终值':>9}"
        f"{'韵律终顿':>9}{'语义完整':>9}{'判停@':>9}{'判停延迟':>9}"
    )
    print("-" * 88)

    for gating_name in args.gating.split(","):
        gating = GatingMode(gating_name.strip())
        for text in UTTERANCES:
            pcm = await synthesize(tts, text)
            eou = build_eou(gating)  # 与 Session 走同一条构造路径
            if asr is not None:
                asr.reset() if hasattr(asr, "reset") else None
            r = await run_one(eou, pcm, args.tail_silence_ms, asr)
            at = f"{r['decided_at_ms']:.0f}" if r["decided_at_ms"] else "未判停"
            lat = f"{r['eou_latency_ms']:.0f}ms" if r["eou_latency_ms"] is not None else "—"
            label = text if len(text) <= 14 else text[:13] + "…"
            print(
                f"{label:<16}{gating.value:<14}{str(r['has_speech_ever']):>11}"
                f"{r['final_sil_ms']:>9.0f}{str(r['prosody_terminal_ever']):>9}"
                f"{str(r['semantic_complete_ever']):>9}{at:>9}{lat:>9}"
            )
            if asr is not None:
                print(f"{'':<16}   ASR partial 抽样: {r['partial_samples'][:5]}")
            if args.trace:
                # 关键：看「静音攒够阈值之后」那一格判据是否还成立。
                # 韵律终顿很可能是瞬时判据——静音刚出现时为真，等静音攒够 250ms
                # 它已经翻回 False，于是两个条件在时间上永远错开。
                print(f"{'':<16}   {'时刻':>7}{'静音':>7}{'人声':>6}{'韵律':>6}{'语义':>6}")
                for elapsed, sil, hs, pros, sem in r["trace"]:
                    if elapsed % 100 == 0:
                        print(
                            f"{'':<16}   {elapsed:>7.0f}{sil:>7.0f}"
                            f"{str(hs):>6}{str(pros):>6}{str(sem):>6}"
                        )
            if r["decided_at_ms"] is None:
                no_speech = not r["has_speech_ever"]
                sil_short = r["final_sil_ms"] < r["max_silence_ms"]
                no_cue = (
                    r["has_speech_ever"]
                    and not r["prosody_terminal_ever"]
                    and not r["semantic_complete_ever"]
                )
                print(
                    f"{'':<16}→ 卡在哪："
                    f"{'VAD 没当人声；' if no_speech else ''}"
                    f"{'静音不够；' if sil_short else ''}"
                    f"{'韵律/语义都不成立' if no_cue else ''}"
                )


if __name__ == "__main__":
    asyncio.run(main())
