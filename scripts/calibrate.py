"""本机（Apple M5 芯片）延迟校准：把 300ms 预算里的每一段量出来。

做法：kokoro 合成中文句 → soxr 降到 16k → 按 100ms/帧喂给流式 ASR，逐帧记录
partial 文本与 CPU 开销。合成侧同时量「首片到手延迟」。

关键实验设计（别省）：
  * 校准音频尾部追加 silence_ms 静音。真实链路里 EOU 之前本来就有静音窗，
    硬切断音频会系统性高估尾延迟、丢尾部 token。
  * 出词粒度 = partial 更新间隔，它是模型固有属性，不由上层喂帧节奏决定。

用法：
  uv run python scripts/calibrate.py                 # 全量
  uv run python scripts/calibrate.py --only transducer
  uv run python scripts/calibrate.py --silence-ms 200,300,600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_pipeline.providers import sherpa as R  # noqa: E402

# 探针句：短/中/长各一档，覆盖日常问答句式
PROBES: list[str] = [
    "今天天气怎么样",
    "我想问一下明天的天气怎么样",
    "帮我把明天的日程念一遍，顺便提醒我下午三点开会",
]

# 超短句单独一组：TTS 有固定开销，第一句越短首音越早，
# 但短到一定程度就是无信息量的填充词。这条权衡线要用数据画出来。
SHORT_PROBES: list[str] = ["好", "在的", "好的，我看看"]

ENGINES: dict[str, type[R.StreamingAsr]] = {
    "paraformer": R.ParaformerStreamingAsr,
    "transducer": R.TransducerStreamingAsr,
}


def machine_name() -> str:
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    mem = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    gib = int(mem) / 1024**3 if mem.isdigit() else 0
    return f"{chip or platform.processor()} / {gib:.0f}GB / macOS {platform.mac_ver()[0]}"


def cer(reference: str, hypothesis: str) -> float:
    """字符错误率（编辑距离 / 参考长度），中文性能的最简可复核口径。"""
    if not reference:
        return 0.0
    prev = list(range(len(hypothesis) + 1))
    for i, rc in enumerate(reference, start=1):
        cur = [i]
        for j, hc in enumerate(hypothesis, start=1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)),
            )
        prev = cur
    return prev[-1] / len(reference)


@dataclass
class TtsProbe:
    text: str
    audio_ms: float
    synth_ms: float
    rtf: float
    first_chunk_ms: float
    first_chunk_samples: int
    callback_calls: int


@dataclass
class AsrProbe:
    engine: str
    text: str
    silence_ms: int
    audio_ms: float
    first_char_ms: float
    settle_ms: float
    last_change_ms: float
    tail_ms: float
    gap_ms: float
    gaps: list[int]
    cpu_ms: float
    rtf: float
    final: str
    cer: float
    updates: list[list[Any]] = field(default_factory=list)


async def probe_tts(tts: R.KokoroTts, text: str) -> tuple[TtsProbe, np.ndarray, int]:
    start = time.perf_counter()
    samples, sr = tts.generate(text)
    synth_ms = (time.perf_counter() - start) * 1000.0
    audio_ms = len(samples) / sr * 1000.0
    latency, got, _ = tts.first_chunk_latency_ms(text)
    calls = tts.callback_profile(text)
    return (
        TtsProbe(
            text=text,
            audio_ms=round(audio_ms, 1),
            synth_ms=round(synth_ms, 1),
            rtf=round(synth_ms / audio_ms, 3) if audio_ms else 0.0,
            first_chunk_ms=round(latency, 1),
            first_chunk_samples=got,
            callback_calls=len(calls),
        ),
        samples,
        sr,
    )


async def probe_asr(
    engine_cls: type[R.StreamingAsr],
    name: str,
    text: str,
    samples: np.ndarray,
    sr: int,
    silence_ms: int,
) -> AsrProbe:
    pcm16 = soxr.resample(samples, sr, 16000)
    tail = int(16000 * silence_ms / 1000)
    pcm16 = np.append(pcm16, np.zeros(tail, dtype=np.float32))

    asr = engine_cls()
    asr.wall_ms = 0.0
    block = 1600  # 100ms @16k
    marks: list[tuple[float, str]] = []
    for i in range(0, len(pcm16), block):
        await asr.push_audio(R.pcm16_bytes(pcm16[i : i + block]))
        cur = asr.text()
        if not marks or marks[-1][1] != cur:
            marks.append((asr.fed_ms, cur))
    final = await asr.finalize()

    audio_ms = (len(pcm16) - tail) / 16000 * 1000.0
    changes = [(round(m), t) for m, t in marks if t]
    first_char = changes[0][0] if changes else -1.0
    # settle：partial 首次等于最终结果的那一点（此后不再变化）
    settle = next((m for m, t in changes if t == final), -1.0)
    last_change = changes[-1][0] if changes else -1.0
    gaps = [b[0] - a[0] for a, b in zip(changes, changes[1:], strict=False)]
    return AsrProbe(
        engine=name,
        text=text,
        silence_ms=silence_ms,
        audio_ms=round(audio_ms, 1),
        first_char_ms=round(first_char, 1),
        settle_ms=round(settle, 1),
        last_change_ms=round(last_change, 1),
        tail_ms=round(settle - audio_ms, 1) if settle >= 0 else -1.0,
        gap_ms=round(statistics.median(gaps), 1) if gaps else -1.0,
        gaps=gaps,
        cpu_ms=round(asr.wall_ms, 1),
        rtf=round(asr.wall_ms / asr.fed_ms, 4) if asr.fed_ms else 0.0,
        final=final,
        cer=round(cer(text, final), 3),
        updates=[[m, t] for m, t in changes],
    )


def print_short(rows: list[TtsProbe]) -> None:
    print("\n=== TTS 首音地板（第一句越短，首音越早）===")
    print(f"{'句':<12}{'音频ms':>8}{'合成ms':>8}{'RTF':>7}{'首片ms':>8}")
    for r in rows:
        print(
            f"{r.text:<12}{r.audio_ms:>8.0f}{r.synth_ms:>8.0f}{r.rtf:>7.2f}{r.first_chunk_ms:>8.0f}"
        )


def print_report(tts_rows: list[TtsProbe], asr_rows: list[AsrProbe]) -> None:
    print("\n=== TTS (kokoro int8, 24kHz) ===")
    print(
        f"{'句':<26}{'音频ms':>8}{'合成ms':>9}{'RTF':>7}{'首片ms':>9}{'回调次数':>10}",
    )
    for r in tts_rows:
        label = r.text if len(r.text) <= 24 else r.text[:23] + "…"
        print(
            f"{label:<26}{r.audio_ms:>8.0f}{r.synth_ms:>9.0f}{r.rtf:>7.2f}"
            f"{r.first_chunk_ms:>9.0f}{r.callback_calls:>10}",
        )
    if any(r.callback_calls <= 1 for r in tts_rows):
        print(
            "  注意：回调次数=1 说明 sherpa-onnx 的 kokoro 在整句合成完后才一次性"
            "交付样本，没有增量流——首片到手时间 ≈ 整句合成时间。",
        )

    print("\n=== ASR 流式出词 ===")
    print(
        f"{'engine':<12}{'silence':>8}{'音频ms':>8}{'首字ms':>8}{'稳定ms':>8}"
        f"{'尾延迟':>8}{'出词粒度':>10}{'CPUms':>8}{'RTF':>7}{'CER':>7}",
    )
    for r in asr_rows:
        print(
            f"{r.engine:<12}{r.silence_ms:>8}{r.audio_ms:>8.0f}{r.first_char_ms:>8.0f}"
            f"{r.settle_ms:>8.0f}{r.tail_ms:>8.0f}{r.gap_ms:>10.0f}"
            f"{r.cpu_ms:>8.0f}{r.rtf:>7.3f}{r.cer:>7.2f}",
        )
    print("\n注：")
    print("  · 尾延迟 = 文本不再变化 − 用户闭嘴时刻；负值说明最终结果比中间状态早稳定，")
    print("    通常是尾部被漏识别（看同行的 CER）。")
    print("  · settle=-1 表示最终结果只在 input_finished() 之后才出现，中途没提前拿到。")


async def main() -> int:
    ap = argparse.ArgumentParser(description="本机延迟校准")
    ap.add_argument("--only", choices=[*ENGINES, "all"], default="all")
    ap.add_argument("--silence-ms", default="300", help="逗号分隔，尾部追加静音毫秒数")
    ap.add_argument("--out", default="bench", help="结果输出目录")
    args = ap.parse_args()

    silences = [int(s) for s in args.silence_ms.split(",")]
    engines = ENGINES if args.only == "all" else {args.only: ENGINES[args.only]}

    print(f"机器: {machine_name()}")
    t0 = time.perf_counter()
    tts = R.KokoroTts()
    print(f"TTS 加载 {(time.perf_counter() - t0) * 1000:.0f}ms  sr={tts.sample_rate}")

    tts_rows: list[TtsProbe] = []
    audios: list[tuple[str, np.ndarray, int]] = []
    for text in PROBES:
        row, samples, sr = await probe_tts(tts, text)
        tts_rows.append(row)
        audios.append((text, samples, sr))
        print(f"  合成 {row.audio_ms:.0f}ms / RTF {row.rtf:.2f}  {text[:20]}")

    short_rows: list[TtsProbe] = []
    for text in SHORT_PROBES:
        row, _, _ = await probe_tts(tts, text)
        short_rows.append(row)

    asr_rows: list[AsrProbe] = []
    for name, cls in engines.items():
        for silence in silences:
            for text, samples, sr in audios:
                row = await probe_asr(cls, name, text, samples, sr, silence)
                asr_rows.append(row)
        print(f"  {name} done")

    print_report(tts_rows, asr_rows)
    print_short(short_rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "machine": machine_name(),
        "created_at": stamp,
        "probes": PROBES,
        "silences_ms": silences,
        "tts": [asdict(r) for r in tts_rows],
        "tts_short": [asdict(r) for r in short_rows],
        "asr": [asdict(r) for r in asr_rows],
    }
    path = out_dir / f"calibrate-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
