#!/usr/bin/env python3
"""预合成 filler 音频（给 `core/filler.py` 用）。

用法：
    uv run python scripts/make_filler.py                 # 默认 3 条 → ~/.voice-pilot/filler
    uv run python scripts/make_filler.py --out /tmp/f --texts "嗯，我看一下。" "好的。"

为什么离线生成、而不是运行时懒合成：filler 的意义就是「**它已经在那了**」——运行时现合成
的话第一轮等于没开（那一格本来就是 650ms 的 TTS 首片）。所以离线生成、落盘、启动时直接读。

输出：目录里 index.json + 每条一个 .pcm（24k PCM16 mono，与下行同格式、零转换）。
文件名 = 文本 sha1 前 12 位；重复跑覆盖同名文件。用同一个 TTS 引擎与音色，
启动服务时 **不要**加 `--repeat`：一次合成一条即可，观察的是首片延迟。

打开开关（起服务时）：
    VOICE_FILLER=1 ./scripts/dev_server.sh
"""

from __future__ import annotations

import argparse
import array
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from voice_pipeline.core.filler import DEFAULT_FILLER_DIR
from voice_pipeline.providers.tts.dashscope_tts import DEFAULT_MODEL, DEFAULT_VOICE, DashScopeTts

# 三条中性、短、互相不同：轮转着用，避免每条都是同一句。
# ⚠️ 必须短：filler 与真回答走同一条 FIFO 下行，**比 TTS 首片（~650ms）长的 filler 会把
# 真回答的「被听见起点」往后推**（2026-10-17 实测：1190ms 的片段把回答推后约 400ms）。
# 目标是「搭话不占位」：片段结束前，真回答的首片已经就绪。
DEFAULT_TEXTS = ("嗯。", "好的。", "我看看。")
# 片段时长上限（ms）：超过它 warning 并以非 0 退出（除非 --allow-long）——
# 这不是美学约束，是上面那条 FIFO 的硬边界。
DEFAULT_MAX_MS = 600.0
SAMPLE_RATE = 24000


def trim_silence(pcm: bytes, *, head_ms: int = 50, tail_ms: int = 80) -> bytes:
    """裁掉首尾静音，各留 head/tail 的余量。

    为什么必须裁：实测「嗯，我看一下。」这条原生输出 4.09s，其中**首部静音就有
    2.74s**——不裁的话 filler 会把真实回答往后推几秒，等于自己制造延迟（正是我们想
    盖住的那段）。裁完只留语音 + 一点余量。
    """
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    frame = int(SAMPLE_RATE * 0.02)
    if len(samples) < frame * 2:
        return pcm
    rms: list[float] = []
    peak = 0.0
    for i in range(0, len(samples) - frame, frame):
        seg = samples[i : i + frame]
        value = (sum(float(x) * x for x in seg) / len(seg)) ** 0.5
        rms.append(value)
        peak = max(peak, value)
    threshold = max(peak * 0.05, 60.0)
    voiced = [i for i, value in enumerate(rms) if value >= threshold]
    if not voiced:
        return pcm
    start = max(0, voiced[0] * frame - int(SAMPLE_RATE * head_ms / 1000))
    end = min(len(samples), (voiced[-1] + 1) * frame + int(SAMPLE_RATE * tail_ms / 1000))
    return samples[start:end].tobytes()


def duration_ms(pcm: bytes) -> float:
    return len(pcm) / 2 / SAMPLE_RATE * 1000.0


async def synth_all(
    texts: list[str], out: Path, *, max_ms: float, allow_long: bool
) -> list[dict[str, str]]:
    """在**同一个事件循环**里依次合成（SDK 的连接/回调绑 loop，见 probe_tts_firstchunk）。"""
    tts = DashScopeTts()
    clips: list[dict[str, str]] = []
    for text in texts:
        pcm_raw = b"".join([chunk async for chunk in tts.synth(text)])
        if not pcm_raw:
            print(f"⚠️ {text!r} 合成结果为空——跳过（不写进 index）", file=sys.stderr)
            continue
        pcm = trim_silence(pcm_raw)
        name = hashlib.sha1(text.encode()).hexdigest()[:12] + ".pcm"
        ms = duration_ms(pcm)
        if ms > max_ms and not allow_long:
            print(
                f"✗ {text!r} 裁后 {ms:.0f}ms > 上限 {max_ms:.0f}ms——"
                "这段 filler 会比 TTS 首片长，把真回答的「被听见起点」推后。"
                "换更短的文本（或确认后加 --allow-long）",
                file=sys.stderr,
            )
            raise SystemExit(2)
        (out / name).write_bytes(pcm)
        clips.append({"text": text, "file": name})
        print(f"✓ {text!r} → {name}（原生 {duration_ms(pcm_raw):.0f}ms → 裁静音后 {ms:.0f}ms）")
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description="预合成 filler 音频")
    ap.add_argument("--out", default=str(DEFAULT_FILLER_DIR), help="输出目录")
    ap.add_argument("--texts", nargs="*", default=list(DEFAULT_TEXTS), help="要合成的文本")
    ap.add_argument(
        "--max-ms",
        type=float,
        default=DEFAULT_MAX_MS,
        help="单条片段时长上限（默认 600ms；超过就失败，因为会推后真回答起点）",
    )
    ap.add_argument("--allow-long", action="store_true", help="明知片段过长仍要写入")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    texts = list(args.texts) or list(DEFAULT_TEXTS)
    clips = asyncio.run(synth_all(texts, out, max_ms=args.max_ms, allow_long=args.allow_long))
    if not clips:
        raise SystemExit("没有一条合成成功，index.json 未写（服务端会当 filler 关闭处理）")

    index = out / "index.json"
    index.write_text(
        json.dumps(
            {
                "sample_rate": SAMPLE_RATE,
                "model": DEFAULT_MODEL,
                "voice": DEFAULT_VOICE,
                "ts": datetime.now().isoformat(timespec="seconds"),
                "clips": clips,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写出 {index}（{len(clips)} 条）")
    print(f"起服务时打开开关：VOICE_FILLER=1 VOICE_FILLER_DIR={out} ./scripts/dev_server.sh")


if __name__ == "__main__":
    main()
