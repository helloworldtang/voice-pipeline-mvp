#!/usr/bin/env python3
"""scripted WS 客户端：回放固定音频文件走完整链路，确定性取证（差距表补齐）。

与 bench_ws / probe_multi_turn 的分工：
- bench_ws：自己合成上行语音 + 多轮参数扫描，A/B 实验用
- probe_multi_turn：自起服务端，验「多轮 + reset」这一个机制
- **本脚本：不合成、不起服务端**——吃现成 WAV、打现成服务端，
  同一文件回放 N 次结果可复现。真人验收前后用它做「同一句话的对照轮」，
  把真人差异和网络抖动从链路差异里剥出来。

用法：
    uv run python scripts/scripted_client.py --wav up-wavs/xxx.wav --repeat 3 \\
        --url ws://127.0.0.1:8000/ws --mode real-serial --token s3cret
输出：逐轮事件序列摘要 + 账单关键格 + p50，全部来自服务端账单，不经人工修饰。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ws import BLOCK, pcm16_bytes  # type: ignore[import-not-found]


def load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        import soxr

        data = soxr.resample(data, sr, 16000).astype(np.float32)
    return np.asarray(data, dtype=np.float32)


async def run_turn(
    ws: Any,
    pcm: np.ndarray,
    mode: str,
    gating: str,
    token: str | None,
    eou_wait_s: float,
    first_turn: bool,
) -> dict[str, Any]:
    if first_turn:
        await ws.recv()  # 首个默认 session.ready
        begin: dict[str, Any] = {"type": "session.begin", "mode": mode, "gating": gating}
        if token:
            begin["token"] = token
        await ws.send(json.dumps(begin))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready

    for i in range(0, len(pcm), BLOCK):
        await ws.send(pcm16_bytes(pcm[i : i + BLOCK]))
        await asyncio.sleep(BLOCK / 16000)

    silence = b"\x00" * (BLOCK * 2)
    deadline = time.monotonic() + eou_wait_s
    audio_frames = 0
    audio_bytes = 0
    asr_final: str | None = None
    while time.monotonic() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
        except TimeoutError:
            await ws.send(silence)
            continue
        if isinstance(msg, bytes):
            audio_frames += 1
            audio_bytes += len(msg)
            continue
        event = json.loads(msg)
        if event.get("type") == "asr.final":
            asr_final = event["text"]
        if event.get("type") == "turn.end":
            return {
                "asr_final": asr_final,
                "audio_frames": audio_frames,
                "audio_bytes": audio_bytes,
                "bill": event["metrics"],
            }
    msg = f"{eou_wait_s}s 内未出账单（EOU 没判停或链路卡死）"
    raise RuntimeError(msg)


async def main_async(args: argparse.Namespace) -> None:
    pcm = load_wav(args.wav)
    bills = []
    async with websockets.connect(args.url, max_size=None, open_timeout=30) as ws:
        for i in range(args.repeat):
            row = await run_turn(
                ws,
                pcm,
                args.mode,
                args.gating,
                args.token,
                args.eou_wait_s,
                first_turn=(i == 0),
            )
            bills.append(row["bill"])
            bill = row["bill"]
            print(
                f"轮 {i + 1}: asr={row['asr_final']!r} "
                f"口径B={bill.get('caliber_b_ms')}ms 判停={bill.get('eou_latency_ms')}ms "
                f"抢话={bill.get('premature_eou')} 打断={bill.get('barge_in_count')} "
                f"音频={row['audio_bytes']}B"
            )
    calibers = [b["caliber_b_ms"] for b in bills if b.get("caliber_b_ms", -1) >= 0]
    if len(calibers) >= 2:
        p50, lo, hi = statistics.median(calibers), min(calibers), max(calibers)
        print(f"口径B: p50={p50:.0f}ms min={lo:.0f}ms max={hi:.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="回放固定 WAV 走 WS 全链路，确定性取证")
    ap.add_argument("--wav", required=True, help="16k 或任意采样率的单声道 WAV")
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--mode", default="real-serial")
    ap.add_argument("--gating", default="conservative")  # 第四次定档（2026-10-04）
    ap.add_argument("--token", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--eou-wait-s", type=float, default=20.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
