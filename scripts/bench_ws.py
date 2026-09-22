#!/usr/bin/env python3
"""A/B 打在真实 WS 链路上（§4.5 的硬要求）。

与 `bench_schedulers.py` 的差别：

| | bench_schedulers | **bench_ws（本脚本）** |
|---|---|---|
| 链路 | 直接 new 调度器，进程内函数调用 | 真 uvicorn 进程 + 真 TCP + JSON 编解码 |
| EOU | 脚本手动补静音窗代替 `FusedEou` | **服务端 `FusedEou` 自己判停** |
| 收尾 | 脚本调 `on_end_of_utterance()` | 等服务端判停；判不了才兜底 `session.end` |
| 代价 | 快，但不含判停开销与网络开销 | 慢，且数字才配叫「端到端」 |

换句话说：bench_schedulers 里的 179ms 不含 EOU 判停开销，本脚本是补上那一格。

跑法：
    uv run python scripts/bench_ws.py
    uv run python scripts/bench_ws.py --repeat 3 --silence-ms 250,600 --grace-ms 0,300
    uv run python scripts/bench_ws.py --modes serial,pipelined   # 零延迟替身，只验结构

诚实声明：上行语音是 TTS 合成的（方案 §13.1），不是真人录音 —— 数字用于比较
调度器之间的**相对差值**，不是真人场景的绝对值。另外 `eou_missed` 一列为 1 表示
服务端在超时前没判停、由脚本兜底收尾，那一行不是「自动判停」的成绩。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soxr
import websockets

from voice_pipeline.providers import sherpa as R
from voice_pipeline.providers.sherpa import pcm16_bytes
from voice_pipeline.server.protocol import SessionBegin, SessionEnd

HOST = "127.0.0.1"
PORT = 8765
BLOCK = 1600  # 100ms @ 16kHz

USER_UTTERANCES = [
    "今天天气怎么样",
    "我想问一下明天的天气怎么样",
]


@dataclass
class Row:
    mode: str
    utterance: str
    silence_ms: int
    grace_ms: float
    caliber_b_ms: float
    spec_hit: float
    eou_missed: int
    audio_bytes: int
    wall_ms: float
    rep: int = 0
    note: str = ""
    extra: dict[str, float] = field(default_factory=dict)


async def wait_ready(url: str, timeout_s: float = 60.0) -> None:
    """等服务端 healthz。起进程后立刻连 WS 会 connection refused。"""
    import httpx

    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                r = await c.get(f"http://{HOST}:{PORT}/healthz", timeout=2)
                if r.status_code == 200:
                    return
            await asyncio.sleep(0.3)
    msg = f"服务端 {url} 在 {timeout_s}s 内未就绪"
    raise RuntimeError(msg)


async def synthesize(text: str, tts: object) -> np.ndarray:
    """合成「用户语音」并降到 16k。音色固定 kokoro —— 见 bench_schedulers 里的说明，
    melo 合成的话喂给同一个 ASR 识别结果为空。"""
    samples, sr = tts.generate(text)  # type: ignore[attr-defined]
    if sr != 16000:
        samples = soxr.resample(samples, sr, 16000)
    return np.asarray(samples, dtype=np.float32)


async def run_once(
    url: str,
    mode: str,
    pcm: np.ndarray,
    silence_ms: int,
    grace_ms: float,
    eou_wait_ms: float,
    utterance: str,
    rep: int,
) -> Row:
    t0 = time.perf_counter()
    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
        await ws.recv()  # 首个默认 session.ready
        await ws.send(
            SessionBegin(
                mode=mode,  # type: ignore[arg-type]
                gating="semantic",  # type: ignore[arg-type]
                grace_ms=grace_ms,
            ).model_dump_json()
        )
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready

        # 上行：语音 + 静音窗，按真实速度（100ms/帧）
        buf = np.concatenate([pcm, np.zeros(int(16000 * silence_ms / 1000), dtype=np.float32)])
        for i in range(0, len(buf), BLOCK):
            await ws.send(pcm16_bytes(buf[i : i + BLOCK]))
            await asyncio.sleep(BLOCK / 16000)

        # 等服务端自己判停。EOU 若没判，超时后兜底 session.end —— 并记下来，
        # 因为「判不了停」和「判得很慢」是两种完全不同的失败。
        deadline = time.monotonic() + eou_wait_ms / 1000.0
        bill: dict[str, float] = {}
        audio_bytes = 0
        eou_missed = 0

        async def drain() -> dict[str, float] | None:
            nonlocal audio_bytes
            while True:
                msg = await asyncio.wait_for(
                    ws.recv(), timeout=max(deadline - time.monotonic(), 0.1)
                )
                if isinstance(msg, bytes):
                    audio_bytes += len(msg)
                    continue
                payload = json.loads(msg)
                if payload["type"] == "turn.end":
                    return payload["metrics"]
                if payload["type"] == "error":
                    return None

        try:
            bill = await drain() or {}
        except (TimeoutError, asyncio.TimeoutError):
            eou_missed = 1
            await ws.send(SessionEnd().model_dump_json())
            try:
                bill = await drain() or {}
            except (TimeoutError, asyncio.TimeoutError):
                bill = {}

    wall = (time.perf_counter() - t0) * 1000
    return Row(
        mode=mode,
        utterance=utterance,
        silence_ms=silence_ms,
        grace_ms=grace_ms if "pipelined" in mode else 0.0,
        caliber_b_ms=float(bill.get("caliber_b_ms", -1.0)),
        spec_hit=float(bill.get("spec_hit", 0.0)),
        eou_missed=eou_missed,
        audio_bytes=audio_bytes,
        wall_ms=round(wall, 1),
        rep=rep,
        extra={k: float(v) for k, v in bill.items() if isinstance(v, (int, float))},
    )


def _stats(vals: list[float]) -> str:
    if not vals:
        return "-"
    return f"{statistics.median(vals):.0f} (min {min(vals):.0f} / max {max(vals):.0f})"


def print_report(rows: list[Row], eou_wait_ms: float) -> None:
    print("\n=== WS 链路 口径B（用户说完 → AI 首音）: p50 (min / max) ===")
    print(f"{'句':<16}{'静音':>6}{'grace':>7}{'mode':>16}{'口径B':>26}{'命中':>7}{'EOU漏判':>9}")
    groups: dict[tuple[str, int, float, str], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.utterance, r.silence_ms, r.grace_ms, r.mode), []).append(r)
    for key in sorted(groups):
        grp = groups[key]
        label = key[0] if len(key[0]) <= 14 else key[0][:13] + "…"
        hits = sum(r.spec_hit for r in grp)
        missed = sum(r.eou_missed for r in grp)
        print(
            f"{label:<16}{key[1]:>6}{key[2]:>7.0f}{key[3]:>16}"
            f"{_stats([r.caliber_b_ms for r in grp]):>26}"
            f"{f'{hits:.0f}/{len(grp)}':>7}{f'{missed}/{len(grp)}':>9}"
        )

    pipes = [r for r in rows if "pipelined" in r.mode]
    sers = [r for r in rows if "pipelined" not in r.mode]
    if pipes and sers:
        hits = sum(r.spec_hit for r in pipes)
        print(f"\nserial    p50 {statistics.median([r.caliber_b_ms for r in sers]):.0f}ms")
        print(f"pipelined p50 {statistics.median([r.caliber_b_ms for r in pipes]):.0f}ms")
        print(f"WS 链路投机命中率: {hits:.0f}/{len(pipes)} = {hits / len(pipes) * 100:.0f}%")
        print(
            f"⚠️ 最快一次 {min(r.caliber_b_ms for r in pipes):.0f}ms 只占 "
            f"{1 / len(pipes) * 100:.0f}% 轮次 —— 报 p50，别报 min。"
        )
    total_missed = sum(r.eou_missed for r in rows)
    if total_missed:
        print(
            f"\n⚠️ {total_missed}/{len(rows)} 轮服务端 EOU 未在 {eou_wait_ms:.0f}ms 内判停，"
            "由脚本兜底 session.end —— 这些行不代表自动判停的成绩。"
        )


async def main() -> None:
    ap = argparse.ArgumentParser(description="serial vs pipelined 的 WS 链路 A/B")
    ap.add_argument("--modes", default="real-serial,real-pipelined")
    ap.add_argument("--silence-ms", default="250,600")
    ap.add_argument("--grace-ms", default="0,300")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--eou-wait-ms", type=float, default=2000.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    url = f"ws://{HOST}:{PORT}/ws"
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "voice_pipeline.server.app:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await wait_ready(url)
        print(
            f"机器: {platform.machine()} / macOS {platform.mac_ver()[0]}   "
            f"Python {platform.python_version()}"
        )
        print(f"服务端已就绪 {url}   modes={args.modes}")

        up_tts = R.build_tts_engine("kokoro", num_threads=2)
        audios = {u: await synthesize(u, up_tts) for u in USER_UTTERANCES}

        rows: list[Row] = []
        for utterance, pcm in audios.items():
            for silence in [int(s) for s in args.silence_ms.split(",") if s]:
                for grace in [float(g) for g in args.grace_ms.split(",") if g]:
                    for mode in args.modes.split(","):
                        if mode.strip() == "":
                            continue
                        if "pipelined" not in mode and grace != float(args.grace_ms.split(",")[0]):
                            continue
                        for rep in range(args.repeat):
                            row = await run_once(
                                url,
                                mode.strip(),
                                pcm,
                                silence,
                                grace,
                                args.eou_wait_ms,
                                utterance,
                                rep,
                            )
                            print(
                                f"  {utterance[:8]:<10} sil={silence:<4} grace={grace:<5.0f} "
                                f"{mode:<15} 口径B={row.caliber_b_ms:>7.0f}ms "
                                f"hit={row.spec_hit:.0f} eou漏判={row.eou_missed}"
                            )
                            rows.append(row)
        print_report(rows, args.eou_wait_ms)

        out = args.json_out or f"bench/ws-{datetime.now():%Y%m%d-%H%M%S}.json"
        Path("bench").mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(
                {
                    "machine": f"{platform.machine()} macOS {platform.mac_ver()[0]}",
                    "link": "uvicorn + websockets（真 TCP + JSON 编解码）",
                    "eou": "服务端 FusedEou 自动判停（超时才兜底 session.end）",
                    "rows": [
                        {**{k: v for k, v in r.__dict__.items() if k != "extra"}, **r.extra}
                        for r in rows
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n→ {out}")
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
