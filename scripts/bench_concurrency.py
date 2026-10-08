#!/usr/bin/env python3
"""并发压测：N 路真人级会话同时打，看每路的延迟退化多少（成熟度判据 9 的取证）。

为什么单独一个脚本
------------------

「并发」此前是 11 项判据里唯一 ❌，根因是本地推理资源**跨连接共享**：同一份 ASR
recognizer / 同一份本地 TTS 实例，两个会话的 ORT 推理在同一个 session 上排队。
它不会报错，只是两边一起变慢——**体感层面的事故，日志里查不到**，所以必须量化。

本脚本做两件事：

1. **A/B**：同一台机器、同一批 provider，只拧 `VOICE_SHARED_INFERENCE` 一个开关——
   - `shared=1`（改造前）：所有会话共用一份推理；
   - `shared=0`（现状）：每会话独占一个槽，`VOICE_MAX_SESSIONS` 之外明确拒绝。
   报的是**每路自己的口径B**，外加总墙钟（真并行的话 N 路总耗时 ≈ 单路，不是 N 倍）。
2. **背压取证**：第 N+1 路必须收到 `error(session_limit)` + 1013，而不是悄悄挤进来。

跑法
----

    uv run python scripts/bench_concurrency.py                 # 默认 2 路 × 2 轮
    uv run python scripts/bench_concurrency.py --sessions 3 --max-sessions 3
    uv run python scripts/bench_concurrency.py --modes serial  # sim 替身，纯结构压测

诚实声明：上行语音是 TTS 合成的（不是真人录音），用户内容在并发下与真实场景不完全
同分布；这里比的是**同一份数据下 shared / pooled 的相对差值**，绝对值请参考
`docs/验收-本机真人-*.md`。另外 sim 模式（mode=serial）不占槽——它不跑 ONNX，
用它验并发数量会得到「无限扩容」的假结论。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soxr
import websockets

from voice_pipeline.providers import sherpa as R
from voice_pipeline.providers.pool import needs_exclusive_slot
from voice_pipeline.providers.sherpa import pcm16_bytes
from voice_pipeline.server.protocol import SessionBegin, SessionEnd

HOST = "127.0.0.1"
PORT = 8766  # 与 bench_ws 错开，防止两个 bench 抢端口
BLOCK = 1600  # 100ms @ 16kHz

USER_UTTERANCES = [
    "今天天气怎么样",
    "帮我查一下明天的安排",
]


@dataclass
class Turn:
    worker: int
    caliber_b_ms: float
    eou_latency_ms: float
    wall_ms: float
    audio_bytes: int
    eou_missed: int
    note: str = ""


@dataclass
class RejectionProbe:
    """第 N+1 路的下场。这是**判据 9 的另一半**：满载必须明确拒绝，不是静默降级。"""

    got_error: bool
    code: str
    close_code: int | None


async def wait_ready(url: str, timeout_s: float = 90.0) -> None:
    """等服务端 healthz。real 模式首次会话要加载模型，给足时间。"""
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
    """合成「用户语音」并降到 16k。上行用 kokoro 固定——换成 melo 喂同一个 ASR
    会识别出空文本（bench_ws 里踩过），两组的数据必须同源才可比。"""
    samples, sr = tts.generate(text)  # type: ignore[attr-defined]
    if sr != 16000:
        samples = soxr.resample(samples, sr, 16000)
    return np.asarray(samples, dtype=np.float32)


async def run_worker(
    url: str,
    mode: str,
    worker: int,
    pcms: list[np.ndarray],
    gating: str,
    eou_wait_ms: float,
    start_barrier: asyncio.Event,
    acquired: asyncio.Event | None = None,
) -> list[Turn]:
    """一个「用户」连续说 rounds 句话。**所有 worker 在同一时刻被放行**，这样才有并发。

    `acquired`：借到槽（session.begin 成功）后点亮——满载探测必须**在 worker 正占用
    资源时**发车，跑完再测就永远测不到拒绝（名额已经还了）。
    """
    await start_barrier.wait()
    turns: list[Turn] = []
    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
        await ws.recv()  # 首个默认 session.ready
        await ws.send(
            SessionBegin(
                mode=mode,  # type: ignore[arg-type]
                gating=gating,  # type: ignore[arg-type]
            ).model_dump_json()
        )
        ready = json.loads(await ws.recv())
        if ready["type"] == "error":
            msg = f"worker {worker}: begin 被拒 {ready}"
            raise RuntimeError(msg)
        if acquired is not None:
            acquired.set()
        for idx, pcm in enumerate(pcms):
            t0 = time.perf_counter()
            for i in range(0, len(pcm), BLOCK):
                await ws.send(pcm16_bytes(pcm[i : i + BLOCK]))
                await asyncio.sleep(BLOCK / 16000)
            stop = asyncio.Event()

            # 闭包必须**绑定**循环的 stop：`async def` 不会捕获当时的值，等到 await
            # 执行时循环变量可能已经换了对象（B023）。
            async def feed_silence(stop_flag: asyncio.Event = stop) -> None:
                zeros = np.zeros(BLOCK, dtype=np.float32)
                while not stop_flag.is_set():
                    try:
                        await ws.send(pcm16_bytes(zeros))
                    except Exception:  # noqa: BLE001  连接关了就退出
                        return
                    await asyncio.sleep(BLOCK / 16000)

            feeder = asyncio.create_task(feed_silence())
            bill: dict[str, float] = {}
            audio_bytes = 0
            eou_missed = 0
            try:
                bill, audio_bytes, eou_missed = await _drain(ws, eou_wait_ms)
                if not bill:
                    await ws.send(SessionEnd().model_dump_json())
                    bill, audio_bytes, eou_missed = await _drain(ws, eou_wait_ms)
            finally:
                stop.set()
                feeder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await feeder
            turns.append(
                Turn(
                    worker=worker,
                    caliber_b_ms=float(bill.get("caliber_b_ms", -1.0)),
                    eou_latency_ms=float(bill.get("eou_latency_ms", -1.0)),
                    wall_ms=round((time.perf_counter() - t0) * 1000, 1),
                    audio_bytes=audio_bytes,
                    eou_missed=eou_missed,
                    note=f"utt{idx}",
                )
            )
    return turns


async def _drain(
    ws: websockets.ClientConnection, eou_wait_ms: float
) -> tuple[dict[str, float], int, int]:
    """收这一轮的下行，直到 turn.end / error / 超时。"""
    bill: dict[str, float] = {}
    audio_bytes = 0
    missed = 0
    end = time.monotonic() + eou_wait_ms / 1000.0
    while True:
        remaining = max(end - time.monotonic(), 0.05)
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except (TimeoutError, asyncio.TimeoutError):
            missed = 1
            return bill, audio_bytes, missed
        if isinstance(msg, bytes):
            audio_bytes += len(msg)
            continue
        payload = json.loads(msg)
        if payload["type"] == "turn.end":
            return payload["metrics"], audio_bytes, missed
        if payload["type"] == "error":
            raise RuntimeError(f"server error: {payload}")


async def probe_rejection(url: str, mode: str, gating: str) -> RejectionProbe:
    """满载后第 N+1 路：必须收到明确的 session_limit，不是被静默排队。"""
    got_error = False
    code = ""
    close_code: int | None = None
    try:
        async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
            await ws.recv()
            await ws.send(
                SessionBegin(
                    mode=mode,
                    gating=gating,  # type: ignore[arg-type]
                ).model_dump_json()
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                if isinstance(raw, bytes):
                    continue
                payload = json.loads(raw)
                if payload["type"] == "error":
                    got_error = True
                    code = str(payload.get("code", ""))
                    break
                if payload["type"] == "session.ready":
                    # 没被拦住 = 满载仍在放行（要么没生效，要么 sim 模式不占槽）
                    return RejectionProbe(False, "no_rejection", None)
            # 等服务端关连接，拿到 close code
            try:
                await ws.wait_closed()
                close_code = ws.close_code
            except Exception:  # noqa: BLE001
                close_code = None
    except Exception as exc:  # noqa: BLE001
        return RejectionProbe(got_error, f"probe_failed:{exc}", close_code)
    return RejectionProbe(got_error, code, close_code)


async def run_case(
    url: str,
    mode: str,
    sessions: int,
    rounds: int,
    audios: list[np.ndarray],
    gating: str,
    eou_wait_ms: float,
    *,
    probe_while_busy: bool = False,
) -> tuple[list[Turn], float, RejectionProbe | None]:
    pcms = audios[:rounds]
    barrier = asyncio.Event()
    acquired = [asyncio.Event() for _ in range(sessions)]
    t0 = time.perf_counter()
    workers = [
        asyncio.create_task(
            run_worker(url, mode, w, pcms, gating, eou_wait_ms, barrier, acquired[w])
        )
        for w in range(sessions)
    ]
    # 先让所有 worker 建好连接（含模型加载），再同时放行——否则「并发」其实是错峰。
    barrier.set()
    probe_task: asyncio.Task[RejectionProbe] | None = None
    if probe_while_busy:
        # 必须在 worker **占着槽**的时候发车：等它们跑完再连，名额早归还了，
        # 会得到「满载不拒绝」的假结论——而这条结论正好是本判据要证明的反面。
        await asyncio.wait_for(asyncio.gather(*[a.wait() for a in acquired]), timeout=120)
        probe_task = asyncio.create_task(probe_rejection(url, mode, gating))
    results = await asyncio.gather(*workers, return_exceptions=True)
    probe = await probe_task if probe_task is not None else None
    wall = (time.perf_counter() - t0) * 1000
    turn_task: list[Turn] = []
    for r in results:
        if isinstance(r, BaseException):
            print(f"  ⚠️ worker 异常：{r}")
            continue
        turn_task.extend(r)
    return turn_task, wall, probe


async def main() -> None:
    ap = argparse.ArgumentParser(description="并发压测：shared（改造前）vs pooled（现状）")
    ap.add_argument("--mode", default="real-serial")
    ap.add_argument("--sessions", type=int, default=2, help="并发路数")
    ap.add_argument("--rounds", type=int, default=2, help="每路说几句")
    ap.add_argument("--max-sessions", type=int, default=2, help="服务端并发上限")
    ap.add_argument("--gating", default="conservative")
    ap.add_argument("--eou-wait-ms", type=float, default=3000.0)
    ap.add_argument(
        "--only",
        choices=("pooled", "shared_before"),
        default=None,
        help="只跑一组（重复取样时用：多跑几次同一个开关，样本量才够下结论）",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    url = f"ws://{HOST}:{PORT}/ws"
    up_tts = R.build_tts_engine("kokoro", num_threads=2)
    audios = [await synthesize(u, up_tts) for u in USER_UTTERANCES]

    report: dict[str, object] = {
        "machine": f"{platform.machine()} macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
        "mode": args.mode,
        "sessions": args.sessions,
        "rounds": args.rounds,
        "max_sessions": args.max_sessions,
        "gating": args.gating,
        "uplink": "kokoro 合成语音 → 16k（两组同一份数据，只拧服务端开关）",
        "cases": {},
    }

    cases: list[tuple[str, bool]] = []
    if args.only != "pooled":
        cases.append(("shared_before", True))
    if args.only != "shared_before":
        cases.append(("pooled", False))

    cases_out: dict[str, object] = {}
    report["cases"] = cases_out

    def _serve(shared: bool) -> subprocess.Popen[bytes]:
        env = {
            **os.environ,
            "VOICE_MAX_SESSIONS": str(args.max_sessions),
            "VOICE_SHARED_INFERENCE": "1" if shared else "0",
        }
        return subprocess.Popen(  # noqa: S603
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
            env=env,
        )

    for name, shared in cases:
        proc = _serve(shared)
        try:
            await wait_ready(url)
            print(f"\n=== {name}（VOICE_SHARED_INFERENCE={int(shared)}）===")
            turns, wall, probe_from_case = await run_case(
                url,
                args.mode,
                args.sessions,
                args.rounds,
                audios,
                args.gating,
                args.eou_wait_ms,
                # 只有 real 系列占槽：sim 模式本来就不受并发上限约束，拿它验
                # 「满载拒绝」会得到永远拒绝不了的假结论。
                probe_while_busy=needs_exclusive_slot(args.mode),
            )
            good = [t for t in turns if t.caliber_b_ms >= 0]
            per_worker: dict[int, list[float]] = {}
            for t in good:
                per_worker.setdefault(t.worker, []).append(t.caliber_b_ms)
            if good:
                print(f"  每路口径B p50：{_per_worker_line(per_worker)}")
                serial_total = sum(t.wall_ms for t in good)
                print(f"  全体口径B p50：{statistics.median([t.caliber_b_ms for t in good]):.0f}ms")
                print(f"  N 路总墙钟：{wall:.0f}ms（各轮时长之和 {serial_total:.0f}ms）")
            else:
                print("  没有测到完整轮次——查服务端日志")

            if probe_from_case is not None:
                ok = probe_from_case.got_error and probe_from_case.code == "session_limit"
                print(
                    f"  满载后第 {args.sessions + 1} 路："
                    f"{'✅ 明确拒绝' if ok else '❌ 没拦住'} "
                    f"code={probe_from_case.code or '-'} close={probe_from_case.close_code}"
                )
            case_payload: dict[str, object] = {
                "shared": shared,
                "per_worker_p50": {
                    str(k): round(statistics.median(v), 1) for k, v in per_worker.items()
                },
                "p50_ms": round(statistics.median([t.caliber_b_ms for t in good]), 1)
                if good
                else None,
                "wall_total_ms": round(wall, 1),
                "turns": [t.__dict__ for t in turns],
                "rejection": probe_from_case.__dict__ if probe_from_case else None,
            }
            cases_out[name] = case_payload
        finally:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)

    _final_line(report)
    out = args.json_out or f"bench/concurrency-{datetime.now():%Y%m%d-%H%M%S}.json"
    Path("bench").mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始账单：{out}")


def _per_worker_line(per_worker: dict[int, list[float]]) -> str:
    return "  ".join(
        f"worker{k}={statistics.median(v):.0f}ms" for k, v in sorted(per_worker.items())
    )


def _final_line(report: dict[str, object]) -> None:
    """把 A/B 的差值说清楚：并发下的**每路延迟**，改造前进了多少。"""
    cases = report.get("cases")
    if not isinstance(cases, dict):
        return
    before = cases.get("shared_before")
    after = cases.get("pooled")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return
    p50_before = before.get("p50_ms")
    p50_after = after.get("p50_ms")
    if not isinstance(p50_before, (int, float)) or not isinstance(p50_after, (int, float)):
        return
    delta = p50_before - p50_after
    n_before = len(before.get("turns") or [])
    n_after = len(after.get("turns") or [])
    print("\n=== 结论（同一台机器、同一份上行语音，只拧一个开关）===")
    print(f"  改造前 shared  p50 {p50_before:.0f}ms  (n={n_before})")
    print(f"  现状   pooled  p50 {p50_after:.0f}ms  (n={n_after})")
    print(f"  并发下的每路 delay 变化：{delta:+.0f}ms / {delta / p50_before * 100:+.0f}%")
    print(f"  N 路总墙钟：{before.get('wall_total_ms')}ms → {after.get('wall_total_ms')}ms")


if __name__ == "__main__":
    asyncio.run(main())
