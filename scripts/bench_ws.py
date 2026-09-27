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
    uv run python scripts/bench_ws.py --repeat 3 --grace-ms 0,300
    uv run python scripts/bench_ws.py --modes serial,pipelined   # 零延迟替身，只验结构

诚实声明：上行语音是 TTS 合成的（方案 §13.1），不是真人录音 —— 数字用于比较
调度器之间的**相对差值**，不是真人场景的绝对值。另外 `eou_missed` 一列为 1 表示
服务端在超时前没判停、由脚本兜底收尾，那一行不是「自动判停」的成绩。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
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
    eou_latency_ms: float
    grace_ms: float
    caliber_b_ms: float
    spec_hit: float
    eou_missed: int
    audio_bytes: int
    wall_ms: float
    rep: int = 0
    tts_engine: str = ""
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


def _wav_name(utterance: str) -> str:
    """上行语音的落盘名：文本哈希，跨机器稳定——Mac 上存、ECS 上读，名字对得上。"""
    return hashlib.sha1(utterance.encode()).hexdigest()[:12] + ".wav"


def save_wavs(directory: str, audios: dict[str, np.ndarray]) -> None:
    """把上行语音存成 16k PCM16 WAV（在 Mac 上跑一次，内存受限的机器就免加载 kokoro）。"""
    import soundfile as sf

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for utterance, pcm in audios.items():
        sf.write(target / _wav_name(utterance), pcm, 16000, subtype="PCM_16")
    print(f"上行语音已存 {target}（{len(audios)} 段，拷到目标机用 --wavs-dir 消费）")


def load_wavs(directory: str) -> dict[str, np.ndarray]:
    """读取预合成的上行语音。2G 内存的 ECS 上 bench 进程若再加载 kokoro（~500M）
    会把服务端模型挤出内存——上行语音在 Mac 上合成好带过来，数字仍然可比。"""
    import soundfile as sf

    target = Path(directory)
    audios: dict[str, np.ndarray] = {}
    for utterance in USER_UTTERANCES:
        path = target / _wav_name(utterance)
        if not path.exists():
            msg = f"缺 {path}——先在有模型的机器上跑 --save-wavs 生成并拷过来"
            raise FileNotFoundError(msg)
        data, sr = sf.read(path, dtype="float32")
        if sr != 16000:
            msg = f"{path} 采样率 {sr} ≠ 16000，重存"
            raise ValueError(msg)
        audios[utterance] = data
    return audios


async def run_once(
    url: str,
    mode: str,
    pcm: np.ndarray,
    grace_ms: float,
    eou_wait_ms: float,
    utterance: str,
    rep: int,
    gating: str = "neural",
    tts_engine: str = "",
) -> Row:
    t0 = time.perf_counter()
    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
        await ws.recv()  # 首个默认 session.ready
        await ws.send(
            SessionBegin(
                mode=mode,  # type: ignore[arg-type]
                gating=gating,  # type: ignore[arg-type]
                grace_ms=grace_ms,
            ).model_dump_json()
        )
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready

        # 上行语音，按真实速度（100ms/帧）
        for i in range(0, len(pcm), BLOCK):
            await ws.send(pcm16_bytes(pcm[i : i + BLOCK]))
            await asyncio.sleep(BLOCK / 16000)

        # ⚠️ 用户闭嘴后**麦克风仍在送静音帧** —— 真实系统里静音累计会持续增长，
        # 判停由服务端的 FusedEou 决定。旧版这里发完 silence_ms 静音就停喂，
        # 静音累计被冻结在 250ms（刚好卡在 semantic 阈值的边界上），于是
        # 「判不了停」——那是测量方式造成的人为漏判，不是产品行为。
        # 已改为持续喂静音直到服务端判停或超时。
        stop = asyncio.Event()

        async def feed_silence() -> None:
            zeros = np.zeros(BLOCK, dtype=np.float32)
            while not stop.is_set():
                try:
                    await ws.send(pcm16_bytes(zeros))
                except Exception:  # noqa: BLE001  连接已关就退出
                    return
                await asyncio.sleep(BLOCK / 16000)

        feeder = asyncio.create_task(feed_silence())
        bill: dict[str, float] = {}
        audio_bytes = 0
        eou_missed = 0

        async def drain(timeout_s: float) -> dict[str, float] | None:
            nonlocal audio_bytes
            end = time.monotonic() + timeout_s
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=max(end - time.monotonic(), 0.1))
                if isinstance(msg, bytes):
                    audio_bytes += len(msg)
                    continue
                payload = json.loads(msg)
                if payload["type"] == "turn.end":
                    return payload["metrics"]
                if payload["type"] == "error":
                    return None

        try:
            bill = await drain(eou_wait_ms / 1000.0) or {}
        except (TimeoutError, asyncio.TimeoutError):
            eou_missed = 1
            await ws.send(SessionEnd().model_dump_json())
            try:
                bill = await drain(eou_wait_ms / 1000.0) or {}
            except (TimeoutError, asyncio.TimeoutError):
                bill = {}
        finally:
            stop.set()
            feeder.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feeder

    wall = (time.perf_counter() - t0) * 1000
    return Row(
        mode=mode,
        utterance=utterance,
        eou_latency_ms=float(bill.get("eou_latency_ms", -1.0)),
        grace_ms=grace_ms if "pipelined" in mode else 0.0,
        caliber_b_ms=float(bill.get("caliber_b_ms", -1.0)),
        spec_hit=float(bill.get("spec_hit", 0.0)),
        eou_missed=eou_missed,
        audio_bytes=audio_bytes,
        wall_ms=round(wall, 1),
        rep=rep,
        tts_engine=tts_engine,
        extra={k: float(v) for k, v in bill.items() if isinstance(v, (int, float))},
    )


def _stats(vals: list[float]) -> str:
    if not vals:
        return "-"
    return f"{statistics.median(vals):.0f} (min {min(vals):.0f} / max {max(vals):.0f})"


def print_report(rows: list[Row], eou_wait_ms: float) -> None:
    print("\n=== WS 链路 口径B（用户说完 → AI 首音）: p50 (min / max) ===")
    print(
        f"{'句':<16}{'判停延迟':>9}{'grace':>7}{'mode':>16}{'口径B':>26}{'命中':>7}{'EOU漏判':>9}"
    )
    groups: dict[tuple[str, float, str], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.utterance, r.grace_ms, r.mode), []).append(r)
    for key in sorted(groups):
        grp = groups[key]
        label = key[0] if len(key[0]) <= 14 else key[0][:13] + "…"
        hits = sum(r.spec_hit for r in grp)
        missed = sum(r.eou_missed for r in grp)
        eou = _stats([r.eou_latency_ms for r in grp if r.eou_latency_ms >= 0])
        print(
            f"{label:<16}{eou:>9}{key[1]:>7.0f}{key[2]:>16}"
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
    ap.add_argument("--grace-ms", default="0,300")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--eou-wait-ms", type=float, default=2000.0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--wavs-dir",
        default=None,
        help="从预合成 WAV 加载上行语音（文本哈希命名），不在本机加载 kokoro",
    )
    ap.add_argument(
        "--save-wavs",
        default=None,
        help="合成后把上行语音存成 WAV（在有模型的机器上生成，供 --wavs-dir 消费）",
    )
    ap.add_argument(
        "--gating",
        default="neural",
        help="判停档位（conservative/semantic/neural），可逗号分隔做 A/B",
    )
    ap.add_argument(
        "--tts",
        default=os.environ.get("VOICE_PIPELINE_TTS_ENGINE", "melo"),
        help="服务端 TTS 引擎，逗号分隔做 A/B（每个引擎各起一个独立服务端进程）",
    )
    args = ap.parse_args()

    url = f"ws://{HOST}:{PORT}/ws"
    tts_engines = [e.strip() for e in args.tts.split(",") if e.strip()]
    all_rows: list[Row] = []
    for tts_engine in tts_engines:
        # 引擎只能经 env 传给服务端子进程（部署配置不进 WS 协议，见 factory.resolve_tts_engine）
        env = {**os.environ, "VOICE_PIPELINE_TTS_ENGINE": tts_engine}
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
            env=env,
        )
        try:
            await wait_ready(url)
            print(
                f"机器: {platform.machine()} / macOS {platform.mac_ver()[0]}   "
                f"Python {platform.python_version()}"
            )
            print(f"服务端已就绪 {url}   modes={args.modes}   tts={tts_engine}")

            # 上行语音用固定引擎合成：它只是「用户说的话」，必须两组完全一致，
            # 否则 A/B 比的是两段的语音，不是服务端 TTS。
            # --wavs-dir 时直接读盘——内存受限机器不加载 kokoro（见 load_wavs）。
            if args.wavs_dir:
                audios = load_wavs(args.wavs_dir)
            else:
                up_tts = R.build_tts_engine("kokoro", num_threads=2)
                audios = {u: await synthesize(u, up_tts) for u in USER_UTTERANCES}
                if args.save_wavs:
                    save_wavs(args.save_wavs, audios)

            rows: list[Row] = []
            for utterance, pcm in audios.items():
                for gating in [g.strip() for g in args.gating.split(",") if g.strip()]:
                    for grace in [float(g) for g in args.grace_ms.split(",") if g]:
                        for mode in args.modes.split(","):
                            if mode.strip() == "":
                                continue
                            if "pipelined" not in mode and grace != float(
                                args.grace_ms.split(",")[0]
                            ):
                                continue
                            for rep in range(args.repeat):
                                row = await run_once(
                                    url,
                                    mode.strip(),
                                    pcm,
                                    grace,
                                    args.eou_wait_ms,
                                    utterance,
                                    rep,
                                    gating=gating,
                                    tts_engine=tts_engine,
                                )
                                print(
                                    f"  {utterance[:8]:<10} {gating:<13} grace={grace:<5.0f} "
                                    f"{mode:<15} 口径B={row.caliber_b_ms:>7.0f}ms "
                                    f"判停={row.eou_latency_ms:>6.0f}ms "
                                    f"hit={row.spec_hit:.0f} eou漏判={row.eou_missed}"
                                )
                                rows.append(row)
            print_report(rows, args.eou_wait_ms)
            all_rows.extend(rows)

            out = args.json_out or f"bench/ws-{tts_engine}-{datetime.now():%Y%m%d-%H%M%S}.json"
            Path("bench").mkdir(parents=True, exist_ok=True)
            Path(out).write_text(
                json.dumps(
                    {
                        "machine": f"{platform.machine()} macOS {platform.mac_ver()[0]}",
                        "link": "uvicorn + websockets（真 TCP + JSON 编解码）",
                        "eou": "服务端 FusedEou 自动判停（超时才兜底 session.end）",
                        "tts_engine": tts_engine,
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

    if len(tts_engines) > 1:
        print("\n=== 引擎 A/B 汇总（同一句话、同一判停档）===")
        print(f"{'句':<16}{'引擎':>8}{'口径B p50':>12}{'判停':>9}")
        groups: dict[tuple[str, str], list[float]] = {}
        for r in all_rows:
            groups.setdefault((r.utterance, r.tts_engine), []).append(r.caliber_b_ms)
        for (utt, eng), vals in sorted(groups.items()):
            print(f"{utt[:14]:<16}{eng:>8}{statistics.median(vals):>11.0f}ms")

        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
