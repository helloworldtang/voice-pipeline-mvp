#!/usr/bin/env python3
"""云流式 TTS 首片拆解：账单里那 ~650ms 是什么构成的。

用法：
    uv run python scripts/probe_tts_firstchunk.py --repeat 20 \\
        --out bench/tts-firstchunk-$(date +%Y%m%d-%H%M%S).json

为什么需要它：`docs/优化空间-20261010.md` §三② 的目标——TTS 首片是口径B 里最大的一格
（真人 26 轮 p50 = 652ms，占 63%），而本机 654ms 与 ECS 600ms 同量级，**不像网络 RTT 主导**。
这份探针把它的构成拆开，回答「还有没有 100~200ms 可榨」：

1. **同句重复 n 次**：首片 p50/p90/min/max——方差本身是结论（p90 远高于 p50 说明有排队/抖动）
2. **冷 vs 热**：第 1 次（冷）与去掉它之后的分布对比——差值是「服务端冷启动」的证据
3. **音色对照**：换 v2 音色各测几次（失败即如实记录错误，不假装测过）
4. **平台性复验**：1 字 vs 25 字（首片应相近——账单里已见「与首句长度无关」）
5. **网络 RTT 上界**：到 DashScope 端点的 TCP 握手耗时（首片里的网络成分不可能超过它）

两条工程纪律（都是被现实逼出来的）：

- **所有阶段在同一个事件循环里跑**：DashScope SDK 的回调/连接状态绑在创建它的 loop 上，
  阶段之间再起一个 `asyncio.run` 会死锁（首版这样挂住过）。
- **每次调用都有超时，超时即重建 provider**：实测同一 provider 连续调用时**偶发卡死**
  （`synth()` 没有超时，卡住就永远等）。探针不能因此挂死，而且失败率本身就是要报的数字——
  `errors` 字段如实记录在第几次、等了多久、什么错。

不测什么：不评音质、不测整句耗时曲线（那是 `scripts/probe_tts_curve.py` 的活）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from voice_pipeline.providers.tts.dashscope_tts import DEFAULT_MODEL, DEFAULT_VOICE, DashScopeTts

HOST = "dashscope.aliyuncs.com"
PORT = 443
# 阶段之间的喘息：别把服务端打成压测——这里测的是常态，不是极限
GAP_S = 0.2


async def synth_once(tts: DashScopeTts, text: str, timeout_s: float) -> dict[str, Any]:
    """合成一次：成功返回首片/整句耗时，失败/超时如实返回 error（不抛给调用方）。"""
    t0 = time.perf_counter()
    state = {"first_ms": -1.0, "chunks": 0, "nbytes": 0}

    async def run() -> None:
        async for chunk in tts.synth(text):
            if state["first_ms"] < 0:
                state["first_ms"] = (time.perf_counter() - t0) * 1000.0
            state["chunks"] += 1
            state["nbytes"] += len(chunk)

    try:
        await asyncio.wait_for(run(), timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - 超时/回调错误都要如实入账
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "waited_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
    return {
        "first_chunk_ms": state["first_ms"],
        "total_ms": (time.perf_counter() - t0) * 1000.0,
        "chunks": state["chunks"],
        "bytes": state["nbytes"],
    }


def stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(q: float) -> float:
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {
        "n": float(len(ordered)),
        "min": round(ordered[0], 1),
        "p50": round(statistics.median(ordered), 1),
        "p90": round(pct(0.9), 1),
        "max": round(ordered[-1], 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


class Runner:
    """连发 N 次合成；失败即换一个 provider（连带换掉那条卡住的连接）。"""

    def __init__(self, voice: str, timeout_s: float) -> None:
        self.voice = voice
        self.timeout_s = timeout_s
        self._tts = DashScopeTts(voice=voice)

    async def once(self, text: str, idx: int) -> dict[str, Any]:
        result = await synth_once(self._tts, text, self.timeout_s)
        if "error" in result:
            result["index"] = idx
            self._tts = DashScopeTts(voice=self.voice)  # 重建：卡死的那条连接不复用
        return result


def split_runs(runs: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
    firsts = [float(r["first_chunk_ms"]) for r in runs if "first_chunk_ms" in r]
    errors = [r for r in runs if "error" in r]
    return firsts, errors


async def phase_repeat(voice: str, text: str, repeat: int, timeout_s: float) -> dict[str, Any]:
    runner = Runner(voice, timeout_s)
    runs: list[dict[str, Any]] = []
    for i in range(repeat):
        runs.append(await runner.once(text, i))
        await asyncio.sleep(GAP_S)
    firsts, errors = split_runs(runs)
    cold = float(runs[0]["first_chunk_ms"]) if "first_chunk_ms" in runs[0] else -1.0
    return {
        "text": text,
        "runs_first_chunk_ms": [round(v, 1) for v in firsts],
        "n_ok": len(firsts),
        "n_err": len(errors),
        "errors": errors,
        "cold_first_ms": round(cold, 1),
        "all": stat(firsts),
        "warm_excl_first": stat(firsts[1:]),
        "total_ms": stat([float(r["total_ms"]) for r in runs if "total_ms" in r]),
        "bytes": stat([float(r["bytes"]) for r in runs if "bytes" in r]),
    }


async def phase_voices(
    voices: list[str], text: str, repeat: int, timeout_s: float
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for voice in voices:
        try:
            runner = Runner(voice, timeout_s)
            runs = []
            for i in range(repeat):
                runs.append(await runner.once(text, i))
                await asyncio.sleep(GAP_S)
            firsts, errors = split_runs(runs)
            out.append(
                {
                    "voice": voice,
                    "first_chunk_ms": stat(firsts),
                    "n_err": len(errors),
                    "errors": errors,
                }
            )
        except Exception as exc:  # noqa: BLE001 - 音色名可能不存在，如实记录而不是假装测过
            out.append({"voice": voice, "error": f"{type(exc).__name__}: {exc}"})
    return out


async def phase_lengths(
    voice: str, texts: list[str], repeat: int, timeout_s: float
) -> list[dict[str, Any]]:
    runner = Runner(voice, timeout_s)
    out: list[dict[str, Any]] = []
    for text in texts:
        runs = []
        for i in range(repeat):
            runs.append(await runner.once(text, i))
            await asyncio.sleep(GAP_S)
        firsts, errors = split_runs(runs)
        out.append(
            {
                "text": text,
                "chars": len(text),
                "first_chunk_ms": stat(firsts),
                "n_err": len(errors),
                "errors": errors,
            }
        )
    return out


def phase_network(times: int = 5) -> dict[str, Any]:
    """到端点的 TCP 握手耗时——首片里网络成分的上界（还没算服务端处理）。"""
    samples: list[float] = []
    for _ in range(times):
        t0 = time.perf_counter()
        with socket.create_connection((HOST, PORT), timeout=5.0):
            samples.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.1)
    return {"host": f"{HOST}:{PORT}", "tcp_connect_ms": stat(samples)}


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": DEFAULT_MODEL,
        "voice": args.voice,
        "repeat": args.repeat,
        "call_timeout_s": args.call_timeout,
    }
    print("[1/4] 同句重复…", flush=True)
    report["same_sentence"] = await phase_repeat(
        args.voice, args.text, args.repeat, args.call_timeout
    )
    print("[2/4] 音色对照…", flush=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    report["voices"] = await phase_voices(voices, args.text, 3, args.call_timeout)
    print("[3/4] 长度对照…", flush=True)
    report["lengths"] = await phase_lengths(args.voice, ["好", args.text], 3, args.call_timeout)
    print("[4/4] 网络握手…", flush=True)
    report["network"] = phase_network()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="云流式 TTS 首片拆解（那 650ms 的构成）")
    ap.add_argument("--repeat", type=int, default=20, help="同句重复次数（默认 20）")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="主测音色")
    ap.add_argument(
        "--voices",
        default=f"{DEFAULT_VOICE},longwan_v2",
        help="音色对照（逗号分隔；不存在的音色会如实记录错误）",
    )
    ap.add_argument(
        "--text", default="今天上海天气怎么样，多云转晴，气温十八到二十五度。", help="基准句"
    )
    ap.add_argument("--call-timeout", type=float, default=20.0, help="单次合成超时（秒）")
    ap.add_argument(
        "--out", default=None, help="结果 JSON 路径（默认 bench/tts-firstchunk-<ts>.json）"
    )
    args = ap.parse_args()

    report = asyncio.run(run_probe(args))

    out = Path(args.out or f"bench/tts-firstchunk-{datetime.now():%Y%m%d-%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["same_sentence"]
    print(f"\n同句重复：成功 {s['n_ok']} / 失败 {s['n_err']}")
    if s["n_ok"]:
        print(
            f"  首片 p50={s['all']['p50']:.0f} p90={s['all']['p90']:.0f} "
            f"min={s['all']['min']:.0f} max={s['all']['max']:.0f} ms"
        )
        warm = s["warm_excl_first"]["p50"]
        print(f"  冷启动第 1 次={s['cold_first_ms']:.0f}ms；去掉它后 p50={warm:.0f}ms")
        print(f"  整句合成 total p50={s['total_ms']['p50']:.0f}ms")
    for err in s["errors"]:
        print(f"  ⚠️ 第 {err['index']} 次：{err['error']}（等了 {err['waited_ms']}ms）")
    print("音色对照:")
    for v in report["voices"]:
        if "error" in v:
            print(f"  {v['voice']}: 失败 {v['error']}")
        else:
            print(f"  {v['voice']}: 首片 p50={v['first_chunk_ms']['p50']:.0f}ms 失败 {v['n_err']}")
    print("长度对照（应相近）:")
    for item in report["lengths"]:
        print(
            f"  {item['chars']:>2} 字: 首片 p50={item['first_chunk_ms']['p50']:.0f}ms"
            f" 失败 {item['n_err']}"
        )
    net = report["network"]["tcp_connect_ms"]
    print(f"网络：TCP 握手 p50={net['p50']:.0f}ms（本机→{HOST}）")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
