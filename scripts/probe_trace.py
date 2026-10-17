"""链路追踪探针：跑真实链路，把耗时拆成 span 树 + 残差。

**它要回答的问题**：`口径B（说完→AI首音）` 的每一毫秒是谁烧的，以及**还有多少毫秒
没有被任何一格记账**。后者是 waterfall 打点给不出的东西——两个时刻之差永远是
"某个链"上的差，漏掉的分支不会显形。

用法（默认就是「最接近线上」的一组）：
    uv run python scripts/probe_trace.py                      # real-serial + melo + qwen，6 轮
    uv run python scripts/probe_trace.py --repeat 8 --tts kokoro
    uv run python scripts/probe_trace.py --llm sim            # 只看本机侧（无 LLM 那一格）
    uv run python scripts/probe_trace.py --tts dashscope --no-prewarm   # 预热 A/B 对照臂
    uv run python scripts/probe_trace.py --tts dashscope --no-energy-gate  # 能量门 A/B 对照臂
    uv run python scripts/probe_trace.py --tree               # 额外打一轮完整 span 树

输出：
  - stdout：分格表（p50 / 占比）+ 残差分解（未归因 = 同循环竞争者 + 真·未埋到）
  - --out：每轮一行 JSONL，含 spans（供 `scripts/report_trace.py` 二次分析）

纪律（与 report_turns.py 一致）：缺失写 -1 不写 0；报分布不只报 p50；样本少就明说。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_pipeline.core.eou import GatingMode  # noqa: E402
from voice_pipeline.core.scheduler.base import AudioFragment  # noqa: E402
from voice_pipeline.metrics.trace import (  # noqa: E402
    MISSING,
    Span,
    exclusive_times,
    parse_spans,
    percentile,
    self_time_map,
    union_ms,
)
from voice_pipeline.providers import sherpa as R  # noqa: E402
from voice_pipeline.providers.factory import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    build_eou,
    build_providers,
)
from voice_pipeline.server.session import Session  # noqa: E402

DEFAULT_TEXT = "我想问一下明天上海天气怎么样，适合出门吗"

# 上行语音必须用 kokoro 合成：melo 喂给 ASR 识别结果为空（bench_ws.py:91 已记）
VOICE_TTS = "kokoro"


def percentile_of(values: Sequence[float], q: float) -> float:
    return percentile(list(values), q)


def _fmt(values: Sequence[float]) -> str:
    if not values:
        return "—"
    return (
        f"n={len(values)} min={min(values):.0f} "
        f"p50={statistics.median(values):.0f} max={max(values):.0f}"
    )


def build_utterance(text: str, sample_rate: int = 16000) -> np.ndarray:
    """合成「用户说的话」并重采样到上行采样率。"""
    tts = R.build_tts_engine(VOICE_TTS, num_threads=4)
    samples, sr = tts.generate(text)
    return soxr.resample(samples, sr, sample_rate)


async def run_turn(
    pcm: Any,
    *,
    tts_engine: str | None,
    llm: str,
    llm_model: str | None = None,
    gating: GatingMode,
    tail_silence_ms: int,
    prewarm: bool = True,
    energy_gate: bool = True,
    max_turn_s: float = 30.0,
) -> dict[str, Any]:
    """喂一轮：上行 → EOU 判停 → run_turn → 并行 drain outbox（模拟 pump+sender）。"""
    # 每轮取一份新 provider：ASR 的 stream 是会话状态必须新；LLM/TTS 由 factory 缓存
    providers = build_providers(
        "real",
        llm_engine=llm,
        llm_model=llm_model or DEFAULT_LLM_MODEL,
        tts_engine=tts_engine or "melo",
    )
    session = Session(
        mode="real",
        asr=providers.asr,
        llm=providers.llm,
        tts=providers.tts,
        eou=build_eou(gating),
        prewarm_tts=prewarm,
        energy_gate=energy_gate,
    )

    frame = 3200  # 100ms @16k
    parts = [R.pcm16_bytes(pcm[i : i + frame]) for i in range(0, len(pcm), frame)]
    silence = R.pcm16_bytes(np.zeros(int(16000 * tail_silence_ms / 1000), dtype=np.float32))
    feed = [*parts, silence]

    turn_task: asyncio.Task[None] | None = None
    sent_bytes = 0
    eou_wait_ms = -1.0

    async def drain() -> None:
        """transport 的替身：把 outbox 取干，并在首片"落网线"时通知 Session。

        真实链路里这一步由 app.py 的 pump + sender 完成；这里同步做是为了让
        `tx.first_audio`（outbox → 下发）这一跳在进程内也能量出来。
        """
        nonlocal sent_bytes
        while True:
            item = await session.outbox.get()
            session.outbox.task_done()
            if isinstance(item, AudioFragment):
                session.note_first_audio_send()  # 真正下发的那一刻
                sent_bytes += len(item.data)
            if turn_task is not None and turn_task.done() and session.outbox.empty():
                return

    t_feed = time.perf_counter()
    for i, chunk in enumerate(feed):  # noqa: B007 - i 是判停时喂到第几帧，feeder 要用
        await session.on_audio(chunk)
        if turn_task is None and session.pending_end():
            eou_at = (time.perf_counter() - t_feed) * 1000
            eou_wait_ms = eou_at
            turn_task = asyncio.create_task(session.run_turn())
            break
    drain_task = asyncio.create_task(drain())

    # **判停后继续按 100ms 节奏喂静音帧**——麦克风不会因为 AI 开口就停，barge-in
    # 也要求上行持续。这一路与下行抢同一个事件循环与同一批 CPU 核，正是要量出来的
    # 竞争（真实客户端永远在喂；停下喂等于把竞争藏起来）。
    async def keep_feeding() -> None:
        """判停后继续按 100ms 节奏喂**新造的**静音帧，直到本轮结束。

        用新造的帧而不是 feed 的剩余部分：EOU 往往正好落在尾部静音里，剩余可能为空，
        那样量出来的"上行竞争 0ms"是假阴性——客户端麦克风根本不关。
        """
        silence_100ms = R.pcm16_bytes(np.zeros(1600, dtype=np.float32))
        for _ in range(300):  # 30s 上限，远长于任何一轮
            if turn_task is not None and turn_task.done():
                return
            await session.on_audio(silence_100ms)
            await asyncio.sleep(0.1)

    feeder = asyncio.create_task(keep_feeding()) if turn_task is not None else None
    if turn_task is None:
        # 一直没判停：把剩余喂完再强制收尾（保守档 + 长尾句可能出现）
        for chunk in feed[1:]:
            await session.on_audio(chunk)
        turn_task = asyncio.create_task(session.run_turn())
    await asyncio.wait_for(turn_task, timeout=max_turn_s)
    await asyncio.wait_for(drain_task, timeout=5.0)
    if feeder is not None:
        feeder.cancel()
    wall_ms = (time.perf_counter() - t_feed) * 1000

    bill = session.metrics.bill()
    tracer = session.metrics.tracer
    spans = tracer.to_list() if tracer else []
    llm_name = type(providers.llm).__name__
    pool = providers.tts.pool_stats() if hasattr(providers.tts, "pool_stats") else {}
    return {
        "bill": bill,
        "spans": spans,
        "wall_ms": wall_ms,
        "eou_wait_ms": eou_wait_ms,
        "sent_bytes": sent_bytes,
        "llm": llm_name,
        "llm_model": getattr(providers.llm, "model", ""),
        "user_text": bill.get("user_text", ""),
        "pool": pool,
    }


def _window(spans: list[Span]) -> Span | None:
    """口径B 窗口 span：**按名字找**，不按 parent 找——它现在挂在 turn 根下面。"""
    for s in spans:
        if s.name.endswith("caliber_b"):
            return s
    return None


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把多轮 span 汇总：分格**自有时间** + 残差分解 + 上行竞争。

    为什么必须算自有时间：真实链路是嵌套的——`llm.request` 里包着 `tts.synth`
    （serial 在 LLM 循环体内 await 合成）。直接拿子 span 时长相加会重复计数：
    第一轮实测 llm.request 2406 + tts.synth 1918 = 4324ms > 口径B 2406ms。
    只有 `Σ自有 + 未归因 == 总长` 的表才能拿来做优化决策。
    """
    per_turn: list[dict[str, Any]] = []
    for row in rows:
        spans, _ = parse_spans(row["spans"])
        root = _window(spans)
        if root is None or root.dur_ms < 0:
            per_turn.append({"ok": False})
            continue
        lo, hi = root.start_ms, root.end_ms
        own = exclusive_times(spans, lo, hi)
        by_id = {s.span_id: s for s in spans}
        covered = union_ms([(s.start_ms, s.end_ms) for s in spans if s.dur_ms >= 0], lo, hi)
        # 同循环竞争者：窗口内**上行帧**与 EOU 的自有时间（它们把下行往后推）
        uplink_names = {"uplink.frame", "asr.push_audio", "eou.update"}
        uplink = sum(v for sid, v in own.items() if by_id[sid].name in uplink_names)
        stages = dict(self_time_map(spans, lo, hi))
        # 身份校验：Σ自有 == 覆盖并集（每毫秒只归一个 span，差 1ms 以上就是口径漏了东西）
        sum_own = sum(own.values())
        per_turn.append(
            {
                "ok": True,
                "total_ms": root.dur_ms,
                "covered_ms": covered,
                "unattributed_ms": root.dur_ms - covered,
                "sum_self_ms": sum_own,
                "identity_error_ms": covered - sum_own,
                "uplink_ms": uplink,
                "true_gap_ms": max(0.0, root.dur_ms - covered - uplink),
                "stages": stages,
                "inclusive": {s.name: s.dur_ms for s in spans if s.parent_id == root.span_id},
                # 能量门（P1）：这一轮多少上行帧被拦在 ASR 之外（span 上打 gated=1）
                "gated_frames": sum(
                    1 for s in spans if s.name == "uplink.frame" and s.attrs.get("gated") == 1
                ),
            }
        )
    ok = [t for t in per_turn if t["ok"]]
    if not ok:
        return {"ok": 0, "total": len(per_turn), "turns": per_turn, "stages": {}}
    stage_names: list[str] = []
    for t in ok:
        for name in t["stages"]:
            if name not in stage_names:
                stage_names.append(name)
    return {
        "ok": len(ok),
        "total": len(per_turn),
        "n": len(ok),
        "total_ms": [t["total_ms"] for t in ok],
        "unattributed_ms": [t["unattributed_ms"] for t in ok],
        "uplink_ms": [t["uplink_ms"] for t in ok],
        "true_gap_ms": [t["true_gap_ms"] for t in ok],
        "identity_error_ms": [t["identity_error_ms"] for t in ok],
        "gated_frames": [t["gated_frames"] for t in ok],
        "stages": {
            name: [t["stages"][name] for t in ok if name in t["stages"]] for name in stage_names
        },
        "inclusive": {
            name: [t["inclusive"][name] for t in ok if name in t["inclusive"]]
            for name in sorted({n for t in ok for n in t["inclusive"]})
        },
        "turns": per_turn,
    }


STAGE_NOTES = {
    "asr.finalize": "utterance_end → transcript（过去混在 LLM 那一格里）",
    "llm.ttft": "首 token：模型 RTT + prefill（= headers + prefill）",
    "llm.headers": "发包→响应头：网络 + 路由（共享连接，TLS 已摊薄）",
    "llm.prefill": "响应头→首 content：服务端排队 + prefill",
    "tts.connect": "call 提交→WS on_open：建连 + 认证（P0.5 后热连接只剩一个往返）",
    "tts.first_byte": "on_open→首个音频分片：服务端合成首片",
    "llm.first_sentence": "首 token → 首句末标点（分句攒句）",
    "llm.request": "整段生成，扣掉嵌套后剩下的「LLM 自己」的份额",
    "tts.synth": "合成整句（本地 melo/kokoro 无增量流，首片=整句）",
    "tts.first_chunk": "合成 → 第一个可播分片",
    "tx.first_audio": "outbox 入队 → 真正下发（传输层那一跳）",
    "spec.miss": "投机未命中，走串行兜底（pipelined）",
}


def print_report(agg: dict[str, Any], mode: str, engine: str, llm: str) -> None:
    print(f"\n=== span 分格（mode={mode} tts={engine} llm={llm}）===")
    if not agg.get("ok"):
        print("没有任何完整轮次（span 根缺失或未闭合）")
        return
    totals = agg["total_ms"]
    p50_total = statistics.median(totals)
    print(f"口径B 根 span：{_fmt(totals)}")
    covered = [t - u for t, u in zip(agg["total_ms"], agg["unattributed_ms"], strict=True)]
    print(f"覆盖（有 span 盖住的部分）：{_fmt(covered)}")
    print(f"未归因残差：{_fmt(agg['unattributed_ms'])}")
    print(f"  ├─ 其中上行/EOU 竞争：{_fmt(agg['uplink_ms'])}")
    print(f"  └─ 真·未埋到：{_fmt(agg['true_gap_ms'])}")
    ident = [abs(v) for v in agg["identity_error_ms"]]
    print(f"恒等式校验 |Σ自有−覆盖|：max={max(ident):.2f}ms（>1ms 说明分格口径漏了东西）")
    print()
    print("| 段（span，自有时间） | p50 | max | 占口径B | 含嵌套 | 说明 |")
    print("|---|---|---|---|---|---|")
    uplink_rows = {"turn", "uplink.frame", "asr.push_audio", "eou.update"}
    for name, vals in agg["stages"].items():
        if not vals or name == "turn.caliber_b" or name in uplink_rows:
            continue  # 根/上行系单独报，列出来只是噪声
        p50 = statistics.median(vals)
        inc = agg["inclusive"].get(name) or []
        inc_txt = f"{statistics.median(inc):.0f}ms" if inc else "—"
        print(
            f"| {name} | {p50:.0f}ms | {max(vals):.0f}ms | "
            f"{p50 / p50_total * 100 if p50_total > 0 else 0:.0f}% | {inc_txt} | "
            f"{STAGE_NOTES.get(name, '')} |"
        )
    up_p50 = statistics.median(agg["uplink_ms"])
    if up_p50 > 0.5:
        print(
            f"| 上行帧 CPU（窗口内） | {up_p50:.0f}ms | {max(agg['uplink_ms']):.0f}ms | "
            f"{up_p50 / p50_total * 100 if p50_total > 0 else 0:.0f}% | — |"
            " 与下行抢同一个事件循环 |"
        )
    gated = agg.get("gated_frames") or []
    if gated:
        print(
            f"| 能量门拦截帧（P1） | p50 {statistics.median(gated):.0f} 帧 | max {max(gated)} 帧 |"
            " — | — | 判停后的静音帧不进 ASR；EOU 仍收全部帧 |"
        )
    gap_p50 = statistics.median(agg["true_gap_ms"])
    print(
        f"| 真·未归因 | {gap_p50:.0f}ms | {max(agg['true_gap_ms']):.0f}ms | "
        f"{gap_p50 / p50_total * 100 if p50_total > 0 else 0:.0f}% | — | 没人埋到的等待 |"
    )


def print_tree(rows: list[dict[str, Any]]) -> None:
    """打一棵完整树：父子关系、时长、属性。根 = 口径B 窗口。"""
    spans, _ = parse_spans(rows[0]["spans"])
    kids: dict[str, list[Span]] = {}
    for s in spans:
        kids.setdefault(s.parent_id, []).append(s)
    print("\n=== span 树（第一轮，相对会话钟 ms）===")

    def walk(span: Span, depth: int) -> None:
        dur = f"{span.dur_ms:7.1f}" if span.dur_ms >= 0 else "  未闭合"
        attrs = " ".join(f"{k}={v}" for k, v in span.attrs.items())
        print("  " * depth + f"├─ {span.name:<20} {dur}  {attrs}")
        for k in sorted(kids.get(span.span_id, []), key=lambda s: s.start_ms):
            walk(k, depth + 1)

    roots = [s for s in spans if not s.parent_id]
    for r in sorted(roots, key=lambda s: s.start_ms):
        walk(r, 0)


async def main() -> int:
    ap = argparse.ArgumentParser(description="链路追踪探针：span 树 + 残差")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--tts", default=None, help="被测 TTS：melo/kokoro/dashscope（默认 melo）")
    ap.add_argument("--llm", default="auto", choices=["auto", "qwen", "sim"])
    ap.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"LLM 模型（默认 {DEFAULT_LLM_MODEL}；换模型 A/B 用它，清单见 DashScope /models）",
    )
    ap.add_argument(
        "--gating",
        default="conservative",
        choices=["conservative", "semantic", "aggressive", "neural"],
    )
    ap.add_argument("--silence-ms", type=int, default=300, help="判停前追加的静音")
    ap.add_argument("--tree", action="store_true", help="打印第一轮完整 span 树")
    ap.add_argument(
        "--no-prewarm",
        action="store_true",
        help="对照臂：关掉 TTS 连接预热（池仍会跨句复用，量的就是预热那一笔）",
    )
    ap.add_argument(
        "--no-energy-gate",
        action="store_true",
        help="对照臂：关掉 ASR 前置能量门（静音帧照常推 ASR，量的就是循环竞争那一格）",
    )
    ap.add_argument("--out", default=None, help="逐轮 span JSONL 输出路径")
    args = ap.parse_args()

    gating = GatingMode(args.gating)
    machine = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"机器: {machine}")
    print(
        f"ASR=transducer  TTS={args.tts or 'melo'}  LLM={args.llm}({args.llm_model})  "
        f"判停={args.gating}  n={args.repeat}  "
        f"预热={'关' if args.no_prewarm else '开'}  "
        f"能量门={'关' if args.no_energy_gate else '开'}"
    )

    pcm = build_utterance(args.text)
    print(f"上行语音 {len(pcm) / 16000 * 1000:.0f}ms（{args.text}）")

    rows: list[dict[str, Any]] = []
    for i in range(args.repeat):
        row = await run_turn(
            pcm,
            tts_engine=args.tts,
            llm=args.llm,
            llm_model=args.llm_model,
            gating=gating,
            tail_silence_ms=args.silence_ms,
            prewarm=not args.no_prewarm,
            energy_gate=not args.no_energy_gate,
        )
        row["repeat"] = i + 1
        rows.append(row)
        b = row["bill"]
        cal = b.get("caliber_b_ms", MISSING)
        print(
            f"轮 {i + 1}: 口径B={cal:.0f}ms "
            f"判停等待={row['eou_wait_ms']:.0f}ms 下行音频={row['sent_bytes']}B "
            f"LLM={row['llm']}"
        )

    agg = analyse(rows)
    print_report(agg, "real", args.tts or "melo", args.llm)
    if rows and rows[-1].get("pool"):
        print(f"TTS 连接池（末轮快照）：{rows[-1]['pool']}")
    if args.tree:
        print_tree(rows)

    out = (
        Path(args.out) if args.out else Path(f"/tmp/traces-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    )
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"\nspan 明细已写入 {out}（每轮一行，含 spans 字段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
