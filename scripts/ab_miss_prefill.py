"""miss 轮 prefill A/B：投机作废后 fallback 的首 token 延迟，修复前后对照。

场景（不依赖真人/麦克风/讯飞）：SimAsr 注入错误 partial → 投机在飞时 EOU →
finalize 给出修正文本 → text_differs 判 miss → fallback 用正确全文重新生成。
LLM 是**真 MLX 模型**，量的就是 miss 轮 `llm.prefill`（headers→first_content）。

两臂对照：
- 修复前（52b07fa）：作废的投机生成继续跑到 EOS，与 fallback 抢同一个模型；
- 修复后（当前 HEAD）：miss 即 cancel + stop 旗标让线程停手，fallback 独占模型。

用法（模型默认 1.5B，与调研 §4.1 的基线同配置）：
    uv run --with mlx-lm python scripts/ab_miss_prefill.py --repeats 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.mlx_llm import MlxLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import TurnEnd

TRANSCRIPT = "上海今天天气怎么样"
WRONG = "上海今天天色"  # 会被 finalize 修正 → text_differs → miss
FRAME = b"\x00" * 960


async def one_turn(llm: MlxLlm, lead_ms: float) -> dict[str, object]:
    """跑一轮注定 miss 的对话，返回 fallback 那次 astream 的分格耗时(ms)。"""
    asr = SimAsr(TRANSCRIPT, partials=[WRONG] * 5)
    sched = PipelinedScheduler(
        asr=asr,
        llm=llm,
        tts=SimTts(cpu_ms=0.0),
        metrics=TurnMetrics(clock=SessionClock()),
        grace_ms=0.0,  # 没就绪立刻判 miss，不等
    )
    await sched.on_audio(FRAME)  # 起投机（基于错误前缀）
    await asyncio.sleep(lead_ms / 1000.0)  # 让投机生成到一半再 EOU
    async for item in sched.on_end_of_utterance():
        if isinstance(item, TurnEnd):
            assert item.metrics["spec_hit"] == 0.0, item.metrics
    ph = llm.last_phases
    wf = ph.get("worker_first_token")  # 修复前没有这个键 → -1（宁空不填 0）
    return {
        "prefill_ms": round((ph["first_content"] - ph["headers"]) * 1000, 1),
        "worker_ttft_ms": round((wf - ph["headers"]) * 1000, 1) if wf else -1.0,
        "wakeup_ms": round((ph["first_content"] - wf) * 1000, 1) if wf else -1.0,
        "miss_reason": sched.spec_last_reason,
    }


async def main_async(repeats: int, lead_ms: float, out: Path) -> None:
    llm = MlxLlm()
    llm.preload()  # load + Metal 编译不能算进第一轮
    rows: list[dict[str, object]] = []
    for _i in range(repeats):
        row = await one_turn(llm, lead_ms)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        # 修复前的臂里作废生成会继续跑：等它彻底结束，别污染下一轮
        await asyncio.sleep(1.5)
    pre = [float(r["prefill_ms"]) for r in rows]
    summary = {
        "n": repeats,
        "lead_ms": lead_ms,
        "prefill_p50": round(statistics.median(pre), 1),
        "prefill_min": round(min(pre), 1),
        "prefill_max": round(max(pre), 1),
    }
    for key in ("worker_ttft_ms", "wakeup_ms"):
        vals = [float(r[key]) for r in rows if float(r[key]) >= 0]
        if vals:
            summary[key + "_p50"] = round(statistics.median(vals), 1)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--lead-ms", type=float, default=300.0, help="投机起跑后多久 EOU")
    ap.add_argument("--out", type=Path, default=Path("/tmp/vp-ac/ab-miss-prefill.jsonl"))
    args = ap.parse_args()
    asyncio.run(main_async(args.repeats, args.lead_ms, args.out))


if __name__ == "__main__":
    main()
