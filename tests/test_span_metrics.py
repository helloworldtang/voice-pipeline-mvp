"""每轮 span 拆解要能到页面：`run_turn` 在 `turn.end` 上挂 `span.<name>` 扁平键。

为什么单独一条：浏览器那张「拆解」表完全依赖这几个键。它们走 **metrics 的扁平命名**，
不是新协议字段——所以既不会静默丢（没有就会显示空表），也不会改掉账单口径
（记账仍用调度器给的 bill，spans 原树仍落 JSONL）。这条用例钉住三件事：

1. 键真的存在（页面才有东西画）；
2. `span.unattributed` 也在（残差是"还有没埋到"的证据，不能只报好听的）；
3. 各格之和不超过口径B（超了就是重复计数——自有时间的核心不变量）。
"""

from __future__ import annotations

import asyncio

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import TurnEnd
from voice_pipeline.server.session import Session


async def test_turn_end_carries_span_breakdown() -> None:
    session = Session(
        mode="serial",
        asr=SimAsr(transcript="上海今天多少度"),
        llm=SimLlm(reply="今天晴，气温 22 度。"),
        tts=SimTts(chunk_per_sentence=2),
    )
    await session.on_audio(b"\x00" * 3200)
    task = asyncio.create_task(session.run_turn())
    end: TurnEnd | None = None
    while end is None:
        item = await asyncio.wait_for(session.outbox.get(), timeout=5)
        if isinstance(item, TurnEnd):
            end = item
    await task

    keys = [k for k in end.metrics if k.startswith("span.")]
    assert keys, f"turn.end 里没有 span 拆解：{sorted(end.metrics)}"
    assert "span.unattributed" in end.metrics
    parts = sum(
        v for k, v in end.metrics.items() if k.startswith("span.") and k != "span.unattributed"
    )
    total = end.metrics["caliber_b_ms"]
    assert total >= 0
    assert parts <= total + 1.0, f"各格之和 {parts:.1f} 超过口径B {total:.1f}：重复计数了"
