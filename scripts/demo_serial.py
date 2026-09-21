"""一条命令跑一轮 serial 链路，打印 waterfall 时间线与双口径账单。

M1 的验收证据之一。real 模式（M2）接入后，同一脚本换成真实 provider 即为实验数据。

    uv run python scripts/demo_serial.py
"""

from __future__ import annotations

import asyncio

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.scheduler.base import AudioFragment
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts

TRANSCRIPT = "上海今天多少度"
REPLY = "今天晴，气温 22 度。适合出门。"
AUDIO_FRAME = b"\x00" * 3200  # 100ms @16k PCM16


async def main() -> None:
    clock = SessionClock()
    metrics = TurnMetrics(clock=clock)
    scheduler = SerialScheduler(
        asr=SimAsr(transcript=TRANSCRIPT),
        llm=SimLlm(reply=REPLY),
        tts=SimTts(chunk_per_sentence=2),
        metrics=metrics,
        splitter=SentenceSplitter(),
        chunker=Chunker(),
    )

    await scheduler.on_audio(AUDIO_FRAME)

    print("== 事件序列 ==")
    audio_count = 0
    audio_bytes = 0
    async for item in scheduler.on_end_of_utterance():
        if isinstance(item, AudioFragment):
            audio_count += 1
            audio_bytes += len(item.data)
            continue
        extra = f" metrics={item.metrics}" if hasattr(item, "metrics") else ""
        print(f"  {item.type}{extra}")

    print(f"\n== 下行音频 ==\n  {audio_count} 分片 / {audio_bytes} bytes（每片 9600）")

    print("\n== waterfall（相对会话钟, ms）==")
    previous = 0.0
    for name, at in metrics.waterfall():
        print(f"  {name:<22} {at:>9.3f}   (+{at - previous:.3f})")
        previous = at

    print("\n== 账单 ==")
    for key, value in metrics.bill().items():
        print(f"  {key:<22} {value}")


if __name__ == "__main__":
    asyncio.run(main())
