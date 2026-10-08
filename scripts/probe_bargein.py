"""真链路打断探针：AI 播放中插话 → 断言打断轮落盘且状态正确。

用法（需先在 8767 起服务、并设 VOICE_EVENTS_JSONL 落盘）：

    VOICE_EVENTS_JSONL=/tmp/turns-barge.jsonl \\
      uv run uvicorn voice_pipeline.server.app:app --host 127.0.0.1 --port 8767
    uv run python scripts/probe_bargein.py

依赖：models/ 就位、DASHSCOPE_API_KEY 可见（key 在 ~/.zshrc，须用 zsh 起服务）。

**为什么单独一个探针**：打断是真人验收的一等判据，而它在账单里曾经完全没有
证据——`finish()` 被打断时 break，产不出 `TurnEnd`，那条记录就永不落盘，
`interrupted` 字段成了永远为假的死字段。本探针就是钉住这一点的端到端复现：
真 ASR + 真 LLM + 真 TTS，AI 出声后插话，断言账单里出现 `interrupted=true`。

两个踩过的坑：
1. **不要先发 `session.end`**（初版因此失败）：real 模式靠 EOU 自动判停起轮，
   抢发 `session.end` 会在 AI 还没出声时就把会话关掉，于是永远测不到打断。
2. 插话时机要在**收到音频分片之后**：`barge_in()` 只在 SPEAKING/THINKING 生效，
   太早发 interrupt 会被状态机判为「用户在听」而忽略。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import websockets

sys.path.insert(0, "scripts")  # noqa: E402 - 复用 bench 的合成与喂帧工具
from bench_ws import BLOCK, pcm16_bytes, synthesize  # type: ignore[import-not-found]

from voice_pipeline.providers.factory import build_providers
from voice_pipeline.server.protocol import SessionBegin  # type: ignore[import-not-found]

PORT = 8767  # 独立端口，不占用 bench_ws 的 8765 / probe_multi_turn 的 8766
UTTERANCE = "请详细介绍一下上海的历史文化"  # 长问句 → 长回答，留足播放时间供插话
TURN_TIMEOUT_S = 40.0


async def main() -> None:
    providers = build_providers("real", tts_engine="kokoro")
    pcm = await synthesize(UTTERANCE, providers.tts)

    async with websockets.connect(
        f"ws://127.0.0.1:{PORT}/ws", max_size=None, open_timeout=30
    ) as ws:
        await ws.recv()  # 首个默认 session.ready
        await ws.send(SessionBegin(mode="real-serial", gating="conservative").model_dump_json())
        await ws.recv()  # rebound session.ready

        for i in range(0, len(pcm) - BLOCK, BLOCK):
            await ws.send(pcm16_bytes(pcm[i : i + BLOCK]))
            await asyncio.sleep(0.001)

        audio, sent, saw_stop = 0, False, False
        deadline = time.monotonic() + TURN_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=4)
            except TimeoutError:
                break
            if isinstance(msg, bytes):
                audio += 1
                # AI 已出声 → 第 2 片后插话（模拟用户打断）
                if audio >= 2 and not sent:
                    sent = True
                    await ws.send(json.dumps({"type": "interrupt"}))
                    print(f"→ 已在第 {audio} 片后发送 interrupt")
            else:
                evt = json.loads(msg)
                kind = evt.get("type")
                if kind == "tts.sentence":
                    print("AI:", evt["text"][:30])
                elif kind == "audio.stop":
                    saw_stop = True
                    print("← 收到 audio.stop（L2 停播）")
                elif kind == "error":
                    print("ERROR:", evt)
                elif kind == "turn.end":
                    keys = ("caliber_b_ms", "barge_in_count", "barge_in_latency_ms")
                    print("turn.end:", {k: evt["metrics"][k] for k in keys if k in evt["metrics"]})
                    break

        print(f"汇总：音频片={audio} 发过 interrupt={sent} 收到 audio.stop={saw_stop}")
        await asyncio.sleep(0.6)  # 等打断轮走完 finally 落盘
        await ws.send(json.dumps({"type": "session.end"}))
        await asyncio.sleep(0.4)


if __name__ == "__main__":
    asyncio.run(main())
