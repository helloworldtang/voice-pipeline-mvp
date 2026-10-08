"""多轮真链路验证：sherpa reset 修复在真引擎上是否成立。

sim 替身（DeadWithoutResetAsr）只能证明调度器**调了** reset；reset 在 sherpa
真实流上是否真的让第 2 轮转写复活，只有真模型能回答。做法：一条 WS 连接上
先后喂两段不同的合成语音，各自等 EOU 自动判停出账，断言两轮的 asr.final
都非空且互不相同。

用法：uv run python scripts/probe_multi_turn.py
前置：models/ 已就位（download_models.sh）、DASHSCOPE_API_KEY 已配、
      服务已在 ws://127.0.0.1:8766/ws 监听（脚本会自己拉起 uvicorn）。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from typing import Any

import numpy as np
import websockets

sys.path.insert(0, "scripts")  # noqa: E402 - 复用 bench 的合成与喂帧工具
from bench_ws import BLOCK, pcm16_bytes, synthesize  # type: ignore[import-not-found]

from voice_pipeline.providers.factory import build_providers
from voice_pipeline.server.protocol import SessionBegin  # type: ignore[import-not-found]

PORT = 8766
TURN_TIMEOUT_S = 20.0
UTTERANCES = ["今天上海天气怎么样", "那明天呢"]  # 第二句语义上承接第一句


async def wait_ready(timeout_s: float = 30.0) -> None:
    """等 healthz。bench_ws 的同名函数绑死它自己的 8765 端口，这里自己轮询。"""
    import httpx

    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            try:
                if (await c.get(f"http://127.0.0.1:{PORT}/healthz", timeout=2)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.3)
    msg = f"服务端 {PORT} 端口在 {timeout_s}s 内未就绪"
    raise RuntimeError(msg)


async def feed_speech(ws: Any, pcm: np.ndarray) -> None:
    for i in range(0, len(pcm), BLOCK):
        await ws.send(pcm16_bytes(pcm[i : i + BLOCK]))
        await asyncio.sleep(BLOCK / 16000)


async def drain_turn(ws: Any) -> dict[str, Any]:
    """持续喂静音直到服务端 EOU 判停并出完本轮账单，返回 (asr_final, bill)。

    二进制帧（下行音频片）直接跳过——本探针只关心 JSON 事件。
    """
    deadline = time.monotonic() + TURN_TIMEOUT_S
    silence = b"\x00" * (BLOCK * 2)
    asr_final: str | None = None
    while time.monotonic() < deadline:
        try:
            payload = await asyncio.wait_for(ws.recv(), timeout=0.1)
        except TimeoutError:
            await ws.send(silence)
            continue
        if isinstance(payload, bytes):
            continue
        event = json.loads(payload)
        if event.get("type") == "asr.final":
            asr_final = event["text"]
        if event.get("type") == "turn.end":
            return {"asr_final": asr_final, "bill": event["metrics"]}
    msg = f"一轮 {TURN_TIMEOUT_S}s 未出账单：EOU 没判停或链路卡死"
    raise RuntimeError(msg)


async def run_probe() -> None:
    # 上行测试语音必须 kokoro（melo 合成的语音 ASR 识别为空——bench 里踩过）
    providers = build_providers("real", tts_engine="kokoro")
    pcms = [await synthesize(u, providers.tts) for u in UTTERANCES]

    async with websockets.connect(
        f"ws://127.0.0.1:{PORT}/ws", max_size=None, open_timeout=30
    ) as ws:
        await ws.recv()  # 首个默认 session.ready
        await ws.send(SessionBegin(mode="real-serial", gating="conservative").model_dump_json())
        ready = json.loads(await ws.recv())
        assert ready["type"] == "session.ready", ready

        results = []
        for pcm in pcms:
            await feed_speech(ws, pcm)
            results.append(await drain_turn(ws))

        for utterance, result in zip(UTTERANCES, results, strict=True):
            bill = result["bill"]
            print(f"问: {utterance}")
            print(f"  asr.final: {result['asr_final']!r}")
            print(f"  口径B: {bill.get('caliber_b_ms')}ms  判停: {bill.get('eou_latency_ms')}ms")

        first, second = (r["asr_final"] for r in results)
        assert first, "第 1 轮转写为空"
        assert second, "第 2 轮转写为空 —— ASR reset 修复没生效（音频进了死流）"
        assert first != second, "两轮转写相同 —— 可疑：像旧流残留而不是新流识别"
        print()
        print("✓ 两轮转写都非空且互不相同：reset 修复在真链路上成立")


async def main() -> None:
    server = subprocess.Popen(  # noqa: S603 - 固定命令，无外部输入
        [
            sys.executable,
            "-m",
            "uvicorn",
            "voice_pipeline.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
    )
    try:
        await wait_ready()
        await run_probe()
    finally:
        server.terminate()


if __name__ == "__main__":
    asyncio.run(main())
