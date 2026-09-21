#!/usr/bin/env python3
"""实时语音对话 live 客户端（M3：barge-in 的端到端验证载体）。

麦克风采集 16k PCM16 → WS 上行；下行 24k PCM16 → 声卡播放。播放期间本地检测
麦克风能量，用户开口即发 interrupt 帧触发服务端三层 barge-in。

⚠️ 回声（一期必须正视）：纯能量检测会把 AI 自己的回声当成「用户开口」疯狂打断。
一期依赖**入耳式耳机**的物理隔离（软件 AEC 列二期差距表），能量阈值因此要调高。
没有物理隔离就不要开这个客户端——它会自激。

依赖（不在主依赖树，独立装）：
    uv run pip install pyaudio websockets

运行：
    uv run python scripts/live_client.py --url ws://127.0.0.1:8000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys

try:
    import pyaudio
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyaudio：uv run pip install pyaudio")

try:
    import websockets
except ImportError:  # pragma: no cover
    sys.exit("缺少 websockets：uv run pip install websockets")

from voice_pipeline.core.eou import rms_of

MIC_RATE = 16000
PLAY_RATE = 24000
CHUNK = 512  # 每帧样本数 @16k ≈ 32ms


async def main(url: str, threshold: float, frames: int) -> None:
    mic_q: asyncio.Queue[bytes] = asyncio.Queue()
    play_q: queue.Queue[bytes] = queue.Queue()
    loop = asyncio.get_running_loop()

    pa = pyaudio.PyAudio()
    mic_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=lambda data, n, t, s: (  # 回调在 pa 线程，跨线程投进 asyncio 队列
            loop.call_soon_threadsafe(mic_q.put_nowait, data),
            pyaudio.paContinue,
        )[1],
    )
    play_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=PLAY_RATE,
        output=True,
        frames_per_buffer=1024,
        stream_callback=lambda data, n, t, s: _play_cb(play_q, n),
    )
    mic_stream.start_stream()
    play_stream.start_stream()

    async with websockets.connect(url) as ws:  # type: ignore[union-attr]
        recv_task = asyncio.create_task(_recv_loop(ws, play_q))
        send_task = asyncio.create_task(_send_loop(ws, mic_q, threshold, frames))
        try:
            await asyncio.gather(recv_task, send_task)
        finally:
            recv_task.cancel()
            send_task.cancel()


async def _recv_loop(ws: websockets.WebSocketClientProtocol, play_q: queue.Queue[bytes]) -> None:
    """下行：二进制帧写播放队列；audio.stop 事件清空队列（L2 停播）。"""
    async for message in ws:
        if isinstance(message, bytes):
            play_q.put(message)
        elif isinstance(message, str):
            try:
                evt = json.loads(message)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "audio.stop":
                # 丢弃已排队、尚未出声的块——AI 嘴停了耳朵里别还回响
                with play_q.mutex:
                    play_q.queue.clear()


async def _send_loop(
    ws: websockets.WebSocketClientProtocol,
    mic_q: asyncio.Queue[bytes],
    threshold: float,
    frames: int,
) -> None:
    """上行：麦克风帧发给服务端；连续高能量（用户开口）→ 发 interrupt 触发 barge-in。"""
    above = 0
    while True:
        chunk = await mic_q.get()
        await ws.send(chunk)
        if rms_of(chunk) >= threshold:
            above += 1
            if above >= frames:
                await ws.send(json.dumps({"type": "interrupt"}))
                above = 0  # 去抖后复位，避免每一帧都重发
        else:
            above = 0


def _play_cb(play_q: queue.Queue[bytes], frame_count: int) -> tuple[bytes, int]:
    need = frame_count * 2
    buf = bytearray()
    while len(buf) < need:
        try:
            buf += play_q.get_nowait()
        except queue.Empty:
            buf += b"\x00" * (need - len(buf))
            break
    return bytes(buf[:need]), pyaudio.paContinue


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="voice-pipeline live 客户端（含 barge-in 打断）")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="WS 服务端地址")
    p.add_argument("--threshold", type=float, default=0.05, help="开口能量阈值（RMS，满幅=1）")
    p.add_argument("--frames", type=int, default=3, help="连续多少帧超阈值才判开口（去抖）")
    return p.parse_args()


if __name__ == "__main__":
    import contextlib

    args = _parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(args.url, args.threshold, args.frames))
