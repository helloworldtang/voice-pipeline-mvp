"""双优先级队列：控制帧（打断/取消）永不被音频或文本积压阻塞。

铁律来源：技术方案 §2.6。打断信号若排在音频数据后面，用户按下打断到真正生效之间
会隔着一整段积压，体验上就是「叫不停」。
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

PRIO_CONTROL = 0
PRIO_AUDIO = 1


@dataclass(slots=True)
class Envelope:
    priority: int
    seq: int
    payload: bytes | str


class DualPriorityQueue:
    """控制帧无界优先；音频帧有界，超限丢弃最旧的分片并计数（背压可见）。"""

    def __init__(self, max_audio_frames: int = 200) -> None:
        self._control: deque[Envelope] = deque()
        self._audio: deque[Envelope] = deque()
        self._max_audio = max_audio_frames
        self._seq = 0
        self._event = asyncio.Event()
        self.dropped_audio = 0

    def put_control(self, payload: str) -> None:
        self._seq += 1
        self._control.append(Envelope(PRIO_CONTROL, self._seq, payload))
        self._event.set()

    def put_audio(self, payload: bytes) -> None:
        self._seq += 1
        if len(self._audio) >= self._max_audio:
            self._audio.popleft()
            self.dropped_audio += 1
        self._audio.append(Envelope(PRIO_AUDIO, self._seq, payload))
        self._event.set()

    async def get(self) -> Envelope:
        """先控制后音频；都空则等待。"""
        while True:
            if self._control:
                return self._control.popleft()
            if self._audio:
                return self._audio.popleft()
            self._event.clear()
            await self._event.wait()

    def qsize(self) -> tuple[int, int]:
        return len(self._control), len(self._audio)

    def empty(self) -> bool:
        return not self._control and not self._audio
