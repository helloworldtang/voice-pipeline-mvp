"""下行发送队列：杀停帧优先、其余严格保序、音频有界丢旧。

铁律来源（技术方案 §2.6）：打断信号若排在音频数据后面，用户按下打断到真正生效
之间会隔着一整段积压，体验上就是「叫不停」。

但**只有杀停类帧**（`audio.stop`）允许跳队。`turn.end` / `session.closed` 若也走
优先道，会越过尚未发完的音频先到客户端——账单先于音频、收流方提前退出收流循环
（bench_ws 正是收到 turn.end 就 break），那是另一种形式的「叫不停」。初版
`DualPriorityQueue` 把所有控制帧都放进优先道，接线分析时发现这个顺序倒置，
遂重构为本形态（2026-10-03，差距表「有界队列未接线」的兑现）。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Entry:
    """一个发送单元：要么单文本帧，要么 audio.chunk 通知 + 二进制数据成对。

    音频的 JSON 通知与二进制必须**原子地**同进同出——背压丢弃若只丢一半，
    客户端的「收到 audio.chunk 后跟一帧二进制」解析就永久错位。
    """

    text: str
    data: bytes | None = None  # 非 None = 音频对：先发 text 再发 data
    # 杀停帧标记（audio.stop）。发送协程据此回调 Session 记 `tx.stop` span——
    # 打断那一跳「检测 → 停播帧落网线」的长度是「叫不停」的直接证据，
    # 而只有真正发送出去的那一刻才量得到（入队时刻量的是队列左侧）。
    kill: bool = False


class OutboundQueue:
    """双车道下行队列：杀停道无界优先；主道严格保序、音频有界丢最旧并计数。"""

    def __init__(self, max_audio_frames: int = 200) -> None:
        self._kill = deque[Entry]()  # audio.stop 等「晚到不如早到」的帧
        self._main = deque[Entry]()  # 其余一切，含音频
        self._audio_in_main = 0
        self._max_audio = max_audio_frames
        self._event = asyncio.Event()
        self._unfinished = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self.dropped_audio = 0

    # ---- 生产（多协程安全：同一事件循环内无 await，天然原子） -----------------

    def put_kill(self, text: str) -> None:
        """杀停帧：插队到一切积压之前。仅限 audio.stop 语义的帧。"""
        self._put(self._kill, Entry(text, kill=True))

    def put_event(self, text: str) -> None:
        """保序事件（turn.end / session.closed / 转写流）。"""
        self._put(self._main, Entry(text))

    def put_audio(self, notice: str, data: bytes) -> None:
        """音频对（audio.chunk JSON + PCM）。主道积压超限时丢**最旧的音频对**
        （越过中间的事件单元），并计数——背压可见。"""
        if self._audio_in_main >= self._max_audio:
            # 定位主道里**最旧的一个音频对**原地删除——夹在它前面的事件单元
            # 保序保留（turn.end 被背压丢掉比丢音频严重得多）。
            for i, entry in enumerate(self._main):
                if entry.data is not None:
                    del self._main[i]
                    self._audio_in_main -= 1
                    self.dropped_audio += 1
                    self._unfinished -= 1
                    break
            logger.warning(
                "下行音频积压超限（≥%s 帧），丢弃最旧音频对（累计 %s）",
                self._max_audio,
                self.dropped_audio,
            )
        self._put(self._main, Entry(notice, data))
        self._audio_in_main += 1

    def _put(self, lane: deque[Entry], entry: Entry) -> None:
        lane.append(entry)
        self._unfinished += 1
        self._event.set()

    # ---- 消费（唯一发送协程调用） ---------------------------------------------

    async def get(self) -> Entry:
        """先杀停后主道；都空则等待。

        **出队不递减完成数**（那是 task_done 的职责），但音频占用计数在这里减——
        它度量的是车道占用而非未完成发送。真人验收实测踩坑：只增不减的话，
        累计下发 200 帧音频（约 40s 内容）后每个新音频都误报「积压超限」并
        可能真丢队列里的新音频（2026-10-04 验收日志 100+ 条误报、误丢 7 对）。
        """
        while True:
            if self._kill:
                return self._kill.popleft()
            if self._main:
                entry = self._main.popleft()
                if entry.data is not None:
                    self._audio_in_main -= 1
                return entry
            self._event.clear()
            await self._event.wait()

    def task_done(self) -> None:
        """一个发送单元**落地后**调用。join 以此为准——出队即计数会让调用方在
        发送仍在飞时就退出收尾，随后的 cancel 把在飞的发送掐死在半路。"""
        self._unfinished -= 1
        if self._unfinished == 0:
            self._drained.set()

    async def join(self) -> None:
        """等所有已入队单元发送完毕（与 asyncio.Queue.join 同语义）。"""
        while self._unfinished > 0:
            self._drained.clear()
            await self._drained.wait()

    def qsize(self) -> tuple[int, int]:
        return len(self._kill), len(self._main)
