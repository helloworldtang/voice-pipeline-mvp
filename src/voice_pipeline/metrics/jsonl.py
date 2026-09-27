"""每轮账单的 JSONL 落盘（差距表「JSONL 事件流落盘未做」的兑现）。

为什么需要它：账单此前只活在内存里，脚本一打印就没了——事后无法回答「那一轮
到底慢在哪」。落盘后每完成一轮追加一行 JSON Lines，可直接 `jq` / `grep`，
真人验收与线上排查共用同一条证据链。

三条设计约束（都不是洁癖，各有代价）：
- **默认关闭**：`VOICE_EVENTS_JSONL` 未设时不碰文件系统，CI 与单测零副作用。
- **只写已完成轮次**：拿到 `turn.end` 账单才落。被打断的半截轮次没有完整账单，
  写进去就是假账——「听到的才是真的」这条纪律在账单上同样成立。
- **追加 + 锁**：单进程内串行写。多 worker 部署需换成集中式收集，故此处只保证
  单进程正确；`uvicorn` 默认单 worker 符合当前形态。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ENV_PATH = "VOICE_EVENTS_JSONL"


class TurnRecorder:
    """把每轮账单追加到 JSONL。`path` 为空即静默丢弃（默认行为）。"""

    def __init__(self, path: str = "") -> None:
        self.path = path
        self._lock = threading.Lock()
        if path:
            # 目录不存在就建：部署时容易忘，忘了的代价是第一轮账单静默丢失。
            # 建目录失败**不在这里抛**——构造 Session 的路径上抛异常等于整条
            # 通话链路起不来，而账单只是观测面。失败留给 record() 静默处理。
            try:
                Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                print(f"[metrics] JSONL 目录不可用（落盘将失败）：{exc}")

    @classmethod
    def from_env(cls) -> TurnRecorder:
        return cls(os.environ.get(ENV_PATH, ""))

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def record(
        self,
        *,
        session_id: str,
        trace_id: str,
        mode: str,
        turn: int,
        bill: Mapping[str, float],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """追加一轮账单。写盘失败不能拖垮会话——宁可丢一行日志，不能断一次通话。"""
        if not self.path:
            return
        record: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "session_id": session_id,
            "trace_id": trace_id,
            "mode": mode,
            "turn": turn,
            **bill,
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # pragma: no cover - 取决于本机文件系统
            # 不抛：账单是观测面，不该成为通话链路的单点故障。
            print(f"[metrics] JSONL 落盘失败（已忽略）：{exc}")
