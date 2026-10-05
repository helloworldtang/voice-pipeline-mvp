"""FastAPI 会话服务：上行音频帧 + JSON 控制帧，下行音频分片 + JSON 事件帧。

两种模式（session.begin.mode）：
- `echo`：上行音频原样回发，用于量 WS 链路本身的序列化开销（M0 回归）
- `serial` / `pipelined`：真实链路 ASR → LLM → 分句 → TTS（M1 起）

M3 并发模型：上行接收与下行发送是两条并发路径。
- 接收循环：audio 帧 → session.on_audio（内含 barge-in L1 检测）；控制帧 → 状态变更。
- 一轮的生成（run_turn）跑在后台任务里，把 TurnItem 投进 session.outbox。
- pump 协程从 outbox 取出发送，独占 websocket.send，避免与 AudioStop 竞态。

这样 AI 播放期间客户端仍在上行麦克风帧，打断检测能持续运行——barge-in 才触发得了。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from voice_pipeline.config import Settings
from voice_pipeline.core.scheduler.base import AudioFragment, TurnItem

from .protocol import (
    AudioChunk,
    AudioStop,
    ErrorEvent,
    Interrupt,
    ProtocolError,
    SessionBegin,
    SessionClosed,
    SessionEnd,
    SessionReady,
    dump_event,
    parse_client_frame,
)
from .queues import OutboundQueue
from .session import Session

# 浏览器客户端默认位置：仓根 web/（src 布局运行时可达；Settings.web_dir 可覆盖）
logger = logging.getLogger(__name__)

DEFAULT_WEB_DIR = Path(__file__).resolve().parents[3] / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="voice-pipeline", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session = Session()
        await websocket.send_text(
            dump_event(SessionReady(session_id=session.session_id, trace_id=session.trace_id))
        )

        # 下行两级：outbox →（pump 分类）→ tx →（sender 发送）→ wire。
        # 分成两个协程的原因：sender 可能阻塞在慢客户端的 send 上，此时 pump
        # 仍要继续分类——audio.stop 才能进杀停道、越过积压音频先行（queues.py）。
        tx = OutboundQueue(max_audio_frames=cfg.max_audio_frames)

        async def pump(outbox: asyncio.Queue[TurnItem | AudioStop]) -> None:
            """分类协程：TurnItem → 杀停道 / 保序事件道 / 有界音频道。"""
            try:
                while True:
                    item = await outbox.get()
                    if isinstance(item, AudioFragment):
                        tx.put_audio(
                            dump_event(AudioChunk(seq=item.seq, bytes_len=len(item.data))),
                            item.data,
                        )
                    elif isinstance(item, AudioStop):
                        tx.put_kill(dump_event(item))
                    else:
                        tx.put_event(dump_event(item))
                    outbox.task_done()
            except asyncio.CancelledError:
                return

        async def sender() -> None:
            """唯一常驻发送协程：从 tx 取发送单元下发（音频对原子发送）。"""
            try:
                while True:
                    entry = await tx.get()
                    await websocket.send_text(entry.text)
                    if entry.data is not None:
                        await websocket.send_bytes(entry.data)
                    # join 的完成计数在**发送落地后**才递减：join 返回即代表客户端
                    # 收得到，调用方随后 cancel 本协程不会掐掉在飞的发送
                    # （TestClient 下实测：出队即计数的版本，session.closed 被
                    # cancel 掐在 send 里，测试端永远收不到）。
                    tx.task_done()
            except asyncio.CancelledError:
                return

        pump_task = asyncio.create_task(pump(session.outbox))
        sender_task = asyncio.create_task(sender())
        turn_task: asyncio.Task[None] | None = None
        try:
            while True:
                message: MutableMapping[str, Any] = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                audio = message.get("bytes")
                if isinstance(audio, bytes):
                    for fragment in await session.on_audio(audio):
                        await websocket.send_bytes(fragment.data)
                    # 播放期间的新帧已在 on_audio 内做 barge-in 检测并往 outbox 投 AudioStop；
                    # real 模式下 EOU 判停触发即自动起一轮（不依赖客户端 session.end）。
                    if not session._turn_running and session.pending_end():
                        turn_task = asyncio.create_task(session.run_turn())
                    continue

                text = message.get("text")
                if not isinstance(text, str):
                    continue

                try:
                    frame = parse_client_frame(text)
                except ProtocolError as exc:
                    await websocket.send_text(
                        dump_event(ErrorEvent(code=exc.code, message=exc.message))
                    )
                    continue

                if isinstance(frame, SessionBegin):
                    # 静态 token 鉴权：cfg.ws_token 设了才启用（不设=完全开放，
                    # 本机开发/bench/测试零配置）。挂在 session.begin 而不是连接层，
                    # 因为首个 session.ready 先于任何客户端帧发出，鉴权早于此就得重设计握手。
                    if cfg.ws_token is not None and not secrets.compare_digest(
                        frame.token or "", cfg.ws_token
                    ):
                        await websocket.send_text(
                            dump_event(
                                ErrorEvent(code="unauthorized", message="bad or missing token")
                            )
                        )
                        await websocket.close(code=1008)
                        return
                    # 换模式即重建会话（含新的 outbox 与 pump）。
                    # gating / grace_ms 必须透传：它们是 300ms 实验的两个旋钮，
                    # 协议里有、构造时丢掉，就等于旋钮是画上去的。
                    session = Session(
                        mode=frame.mode,
                        gating=frame.gating,
                        grace_ms=frame.grace_ms,
                        # 账单落盘走环境变量 VOICE_EVENTS_JSONL（Session 默认
                        # 从 env 读，见 metrics/jsonl.py）：这是运维开关，
                        # 不是每次会话的实验旋钮，所以不上 wire 协议。
                    )
                    pump_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await pump_task
                    pump_task = asyncio.create_task(pump(session.outbox))
                    await websocket.send_text(
                        dump_event(
                            SessionReady(session_id=session.session_id, trace_id=session.trace_id)
                        )
                    )
                    continue

                if isinstance(frame, SessionEnd):
                    # 显式结束会话：确保本轮跑完（sim 模式靠这个驱动一轮；real 模式若已在跑
                    # 则不重复），然后回 session.closed 并关闭连接。real 模式持续对话时不靠
                    # 这个，靠 EOU 自动判停。
                    if not session._turn_running:
                        turn_task = asyncio.create_task(session.run_turn())
                    if turn_task is not None:
                        with contextlib.suppress(asyncio.CancelledError):
                            await turn_task
                    # 关键：run_turn 把本轮所有 TurnItem 投进了 outbox，但 pump/sender
                    # 是独立协程，此刻可能还没发完（尤其是 turn.end）。必须先等两级
                    # 队列排空再收尾——否则 break 后的 finally 会 cancel 两个协程，
                    # 剩余事件（含 turn.end）永远到不了客户端，测试/客户端就卡在等 turn.end。
                    # closed 走保序事件道（排在残余音频之后），不抢道。
                    await session.outbox.join()
                    tx.put_event(
                        dump_event(
                            SessionClosed(
                                session_id=session.session_id,
                                frames_echoed=session.frames_echoed,
                            )
                        )
                    )
                    await tx.join()
                    break

                if isinstance(frame, Interrupt) and not session.barge_in(source="client"):
                    # 客户端本地打断是「快路径」，经常在服务端轮次已结束后由播放
                    # 尾音再触发一次（2026-10-04 验收：两轮打断各跟一条误报）。
                    # 此时无事可停，静默丢弃即可——报错只会污染验收界面。
                    logger.debug("interrupt with no active turn: dropped")
        except WebSocketDisconnect:
            return
        finally:
            pump_task.cancel()
            sender_task.cancel()
            if turn_task is not None:
                turn_task.cancel()

    # 浏览器客户端（web/），必须挂在所有显式路由之后：Starlette 按注册顺序匹配，
    # mount("/") 先注册会吞掉 /healthz 与 /ws。目录不存在不挂载——纯 WS 部署
    # （bench / 只跑 API）不因缺前端文件而起不来。
    web_dir = Path(cfg.web_dir) if cfg.web_dir else DEFAULT_WEB_DIR
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app


app = create_app()
