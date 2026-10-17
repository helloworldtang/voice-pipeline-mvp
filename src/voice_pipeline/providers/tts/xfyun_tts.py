"""讯飞在线语音合成（v2 流式 WS）——本机实测首包 p50 **137ms**。

为什么加它：DashScope 的 TTS 首包是 344~421ms，占口径B 的 42%，是 300ms 的唯一门槛。
讯飞这家的**服务端首包 p50 137ms**（n=8，min 132 / max 146），加握手 38ms ≈ 175ms，
比 DashScope 省 ~210ms（探针：`scripts/probe_vendor_tts.py`）。

三条实测约束（都写进实现了）：

1. **只支持 MP3 或 16k/8k 的 raw**：`aue=raw + auf=audio/L16;rate=24000` 会返回
   `code=0 success` 但**音频长度为 0**（静默失败）。所以取 16k PCM，再用 `soxr` **流式**
   重采样到链路的 24k——按块重采样会有边界爆音，必须用带状态的 `ResampleStream`。
2. **连接不能复用**：同一 WS 上发第二个任务会超时（服务端一个任务后即收尾），
   所以每条句子各自建连（握手 38ms），没有"热连接池"可省。
3. **凭证走环境变量**：`XFYUN_TTS_APPID` / `XFYUN_TTS_API_KEY` / `XFYUN_TTS_API_SECRET`；
   缺了**响亮失败**，不静默退回本地/替身（否则账单会假装一切正常）。

接口与 DashScope 那个同构：`tts_streams = True`，`synth(sentence)` 吐 24k PCM 片。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from email.utils import format_datetime
from urllib.parse import urlencode

import numpy as np
import soxr
import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

HOST = "tts-api.xfyun.cn"
PATH = "/v2/tts"
APP_ID_ENV = "XFYUN_TTS_APPID"
KEY_ENV = "XFYUN_TTS_API_KEY"
SECRET_ENV = "XFYUN_TTS_API_SECRET"
DEFAULT_VCN = "xiaoyan"
PREWARM_TTL_S = 10.0  # 预建连最多等这么久；超了就重连（也防服务端静默断链）
SRC_RATE = 16000  # 实测：24k raw 静默返回空音频，16k 正常
OUT_RATE = 24000
FIELD_KEY = "api_key"


def _decode(audio_b64: str) -> np.ndarray:
    """base64 的 16k/16bit 小端 PCM → float32 [-1, 1]。"""
    pcm16 = base64.b64decode(audio_b64)
    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


def _encode(samples: np.ndarray) -> bytes:
    """float32 → 24k/16bit 小端 PCM 字节，交给下行分片器。"""
    pcm: bytes = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return pcm


async def _safe_close(ws: ClientConnection) -> None:
    """关闭且吞掉异常：清理路径不该把一次打断变成一次报错。"""
    try:
        await ws.close()
    except Exception:
        logger.debug("[xfyun] 关连接失败（忽略）", exc_info=True)


class XfyunTts:
    """讯飞 v2 流式 TTS。与 DashScopeTts 同契约（`tts_streams = True`）。"""

    tts_streams = True

    def __init__(self, vcn: str = DEFAULT_VCN, *, speed: int = 50, timeout_s: float = 20.0) -> None:
        self.vcn = vcn
        self.speed = speed
        self.timeout_s = timeout_s
        self._app_id = os.environ.get(APP_ID_ENV, "").strip()
        self._api_key = os.environ.get(KEY_ENV, "").strip()
        self._api_secret = os.environ.get(SECRET_ENV, "").strip()
        # 预建连槽位：(建立时刻, 连接)。讯飞一个连接只能做一个任务，所以这是「一次性」的，
        # 判停时先握上、等首句文本到了直接用 —— 把 114~129ms 的握手藏进 finalize+LLM 窗口。
        self._pending: tuple[float, ClientConnection] | None = None
        self._opening = False
        self._prewarm_on = os.environ.get("VOICE_XFYUN_PREWARM", "1").strip() != "0"
        missing = [
            name
            for name, value in (
                (APP_ID_ENV, self._app_id),
                (KEY_ENV, self._api_key),
                (SECRET_ENV, self._api_secret),
            )
            if not value
        ]
        if missing:
            msg = (
                f"缺环境变量 {', '.join(missing)}：讯飞 TTS 需要它们"
                "（建议放 ~/.voice-pilot/vendors.env）"
            )
            raise RuntimeError(msg)

    # ---- 鉴权 ----------------------------------------------------------------

    def _url(self) -> str:
        """按讯飞 v2 规则签名：HMAC-SHA256 over `host / date / GET /v2/tts HTTP/1.1`。"""
        date = format_datetime(datetime.now(timezone.utc), usegmt=True)
        origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
        signature = base64.b64encode(
            hmac.new(self._api_secret.encode(), origin.encode(), hashlib.sha256).digest()
        ).decode()
        fields = [
            FIELD_KEY + '="' + self._api_key + '"',
            'algorithm="hmac-sha256"',
            'headers="host date request-line"',
            'signature="' + signature + '"',
        ]
        authorization_origin = ", ".join(fields)
        authorization = base64.b64encode(authorization_origin.encode()).decode()
        return f"wss://{HOST}{PATH}?" + urlencode(
            {"authorization": authorization, "date": date, "host": HOST}
        )

    def _payload(self, sentence: str) -> str:
        return json.dumps(
            {
                "common": {"app_id": self._app_id},
                "business": {
                    "aue": "raw",
                    "auf": f"audio/L16;rate={SRC_RATE}",  # 24k 会静默返回空音频（实测）
                    "vcn": self.vcn,
                    "tte": "UTF8",
                    "speed": self.speed,
                },
                "data": {"status": 2, "text": base64.b64encode(sentence.encode()).decode()},
            }
        )

    # ---- 预建连（判停即握手） --------------------------------------------------

    def _fresh(self, pending: tuple[float, ClientConnection]) -> bool:
        return (time.monotonic() - pending[0]) < PREWARM_TTL_S

    def _usable(self, ws: ClientConnection) -> bool:
        # websockets 两者都要看：close_code 非空 = 已收关闭帧；close_sent = 我们已经关它了
        return getattr(ws, "close_code", None) is None and not getattr(ws, "close_sent", False)

    async def _take(self) -> ClientConnection:
        """取连接：有新鲜预建连就用它，否则现连。"""
        pending, self._pending = self._pending, None
        if pending is not None:
            if self._fresh(pending) and self._usable(pending[1]):
                return pending[1]
            await _safe_close(pending[1])
        return await websockets.connect(self._url(), max_size=None, open_timeout=self.timeout_s)

    def prewarm(self) -> None:
        """EOU 触发时调用（非阻塞）：把握手提前跑掉。

        讯飞一个连接只服务一个任务，所以这里不是"池"，是"一次性预建连"：
        握手 114~129ms（uvicorn 里）整段藏进 finalize + LLM 首句的窗口。
        """
        if not self._prewarm_on or self._opening:
            return
        if (
            self._pending is not None
            and self._fresh(self._pending)
            and self._usable(self._pending[1])
        ):
            return
        self._opening = True
        task = asyncio.create_task(self._preopen())
        task.add_done_callback(lambda _: setattr(self, "_opening", False))

    async def _preopen(self) -> None:
        try:
            ws = await websockets.connect(self._url(), max_size=None, open_timeout=self.timeout_s)
        except Exception:  # 预建连失败不影响正确性：synth 会自己重连
            logger.debug("[xfyun] 预建连失败，synth 时重连", exc_info=True)
            return
        stale, self._pending = self._pending, (time.monotonic(), ws)
        if stale is not None:
            await _safe_close(stale[1])

    async def aclose(self) -> None:
        """关闭预建连（会话结束时用；没被用掉不留半开连接）。"""
        pending, self._pending = self._pending, None
        if pending is not None:
            await _safe_close(pending[1])

    # ---- 合成 ----------------------------------------------------------------

    async def synth(self, sentence: str) -> AsyncIterator[bytes]:
        """边收边转：每个 WS 音频包 → 16k PCM → 流式重采样 → 24k PCM 片。"""
        # 流式重采样器（带状态）：按块独立重采样会在每 200ms 边界爆音
        resampler = soxr.ResampleStream(SRC_RATE, OUT_RATE, 1, dtype="float32")
        _t0 = time.perf_counter()
        ws = await self._take()
        try:
            _conn_ms = (time.perf_counter() - _t0) * 1000
            if os.environ.get("XFYUN_DEBUG"):
                logger.warning("[xfyun] connect=%.0fms", _conn_ms)
            await ws.send(self._payload(sentence))
            _t_send = time.perf_counter()
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
                event = json.loads(raw)
                if event.get("code", 0) != 0:
                    msg = f"讯飞 TTS 失败 {event.get('code')}: {event.get('message')}"
                    raise RuntimeError(msg)
                data = event.get("data") or {}
                audio = data.get("audio")
                if audio:
                    if os.environ.get("XFYUN_DEBUG") and not locals().get("_logged"):
                        _logged = True
                        logger.warning(
                            "[xfyun] 首帧=%.0fms (connect %.0f + 收帧 %.0f)",
                            (time.perf_counter() - _t0) * 1000,
                            _conn_ms,
                            (time.perf_counter() - _t_send) * 1000,
                        )
                    out = resampler.resample_chunk(_decode(audio), last=False)
                    if out.size:
                        yield _encode(out)
                if data.get("status") == 2:
                    tail = resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                    if tail.size:
                        yield _encode(tail)
                    return

        finally:
            # 讯飞一条连接只做一个任务，用完就关，不留半开连接
            await _safe_close(ws)

    def synth_blocking(self, sentence: str) -> bytes:
        """协议兼容（整句）：走同一路径收齐。流水线本身用 `synth()`。"""

        async def _collect() -> bytes:
            return b"".join([chunk async for chunk in self.synth(sentence)])

        return asyncio.run(_collect())


def build(**kwargs: object) -> XfyunTts:
    return XfyunTts(**kwargs)  # type: ignore[arg-type]
