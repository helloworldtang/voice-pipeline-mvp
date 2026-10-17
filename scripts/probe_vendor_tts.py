#!/usr/bin/env python3
"""厂商 TTS 首包探针（讯飞 v2 在线语音合成）：值不值得接，先量首包。

背景：300ms 口径B 的预算里，TTS 首包只能占 ~150ms；当前云端 DashScope 是 344~421ms。
换厂商的唯一判据就是这一格——**首包 p50 ≤150ms 才值得接**，否则如实记否证。

凭证（**不进仓、不打印**）：从环境变量读，建议放 `~/.voice-pilot/vendors.env`
    export XFYUN_TTS_APPID=...
    export XFYUN_TTS_API_KEY=...
    export XFYUN_TTS_API_SECRET=...

用法：
    zsh -c 'source ~/.voice-pilot/vendors.env &&
        uv run python scripts/probe_vendor_tts.py --repeat 5'

测什么：从「发 run-task 请求」到「第一个音频包」的毫秒数（n 次，报 min/p50/p90/max）。
不测什么：音质、整句时长——那些等首包达标再谈。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import statistics
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from urllib.parse import urlencode

import websockets

HOST = "tts-api.xfyun.cn"
PATH = "/v2/tts"
APP_ID_ENV = "XFYUN_TTS_APPID"
KEY_ENV = "XFYUN_TTS_API_KEY"
SECRET_ENV = "XFYUN_TTS_API_SECRET"
DEFAULT_VCN = "xiaoyan"  # 讯飞默认发音人；需要别的音色改 --vcn
DEFAULT_TEXT = "今天上海多云，气温18到25度。"


def _auth_url(app_id: str, api_secret: str) -> str:
    """按讯飞 v2 的签名规则拼 URL（HMAC-SHA256 over host/date/request-line）。"""
    date = format_datetime(datetime.now(timezone.utc), usegmt=True)
    origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode(), origin.encode(), hashlib.sha256).digest()
    ).decode()
    authorization_origin = (
        f'api_key="{app_id}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode()).decode()
    return f"wss://{HOST}{PATH}?" + urlencode(
        {"authorization": authorization, "date": date, "host": HOST}
    )


async def one_run(
    url: str, app_id: str, text: str, vcn: str, timeout_s: float
) -> dict[str, float | str]:
    """一次合成：返回首包毫秒数（失败返回 error）。"""
    payload = json.dumps(
        {
            "common": {"app_id": app_id},
            "business": {"aue": "raw", "auf": "audio/L16;rate=24000", "vcn": vcn, "tte": "UTF8"},
            "data": {"status": 2, "text": base64.b64encode(text.encode()).decode()},
        }
    )
    t0 = time.perf_counter()
    try:
        async with websockets.connect(url, max_size=None, open_timeout=timeout_s) as ws:
            await ws.send(payload)
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                ev = json.loads(msg)
                if ev.get("code", 0) != 0:
                    return {"first_ms": -1.0, "error": f"{ev.get('code')}: {ev.get('message')}"}
                audio = (ev.get("data") or {}).get("audio")
                if audio:
                    return {"first_ms": (time.perf_counter() - t0) * 1000.0, "error": ""}
                if (ev.get("data") or {}).get("status") == 2:
                    return {"first_ms": -1.0, "error": "completed without audio"}
    except Exception as exc:  # noqa: BLE001 - 探针要如实记错
        return {"first_ms": -1.0, "error": f"{type(exc).__name__}: {exc}"}


def _pct(values: list[float], q: float) -> float:
    if not values:
        return -1.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


async def main_async(args: argparse.Namespace) -> None:
    app_id = os.environ.get(APP_ID_ENV, "").strip()
    api_key = os.environ.get(KEY_ENV, "").strip()
    api_secret = os.environ.get(SECRET_ENV, "").strip()
    missing = [
        n for n, v in ((APP_ID_ENV, app_id), (KEY_ENV, api_key), (SECRET_ENV, api_secret)) if not v
    ]
    if missing:
        raise SystemExit(
            f"缺环境变量 {', '.join(missing)}。建议放 ~/.voice-pilot/vendors.env 后 source 再跑。"
        )

    url = _auth_url(api_key, api_secret)
    runs: list[dict[str, float | str]] = []
    for i in range(args.repeat):
        runs.append(await one_run(url, app_id, args.text, args.vcn, args.timeout))
        got = float(runs[-1]["first_ms"])
        print(f"  run{i + 1}: {got if got < 0 else round(got)}ms {runs[-1]['error']}", flush=True)

    firsts = [float(r["first_ms"]) for r in runs if float(r["first_ms"]) >= 0]
    errs = [str(r["error"]) for r in runs if r["error"]]
    print(f"\n[讯飞 v2 TTS] 文本 {args.text!r} vcn={args.vcn} n={args.repeat}")
    if not firsts:
        print(f"✗ 全部失败：{errs[:3]}")
        return
    print(
        f"首包：min={min(firsts):.0f} p50={statistics.median(firsts):.0f} "
        f"p90={_pct(firsts, 0.9):.0f} max={max(firsts):.0f}ms（失败 {len(errs)}）"
    )
    verdict = "✅ 达标（≤150ms）" if statistics.median(firsts) <= 150 else "❌ 未达标（>150ms）"
    print(f"判定：{verdict}——门槛是 300ms 口径B 里 TTS 只能占 ~150ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="厂商 TTS 首包探针（讯飞 v2）")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--vcn", default=DEFAULT_VCN, help="发音人（默认 xiaoyan）")
    ap.add_argument("--timeout", type=float, default=15.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
