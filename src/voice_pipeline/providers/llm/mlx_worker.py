"""MLX 推理子进程：模型加载与生成都在这里，父进程只做流式转发。

为什么要拆进程：本仓实测同一份代码，TTS 的 `tts.first_chunk` 在 MLX 线程同时跑时
从 216ms 涨到 411ms；LLM 自己的微基准 TTFT 46ms，进链路变 66~141ms。
两者都是 **GIL / 线程调度** 的代价，不是算力不够。拆进程把这块整段拿掉。

协议（stdin/stdout 各一行 JSON，stdout 只走协议，日志一律 stderr）：

    子进程启动 → {"event": "ready", "model": "..."}
    父进程请求 → {"id": 1, "messages": [...], "max_tokens": 160}
    子进程逐 token → {"id": 1, "text": "今"}
    子进程结束 → {"id": 1, "event": "done"}
    父进程取消 → {"op": "cancel", "id": 1}   （子进程在两 token 之间查一次）
"""

from __future__ import annotations

import json
import logging
import select
import sys
import time

logger = logging.getLogger(__name__)


def _stdin_pending() -> bool:
    """stdin 上有没有待读的控制命令（非阻塞）。"""
    try:
        return bool(select.select([sys.stdin], [], [], 0)[0])
    except (OSError, ValueError):
        return False


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        from mlx_lm import load, stream_generate
    except ImportError:
        # 协议里回一条 error，父进程才能报"缺依赖"而不是干等 180s 超时
        msg = "子进程缺 mlx-lm（用 `uv run --with mlx-lm ...` 启动，或 `uv sync --extra local`）"
        print(json.dumps({"event": "error", "message": msg}, ensure_ascii=False), flush=True)
        raise SystemExit(3) from None

    from .mlx_llm import _apply_template, _resolve_model

    model_id = _resolve_model(None)
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    # 预热一次：Metal 首次编译实测 +900ms，不能算在第一轮头上
    try:
        stream_generate(model, tokenizer, prompt="你好", max_tokens=1)
    except Exception:  # 预热带不来正确性，失败也要能继续服务
        logger.warning("预热失败（不影响正确性）", exc_info=True)
    load_ms = round((time.perf_counter() - t0) * 1000)
    print(json.dumps({"event": "ready", "model": model_id, "load_ms": load_ms}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "cancel":
            continue  # 没有正在跑的任务，忽略

        rid = req["id"]
        t_recv = time.perf_counter()
        messages = req["messages"]
        max_tokens = int(req.get("max_tokens", 160))
        prompt = _apply_template(tokenizer, messages)
        cancelled = False
        first_ms = -1.0
        for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
            if first_ms < 0:
                first_ms = (time.perf_counter() - t_recv) * 1000
            print(json.dumps({"id": rid, "text": resp.text}, ensure_ascii=False), flush=True)
            if _stdin_pending():
                cmd = sys.stdin.readline().strip()
                if cmd:
                    try:
                        parsed = json.loads(cmd)
                    except json.JSONDecodeError:
                        parsed = {}
                    if parsed.get("op") == "cancel" and parsed.get("id") == rid:
                        cancelled = True
                        break
        print(
            json.dumps(
                {
                    "id": rid,
                    "event": "cancelled" if cancelled else "done",
                    "first_token_ms": round(first_ms),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
