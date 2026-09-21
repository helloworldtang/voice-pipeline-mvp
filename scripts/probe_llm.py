"""LLM 选型探针：量「首 token」与「首完整句」，本机实测，可复跑。

**为什么要量首完整句而不是首 token**：TTS 要攒到句末标点才开工（`SentenceSplitter`），
首 token 早到没用。语音链路的 LLM 那一格 = 首完整句的时刻。

**为什么要分开记 reasoning_content 与 content**：推理模型先吐一整条思考链，
期间 content 恒为 null。只统计「首任意 token」会把思考链的开始当成可用输出，自欺。

用法（不传 key 就只用环境变量里已有的那家）：

    uv run python scripts/probe_llm.py                       # 扫默认候选
    uv run python scripts/probe_llm.py --models qwen-flash,qwen-turbo
    uv run python scripts/probe_llm.py --runs 5 --prompt "讲个笑话"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

import httpx

TERMINATORS = "。！？；!?;\n"
SYSTEM = "你是语音助手。回答要短，一句话说完，不要 markdown，不要用列表。直接给结论。"

ENDPOINTS = {
    "qwen-": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "deepseek-": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}
DEFAULT_MODELS = ["qwen-flash", "qwen-turbo", "deepseek-flash"]


def resolve(model: str) -> tuple[str, str]:
    """模型名 → (base_url, api_key)。按名字前缀认家。"""
    for prefix, (base, env) in ENDPOINTS.items():
        if model.startswith(prefix):
            key = os.environ.get(env, "")
            if not key:
                msg = f"{model} 需要设置 {env}"
                raise SystemExit(msg)
            return base, key
    msg = f"不认得 {model} 属于哪家（前缀需为 qwen- / deepseek-）"
    raise SystemExit(msg)


def extra_for(model: str) -> dict[str, object]:
    """DeepSeek 现役模型默认开思考链，语音场景必须显式关掉。

    实测：只有 `thinking={"type":"disabled"}` 生效；`enable_thinking:False` 与
    `chat_template_kwargs` 两种写法无效，思考链照吐。
    """
    if model.startswith("deepseek-"):
        return {"thinking": {"type": "disabled"}}
    return {}


async def once(model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    base, key = resolve(model)
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    }
    body.update(extra_for(model))

    t0 = time.perf_counter()
    t_tok: float | None = None
    t_sent: float | None = None
    spoken: list[str] = []
    reason_len = 0
    async with (
        httpx.AsyncClient(timeout=90) as client,
        client.stream(
            "POST",
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
        ) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            payload = line.strip()
            if not payload.startswith("data:"):
                continue
            data = payload[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reason_len += len(delta["reasoning_content"])
                content = delta.get("content")
                if not content:
                    continue
                if t_tok is None:
                    t_tok = (time.perf_counter() - t0) * 1000
                spoken.append(content)
                if t_sent is None and any(c in TERMINATORS for c in content):
                    t_sent = (time.perf_counter() - t0) * 1000
    total = (time.perf_counter() - t0) * 1000
    return {
        "t_tok": t_tok,
        "t_sent": t_sent if t_sent is not None else t_tok,
        "total": total,
        "text": "".join(spoken).strip(),
        "reason_len": reason_len,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--prompt", default="我想问一下明天的天气怎么样")
    args = ap.parse_args()

    print(f"prompt={args.prompt!r}  runs={args.runs}（另有 1 次预热不计入）\n")
    print(f"{'模型':<22}{'首token':>9}{'首完整句':>10}{'总耗时':>9}{'思考链':>8}   首答")
    print("-" * 96)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        await once(model, args.prompt, args.max_tokens)  # 预热
        runs = [await once(model, args.prompt, args.max_tokens) for _ in range(args.runs)]
        toks = [r["t_tok"] for r in runs]
        sents = [r["t_sent"] for r in runs]
        tots = [r["total"] for r in runs]
        sd = statistics.pstdev(sents)
        no_content = sum(1 for t in toks if t is None)
        flag = f"  ⚠️ {no_content}/{args.runs} 轮无可播 token" if no_content else ""
        print(
            f"{model:<22}{statistics.median(toks):>8.0f}ms{statistics.median(sents):>9.0f}ms"
            f"{statistics.median(tots):>8.0f}ms{runs[0]['reason_len']:>7}字"
            f"   {str(runs[0]['text'])[:24]!r}{flag}"
        )
        print(f"{'':<22}首完整句波动 ±{sd:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
