"""MLX 小模型 A/B：TTFT / 首句 / 吞吐，同一 system prompt、同一问句。

为什么要量而不是照抄别人的数：网上公开的端侧数字多是 MNN/CPU 后端（首 token ~500ms），
与 MLX/Metal 不是一个量级（本仓 MLX 微基准 TTFT 45ms），不能直接搬。

用法：
    HF_ENDPOINT=https://hf-mirror.com uv run --with mlx-lm python scripts/bench_mlx_models.py \
        --models mlx-community/Qwen3.5-2B-4bit mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from voice_pipeline.providers.llm.mlx_llm import DEFAULT_SYSTEM, build_messages

QUESTION = "今天上海天气怎么样"
PUNCT = "。！？，；"


def first_sentence_ms(tokens: list[str], stamps: list[float]) -> float:
    buf = ""
    for tok, ts in zip(tokens, stamps, strict=False):
        buf += tok
        if any(p in tok for p in PUNCT) and len(buf.strip()) >= 2:
            return ts * 1000
    return -1.0


def bench(model_id: str, repeats: int, question: str = QUESTION) -> dict[str, object]:
    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_id)
    msgs = build_messages(DEFAULT_SYSTEM, [], question)
    try:
        # Qwen3.5 默认开「思考链」：对语音助手是灾难（先说 200 token 的推理过程）。
        # 模板不支持这个参数就退回默认。
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    ttfts: list[float] = []
    firsts: list[float] = []
    texts: list[str] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ttft = -1.0
        first = -1.0
        buf = ""
        for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=60):
            now = (time.perf_counter() - t0) * 1000
            if ttft < 0:
                ttft = now
            buf += resp.text
            if first < 0 and any(p in resp.text for p in PUNCT) and len(buf.strip()) >= 2:
                first = now
        ttfts.append(ttft)
        firsts.append(first)
        texts.append(buf)

    return {
        "model": model_id,
        "repeats": repeats,
        "ttft_ms_p50": round(statistics.median(ttfts)),
        "first_sentence_ms_p50": round(statistics.median(firsts)) if min(firsts) >= 0 else -1,
        "chars": len(texts[-1]),
        "answer": texts[-1][:60].replace("\n", " "),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("/tmp/vp-ac/mlx-ab.jsonl"))
    ap.add_argument("--question", default=QUESTION)
    args = ap.parse_args()

    rows = []
    for m in args.models:
        try:
            row = bench(m, args.repeats, args.question)
        except Exception as exc:  # 下载失败/模型不存在都要留痕，不静默跳过
            row = {"model": m, "error": f"{type(exc).__name__}: {exc}"[:200]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
