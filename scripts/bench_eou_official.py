"""EOU 判停器的**流式**验收：判停延迟 vs 抢话率，跑在官方测试集上。

为什么不用自造 TTS 样本（踩过的坑，2026-09-22）：
  分开合成「前半句」时，melo 会给它单独配一个句末终顿韵律 —— 听起来就是「说完了」，
  于是神经模型判 complete 是**对的**，错的是我的真值。用它评出来的「口径 A/B 都只有
  33~40% 判别力」是假结论。判停评测必须用带真值标签的语料，本脚本用官方
  pipecat-ai/smart-turn-data-v3.1-test（标签 endpoint_bool）。

做法（流式，不是整段判定）：
  每条样本按 100ms 帧喂进判停器（与 Session 上行帧长一致），记录第一次判停的时刻。
    endpoint_bool=True（真值：说完了）→ 延迟 = 判停时刻 − 最后一个人声样本时刻
    endpoint_bool=False（真值：没说完）→ **任何判停都算抢话**
  真值结束时刻用能量法找（最后一个 RMS ≥ floor 的帧），不用文本对齐。

用法：
  uv run python scripts/bench_eou_official.py                      # 中文，全组合
  uv run python scripts/bench_eou_official.py --langs eng --sweep-silence 0,200
数据：`scripts/fetch_eou_dataset.py` 下载并解码到 data/eou-test/（439MB，不入仓）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_pipeline.core.eou import FusedEou, GatingMode  # noqa: E402
from voice_pipeline.providers import sherpa as R  # noqa: E402
from voice_pipeline.providers.eou.smart_turn import SmartTurnEou  # noqa: E402

SR = 16000
FRAME_MS = 100
FRAME = int(SR * FRAME_MS / 1000)
ENERGY_FLOOR = 0.005  # 与 eou.py 一致：低于它算静音


def speech_end_ms(audio: np.ndarray) -> float:
    """真值：最后一个有人声的样本位置（ms）。"""
    win = int(SR * 0.005)  # 5ms
    n = len(audio) // win * win
    if n == 0:
        return 0.0
    rms = np.sqrt((audio[:n].reshape(-1, win) ** 2).mean(axis=1))
    voiced = np.where(rms >= ENERGY_FLOOR)[0]
    return float((voiced[-1] + 1) * win) / SR * 1000 if voiced.size else 0.0


def run_clip(audio: np.ndarray, make_eou) -> tuple[float | None, float]:
    """返回 (判停时刻 ms, 推理总耗时 ms)；未判停返回 (None, _)。"""
    eou = make_eou()
    decided: float | None = None
    for i in range(0, len(audio), FRAME):
        chunk = R.pcm16_bytes(audio[i : i + FRAME])
        eou.push_audio(chunk)
        if eou.should_end():
            decided = (i + FRAME) / SR * 1000
            break
    inf = 0.0
    if isinstance(eou, FusedEou) and eou.neural is not None:
        inf = float(getattr(eou.neural, "inference_ms", 0.0))
    return decided, inf


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:.0f}%)" if d else "—"


def evaluate(rows: list[dict], name: str) -> dict:
    """三类结果分开算，混在一起会把「抢话」和「慢」互相抵消。

    pos(真值说完了) 在 speech_end 之后判停  → 正常，延迟 = 判停 − speech_end
    pos 在 speech_end 之前判停             → 提前抢话（句中就打断了）
    pos 全程未判停                          → 漏判（干等）
    neg(真值没说完) 任何判停                → 抢话
    """
    lat: list[float] = []
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    neg_pre = sum(1 for r in neg if r["decided_ms"] is not None)
    pos_early = sum(
        1 for r in pos if r["decided_ms"] is not None and r["decided_ms"] < r["speech_end_ms"]
    )
    miss = sum(1 for r in pos if r["decided_ms"] is None)
    for r in pos:
        if r["decided_ms"] is not None and r["decided_ms"] >= r["speech_end_ms"]:
            lat.append(r["decided_ms"] - r["speech_end_ms"])
    inf = sum(r["inference_ms"] for r in rows)
    total_pre = neg_pre + pos_early
    return {
        "name": name,
        "n": len(rows),
        "抢话率": pct(total_pre, len(rows)),
        "抢话_n": total_pre,
        "其中没说完": neg_pre,
        "其中提前": pos_early,
        "漏判率": pct(miss, len(pos)),
        "延迟p50": statistics.median(lat) if lat else float("nan"),
        "延迟p90": (sorted(lat)[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else float("nan")),
        "推理ms/条": inf / len(rows) if rows else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/eou-test")
    ap.add_argument("--langs", default="zho")
    ap.add_argument("--limit", type=int, default=0, help="每种语言最多取多少条（0=全部）")
    ap.add_argument("--sweep-silence", default="0,100,200,300")
    ap.add_argument("--sweep-threshold", default="0.5")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    meta = json.loads((root / "meta.json").read_text())
    clips: dict[str, list[tuple[np.ndarray, bool]]] = {}
    for lang in args.langs.split(","):
        lang = lang.strip()
        sel = [m for m in meta if m["lang"] == lang]
        if args.limit:
            sel = sel[: args.limit]
        items = []
        t0 = time.perf_counter()
        for m in sel:
            a = np.load(root / lang / f"{m['i']:04d}.npy")
            items.append((a.astype(np.float32), bool(m["label"])))
        clips[lang] = items
        print(f"{lang}: {len(items)} 条（加载 {(time.perf_counter() - t0) * 1000:.0f}ms）")

    base_model = SmartTurnEou()  # 权重只加载一次
    print(f"神经判停器就绪：{base_model.model_path.name}\n")

    def make_conservative():
        return FusedEou(gating=GatingMode.CONSERVATIVE)

    results: list[dict] = []
    for lang, items in clips.items():
        # 基线
        rows = []
        for a, label in items:
            d, _ = run_clip(a, make_conservative)
            rows.append(
                {
                    "label": label,
                    "decided_ms": d,
                    "speech_end_ms": speech_end_ms(a),
                    "inference_ms": 0.0,
                }
            )
        results.append({"lang": lang, **evaluate(rows, "conservative(500ms)")})

        for thr_s in args.sweep_threshold.split(","):
            thr = float(thr_s)
            for sil_s in args.sweep_silence.split(","):
                sil = float(sil_s)

                def make_neural(thr: float = thr, sil: float = sil) -> FusedEou:
                    return FusedEou(
                        gating=GatingMode.NEURAL,
                        neural=SmartTurnEou(session=base_model._session, threshold=thr),
                        min_silence_neural_ms=sil,
                    )

                rows = []
                for a, label in items:
                    d, inf = run_clip(a, make_neural)
                    rows.append(
                        {
                            "label": label,
                            "decided_ms": d,
                            "speech_end_ms": speech_end_ms(a),
                            "inference_ms": inf,
                        }
                    )
                r = evaluate(rows, f"neural(thr={thr}, sil={sil:.0f}ms)")
                results.append({"lang": lang, **r})
                print(
                    f"  {lang} {r['name']:<28} 抢话 {r['抢话率']:<12} 漏判 {r['漏判率']:<10}"
                    f" 延迟p50 {r['延迟p50']:>7.0f}ms p90 {r['延迟p90']:>7.0f}ms"
                )

    print("\n" + "=" * 108)
    print("判停延迟 vs 抢话率（流式，100ms 帧，官方 smart-turn-data-v3.1-test）")
    print("=" * 108)
    print(
        f"{'语言':<6}{'档位':<30}{'抢话率':>13}{'没说完':>8}{'提前':>6}{'漏判率':>12}"
        f"{'延迟p50':>10}{'延迟p90':>10}{'推理ms/条':>11}"
    )
    for r in results:
        print(
            f"{r['lang']:<6}{r['name']:<30}{r['抢话率']:>13}{r['其中没说完']:>8}{r['其中提前']:>6}"
            f"{r['漏判率']:>12}{r['延迟p50']:>9.0f}ms{r['延迟p90']:>9.0f}ms{r['推理ms/条']:>11.1f}"
        )

    out = Path("bench") / f"eou-official-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n原始结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
