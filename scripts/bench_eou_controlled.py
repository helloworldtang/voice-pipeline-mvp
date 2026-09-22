"""EOU 受控实验：句末 vs 句中停顿，各自补一段**固定长度**的静音，看谁判停。

为什么要这么测（前面两版测量都错了，理由留在文件里）：
  1. 自造 TTS 样本真值不可信 —— 分开合成的前半句自带句末终顿韵律。
  2. 逐帧流式喂真音频时，模型输入末尾是「正在说话的语音」，而训练样本末尾都有
     静音 —— 模型在分布外，概率在 0.01↔0.99 之间乱跳（实测），判停时刻是噪声。
  所以：**只在静音出现后评估模型**（这才是 pipecat 的用法），并把静音长度做成
  受控变量。

构造（两集**对称**，唯一差别是标签）：
  句末集 —— 真值 endpoint_bool=True：裁到最后一个有人声的样本，再补 L ms 静音
  句中集 —— 真值 endpoint_bool=False：同样裁到最后一个有人声的样本，再补 L ms 静音
  数据集里 neg 样本本来就是「切在用户停顿处、话还没说完」，所以两集的声学形式完全
  一样，差别只在「这句话在这儿算不算完」。**别自作聪明另找句中停顿点** —— 我第一版
  去找「后面还有 ≥1s 语音」的中间静音段，那是另一个更难的任务，不是数据集的对比。

对照基线 conservative：静音 ≥500ms 才判停 —— 于是它在 L<500 时检出 0%、抢话 0%；
L≥500 时检出 100%、抢话 100%。神经档要赢，得在 L<500 时检出高而抢话低。

用法：
  uv run python scripts/bench_eou_controlled.py
  uv run python scripts/bench_eou_controlled.py --langs zho,eng --silence 0,100,200,300,400,500,600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_pipeline.providers.eou.smart_turn import SmartTurnEou  # noqa: E402

SR = 16000
ENERGY_FLOOR = 0.005
CONSERVATIVE_MS = 500.0


def voiced_profile(audio: np.ndarray, win_ms: float = 5.0) -> np.ndarray:
    win = int(SR * win_ms / 1000)
    n = len(audio) // win * win
    if n == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((audio[:n].reshape(-1, win) ** 2).mean(axis=1))
    return rms >= ENERGY_FLOOR


def speech_end_sample(audio: np.ndarray) -> int:
    v = voiced_profile(audio)
    win = int(SR * 0.005)
    return int((np.where(v)[0][-1] + 1) * win) if v.any() else 0


def prob_at(det: SmartTurnEou, audio: np.ndarray, silence_ms: float) -> float:
    det.reset()
    sil = np.zeros(int(SR * silence_ms / 1000), dtype=np.float32)
    buf = np.concatenate([audio, sil]) if silence_ms else audio
    det.push_audio((buf * 32767).astype("<i2").tobytes())
    return det.probability()


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.1f}%" if d else "    —"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/eou-test")
    ap.add_argument("--langs", default="zho")
    ap.add_argument("--silence", default="0,100,200,300,400,500,600")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.data_dir)
    meta = json.loads((root / "meta.json").read_text())
    silences = [float(s) for s in args.silence.split(",")]
    det = SmartTurnEou(threshold=args.threshold, min_interval_ms=0.0)
    print(f"模型 {det.model_path.name}  阈值 {args.threshold}\n")

    results = []
    for lang in args.langs.split(","):
        lang = lang.strip()
        sel = [m for m in meta if m["lang"] == lang]
        end_set, pause_set = [], []
        t0 = time.perf_counter()
        for m in sel:
            a = np.load(root / lang / f"{m['i']:04d}.npy").astype(np.float32)
            if m["label"]:  # 真值：说完了
                cut = speech_end_sample(a)
                if cut > SR * 0.3:
                    end_set.append(a[:cut])
            else:  # 真值：没说完（数据集本就切在停顿处）
                cut = speech_end_sample(a)
                if cut > SR * 0.3:
                    pause_set.append(a[:cut])
        print(
            f"{lang}: 句末 {len(end_set)} 条 / 句中停顿 {len(pause_set)} 条"
            f"（构造 {(time.perf_counter() - t0) * 1000:.0f}ms）"
        )

        rows = []
        for L in silences:
            t0 = time.perf_counter()
            pe = [prob_at(det, a, L) for a in end_set]
            pp = [prob_at(det, a, L) for a in pause_set]
            end_hit = sum(1 for p in pe if p >= args.threshold)
            pause_fp = sum(1 for p in pp if p >= args.threshold)
            rows.append(
                {
                    "lang": lang,
                    "silence_ms": L,
                    "句末检出": end_hit,
                    "句末n": len(pe),
                    "句中抢话": pause_fp,
                    "句中n": len(pp),
                    "p_end": float(np.median(pe)) if pe else float("nan"),
                    "p_pause": float(np.median(pp)) if pp else float("nan"),
                    "ms/条": (time.perf_counter() - t0) * 1000 / max(1, len(pe) + len(pp)),
                }
            )
            print(
                f"  L={L:>5.0f}ms  句末检出 {pct(end_hit, len(pe))} ({end_hit}/{len(pe)})  "
                f"句中抢话 {pct(pause_fp, len(pp))} ({pause_fp}/{len(pp)})  "
                f"p中位 句末{rows[-1]['p_end']:.2f}/句中{rows[-1]['p_pause']:.2f}"
            )
        results.extend(rows)

    print("\n" + "=" * 96)
    print("句末检出率 vs 句中抢话率（受控静音长度）")
    print("=" * 96)
    print(f"{'语言':<6}{'静音':>7}{'句末检出':>12}{'句中抢话':>12}{'conservative':>28}")
    for r in results:
        cons = "检出 0% / 抢话 0%" if r["silence_ms"] < CONSERVATIVE_MS else "检出 100% / 抢话 100%"
        print(
            f"{r['lang']:<6}{r['silence_ms']:>6.0f}ms{pct(r['句末检出'], r['句末n']):>12}"
            f"{pct(r['句中抢话'], r['句中n']):>12}{cons:>28}"
        )

    out = Path("bench") / f"eou-controlled-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n原始结果 -> {out}")
    print("\n读法：conservative 在 500ms 处一刀切（检出和抢话同时 0→100%）。")
    print("      神经档要赢，必须出现在「检出 > 0% 而抢话仍很低」的格子。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
