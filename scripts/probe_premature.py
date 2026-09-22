"""抢话率探针：句内停顿会不会被误判成「说完了」。

为什么必须有它（D4/D5 的缺口）：semantic 档在 2026-09-22 修好之后判停变快了
（1804ms → 282ms），代价是更敏感——真人说话的词间停顿 200~400ms 正好落在
「静音 ≥250ms + 最近有韵律终顿」的触发区。**这个代价此前只有一个定性判断
（「抢话率必然上升」），没有数。** 本脚本把它量出来。

做法：
  用 TTS 合成一句中文，在**句中自然停断点**（逗号处）插一段静音，再接后半句。
  判停若发生在后半句开始之前 = 抢话（premature）。

  [ 前半句 ][ pause ][ 后半句 ][ 尾部静音 1500ms ]
             ↑ 这里判停就是抢话

诚实标注（务必一并读）：
  1. 上行是**合成语音**，尾字能量衰减比真人干净、没有换气声和「嗯/啊」填充词。
     真人更容易被误判，所以这里的抢话率应视为**乐观下界**。
  2. ASR partial 是真模型（transducer），不是伪造文本——早期探针用 "啊"*n 冒充
     partial，让语义判据凭空成立，得出过「判停正常」的假结论。
  3. 只测「句内停顿」这一类抢话；「用户被回声打断」「背景噪声」不覆盖。

用法：
  uv run python scripts/probe_premature.py
  uv run python scripts/probe_premature.py --pauses 200,300,400 --gatings semantic
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_pipeline.core.eou import FusedEou, GatingMode  # noqa: E402
from voice_pipeline.providers import sherpa as R  # noqa: E402
from voice_pipeline.providers.factory import build_eou  # noqa: E402

SR = 16000
FRAME = 1600  # 100ms @16k —— 与 bench_ws / Session 的上行帧长一致
TAIL_SILENCE_MS = 2500  # 必须 > max_silence_ms(2000)，否则「漏判」只是尾巴不够长
ENERGY_FLOOR = 0.005  # 与 eou.py 的 ENERGY_FLOOR 一致：低于它算静音

# 句中自然停断点用 ｜ 标记：真人在这里换气/犹豫，是最容易被误判的位置。
UTTERANCES = [
    "我想问一下，｜明天的天气怎么样",
    "今天下午三点，｜我们在老地方见",
    "帮我查一下，｜上海到北京的机票",
    "这个方案吧，｜我觉得还可以再改改",
    "先说结论，｜成本大概要三十万",
]


def trim_silence(pcm: np.ndarray) -> np.ndarray:
    """裁掉合成语音两端的自带静音。

    为什么必须裁：melo 生成的每段话自带句尾衰减 + 句首起振静音，实测接点处
    天然就有 160~345ms 间隙。不裁的话「插入 250ms」实际是 410~595ms，
    conservative（阈值 500ms）会被自己的测量方式喂成「抢话 60%」。
    """
    win = 80  # 5ms 窗
    n = len(pcm) // win * win
    if n == 0:
        return pcm
    rms = np.sqrt((pcm[:n].reshape(-1, win) ** 2).mean(axis=1))
    voiced = np.where(rms >= ENERGY_FLOOR)[0]
    if voiced.size == 0:
        return pcm
    return pcm[voiced[0] * win : (voiced[-1] + 1) * win]


def gap_ms(buf: np.ndarray, from_ms: float, to_ms: float) -> float:
    """量 [from_ms, to_ms] 区间内的静音样本数——用来验证「插入值 == 真实间隙」。"""
    a, b = int(from_ms * SR / 1000), int(to_ms * SR / 1000)
    seg = buf[a:b]
    win = 80
    n = len(seg) // win * win
    if n == 0:
        return 0.0
    rms = np.sqrt((seg[:n].reshape(-1, win) ** 2).mean(axis=1))
    return float((rms < ENERGY_FLOOR).sum()) * win / SR * 1000


def synth(tts: object, text: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if text in cache:
        return cache[text]
    samples, sr = tts.generate(text)  # type: ignore[attr-defined]
    pcm = trim_silence(soxr.resample(samples, sr, SR).astype(np.float32))
    cache[text] = pcm
    return pcm


async def run_one(
    tts: object,
    asr: object,
    cache: dict[str, np.ndarray],
    utterance: str,
    pause_ms: int,
    gating: GatingMode,
    min_silence_short_ms: float | None = None,
) -> dict:
    head, _, tail = utterance.partition("｜")
    head_pcm = synth(tts, head, cache)
    tail_pcm = synth(tts, tail, cache)

    pause = np.zeros(int(SR * pause_ms / 1000), dtype=np.float32)
    tail_sil = np.zeros(int(SR * TAIL_SILENCE_MS / 1000), dtype=np.float32)
    buf = np.concatenate([head_pcm, pause, tail_pcm, tail_sil])

    # 真值：用户真正说完 = 后半句结束的时刻
    speech_end_ms = (len(head_pcm) + len(pause) + len(tail_pcm)) / SR * 1000
    # 自检：插入的 pause 是否等于真实声学间隙（不裁边界静音时两者差 160~345ms）
    measured_gap = gap_ms(
        buf, len(head_pcm) / SR * 1000 - 50, speech_end_ms - len(tail_pcm) / SR * 1000 + 50
    )

    if min_silence_short_ms is not None:
        eou = FusedEou(gating=gating, sample_rate=SR, min_silence_short_ms=min_silence_short_ms)
    else:
        eou = build_eou(gating)
    asr.reset()  # type: ignore[attr-defined]
    decided_at_ms: float | None = None
    elapsed = 0.0

    for i in range(0, len(buf), FRAME):
        chunk = R.pcm16_bytes(buf[i : i + FRAME])
        eou.push_audio(chunk)
        await asr.push_audio(chunk)  # type: ignore[attr-defined]
        eou.update_text(asr.text())  # type: ignore[attr-defined]
        elapsed = (i + FRAME) / SR * 1000
        if eou.should_end() and decided_at_ms is None:
            decided_at_ms = elapsed
            # 继续喂完：FusedEou 靠「判停后窗口内又听到人声」自己标记抢话
            # （premature_eou），只有喂完才能验证这个指标会不会响。

    premature = bool(decided_at_ms is not None and decided_at_ms < speech_end_ms)
    return {
        "utterance": utterance,
        "pause_ms": pause_ms,
        "gating": gating.value,
        "decided_at_ms": decided_at_ms,
        "speech_end_ms": speech_end_ms,
        "premature": premature,
        "missed": bool(decided_at_ms is None),
        # 判停延迟：真值结束 → 判停。抢话时它是负的（提前了多少 ms）
        "latency_ms": (decided_at_ms - speech_end_ms) if decided_at_ms is not None else None,
        "self_flagged": bool(eou.premature_eou),
        "asr_text": str(asr.text()),  # type: ignore[attr-defined]
        "measured_gap_ms": measured_gap,
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:.0f}%)" if d else "—"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pauses", default="0,150,250,350,500", help="句内停顿时长（ms）")
    ap.add_argument("--gatings", default="conservative,semantic,aggressive")
    ap.add_argument(
        "--sweep",
        default="250,300,350,400,500,600",
        help="扫 semantic 档的 min_silence_short_ms（延迟–抢话取舍曲线，D4）",
    )
    ap.add_argument("--sweep-pauses", default="250,350,500")
    ap.add_argument("--tts", default="melo", choices=["melo", "kokoro"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pauses = [int(p) for p in args.pauses.split(",") if p != ""]
    gatings = [GatingMode(g.strip()) for g in args.gatings.split(",") if g.strip()]

    t0 = time.perf_counter()
    tts = R.build_tts_engine(args.tts, num_threads=4)
    asr = R.TransducerStreamingAsr()
    print(f"模型加载 {(time.perf_counter() - t0) * 1000:.0f}ms  TTS={args.tts}  ASR=transducer(真)")
    print(f"句内停顿: {pauses}ms    门控档: {[g.value for g in gatings]}")
    print(f"尾部静音 {TAIL_SILENCE_MS}ms（给非抢话的轮次留出正常判停空间）\n")

    cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for utt in UTTERANCES:
        for pause in pauses:
            for gating in gatings:
                r = await run_one(tts, asr, cache, utt, pause, gating)
                rows.append(r)
                if args.verbose:
                    flag = "抢话" if r["premature"] else ("漏判" if r["missed"] else "正常")
                    lat = r["latency_ms"]
                    print(
                        f"  {utt[:12]:<14} 停顿{pause:>4}ms {gating.value:<13}"
                        f"{flag:<5} 判停延迟 {lat:>+7.0f}ms  自身标记抢话={r['self_flagged']}"
                        if lat is not None
                        else f"  {utt[:12]:<14} 停顿{pause:>4}ms {gating.value:<13}漏判"
                    )

    print("\n" + "=" * 92)
    print("测量自检：插入 0ms 时的真实声学间隙（裁边前是 160~345ms，裁后应 ≈0）")
    print("=" * 92)
    for utt in UTTERANCES:
        g = [r["measured_gap_ms"] for r in rows if r["utterance"] == utt and r["pause_ms"] == 0]
        if g:
            print(f"  {utt[:22]:<24} 实测间隙 {g[0]:>5.0f}ms")

    print("\n" + "=" * 92)
    print("抢话率矩阵（判停发生在后半句开始之前 = 抢话）")
    print("=" * 92)
    print(
        f"{'档位':<14}{'停顿':>6}{'抢话率':>12}{'漏判':>7}{'判停延迟中位':>14}{'自身标记抢话':>14}"
    )
    for gating in gatings:
        for pause in pauses:
            grp = [r for r in rows if r["gating"] == gating.value and r["pause_ms"] == pause]
            pre = [r for r in grp if r["premature"]]
            miss = [r for r in grp if r["missed"]]
            lat = [
                r["latency_ms"] for r in grp if r["latency_ms"] is not None and not r["premature"]
            ]
            med = statistics.median(lat) if lat else float("nan")
            self_flag = sum(1 for r in pre if r["self_flagged"])
            print(
                f"{gating.value:<14}{pause:>6}{_pct(len(pre), len(grp)):>12}{len(miss):>7}"
                f"{med:>13.0f}ms{f'{self_flag}/{len(pre)}' if pre else '—':>14}"
            )
        print()

    sweeps = [float(v) for v in args.sweep.split(",") if v.strip()]
    sweeps_pauses = [int(v) for v in args.sweep_pauses.split(",") if v.strip()]
    if sweeps:
        print("=" * 92)
        print("延迟–抢话取舍曲线（D4）：semantic 档把 min_silence_short_ms 从 250 扫到 600")
        print("=" * 92)
        print(f"{'静音阈值':>9}{'句内停顿':>9}{'抢话率':>12}{'未抢话时判停延迟':>18}")
        for v in sweeps:
            for pause in sweeps_pauses:
                grp = [
                    await run_one(tts, asr, cache, utt, pause, GatingMode.SEMANTIC, v)
                    for utt in UTTERANCES
                ]
                pre = [r for r in grp if r["premature"]]
                lat = [
                    r["latency_ms"]
                    for r in grp
                    if r["latency_ms"] is not None and not r["premature"]
                ]
                med = statistics.median(lat) if lat else float("nan")
                print(f"{v:>7.0f}ms{pause:>8}ms{_pct(len(pre), len(grp)):>12}{med:>17.0f}ms")
        print()

    print("=" * 92)
    print("关键读数：真人词间停顿 250~350ms 这一段")
    print("=" * 92)
    for gating in gatings:
        grp = [r for r in rows if r["gating"] == gating.value and 200 <= r["pause_ms"] <= 400]
        pre = [r for r in grp if r["premature"]]
        lat = [r["latency_ms"] for r in grp if r["latency_ms"] is not None and not r["premature"]]
        med = statistics.median(lat) if lat else float("nan")
        print(
            f"  {gating.value:<14} 抢话 {_pct(len(pre), len(grp)):<12}"
            f" 未抢话时判停延迟中位 {med:>7.0f}ms"
        )
    print("\n（合成语音无换气/填充词，真人只会更糟 → 以上抢话率是乐观下界）")
    return 0


if __name__ == "__main__":
    raise SystemExit(__import__("asyncio").run(main()))
