"""A/B 对照：同一句语音、同一组真模型，只换调度器。

跑法：
    uv run python scripts/bench_schedulers.py
    uv run python scripts/bench_schedulers.py --tts melo --silence-ms 250,400,600
    uv run python scripts/bench_schedulers.py --grace-ms 0,150,300

它回答的是技术方案 §13.4 留下的那个问题：既然串行下 300ms 不可达，
投机究竟能把口径B 压到多少 —— 以及代价是什么。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soxr

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.scheduler.base import AudioFragment
from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers import sherpa as R
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.sherpa import pcm16_bytes

USER_UTTERANCES = [
    "今天天气怎么样",
    "我想问一下明天的天气怎么样",
    "帮我把明天的日程念一遍，顺便提醒我下午三点开会",
]

BLOCK = 1600  # 100ms @ 16kHz


@dataclass
class Row:
    mode: str
    tts: str
    asr: str
    utterance: str
    silence_ms: int
    grace_ms: float
    caliber_b_ms: float
    spec_hit: float
    spec_miss: float
    audio_bytes: int
    sentences: int
    wall_ms: float
    spec_reason: str = "n/a"
    rep: int = 0  # --repeat 的第几轮。保留它才能算分位数，而不是只留最好的一轮
    llm: str = "SimLlm"  # 零延迟替身 vs 真 LLM——两者命中率完全不同，必须标在每行上


def build(mode: str, asr: object, tts: object, grace_ms: float, llm: object) -> object:
    """同一组依赖，只换调度器实现。"""
    metrics = TurnMetrics(clock=SessionClock())
    splitter = SentenceSplitter()
    chunker = Chunker()
    if mode == "pipelined":
        return PipelinedScheduler(
            asr=asr,
            llm=llm,
            tts=tts,
            metrics=metrics,
            splitter=splitter,
            chunker=chunker,
            grace_ms=grace_ms,
        )
    return SerialScheduler(
        asr=asr, llm=llm, tts=tts, metrics=metrics, splitter=splitter, chunker=chunker
    )


async def synthesize_user_speech(tts: object, text: str) -> np.ndarray:
    """用 TTS 合成一段「用户语音」再降到 16k，充当上行输入。

    两处诚实声明：

    1. 这是合成语音，不是真人录音（见技术方案 §13.1 的注意事项），因此下面的数字
       用来比较**调度器之间的相对差值**，不足以得出真人场景的绝对值。
    2. **音色必须固定为 kokoro**：本机实测 melo 合成的话喂给同一个 ASR 识别结果为空，
       final 空串。同一份"合成数据做回归"的套路，换个音色就静默失效——这是最容易
       让人把 bug 当成延迟问题查半天的坑，所以在这里写死并注明。
    """
    samples, sr = tts.generate(text)  # type: ignore[attr-defined]
    if sr != 16000:
        samples = soxr.resample(samples, sr, 16000)
    return samples


async def replay(scheduler: object, pcm: np.ndarray) -> None:
    """按真实速度喂帧（每 100ms 一帧）。

    一次性把整段音频灌进去是错的：那样事件循环从不停歇，后台投机任务根本拿不到
    执行机会，测出来「投机没用」其实是测量方式的问题。真实链路里音频是实时到达的，
    静音窗的 250ms 是真实的 250ms，投机才有时间去跑。
    """
    frame_ms = BLOCK / 16000 * 1000
    for i in range(0, len(pcm), BLOCK):
        await scheduler.on_audio(pcm16_bytes(pcm[i : i + BLOCK]))  # type: ignore[attr-defined]
        await asyncio.sleep(frame_ms / 1000.0)


async def run_once(
    mode: str,
    tts: object,
    asr_engine: type,
    tts_name: str,
    asr_name: str,
    utterance: str,
    user_pcm: np.ndarray,
    silence_ms: int,
    grace_ms: float,
    rep: int = 0,
    llm: object = None,
) -> Row:
    if llm is None:
        llm = SimLlm()
    scheduler: PipelinedScheduler | SerialScheduler = build(  # type: ignore[assignment]
        mode, asr_engine(), tts, grace_ms, llm
    )
    metrics = scheduler._metrics  # type: ignore[union-attr]  # 同一份账单，两边都从这里取

    t0 = time.perf_counter()
    await replay(scheduler, user_pcm)
    # EOU 之前的静音窗：真实系统里 VAD 必须靠它确认用户说完（本机实测 ≥250ms）。
    # 这段时间正是投机唯一能吃到的窗口。
    silence = np.zeros(int(16000 * silence_ms / 1000), dtype=np.float32)
    await replay(scheduler, silence)

    audio_bytes = 0
    async for item in scheduler.on_end_of_utterance():
        if isinstance(item, AudioFragment):
            audio_bytes += len(item.data)
    wall = (time.perf_counter() - t0) * 1000

    bill = metrics.bill()
    return Row(
        mode=mode,
        tts=tts_name,
        asr=asr_name,
        utterance=utterance,
        silence_ms=silence_ms,
        grace_ms=grace_ms if mode == "pipelined" else 0.0,
        caliber_b_ms=bill.get("caliber_b_ms", -1.0),
        spec_hit=float(getattr(scheduler, "spec_hits", 0)),
        spec_miss=float(getattr(scheduler, "spec_misses", 0)),
        audio_bytes=audio_bytes,
        sentences=int(bill.get("sentences", 0)),
        wall_ms=round(wall, 1),
        spec_reason=str(getattr(scheduler, "spec_last_reason", "n/a")),
        rep=rep,
        llm=type(llm).__name__,
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="serial vs pipelined 本机 A/B")
    ap.add_argument("--tts", default="melo", choices=["kokoro", "melo"])
    ap.add_argument("--asr", default="transducer", choices=["transducer", "paraformer"])
    ap.add_argument("--silence-ms", default="250,400,600")
    ap.add_argument("--grace-ms", default="0")
    ap.add_argument(
        "--llm",
        default="sim",
        choices=["sim", "qwen"],
        help="sim=零延迟替身（不含 LLM 首 token）；qwen=真 LLM（含网络，命中率会更低）",
    )
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    silences = [int(s) for s in args.silence_ms.split(",") if s]
    graces = [float(g) for g in args.grace_ms.split(",") if g]

    # 上行语音源固定 kokoro（原因见 synthesize_user_speech 的 docstring），
    # 下行 TTS 才是被比较的对象。两者混用会让「换 TTS」同时污染输入，对比失真。
    t0 = time.perf_counter()
    up_tts = R.build_tts_engine("kokoro", num_threads=2)
    down_tts = R.build_tts_engine(args.tts, num_threads=4)
    asr_cls = {
        "transducer": R.TransducerStreamingAsr,
        "paraformer": R.ParaformerStreamingAsr,
    }[args.asr]
    # 真 LLM 必须显式点名才启用：它慢 500ms 上下，会让命中率塌掉，
    # 所以「这组数含不含 LLM」必须由开关决定，而不是悄悄混在一起。
    if args.llm == "qwen":
        from voice_pipeline.providers.factory import build_llm

        llm = build_llm("qwen")
    else:
        llm = SimLlm()
    load_ms = (time.perf_counter() - t0) * 1000
    print(
        f"机器: {platform.machine()} / macOS {platform.mac_ver()[0]}   "
        f"Python {platform.python_version()}"
    )
    print(
        f"模型加载 {load_ms:.0f}ms   ASR={args.asr}   下行TTS={args.tts}   "
        f"上行语音源=kokoro   LLM={args.llm}"
    )

    rows: list[Row] = []
    for utterance in USER_UTTERANCES:
        user_pcm = await synthesize_user_speech(up_tts, utterance)
        for silence in silences:
            for grace in graces:
                for mode in ("serial", "pipelined"):
                    if mode == "serial" and grace != graces[0]:
                        continue  # serial 不吃 grace，跑一次就够
                    # ⚠️ 这里原来是「跑 repeat 次，只留 caliber_b 最小的那一轮」。
                    # 那是挑好轮次——与本项目「不裁剪、不挑轮次」的纪律直接冲突：
                    # 谁用 --repeat 复跑，谁就拿到一个被挑选过的数字，而挑出来的
                    # 最小值恰恰是 pipelined 唯一那次命中的形态。改为全轮保留，
                    # 由 print_report 报 p50/min/max，命中率才有分母。
                    for rep in range(args.repeat):
                        rows.append(
                            await run_once(
                                mode,
                                down_tts,
                                asr_cls,
                                args.tts,
                                args.asr,
                                utterance,
                                user_pcm,
                                silence,
                                grace,
                                rep,
                                llm,
                            )
                        )
        print(f"  done: {utterance}")

    print_report(rows)

    out = args.json_out or f"bench/schedulers-{datetime.now():%Y%m%d-%H%M%S}.json"
    bench_dir = Path("bench")
    if not bench_dir.exists():
        bench_dir.mkdir(parents=True)
    Path(out).write_text(
        json.dumps(
            {
                "machine": f"{platform.machine()} macOS {platform.mac_ver()[0]}",
                "asr": args.asr,
                "tts": args.tts,
                "rows": [asdict(r) for r in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n→ {out}")


def _stats(vals: list[float]) -> str:
    """p50 (min / max)。只报 min 等于挑好轮次，只报 p50 会藏住长尾——两个都要。"""
    if not vals:
        return "-"
    return f"{statistics.median(vals):.0f} (min {min(vals):.0f} / max {max(vals):.0f})"


def print_report(rows: list[Row]) -> None:
    print("\n=== 口径B（用户说完 → AI 首音）: p50 (min / max) ===")
    print(f"{'句':<16}{'静音窗':>7}{'grace':>7}{'serial':>24}{'pipelined':>24}{'命中':>8}")

    groups: dict[tuple[str, int, float], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.utterance, r.silence_ms, r.grace_ms), []).append(r)

    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        grp = groups[key]
        ser = [r.caliber_b_ms for r in grp if r.mode == "serial"]
        pip = [r for r in grp if r.mode == "pipelined"]
        if not pip:
            continue
        label = key[0] if len(key[0]) <= 14 else key[0][:13] + "…"
        hits = sum(r.spec_hit for r in pip)
        print(
            f"{label:<16}{key[1]:>7}{key[2]:>7.0f}{_stats(ser):>24}"
            f"{_stats([r.caliber_b_ms for r in pip]):>24}"
            f"{f'{hits:.0f}/{len(pip)}':>8}"
        )

    pipes = [r for r in rows if r.mode == "pipelined"]
    serials = [r for r in rows if r.mode == "serial"]
    if pipes and serials:
        hits = sum(r.spec_hit for r in pipes)
        rate = hits / len(pipes) * 100
        print(f"\nserial    p50 {statistics.median([r.caliber_b_ms for r in serials]):.0f}ms")
        print(f"pipelined p50 {statistics.median([r.caliber_b_ms for r in pipes]):.0f}ms")
        print(f"投机命中率: {hits:.0f}/{len(pipes)} = {rate:.0f}%")
        print(
            "\n⚠️ 报数字时说命中率，不要说最好那一轮：\n"
            "   min 是分布的一端，不是这条链路的常态；p50 才是。\n"
            f"   本轮 pipelined 最快的一次是 {min(r.caliber_b_ms for r in pipes):.0f}ms，"
            f"但它只占 {1 / len(pipes) * 100:.0f}% 的轮次。"
        )
        misses: dict[str, int] = {}
        for r in pipes:
            if r.spec_hit != 1.0:
                misses[r.spec_reason] = misses.get(r.spec_reason, 0) + 1
        if misses:
            print("\n未命中原因分布:")
            for reason, n in sorted(misses.items(), key=lambda kv: -kv[1]):
                print(f"  {n:>3}×  {reason}")
    print(
        "\n注：上行是 TTS 合成语音而非真人录音 —— 数字用于比较调度器之间的相对差值，"
        "\n不作为真人场景绝对值。命中的前提是「partial 已收敛且 EOU 前投机跑完」，"
        "\n所以静音窗越长、句子越短越容易命中。",
    )


if __name__ == "__main__":
    asyncio.run(main())
