"""把落盘的账单 JSONL 汇成验收记录（P4 真人验收的取证工具）。

用法：
    uv run python scripts/report_turns.py ~/.voice-pilot/turns.jsonl
    uv run python scripts/report_turns.py turns.jsonl --out docs/验收-真人.md

为什么需要它：JSONL 是**机器可读的原始行**，但验收要回答的是「这一轮快在哪、
有没有抢话、有没有被截断」——那需要按轮拆账、按分位汇总，并且把每轮的
waterfall 逐格对齐（`口径B` 不等于链路开销之和时，差额本身就是要解释的东西）。

输出口径三条纪律（与 docs/实验报告.md 一致）：
- **缺失值如实留空**，不填 0、不做均值替代（账单里的 `-1` 是「没测到」，不是「测得 0」）；
- **报分布不只报 p50**，同时给 n / min / p50 / max，样本少时明说样本少；
- **被打断的轮次单独统计**，不与完整轮次混在一起平均——它们的口径B 本就不可比。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# waterfall 各点相对会话钟的顺序（与 metrics/events.py 的 WATERFALL_POINTS 对齐）
WATERFALL = (
    "vad_onset",
    "asr_first_char",
    "fragment_commit",
    "eou",
    "utterance_end",
    "llm_first_token",
    "first_sentence_done",
    "tts_first_chunk",
    "first_audio_out",
    "turn_end",
)

# 分格拆账：每一格 = 相邻两个打点之差。口径B 是它们的和（不含被判停等待之外的部分）。
# 注意：`判停` 一格**不走 waterfall 打点**，因为探针里 `eou` 这个点从未 mark 过
# （同 `vad_onset` / `asr_first_char` / `fragment_commit`）——判停时长只存在于
# `eou_latency_ms` 指标里（判停时刻的静音时长）。拿一个不存在的打点做差，
# 结果会是「无打点」而不是数字，所以这里用真实来源并在证据列标明。
SEGMENTS = (
    ("LLM首token", "utterance_end", "llm_first_token"),
    ("首句成", "llm_first_token", "first_sentence_done"),
    ("TTS首片", "first_sentence_done", "tts_first_chunk"),
    ("首音下发", "tts_first_chunk", "first_audio_out"),
)

EOU_MODES = {0.0: "conservative", 1.0: "semantic", 2.0: "aggressive", 3.0: "neural", -1.0: "未用"}


def load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"⚠️ 第 {lineno} 行不是合法 JSON，已跳过：{exc}")
    return records


def _pct(values: list[float], q: float) -> float:
    """线性插值分位。样本少时由调用方负责说明 n。"""
    if not values:
        return -1.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values: list[float]) -> str:
    if not values:
        return "—"
    return (
        f"n={len(values)} min={min(values):.0f} "
        f"p50={statistics.median(values):.0f} max={max(values):.0f}"
    )


def segment_ms(rec: dict[str, Any], start: str, end: str) -> float | None:
    a, b = rec.get(start), rec.get(end)
    if a is None or b is None or a < 0 or b < 0:
        return None
    return float(b) - float(a)


def render(records: list[dict[str, Any]], source: str) -> str:
    if not records:
        return f"# 验收记录\n\n来源 `{source}`：**没有任何账单行**，无法出结论。\n"

    sessions = sorted({r.get("session_id", "?") for r in records})
    llm_engines = {str(r.get("llm", "?")) for r in records}
    llm_names = sorted(llm_engines)
    modes = sorted({r.get("mode", "?") for r in records})
    complete = [r for r in records if not r.get("interrupted")]
    interrupted = [r for r in records if r.get("interrupted")]

    lines: list[str] = [
        "# 真人验收记录（由脚本从账单 JSONL 生成）",
        "",
        f"> 来源：`{source}`　会话数：{len(sessions)}　模式：{', '.join(modes)}　"
        f"轮次：{len(records)}（完整 {len(complete)} / 被打断 {len(interrupted)}）"
        f"　LLM：{', '.join(llm_names)}",
        "",
        *(
            [
                "> ⚠️ **LLM 引擎包含 SimLlm——这些轮次不含真 LLM 延迟，本记录的口径B "
                "只是本机侧下界，不能当真实延迟报**（多半是起服务时缺 DASHSCOPE_API_KEY）。",
                "",
            ]
            if llm_engines - {"QwenLlm"}
            else []
        ),
        "> 本文件由 `scripts/report_turns.py` 生成，数字全部来自落盘账单，未经人工修饰。",
        "",
        "## 一、口径B 分布（完整轮次）",
        "",
    ]

    b_values = [float(r["caliber_b_ms"]) for r in complete if float(r.get("caliber_b_ms", -1)) >= 0]
    if not complete:
        lines.append("**无完整轮次**——所有轮次都被打断或未出账，无法给出延迟结论。")
    else:
        lines.append(f"`口径B（说完→AI首音）`：{summarize(b_values)}")
        if len(b_values) < 5:
            lines.append("")
            lines.append(f"⚠️ **样本仅 n={len(b_values)}，不足 5 轮**，分位数无统计意义，仅供参考。")
    lines.append("")

    if interrupted:
        lines += [
            "### 被打断的轮次（单独列，不与完整轮次混算）",
            "",
            "| 轮 | 打断时刻(ms) | 服务端已出音频(轮) |",
            "|---|---|---|",
        ]
        for r in interrupted:
            lines.append(f"| {r.get('turn', '?')} | {r.get('barge_in_latency_ms', -1):.0f} | 是 |")
        lines.append("")

    lines += [
        "## 二、分格拆账（完整轮次 p50）",
        "",
        "| 段 | p50 | 占口径B | 证据 |",
        "|---|---|---|---|",
    ]
    # 判停一格：用 eou_latency_ms（真实来源），不是 waterfall 打点。
    _eou_vals = [
        float(r["eou_latency_ms"]) for r in complete if float(r.get("eou_latency_ms", -1)) >= 0
    ]
    if _eou_vals and b_values:
        _p = statistics.median(_eou_vals)
        _base = statistics.median(b_values)
        lines.append(
            f"| 判停 | {_p:.0f}ms | {(_p / _base * 100) if _base > 0 else 0:.0f}% "
            f"| `eou_latency_ms`（判停时刻的静音时长） |"
        )
    else:
        lines.append("| 判停 | — | — | 该轮未走判停器 |")
    for name, start, end in SEGMENTS:
        vals = [v for r in complete if (v := segment_ms(r, start, end)) is not None and v >= 0]
        if not vals or not b_values:
            lines.append(f"| {name} | — | — | 该格无打点 |")
            continue
        p50 = statistics.median(vals)
        share = p50 / statistics.median(b_values) * 100 if statistics.median(b_values) > 0 else 0
        lines.append(f"| {name} | {p50:.0f}ms | {share:.0f}% | `{start}` → `{end}` |")
    lines.append("")

    lines += ["## 三、判停质量（一等验收指标）", ""]
    prem = sum(1 for r in complete if r.get("premature_eou", 0) >= 1)
    miss = sum(1 for r in complete if r.get("missed_eou", 0) >= 1)
    lat = [float(r["eou_latency_ms"]) for r in complete if float(r.get("eou_latency_ms", -1)) >= 0]
    gates = sorted({EOU_MODES.get(float(r.get("eou_mode", -1)), "?") for r in records})
    lines += [
        f"- 判停档：{', '.join(gates)}",
        f"- 判停延迟：{summarize(lat)}",
        f"- **抢话（premature_eou）：{prem}/{len(complete)}**"
        + ("　← 真人环境下这项比延迟更能说明问题" if prem else ""),
        f"- 漏判（missed_eou）：{miss}/{len(complete)}",
        f"- 打断次数：{len(interrupted)} 轮",
        "",
    ]

    lines += [
        "## 四、逐轮明细",
        "",
        "| 轮 | 口径B | 判停 | 首句字数 | 用户说 | AI 说 |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        flag = " ⚠️打断" if r.get("interrupted") else ""
        b = float(r.get("caliber_b_ms", -1))
        b_txt = f"{b:.0f}ms" if b >= 0 else "—（未测到）"
        lat_v = float(r.get("eou_latency_ms", -1))
        lines.append(
            f"| {r.get('turn', '?')}{flag} | {b_txt} | "
            f"{f'{lat_v:.0f}ms' if lat_v >= 0 else '—'} | "
            f"{r.get('tts_first_chars', '?')} | "
            f"{(r.get('user_text') or '')[:24]} | {(r.get('assistant_text') or '')[:24]} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="把账单 JSONL 汇成验收记录")
    ap.add_argument("jsonl", help="账单 JSONL 路径（VOICE_EVENTS_JSONL 写出的文件）")
    ap.add_argument("--out", default=None, help="写入文件；不给就打印到 stdout")
    args = ap.parse_args()

    path = Path(args.jsonl).expanduser()
    if not path.exists():
        raise SystemExit(f"找不到账单文件：{path}\n提示：起服务时设 VOICE_EVENTS_JSONL={path}")
    report = render(load(path), args.jsonl)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"已写出 {out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
