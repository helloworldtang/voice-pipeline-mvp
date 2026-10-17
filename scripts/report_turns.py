"""把落盘的账单 JSONL 汇成验收记录（P4 真人验收的取证工具）。

用法：
    uv run python scripts/report_turns.py ~/.voice-pilot/turns.jsonl --list   # 先看会话清单
    uv run python scripts/report_turns.py ~/.voice-pilot/turns.jsonl --session last \
        --out docs/验收-真人.md
    uv run python scripts/report_turns.py ~/.voice-pilot/turns.jsonl --since 2026-10-10 \
        --out docs/验收-真人.md       # 一天里连了两段会话时用这个

为什么需要 `--session` / `--since`：账单文件是**累积**的（`VOICE_EVENTS_JSONL` 一直追加），
而本脚本原先把文件里每一行都当本轮——2026-10-10 真人验收就出过事：拿累积文件出记录，
得到的是「会话数 5 / 轮次 150」的混合版，本轮的 25 轮被旧轮次稀释（口径B p50 跟着变成混合值）。
不给过滤条件时，若文件里有多段会话，记录里会写明这一点并在 stderr 提醒。

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
import sys
import time
from pathlib import Path
from typing import Any

from voice_pipeline.metrics.trace import exclusive_times, parse_spans

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


def sessions_of(records: list[dict[str, Any]]) -> list[str]:
    """会话 id 按**首行时间**排序 —— `--session last` 靠它定义「最新一段」。"""
    first_ts: dict[str, float] = {}
    for r in records:
        first_ts.setdefault(str(r.get("session_id", "?")), float(r.get("ts", 0.0)))
    return [sid for sid, _ in sorted(first_ts.items(), key=lambda kv: kv[1])]


def session_rows(records: list[dict[str, Any]], sid: str) -> list[dict[str, Any]]:
    return [r for r in records if str(r.get("session_id", "?")) == sid]


def session_lines(records: list[dict[str, Any]]) -> list[str]:
    """每段会话一行（id / 轮次 / 起止 / 模式）—— `--list` 与提醒共用。"""
    lines: list[str] = []
    for sid in sessions_of(records):
        rows = session_rows(records, sid)
        t0 = time.strftime("%m-%d %H:%M:%S", time.localtime(float(rows[0].get("ts", 0.0))))
        t1 = time.strftime("%H:%M:%S", time.localtime(float(rows[-1].get("ts", 0.0))))
        modes = ",".join(sorted({str(r.get("mode", "?")) for r in rows}))
        lines.append(f"{sid}  {len(rows):>3} 轮  {t0} → {t1}  mode={modes}")
    return lines


def parse_since(value: str) -> float:
    """`--since` 的解析：接受 YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'，按本地时区。"""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return time.mktime(time.strptime(value, fmt))
        except ValueError:
            continue
    raise SystemExit(f"--since {value!r} 解析不了；用 YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'")


def pick_session(records: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
    """`--session` 的选择器：`last` = 最新一段，其余按 id 精确匹配（认不出就列出可选值）。"""
    ids = sessions_of(records)
    if not ids:
        return records
    if value == "last":
        sid = ids[-1]
    elif value in ids:
        sid = value
    else:
        raise SystemExit(
            f"--session {value!r} 不在文件里。可选：{', '.join(ids)}（last = 最新一段）"
        )
    return session_rows(records, sid)


def pick_since(records: list[dict[str, Any]], since: float) -> list[dict[str, Any]]:
    """只留首行时间 ≥ since 的会话（含整段，不截半个会话）。"""
    keep = {
        sid
        for sid in sessions_of(records)
        if float(session_rows(records, sid)[0].get("ts", 0.0)) >= since
    }
    return [r for r in records if str(r.get("session_id", "?")) in keep]


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
    """n / min / p50 / p90 / p99 / max。

    为什么一定要带 p90/p99：延迟优化最容易骗人的是均值与中位数——长尾（冷启动、网络抖动）
    恰恰是用户喊"卡"的那些轮。样本 <20 时 p99 只能当"最差值附近"看，不具统计意义，
    所以同时留 max 作交叉核对。
    """
    if not values:
        return "—"
    return (
        f"n={len(values)} min={min(values):.0f} "
        f"p50={statistics.median(values):.0f} "
        f"p90={_pct(values, 0.9):.0f} p99={_pct(values, 0.99):.0f} "
        f"max={max(values):.0f}"
    )


def segment_ms(rec: dict[str, Any], start: str, end: str) -> float | None:
    a, b = rec.get(start), rec.get(end)
    if a is None or b is None or a < 0 or b < 0:
        return None
    return float(b) - float(a)


def render(
    records: list[dict[str, Any]], source: str, *, filtered: bool = False, spans: bool = False
) -> str:
    """`filtered`：调用方已按会话/时间切片时不提醒「合并统计」——那是误报（本次就是切出来的）。

    `spans`：追加 span 分格表（口径B 窗口内每个 span 的自有时间）。账单行里的
    `spans` 字段是 2026-10-17 链路追踪上线后才有的，历史行没有就如实说明。
    """
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
                f"> ⚠️ **源文件里有 {len(sessions)} 段会话，本记录是它们的合并统计**——单轮验收出记录"
                "请加 `--session last`（或 `--since <日期>`）重跑，否则旧轮次会稀释本轮数字。"
                "会话清单用 `--list` 看。",
                "",
            ]
            if len(sessions) > 1 and not filtered
            else []
        ),
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
        if 5 <= len(b_values) < 20:
            lines.append("")
            lines.append("⚠️ 样本 <20：p99 只能当「最差值附近」看，不具统计意义（max 作交叉核对）。")
        if len(b_values) < 5:
            lines.append("")
            lines.append(f"⚠️ **样本仅 n={len(b_values)}，不足 5 轮**，分位数无统计意义，仅供参考。")
    # 体感首音（含 filler）：与口径B **分开报**——filler 是预合成音频，它让「第一个声音」
    # 更早，但真实回答的首片一秒没早（`core/filler.py` 的计量纪律）。两个数必须同时在场，
    # 否则单看任何一个都会误读：只看体感会以为链路变快了，只看口径B 会以为 filler 白做。
    fs_values = [
        float(r["first_sound_ms"]) for r in complete if float(r.get("first_sound_ms", -1)) >= 0
    ]
    if fs_values:
        used = sum(1 for r in complete if float(r.get("filler_used", 0)) >= 1)
        lines.append("")
        lines.append(f"`体感首音（说完→第一个声音，含 filler）`：{summarize(fs_values)}")
        lines.append(
            f"> ⚠️ {used}/{len(fs_values)} 轮播了预合成 filler。**口径B 不含 filler**——"
            "它仍只算真实回答的首片；对外两个数一起报。"
        )
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
        "| 轮 | 口径B | 体感 | 判停 | 首句字数 | 用户说 | AI 说 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        flag = " ⚠️打断" if r.get("interrupted") else ""
        b = float(r.get("caliber_b_ms", -1))
        b_txt = f"{b:.0f}ms" if b >= 0 else "—（未测到）"
        fs = float(r.get("first_sound_ms", -1))
        fs_txt = f"{fs:.0f}ms" if fs >= 0 else "—"
        lat_v = float(r.get("eou_latency_ms", -1))
        lines.append(
            f"| {r.get('turn', '?')}{flag} | {b_txt} | {fs_txt} | "
            f"{f'{lat_v:.0f}ms' if lat_v >= 0 else '—'} | "
            f"{r.get('tts_first_chars', '?')} | "
            f"{(r.get('user_text') or '')[:24]} | {(r.get('assistant_text') or '')[:24]} |"
        )
    lines.append("")

    if spans:
        lines += render_span_section(records)
    return "\n".join(lines) + "\n"


def render_span_section(records: list[dict[str, Any]]) -> list[str]:
    """span 分格表：口径B 窗口内每个 span 的**自有时间**（flame graph 口径）。

    两条口径必须写在代码里，否则读表会读反：

    1. **每轮求和再取 p50**。同一 span 一轮会出现多次（`uplink.frame` 一轮 80 帧、
       `eou.update` 同理），逐帧取 p50 会得到 0ms——把「上行 CPU 那一格」读成免费，
       而它每窗口实际 10~15ms（2026-10-17 验收前自查发现）。
    2. **自有时间**（区间扣掉直接后代的覆盖），每毫秒只归一个 span：`Σ自有 ≤ 口径B`，
       差额就是没埋到的残差。直接拿 span 时长加总不行——真实链路是嵌套的
       （`llm.request` 里包着 `tts.synth`），相加会比口径B 还大。
    """
    per_name: dict[str, list[float]] = {}
    n_with_spans = 0
    for r in records:
        spans, _ = parse_spans(r.get("spans") or [])
        window = next((s for s in spans if s.name.endswith("caliber_b")), None)
        if window is None or window.dur_ms < 0:
            continue
        n_with_spans += 1
        own = exclusive_times(spans, window.start_ms, window.end_ms)
        by_id = {s.span_id: s for s in spans}
        round_sum: dict[str, float] = {}
        for sid, ms in own.items():
            name = by_id[sid].name
            round_sum[name] = round_sum.get(name, 0.0) + ms
        for name, ms in round_sum.items():
            per_name.setdefault(name, []).append(ms)

    lines = ["## 五、span 分格（口径B 窗口，自有时间）", ""]
    if not n_with_spans:
        lines += [
            "**这些账单行没有 spans 字段**——链路追踪（trace_id + span）是 2026-10-17 上线的，",
            "更早的轮次只有 waterfall 打点。用 `scripts/probe_trace.py` 重跑即可拿到 span 树。",
            "",
        ]
        return lines

    base = statistics.median(
        [float(r["caliber_b_ms"]) for r in records if float(r.get("caliber_b_ms", -1)) >= 0]
    )
    lines += [
        f"> 有 span 的轮次：{n_with_spans}/{len(records)}。"
        "每轮完整 span 树在账单 JSONL 的 `spans` 字段里（按 `trace_id` 取）。",
        "> 口径：**每轮窗口内先求和、再取 p50**（同一 span 一轮可能多次，如 `uplink.frame`）；",
        "> 父子嵌套只算自有时间。`llm.request` / `tts.synth` 接近 0 是正常的——它们被下游背压",
        "> 拉长（见 `docs/链路追踪-根因分析.md` §4.6），不代表 provider 快。",
        "",
        "| span | 轮数 | 每轮 p50 | p90 | max | 占口径B |",
        "|---|---|---|---|---|---|",
    ]
    ranked = sorted(per_name.items(), key=lambda kv: -statistics.median(kv[1]))
    for name, vals in ranked:
        if name in {"turn", "turn.caliber_b"}:
            continue  # 根/窗口的自有时间恒 0，列出来只是噪声
        p50 = statistics.median(vals)
        share = f"{p50 / base * 100:.0f}%" if base > 0 else "—"
        lines.append(
            f"| {name} | {len(vals)} | {p50:.0f}ms | {_pct(vals, 0.9):.0f}ms | "
            f"{max(vals):.0f}ms | {share} |"
        )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="把账单 JSONL 汇成验收记录")
    ap.add_argument("jsonl", help="账单 JSONL 路径（VOICE_EVENTS_JSONL 写出的文件）")
    ap.add_argument("--out", default=None, help="写入文件；不给就打印到 stdout")
    ap.add_argument(
        "--session",
        default=None,
        help="只统计一段会话：传 session_id，或 last（文件里最新的一那一段）",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="只统计首行时间在这之后的会话（YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'）",
    )
    ap.add_argument("--list", action="store_true", help="只列出文件里的会话清单，不生成记录")
    ap.add_argument(
        "--spans",
        action="store_true",
        help="追加 span 分格表（口径B 窗口，trace 上线后的账单才有）",
    )
    args = ap.parse_args()

    path = Path(args.jsonl).expanduser()
    if not path.exists():
        raise SystemExit(f"找不到账单文件：{path}\n提示：起服务时设 VOICE_EVENTS_JSONL={path}")
    records = load(path)
    if args.list:
        for line in session_lines(records):
            print(line)
        return
    ids = sessions_of(records)
    if args.session:
        records = pick_session(records, args.session)
        if not records:
            raise SystemExit(f"--session {args.session!r} 过滤后没有行")
    elif args.since:
        records = pick_since(records, parse_since(args.since))
        if not records:
            raise SystemExit(f"--since {args.since!r} 之后没有会话（--list 看清单）")
    elif len(ids) > 1:
        # 静默合并正是 2026-10-10 那次的坑：记录看着正常，数字却是整段历史的。
        print(
            f"⚠️ 源文件含 {len(ids)} 段会话，已合并统计；"
            "单轮验收请加 --session last（--list 看清单）",
            file=sys.stderr,
        )
    report = render(
        records, args.jsonl, filtered=bool(args.session or args.since), spans=args.spans
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"已写出 {out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
