"""链路追踪：TraceID + Span 树的延迟归因。

**为什么 waterfall 打点不够，还得写它**（这一段是设计理由，不是注释癖）：

`metrics/events.py` 的 TurnMetrics 记的是**瞬间**（一串绝对时刻），要分析就得两个瞬间相减。
这在单链串行且全埋到的时候够用，但它有三个失效方式，恰好都在找根因的路上会撞上：

1. **重叠不可见**。两个区间并发（TTS 在线程池里算，同时用户还在上行帧）时，
   相减的结果会重复计数——看起来链路变慢了，实际是并发前进了一大步。
2. **混账不可见**。"LLM首token" 那格 = `utterance_end → llm_first_token`，中间还夹着
   `asr.finalize()` 与事件循环调度。除非事先知道要往那插一根针，这几十毫秒永远记到
   LLM 头上（本机首次量出来是多少，见 `docs/链路追踪-根因分析.md`）。
3. **漏测不可见**。瀑布表没有"未覆盖"这一格，缺了看起来像测到 0。

所以这里按分布式追踪的最小模型重写：**区间 + 父子关系 + 属性**，并强制一条会计恒等式：

    root_total = Σ(子区间覆盖并集) + 未归因残差
    Σ(子区间时长) = 覆盖并集 + 重叠（并发折损）

残差不是装饰——**它非 0 就说明还有东西没埋到**，而过去那几十毫秒是静默消失的。

刻意做薄：不做采样、不做导出后端、不上 contextvars 隐式上下文。
理由：本进程是单事件循环的 asyncio，显式传 `parent` 比隐式传播可靠——
跨协程的 contextvars 传播正是最容易出错且最难查的那一类机制。
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from .events import SessionClock

# 未完成/未测到的哨兵：**不写 0**（0 看起来像个正常值，所以能潜伏很久）
MISSING = -1.0

ROOT_SUFFIX = "caliber_b"  # 根 span 的命名约定（口径B 窗口），报告脚本靠它找根


@dataclass(slots=True)
class Span:
    """一个区间。时间是相对会话钟的毫秒，不是"这段代码的墙上耗时"——差别在于

    会话钟里**包含了等待**（await 期间别人占用事件循环的时间也记在这一格上），
    这正是延迟分析要的东西。
    """

    name: str
    span_id: str
    parent_id: str
    start_ms: float
    end_ms: float = MISSING
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def dur_ms(self) -> float:
        """时长；未 finish 返回 MISSING —— 未完成就是未完成，不给 0。"""
        if self.end_ms < 0 or self.start_ms < 0:
            return MISSING
        return self.end_ms - self.start_ms

    @property
    def open(self) -> bool:
        return self.end_ms < 0

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3) if self.end_ms >= 0 else MISSING,
            "dur_ms": round(self.dur_ms, 3) if self.dur_ms >= 0 else MISSING,
        }
        for key, value in self.attrs.items():
            row[f"a.{key}"] = value
        return row


class Tracer:
    """一轮一棵树。`trace_id` 跨轮稳定（一次会话的多轮要靠它串起来看分布）。"""

    def __init__(self, trace_id: str, clock: SessionClock | None = None) -> None:
        self.trace_id = trace_id
        self._clock = clock or SessionClock()
        self._spans: list[Span] = []
        self._seq = itertools.count(1)
        # 两级根：`root` = 整轮（从本轮第一帧上行开始，到 turn.end），
        # `window` = 口径B 窗口（说完 → 首音）。**为什么要有 root**：2026-10-17
        # 实测发现 reset_turn 会把树清空，只按口径B 记的话，判停前的上行帧/EOU/partial
        # 全部被抹掉——而"用户还在说的时候链路在干什么"恰恰是解释尾部的关键一段。
        self.root: Span | None = None
        self.window: Span | None = None

    # ---- 建树 ----------------------------------------------------------

    def _new_id(self) -> str:
        return f"{next(self._seq):04x}"

    def start(
        self,
        name: str,
        parent: Span | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> Span:
        """起一个 span。

        `attrs` 用**显式命名参数**而不是 `**attrs`：后者会把任何叫 `attrs` 的关键字
        整包吞成 `{'attrs': {...}}`（2026-10-17 真实踩到，roundtrip 测试才暴露），
        而且显式参数让 mypy 能查 attrs 的类型——追踪最怕的就是"看着埋了其实埋错键"。
        """
        span = Span(
            name=name,
            span_id=self._new_id(),
            parent_id=parent.span_id if parent is not None else "",
            start_ms=self._clock.elapsed_ms(),
            attrs=dict(attrs) if attrs else {},
        )
        self._spans.append(span)
        return span

    def start_root(self, name: str, attrs: Mapping[str, object] | None = None) -> Span:
        """建根 span。一轮一个；重复调用会改 root 指向，旧 span 仍留在树上（不静默丢弃）。"""
        span = self.start(name, None, attrs)
        self.root = span
        return span

    def start_window(self, name: str, attrs: Mapping[str, object] | None = None) -> Span:
        """建**口径B 窗口** span：root 的孩子（没有 root 时退化成根）。

        延迟分析的所有分格都挂在它下面，而 root 负责兜住"用户说话期间"那一段。
        """
        parent = self.root if self.root is not None else None
        span = self.start(name, parent, attrs)
        self.window = span
        return span

    def finish(self, span: Span, attrs: Mapping[str, object] | None = None) -> Span:
        """收尾。**已闭合的不再改写**——避免"两个出口都 finish 一遍"时后者把前者抹掉。"""
        if span.open:
            span.end_ms = self._clock.elapsed_ms()
            if attrs:
                span.attrs.update(attrs)
        return span

    def record(
        self,
        name: str,
        start_ms: float,
        end_ms: float,
        parent: Span | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> Span:
        """事后补记：给「才知道起止」的那类区间用（如 outbox → 真正下发的排队等待，
        只有发送那一刻才量得出它排了多久）。"""
        span = Span(
            name=name,
            span_id=self._new_id(),
            parent_id=parent.span_id if parent is not None else "",
            start_ms=start_ms,
            end_ms=end_ms,
            attrs=dict(attrs) if attrs else {},
        )
        self._spans.append(span)
        return span

    def finish_root(self, attrs: Mapping[str, object] | None = None) -> None:
        """闭合根 span（整轮结束）。重复调用无效——闭合后的值不接受改写。"""
        if self.root is not None:
            self.finish(self.root, attrs)

    def record_perf(
        self,
        name: str,
        t0_perf: float,
        t1_perf: float,
        parent: Span | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> Span:
        """用 `time.perf_counter()` 原值补记 span。

        与 `record` 的区别只在时间戳的坐标系：这里的入参是 perf_counter 绝对值
        （provider 在别的线程里记的），内部经**时钟**换算成会话 ms。必须走时钟
        统一换算——provider 与 tracer 各自找原点相减的话 span 会整体漂移，
        而账面上看不出来。
        """
        return self.record(
            name,
            self._clock.ms_at(t0_perf),
            self._clock.ms_at(t1_perf),
            parent,
            attrs,
        )

    def finish_window(self, attrs: Mapping[str, object] | None = None) -> None:
        """闭合口径B 窗口（首音出去的那一刻）。重复调用无效。"""
        if self.window is not None:
            self.finish(self.window, attrs)

    @asynccontextmanager
    async def scoped(
        self,
        name: str,
        parent: Span | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> AsyncIterator[Span]:
        """`async with tracer.scoped("llm.request", root, attrs={...}) as sp:` —— 异常也留痕。"""
        span = self.start(name, parent, attrs)
        try:
            yield span
        finally:
            self.finish(span)

    # ---- 读树 ----------------------------------------------------------

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def reset(self) -> None:
        """新一轮：清空树。**不换 trace_id**——一次会话的若干轮要能串起来看分布。"""
        self._spans.clear()
        self.root = None
        self.window = None

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._spans]

    def children(self, parent: Span) -> list[Span]:
        return [s for s in self._spans if s.parent_id == parent.span_id]


# ---- 归因分析 ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decomposition:
    """根区间的三分账：覆盖 / 重叠 / 未归因。

    `total = covered + unattributed` 是恒等式；`busy = covered + overlap` 是子区间时长之和。
    """

    name: str
    total_ms: float
    covered_ms: float  # 子 span 在时间轴上的并集（重叠只算一次）
    busy_ms: float  # 子 span 时长之和（重叠算多次）
    unattributed_ms: float  # total − covered：**非 0 就是还有东西没埋到**
    children: tuple[tuple[str, float], ...]

    @property
    def overlap_ms(self) -> float:
        return self.busy_ms - self.covered_ms

    @property
    def covered_pct(self) -> float:
        return self.covered_ms / self.total_ms * 100 if self.total_ms > 0 else 0.0

    @property
    def unattributed_pct(self) -> float:
        return self.unattributed_ms / self.total_ms * 100 if self.total_ms > 0 else 0.0


def union_ms(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> float:
    """若干区间与窗口 [lo,hi] 交集的并集长度（毫秒）。"""
    clipped = [(max(a, lo), min(b, hi)) for a, b in intervals if b >= 0]
    clipped = [(a, b) for a, b in clipped if b > a]
    if not clipped:
        return 0.0
    clipped.sort()
    total = 0.0
    cur_a, cur_b = clipped[0]
    for a, b in clipped[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return total + (cur_b - cur_a)


def decompose(root: Span, spans: Sequence[Span]) -> Decomposition:
    """把根 span 的时长拆到直接子 span 上。**没埋到的那一层由残差兜住**，不会静默消失。"""
    if root.dur_ms < 0:
        return Decomposition(root.name, MISSING, MISSING, MISSING, MISSING, ())
    lo, hi = root.start_ms, root.end_ms
    kids = [s for s in spans if s.parent_id == root.span_id and s.dur_ms >= 0]
    covered = union_ms([(s.start_ms, s.end_ms) for s in kids], lo, hi)
    busy = sum(s.dur_ms for s in kids)
    return Decomposition(
        name=root.name,
        total_ms=root.dur_ms,
        covered_ms=covered,
        busy_ms=busy,
        unattributed_ms=root.dur_ms - covered,
        children=tuple((s.name, s.dur_ms) for s in kids),
    )


def exclusive_times(spans: Sequence[Span], lo: float, hi: float) -> dict[str, float]:
    """把窗口内每一毫秒指派给「包住它的**最短** span」——flame graph 的自有时间口径。

    为什么不用父子边算自有时间：**树边未必反映时间嵌套**。serial 里 TTS 是在
    `llm.astream` 循环体内 await 的（见 scheduler/serial.py），时间上确实嵌在
    llm.request 里，但埋点时为了可读把 tts.synth 挂在根下。按边扣会漏扣，
    按时间轴算才不重不漏（2026-10-17 实测：按边算 Σ自有 比口径B 多出近一倍）。

    返回值按 span_id 索引。不变量：`Σ自有 == 覆盖并集`（每个时刻只归一个 span），
    并发重叠的毫秒会算给"更内层"的那个，**不会**重复计数——所以加总永远不超标。
    """
    active = [s for s in spans if s.dur_ms >= 0]
    own = {s.span_id: 0.0 for s in active}
    if not active:
        return own
    # 边界必须**夹在窗口内**：直接取 min(end, hi) 会把"整段在窗口之前结束"的 span
    # 塞进来一个 < lo 的边界，段落总长就会超出窗口（2026-10-17 实测 Σ自有 比口径B
    # 多出 54ms，全部来自判停前那 60 个 uplink.frame）。
    bounds = sorted(
        {
            min(max(v, lo), hi)
            for v in {lo, hi} | {s.start_ms for s in active} | {s.end_ms for s in active}
        }
    )
    # strict=False：右参短 1 项是故意的（相邻配对）
    for a, b in zip(bounds, bounds[1:], strict=False):
        if b <= a:
            continue
        inner = [s for s in active if s.start_ms <= a and s.end_ms >= b]
        if not inner:
            continue  # 这一段没有任何 span 盖住 → 未归因残差
        # 最短区间 = 最内层；并列时用 span_id 打破平局，保证同一份输入同一份输出
        winner = min(inner, key=lambda s: (s.end_ms - s.start_ms, s.span_id))
        own[winner.span_id] += b - a
    return own


def self_time_map(spans: Sequence[Span], lo: float, hi: float) -> dict[str, float]:
    """同名 span 的自有时间汇总（供报告直接打印）。"""
    own = exclusive_times(spans, lo, hi)
    out: dict[str, float] = {}
    for s in spans:
        if s.span_id in own:
            out[s.name] = out.get(s.name, 0.0) + own[s.span_id]
    return out


def window_breakdown(tracer: Tracer, *, min_ms: float = 0.5) -> dict[str, Any] | None:
    """口径B 窗口内的自有时间汇总——服务端算好，页面/账单直接用。

    为什么不在前端算：`exclusive_times` 的口径（每毫秒只归**最内层** span、并发重叠
    不重复计）是这一层的关键不变量；前端再实现一遍必然漂。这里只把结果交出去：
    按名字汇总、降序、丢掉 < min_ms 的噪声格子，并给出**未归因残差**（非 0 = 还有
    东西没埋到）。

    没有窗口（echo / 异常轮）返回 None——页面据此不画表，而不是画一张 0ms 的假表。
    """
    window = tracer.window
    if window is None or window.dur_ms < 0:
        return None
    spans = tracer.spans
    own = exclusive_times(spans, window.start_ms, window.end_ms)
    by_id = {s.span_id: s for s in spans}
    per_name: dict[str, float] = {}
    for sid, ms in own.items():
        name = by_id[sid].name
        per_name[name] = per_name.get(name, 0.0) + ms
    items = [
        {"name": name, "ms": round(ms, 1)}
        for name, ms in sorted(per_name.items(), key=lambda kv: -kv[1])
        if ms >= min_ms and name not in {"turn", "turn.caliber_b"}
    ]
    # 未归因 = 窗口 span **自己** 占到的毫秒（没被任何子 span 盖住的那部分）。
    # 不能用 total − Σ各格：exclusive_times 会把未覆盖的时段归给「覆盖它且最短」的 span，
    # 也就是窗口自己——减出来永远是 0（2026-10-17 首版就错在这里，单测才发现）。
    return {
        "total_ms": round(window.dur_ms, 1),
        "unattributed_ms": round(own.get(window.span_id, 0.0), 1),
        "items": items,
    }


@dataclass(frozen=True, slots=True)
class SpanStats:
    name: str
    n: int
    total_ms: float
    p50_ms: float
    max_ms: float

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.n if self.n else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return MISSING
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(spans: Sequence[Span]) -> list[SpanStats]:
    """同名 span 聚合。每帧一次的那种 span（如 `asr.frame`）靠它压成一个数。"""
    buckets: dict[str, list[float]] = {}
    order: list[str] = []
    for s in spans:
        if s.dur_ms < 0:
            continue
        if s.name not in buckets:
            buckets[s.name] = []
            order.append(s.name)
        buckets[s.name].append(s.dur_ms)
    out: list[SpanStats] = []
    for name in order:
        vals = buckets[name]
        # 分位用线性插值（与 report_turns.py 同源口径）；配对-pairwise 排序由 sort 保证
        out.append(
            SpanStats(
                name=name,
                n=len(vals),
                total_ms=sum(vals),
                p50_ms=statistics.median(vals),
                max_ms=max(vals),
            )
        )
    return out


def parse_spans(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Span], dict[str, Span]]:
    """从账单 JSONL 的 `spans` 字段还原 Span 列表，并返回 (根 span 名 → span) 映射。"""
    out: list[Span] = []
    roots: dict[str, Span] = {}
    for row in rows:
        span = Span(
            name=str(row["name"]),
            span_id=str(row["span_id"]),
            parent_id=str(row.get("parent_id", "")),
            start_ms=float(row["start_ms"]),
            end_ms=float(row.get("end_ms", MISSING)),
            attrs={str(k)[2:]: v for k, v in row.items() if str(k).startswith("a.")},
        )
        out.append(span)
        if not span.parent_id and span.name.endswith(ROOT_SUFFIX):
            roots[span.span_id] = span
    return out, roots


def phases_of(provider: object) -> Mapping[str, float]:
    """取 provider 的 phase 时刻表（约定属性名 `last_phases`，perf_counter 原值）。

    没有就返回空表：本地引擎（melo/kokoro）没有可分的事件，**空表比编几个键
    诚实**——调用方据此跳过子 span，而不是记一段 0ms 的假账。
    """
    marks = getattr(provider, "last_phases", None)
    return marks if isinstance(marks, Mapping) else {}


def phase_span(
    tracer: Tracer | None,
    name: str,
    phases: Mapping[str, float],
    lo_key: str,
    hi_key: str,
    parent: Span | None = None,
    attrs: Mapping[str, object] | None = None,
) -> Span | None:
    """从 phase 时刻表补记一个子 span；缺任一端点返回 None。

    例：DashScope TTS 的 `on_open` 没 fire 就不记 `tts.connect`，只记有证据的
    那一段。深拆的原则是**缺证据的格子宁可空着，也不能填 0 假装测过**。
    """
    if tracer is None:
        return None
    a = phases.get(lo_key)
    b = phases.get(hi_key)
    if a is None or b is None or b < a:
        return None
    return tracer.record_perf(name, a, b, parent, attrs)
