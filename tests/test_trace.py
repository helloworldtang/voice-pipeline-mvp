"""追踪层：会计恒等式与 attrs 往返 **就是对延迟归因可信度的兜底**。

为什么值得写这几条而不是直接靠端到端测试端到端：分解错 1ms 不会报错，
只会让人把优化力气投到错的那一格上——那是比崩溃更贵的失败形态。
"""

from __future__ import annotations

from voice_pipeline.core.scheduler.base import AudioFragment
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.metrics.trace import (
    MISSING,
    Span,
    Tracer,
    decompose,
    exclusive_times,
    parse_spans,
    stats,
    union_ms,
)
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.session import Session


def test_unfinished_span_reports_missing_not_zero() -> None:
    """未闭合的 span 必须是 -1：0 看起来像正常值，会把「没测到」伪装成「测得 0」。"""
    tracer = Tracer("abc")
    span = tracer.start("x")
    assert span.dur_ms == MISSING
    assert span.open


def test_decomposition_is_an_identity() -> None:
    """核心恒等式：total = 覆盖并集 + 未归因残差。埋全了残差必须为 0。"""
    tracer = Tracer("abc")
    root = tracer.start_root("turn.caliber_b")
    a = tracer.start("asr.finalize", root)
    tracer.finish(a)
    b = tracer.start("llm.request", root)
    tracer.finish(b)
    tracer.finish(root)

    dec = decompose(root, tracer.spans)
    assert dec.total_ms > 0
    assert dec.covered_ms + dec.unattributed_ms == dec.total_ms
    # 两段首尾相接（a 结束在 b 起点附近）→ 覆盖 ≈ 总长，残差接近 0
    assert abs(dec.unattributed_ms) < 1.0


def test_overlap_is_counted_once_and_residual_surfaces_the_gap() -> None:
    """并发（区间重叠）时，和会重复计数、并集不会——残差负责把没埋到的那段落出来。"""
    spans = [
        Span("root", "1", "", 0.0, 100.0),
        Span("a", "2", "1", 0.0, 40.0),
        Span("b", "3", "1", 30.0, 60.0),  # 与 a 重叠 10ms
        Span("c", "4", "1", 70.0, 80.0),  # 60~70 这段没有子 span
    ]
    root = spans[0]
    dec = decompose(root, spans)
    assert dec.busy_ms == 40.0 + 30.0 + 10.0
    assert dec.covered_ms == 70.0  # [0,60] 连续 + [70,80]；重叠的 10ms 只算一次
    assert dec.overlap_ms == 10.0
    assert abs(dec.unattributed_ms - 30.0) < 1e-9  # 60~70 与 80~100 都没有子 span
    assert union_ms([(0.0, 200.0)], 50.0, 60.0) == 10.0  # 裁剪到窗口


def test_exclusive_times_assign_each_ms_once() -> None:
    """flame graph 口径：Σ自有 == 覆盖，且**窗口外的 span 不得把边界带进来**。

    2026-10-17 实测 bug：判停前那 60 个 uplink.frame 整段在窗口之前结束，
    未夹紧的边界让 Σ自有 比口径B 多出 54ms——账面对不上，而且错的方向是"高估"。
    """
    spans = [
        Span("turn", "1", "", 0.0, 3000.0),
        Span("turn.caliber_b", "2", "1", 500.0, 1500.0),
        Span("llm.request", "3", "2", 505.0, 2900.0),  # 跨过窗口右边界
        Span("llm.ttft", "4", "3", 505.0, 800.0),
        Span("tts.synth", "5", "3", 900.0, 1800.0),  # 也跨过右边界
        Span("uplink.frame", "6", "1", 10.0, 20.0),  # 整段在窗口之前
        Span("uplink.frame", "7", "1", 1520.0, 1530.0),  # 整段在窗口之后
    ]
    own = exclusive_times(spans, 500.0, 1500.0)
    by_name = {next(s.name for s in spans if s.span_id == sid): ms for sid, ms in own.items()}
    assert abs(sum(own.values()) - 1000.0) < 1e-9
    assert abs(by_name["llm.ttft"] - 295.0) < 1e-9
    assert abs(by_name["tts.synth"] - 600.0) < 1e-9  # 900~1500 在窗口内
    assert by_name["uplink.frame"] == 0.0  # 窗口外的帧一分不算
    assert abs(by_name["turn.caliber_b"] - 105.0) < 1e-9  # 505~610 之间没有更细的 span


def test_exclusive_times_handle_overlap_without_double_counting() -> None:
    """两个 span 部分重叠时，重叠段归更内层（更短）的那个，不重复计。"""
    spans = [
        Span("root", "1", "", 0.0, 100.0),
        Span("a", "2", "1", 0.0, 40.0),
        Span("b", "3", "1", 30.0, 60.0),  # 与 a 重叠 10ms，b 更短 → b 拿这段
    ]
    own = exclusive_times(spans, 0.0, 100.0)
    assert abs(own["2"] - 30.0) < 1e-9  # a
    assert abs(own["3"] - 30.0) < 1e-9  # b 拿重叠的 10ms（区间更短=更内层）
    assert abs(own["1"] - 40.0) < 1e-9  # 60~100 没人盖
    assert abs(sum(own.values()) - 100.0) < 1e-9
    tracer = Tracer("abc")
    for i in range(5):
        s = tracer.start("asr.frame", attrs={"thread": 1})
        s.end_ms = s.start_ms + float(i)
    agg = {st.name: st for st in stats(tracer.spans)}
    assert agg["asr.frame"].n == 5
    assert agg["asr.frame"].mean_ms >= 0

    rows = tracer.to_list()
    restored, roots = parse_spans(rows)
    assert restored[0].attrs["thread"] == 1
    assert not roots  # 没有根 suffix → 不认作根（避免报告脚本把普通 span 当窗口）


def test_stats_and_json_roundtrip_keep_attributes() -> None:
    tracer = Tracer("abc")
    for i in range(5):
        s = tracer.start("asr.frame", attrs={"thread": 1})
        s.end_ms = s.start_ms + float(i)
    agg = {st.name: st for st in stats(tracer.spans)}
    assert agg["asr.frame"].n == 5
    assert agg["asr.frame"].mean_ms >= 0

    rows = tracer.to_list()
    restored, roots = parse_spans(rows)
    assert restored[0].attrs["thread"] == 1
    assert not roots  # 没有根 suffix → 不认作根（避免报告脚本把普通 span 当窗口）


def test_attrs_keyword_is_not_swallowed() -> None:
    """**显式 attrs 参数**的理由：`**attrs` 会把叫 `attrs` 的关键字整包吞掉。

    2026-10-17 实测踩到：`tracer.start("x", attrs={...})` 会得到
    `attrs == {"attrs": {...}}`——埋点看着写了，roundtrip 时键名对不上，取不到值。
    """
    tracer = Tracer("abc")
    s = tracer.start("x", attrs={"attrs": {"nested": 1}})
    assert s.attrs == {"attrs": {"nested": 1}}


def test_reset_turn_keeps_the_tree_until_session_rolls_it_over() -> None:
    """reset_turn **不能**清树：判停前的上行帧证据比"新一轮干净开始"更值钱。

    2026-10-17：清过一次，结果 trace 里 0 个 uplink.frame，而 mic 全程在收音——
    看起来"上行零开销"，其实是证据被自己抹了。清树的时机只能是下一轮第一帧
    （见 Session.on_audio）。
    """
    metrics = TurnMetrics(clock=SessionClock())
    metrics.tracer = Tracer("t1", metrics.clock)
    metrics.tracer.start_root("turn")
    metrics.tracer.start("uplink.frame")
    metrics.reset_turn()
    assert len(metrics.tracer.spans) == 2  # 树还在
    metrics.tracer.finish_root()
    metrics.tracer.reset()  # 下一轮第一帧的等价动作
    assert metrics.tracer.spans == []
    assert metrics.tracer.root is None


async def test_serial_turn_produces_root_span_closing_at_first_audio() -> None:
    """端到端契约：一轮跑完，树里必须有根 span 且它在首音频处闭合、带 stage 子 span。"""
    session = Session(
        mode="serial",
        asr=SimAsr(transcript="上海今天多少度"),
        llm=SimLlm(reply="今天晴，22 度。适合出门。"),
        tts=SimTts(chunk_per_sentence=2),
    )
    tracer = session.metrics.tracer
    assert tracer is not None
    await session.on_audio(b"\x00" * 3200)

    first_audio_seen = False
    async for item in session.finish():
        if isinstance(item, AudioFragment):
            if not first_audio_seen:
                first_audio_seen = True
                window = tracer.window
                # 口径B 窗口必须在首片**那一时刻**就闭合：晚闭合会把后续 latency 算进口径B
                assert window is not None and not window.open
                # 外层 turn 此刻还没闭合（要等 turn.end），否则"轮后等待"就没处可查
                assert tracer.root is not None and tracer.root.open
            break

    names = {s.name for s in tracer.spans}
    assert {
        "turn",
        "turn.caliber_b",
        "asr.finalize",
        "llm.request",
        "llm.ttft",
        "tts.synth",
    } <= names
    window = tracer.window
    assert window is not None and window.dur_ms >= 0
    # 判停前的上行帧也要在树上：reset 会把树清掉，所以 turn 根必须建在第一帧
    assert any(s.name == "uplink.frame" for s in tracer.spans)
    dec = decompose(window, tracer.spans)
    # sim 链路是串行的：覆盖之后残差应当接近 0（有值时也是真有东西没埋到，断言量级而非等号）
    assert dec.unattributed_ms < max(5.0, dec.total_ms * 0.2)


async def test_uplink_frames_are_traced() -> None:
    """上行每帧必须有 span：同事件循环里的 CPU 开销靠它才可见。"""
    session = Session(
        mode="serial",
        asr=SimAsr(transcript="喂"),
        llm=SimLlm(reply="你好。"),
        tts=SimTts(chunk_per_sentence=1),
    )
    tracer = session.metrics.tracer
    assert tracer is not None
    await session.on_audio(b"\x00" * 3200)
    names = [s.name for s in tracer.spans]
    assert "uplink.frame" in names
    assert "asr.push_audio" in names
    # 父子关系必须是树，不能是平铺（否则算不出自有时间）
    frame = next(s for s in tracer.spans if s.name == "uplink.frame")
    assert any(s.parent_id == frame.span_id for s in tracer.spans)
