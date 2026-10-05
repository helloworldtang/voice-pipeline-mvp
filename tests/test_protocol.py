"""wire 契约测试：字段名与 type 值就是对外承诺，改动必须显式。"""

from __future__ import annotations

import json

import pytest

from voice_pipeline.core.eou import GatingMode
from voice_pipeline.server.protocol import (
    ErrorEvent,
    Interrupt,
    ProtocolError,
    SessionBegin,
    SessionClosed,
    SessionEnd,
    SessionReady,
    dump_event,
    parse_client_frame,
)


def test_session_begin_defaults() -> None:
    frame = parse_client_frame(SessionBegin().model_dump_json())
    assert isinstance(frame, SessionBegin)
    assert frame.type == "session.begin"
    assert frame.sample_rate == 16000


def test_interrupt_carries_reason() -> None:
    frame = parse_client_frame(Interrupt(reason="barge-in").model_dump_json())
    assert isinstance(frame, Interrupt)
    assert frame.reason == "barge-in"


def test_session_end() -> None:
    frame = parse_client_frame(SessionEnd().model_dump_json())
    assert isinstance(frame, SessionEnd)


def test_bad_frame_raises_protocol_error() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        parse_client_frame("not json")
    assert excinfo.value.code == "bad_frame"


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        parse_client_frame(json.dumps({"type": "session.nope"}))


def test_all_scheduler_modes_are_selectable_on_the_wire() -> None:
    """实验组必须能在 WS 上被点名——选不到就等于 300ms 那条路只存在于脚本里。"""
    for mode in ("echo", "serial", "pipelined", "real", "real-serial", "real-pipelined"):
        raw = json.dumps({"type": "session.begin", "mode": mode})
        frame = parse_client_frame(raw)
        assert isinstance(frame, SessionBegin)
        assert frame.mode == mode


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        parse_client_frame(json.dumps({"type": "session.begin", "mode": "turbo"}))


def test_default_gating_is_conservative_after_human_trials() -> None:
    """默认档是 conservative（第四次定档，2026-10-04 真人验收 D5）。

    定档史：semantic 被否两次（L2 终顿黏性锁存，句内停顿 250~350ms 抢话
    20~40%）→ neural 上位（合成语音受控实验 70.4% 检出/4.6% 抢话）→
    **真人验收否掉 neural 默认**：半秒句中停顿下抢话 3/3（"我想"/"帮我查一下"/
    "我等会儿"全部在停顿处抢答），100ms 确认窗扛不住真人停顿。

    防回退：改回 neural 默认前先跑真人抢话三连测（手册剧本 6-8 句）——
    合成语音的内部静音与真人停顿不是同一分布，这正是 D5 的教训。
    """
    assert SessionBegin().gating is GatingMode.CONSERVATIVE


def test_neural_gating_travels_on_the_wire() -> None:
    frame = parse_client_frame(
        SessionBegin(mode="real-serial", gating=GatingMode.NEURAL).model_dump_json()
    )
    assert isinstance(frame, SessionBegin)
    assert frame.gating is GatingMode.NEURAL


def test_gating_and_grace_travel_on_the_wire() -> None:
    frame = parse_client_frame(
        SessionBegin(
            mode="pipelined", gating=GatingMode.AGGRESSIVE, grace_ms=300.0
        ).model_dump_json()
    )
    assert isinstance(frame, SessionBegin)
    assert frame.gating is GatingMode.AGGRESSIVE
    assert frame.grace_ms == 300.0


def test_server_events_wire_fields() -> None:
    ready = json.loads(dump_event(SessionReady(session_id="s1", trace_id="t1")))
    assert ready == {"type": "session.ready", "session_id": "s1", "trace_id": "t1"}

    closed = json.loads(dump_event(SessionClosed(session_id="s1", frames_echoed=7)))
    assert closed == {"type": "session.closed", "session_id": "s1", "frames_echoed": 7}

    err = json.loads(dump_event(ErrorEvent(code="bad_frame", message="boom")))
    assert err == {"type": "error", "code": "bad_frame", "message": "boom"}
