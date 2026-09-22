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


def test_default_gating_is_semantic_not_conservative() -> None:
    """conservative 要等满 500ms 静音窗，口径B 被托底在 500ms 以上——300ms 就不可达了。

    这条断言防的是「有人把它改回保守档，300ms 悄悄变成数学上不可能」。
    """
    assert SessionBegin().gating is GatingMode.SEMANTIC


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
