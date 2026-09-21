"""WS 集成：真实走 WebSocket（含序列化与网络开销），契约断言 wire 字段。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from voice_pipeline.server.app import create_app
from voice_pipeline.server.protocol import Interrupt, SessionEnd


def test_healthz() -> None:
    client = TestClient(create_app())
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ws_handshake_then_echo_audio() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["session_id"]
        assert ready["trace_id"]

        ws.send_bytes(b"\x01\x02\x03\x04")
        assert ws.receive_bytes() == b"\x01\x02\x03\x04"

        ws.send_text(SessionEnd().model_dump_json())
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["frames_echoed"] == 1


def test_ws_counts_multiple_echoed_frames() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(3):
            ws.send_bytes(b"\x00\x01")
            ws.receive_bytes()
        ws.send_text(SessionEnd().model_dump_json())
        assert ws.receive_json()["frames_echoed"] == 3


def test_ws_reports_bad_frame_and_keeps_session_alive() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_text("not json")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "bad_frame"

        # 会话未被异常打断：仍能正常 echo
        ws.send_bytes(b"\x07")
        assert ws.receive_bytes() == b"\x07"


def test_ws_interrupt_without_turn_reports_no_active_turn() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "interrupt", "reason": "barge-in"}))
        err = ws.receive_json()
        assert err["code"] == "no_active_turn"


def test_interrupt_model_roundtrip() -> None:
    assert json.loads(Interrupt(reason="stop").model_dump_json())["reason"] == "stop"
