"""WS 集成：真实走 WebSocket（含序列化与网络开销），契约断言 wire 字段。"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from voice_pipeline.server.app import create_app
from voice_pipeline.server.protocol import (
    AsrFinal,
    AudioChunk,
    Interrupt,
    SessionBegin,
    SessionEnd,
    TtsSentence,
    TurnEnd,
    TurnStart,
)

PCM_FRAME = b"\x00\x01"


def _open_echo(ws: Any) -> Any:
    ws.send_text(SessionBegin(mode="echo").model_dump_json())
    return ws.receive_json()


def test_healthz() -> None:
    client = TestClient(create_app())
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ws_handshake_then_echo_audio() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()  # 首帧_ready（默认 serial）
        ready = _open_echo(ws)
        assert ready["type"] == "session.ready"
        assert ready["session_id"]

        ws.send_bytes(PCM_FRAME)
        assert ws.receive_bytes() == PCM_FRAME

        ws.send_text(SessionEnd().model_dump_json())
        closed = ws.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["frames_echoed"] == 1


def test_ws_counts_multiple_echoed_frames() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        _open_echo(ws)
        for _ in range(3):
            ws.send_bytes(PCM_FRAME)
            ws.receive_bytes()
        ws.send_text(SessionEnd().model_dump_json())
        assert ws.receive_json()["frames_echoed"] == 3


def test_ws_reports_bad_frame_and_keeps_session_alive() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        _open_echo(ws)

        ws.send_text("not json")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "bad_frame"

        ws.send_bytes(b"\x07")
        assert ws.receive_bytes() == b"\x07"


def test_ws_interrupt_without_turn_reports_no_active_turn() -> None:
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "interrupt", "reason": "barge-in"}))
        err = ws.receive_json()
        assert err["code"] == "no_active_turn"


def test_serial_mode_runs_full_chain_over_ws() -> None:
    """上行音频 → session.end → ASR → LLM → 分句 → TTS → 200ms 分片 → turn.end(账单)。"""
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()  # 默认即 serial
        ws.send_bytes(PCM_FRAME)

        ws.send_text(SessionEnd().model_dump_json())

        types: list[str] = []
        bill: dict[str, float] = {}
        audio_sizes: list[int] = []
        while True:
            payload = ws.receive_json()
            types.append(payload["type"])
            if payload["type"] == "audio.chunk":
                audio_sizes.append(payload["bytes_len"])
                assert len(ws.receive_bytes()) == payload["bytes_len"]
            if payload["type"] == "turn.end":
                bill = payload["metrics"]
                break

        assert types[0] == TurnStart().type == "turn.start"
        assert "asr.final" in types
        assert "tts.sentence" in types
        assert "audio.chunk" in types
        assert types[-1] == "turn.end"

        assert set(audio_sizes) == {9600}  # 24k × 200ms × PCM16
        assert bill["caliber_b_ms"] >= 0
        assert bill["sentences"] >= 1


def test_pipelined_mode_runs_full_chain_over_ws() -> None:
    """实验组必须能在 WS 上跑通，而不只是在 bench 脚本里。

    判据用账单里的 `spec_hit` 键：只有 PipelinedScheduler 会打这个字段，
    serial 的 TurnEnd 不带它。所以这个键出现在 wire 上 = 走的确实是投机调度。
    """
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(SessionBegin(mode="pipelined", grace_ms=100.0).model_dump_json())
        assert ws.receive_json()["type"] == "session.ready"

        ws.send_bytes(PCM_FRAME)
        ws.send_text(SessionEnd().model_dump_json())

        bill: dict[str, float] = {}
        audio_sizes: list[int] = []
        while True:
            payload = ws.receive_json()
            if payload["type"] == "audio.chunk":
                audio_sizes.append(payload["bytes_len"])
                ws.receive_bytes()
            if payload["type"] == "turn.end":
                bill = payload["metrics"]
                break

        assert "spec_hit" in bill, "账单里没有 spec_hit → 走的还是 serial，pipelined 没接进 wire"
        assert "spec_miss" in bill
        assert audio_sizes, "pipelined 也必须真的出音频"
        assert bill["caliber_b_ms"] >= 0


def test_intermediate_event_shapes_are_stable() -> None:
    """事件的 wire 形状就是对外承诺，改动必须显式。"""
    assert AsrFinal(text="上海今天多少度").model_dump() == {
        "type": "asr.final",
        "text": "上海今天多少度",
    }
    assert TtsSentence(seq=1, text="今天晴。").model_dump() == {
        "type": "tts.sentence",
        "seq": 1,
        "text": "今天晴。",
    }
    assert AudioChunk(seq=2, bytes_len=9600).model_dump() == {
        "type": "audio.chunk",
        "seq": 2,
        "bytes_len": 9600,
    }
    assert TurnEnd(metrics={"caliber_b_ms": 1.0}).type == "turn.end"
    assert Interrupt(reason="stop").type == "interrupt"
