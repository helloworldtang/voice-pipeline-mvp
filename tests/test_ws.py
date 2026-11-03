"""WS 集成：真实走 WebSocket（含序列化与网络开销），契约断言 wire 字段。"""

from __future__ import annotations

import json
from typing import Any

import pytest
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


def test_lifespan_closes_pool_on_shutdown(monkeypatch: Any) -> None:
    """停机时 lifespan 必须调 pool.aclose()（2026-11-01 审查缺陷 4）：
    `VOICE_LLM_PROCESS=1` 的 MLX 子进程不 terminate 就是孤儿，常驻吃 Metal+内存。"""
    from voice_pipeline.config import Settings

    app = create_app(Settings())
    pool = app.state.pool
    calls: list[int] = []

    async def _spy() -> None:
        calls.append(1)

    monkeypatch.setattr(pool, "aclose", _spy)
    with TestClient(app):
        assert calls == [], "运行中不该收尾"
    assert calls == [1], "停机时必须收尾池内的重资源"


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


def test_ws_interrupt_without_turn_is_silently_dropped() -> None:
    """无活动轮次的 interrupt 静默丢弃：客户端本地打断是快路径，常在轮次
    结束后由播放尾音再触发一次——报错只会污染验收界面（2026-10-04 实测）。
    契约：不回错误帧、连接保持可用。"""
    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "interrupt", "reason": "tail-audio"}))
        # 没有错误帧回来说明被吞掉；连接仍活着（echo 一帧验证）
        ws.send_text(SessionBegin(mode="echo").model_dump_json())
        assert ws.receive_json()["type"] == "session.ready"


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


def test_ws_token_gate() -> None:
    """静态 token 三态：不设=开放（旧测试全在此态）；设了=begin 必须带对 token。"""
    from voice_pipeline.config import Settings

    app = create_app(Settings(ws_token="s3cret"))
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()  # 首帧 ready 先于任何客户端帧，不受 token 影响

        # 缺 token → error + 1008
        ws.send_text(SessionBegin(mode="echo").model_dump_json())
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "unauthorized"

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        # 错 token → 同样拒绝
        ws.send_text(SessionBegin(mode="echo", token="wrong").model_dump_json())
        assert ws.receive_json()["code"] == "unauthorized"

    with TestClient(app).websocket_connect("/ws") as ws:
        ws.receive_json()
        # 对 token → 正常重建会话
        ws.send_text(SessionBegin(mode="echo", token="s3cret").model_dump_json())
        ready = ws.receive_json()
        assert ready["type"] == "session.ready"


def test_session_begin_token_field_survives_wire_parsing() -> None:
    """pydantic 对多余字段是忽略——字段名拼错会被静默吞掉、鉴权莫名失败。
    这条 round-trip 把 token 字段钉在 wire 契约上。"""
    assert SessionBegin.model_validate({"type": "session.begin", "token": "x"}).token == "x"
    assert SessionBegin(mode="echo").model_dump()["token"] is None


def test_web_client_is_served_and_healthz_still_json() -> None:
    """静态挂载：GET / 出 HTML 页面；显式路由（/healthz）不被 mount("/") 吞掉。

    Starlette 按注册顺序匹配——mount 若先于显式路由注册，/healthz 会被
    StaticFiles 接走返回 404，bench 的就绪探测就废了。
    """
    client = TestClient(create_app())
    page = client.get("/")
    assert page.status_code == 200
    assert "voice-pilot" in page.text
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/app.js").status_code == 200


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


# ---- 引擎档（URL ?profile=cloud|latency）--------------------------------------------


def _light_pool(monkeypatch: Any, engines: list[str], tts_names: list[str]) -> None:
    """把槽的重资产构建与 LLM 组装换成记录器：不加载模型、不联网（CI 零模型零网络）。

    记录的是**传给组装层的引擎名**——档位是否真的展开到引擎，从这采证。
    """
    from voice_pipeline.providers import pool as pool_mod
    from voice_pipeline.providers.asr.sim_asr import SimAsr
    from voice_pipeline.providers.llm.sim_llm import SimLlm
    from voice_pipeline.providers.tts.sim_tts import SimTts

    def fake_build(slot: Any, asr_engine: str, tts_name: str) -> None:
        tts_names.append(tts_name)
        slot._recognizer = object()
        slot._tts = SimTts()
        slot._tts_engine_name = tts_name

    monkeypatch.setattr(pool_mod._Slot, "_build", fake_build)
    monkeypatch.setattr(pool_mod._Slot, "wrap_asr", lambda self, asr_engine: SimAsr())
    monkeypatch.setattr(
        pool_mod, "build_llm", lambda engine, model="m": (engines.append(engine), SimLlm())[1]
    )


def test_ws_bad_profile_rejected_before_ready(monkeypatch: Any) -> None:
    """未知档名：第一帧即 error(bad_profile)，不发 session.ready，随接关闭。"""
    from starlette.websockets import WebSocketDisconnect

    with TestClient(create_app()).websocket_connect("/ws?profile=nope") as ws:
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "bad_profile"
        assert "unknown profile" in err["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_cloud_profile_without_key_rejected(monkeypatch: Any) -> None:
    """cloud 档缺 DASHSCOPE_API_KEY：明确拒绝（profile_unavailable），不静默降级 env 档。"""
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with TestClient(create_app()).websocket_connect("/ws?profile=cloud") as ws:
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "profile_unavailable"
        assert "DASHSCOPE_API_KEY" in err["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_latency_profile_without_mlx_rejected(monkeypatch: Any) -> None:
    """latency 档缺 mlx 依赖（ECS 2C 形态）：明确拒绝，不静默换引擎。"""
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr("voice_pipeline.providers.factory._mlx_available", lambda: False)
    monkeypatch.setenv("XFYUN_TTS_APPID", "app")
    monkeypatch.setenv("XFYUN_TTS_API_KEY", "key")
    monkeypatch.setenv("XFYUN_TTS_API_SECRET", "secret")
    with TestClient(create_app()).websocket_connect("/ws?profile=latency") as ws:
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "profile_unavailable"
        assert "mlx" in err["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_latency_profile_builds_mlx_and_xfyun(monkeypatch: Any) -> None:
    """档位生效采证：latency 档下 LLM 组装收到 mlx、槽按 xfyun 建重资产、ready 回显档名。"""
    monkeypatch.setattr("voice_pipeline.providers.factory._mlx_available", lambda: True)
    monkeypatch.setattr(
        "voice_pipeline.providers.llm.mlx_llm._local_model_path", lambda model: "/models/qwen"
    )
    monkeypatch.setenv("XFYUN_TTS_APPID", "app")
    monkeypatch.setenv("XFYUN_TTS_API_KEY", "key")
    monkeypatch.setenv("XFYUN_TTS_API_SECRET", "secret")
    engines: list[str] = []
    tts_names: list[str] = []
    _light_pool(monkeypatch, engines, tts_names)

    with TestClient(create_app()).websocket_connect("/ws?profile=latency") as ws:
        ready = ws.receive_json()
        assert ready["profile"] == "latency"
        ws.send_text(SessionBegin(mode="real-serial").model_dump_json())
        rebound = ws.receive_json()
        assert rebound["type"] == "session.ready"
        assert rebound["profile"] == "latency"
        assert engines == ["mlx"], "LLM 必须按档展开为 mlx，而非 env/auto 默认"
        assert tts_names == ["xfyun"], "槽的重资产必须按档用 xfyun 建"


def test_ws_no_profile_param_is_env_behavior(monkeypatch: Any) -> None:
    """裸 /ws 零变化：无档覆盖、引擎走 env/默认——systemd 部署与 bench 的后向兼容钉死。"""
    monkeypatch.setenv("VOICE_PIPELINE_TTS_ENGINE", "melo")
    engines: list[str] = []
    tts_names: list[str] = []
    _light_pool(monkeypatch, engines, tts_names)

    with TestClient(create_app()).websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["profile"] is None
        ws.send_text(SessionBegin(mode="real-serial").model_dump_json())
        rebound = ws.receive_json()
        assert rebound["type"] == "session.ready"
        assert rebound["profile"] is None
        assert engines == ["auto"], "无档时不得有 LLM 引擎覆盖"
        assert tts_names == ["melo"], "TTS 仍走 VOICE_PIPELINE_TTS_ENGINE"


def test_ws_profile_overrides_env_tts(monkeypatch: Any) -> None:
    """优先级钉死：显式档压过 env（env=melo + latency 档 → 槽按 xfyun 建）。"""
    monkeypatch.setattr("voice_pipeline.providers.factory._mlx_available", lambda: True)
    monkeypatch.setattr(
        "voice_pipeline.providers.llm.mlx_llm._local_model_path", lambda model: "/models/qwen"
    )
    monkeypatch.setenv("VOICE_PIPELINE_TTS_ENGINE", "melo")
    monkeypatch.setenv("XFYUN_TTS_APPID", "app")
    monkeypatch.setenv("XFYUN_TTS_API_KEY", "key")
    monkeypatch.setenv("XFYUN_TTS_API_SECRET", "secret")
    engines: list[str] = []
    tts_names: list[str] = []
    _light_pool(monkeypatch, engines, tts_names)

    with TestClient(create_app()).websocket_connect("/ws?profile=latency") as ws:
        ws.receive_json()
        ws.send_text(SessionBegin(mode="real-serial").model_dump_json())
        assert ws.receive_json()["type"] == "session.ready"
        assert tts_names == ["xfyun"]
