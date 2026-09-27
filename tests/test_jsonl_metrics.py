"""账单 JSONL 落盘的测试（差距表「JSONL 事件流落盘未做」的验收）。

钉住三件事：默认关闭不碰盘、开启后每轮一行且字段可追溯、写盘失败不冒泡。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_pipeline.core.eou import GatingMode
from voice_pipeline.metrics.events import TurnMetrics
from voice_pipeline.metrics.jsonl import ENV_PATH, TurnRecorder
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import SessionEnd
from voice_pipeline.server.session import Session


def _session(recorder: TurnRecorder) -> Session:
    return Session(
        mode="serial",
        asr=SimAsr(),
        llm=SimLlm(),
        tts=SimTts(),
        recorder=recorder,
    )


def test_default_recorder_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设环境变量时不得创建文件——CI 与单测零副作用。"""
    monkeypatch.delenv(ENV_PATH, raising=False)
    rec = TurnRecorder.from_env()
    assert rec.enabled is False
    rec.record(
        session_id="s",
        trace_id="t",
        mode="serial",
        turn=1,
        bill={"caliber_b_ms": 1.0},
    )  # 不应抛、不应写


async def test_each_turn_lands_one_jsonl_line(tmp_path: Path) -> None:
    """一轮一行，字段够追溯：会话身份 + 模式 + 轮次 + 账单 + 双份文本。"""
    out = tmp_path / "turns.jsonl"
    session = _session(TurnRecorder(str(out)))

    # 必须走 run_turn：落盘发生在它消费 TurnEnd 的那一刻，直接调调度器会绕过它。
    for _ in range(2):
        await session.on_audio(b"\x00\x00" * 160)
        await session.run_turn()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"期望 2 行账单，实际 {len(lines)}"

    first = json.loads(lines[0])
    for key in ("ts", "session_id", "trace_id", "mode", "turn", "caliber_b_ms"):
        assert key in first, f"账单缺字段 {key}"
    assert first["turn"] == 1
    assert first["mode"] == "serial"
    assert first["session_id"] == session.session_id
    assert json.loads(lines[1])["turn"] == 2, "轮次编号必须是递增的，否则无法排序"


async def test_recorder_is_wired_through_run_turn(tmp_path: Path) -> None:
    """走 run_turn 真路径也要落盘——直接调调度器只是单测便利，不是验收口径。"""
    out = tmp_path / "turns.jsonl"
    session = _session(TurnRecorder(str(out)))
    await session.on_audio(b"\x00\x00" * 160)
    await session.run_turn()

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["turn"] == 1
    assert records[0]["interrupted"] is False


def test_write_failure_does_not_raise(tmp_path: Path) -> None:
    """写盘失败不能拖垮会话：把路径指到不可写的位置，构造与 record 都不得抛。"""
    bad = tmp_path / "as_file"
    bad.write_text("occupied", encoding="utf-8")  # 目录位置被普通文件占住
    rec = TurnRecorder(str(bad / "child" / "turns.jsonl"))
    rec.record(
        session_id="s",
        trace_id="t",
        mode="real-serial",
        turn=1,
        bill={"caliber_b_ms": 1.0},
    )  # 不应抛；账单丢了也不能断通话


def test_no_record_before_any_turn(tmp_path: Path) -> None:
    """未跑任何轮次前不该建文件——掉连接不该留下空账单。"""
    out = tmp_path / "turns.jsonl"
    _session(TurnRecorder(str(out)))
    assert SessionEnd().type == "session.end"
    assert out.exists() is False


def test_every_gating_mode_has_a_bill_code() -> None:
    """防回退：每个 GatingMode 都必须能在账单里编出非 -1 的码。

    真链路落盘实测抓到过这个 bug——`neural`（默认档！）不在编码映射里，
    于是默认配置下每轮账单都写 `eou_mode: -1.0`（“未用判停器”），而同一条记录里
    `eou_latency_ms` 明明是 100.0。两个字段自相矛盾，且 -1 看起来像个合法值，
    不会报错——与“缺失值别写 0”同一类错，只能靠断言钉住。
    """
    for mode in GatingMode:
        metrics = TurnMetrics()
        metrics.eou_mode = mode.value
        assert metrics.bill()["eou_mode"] >= 0.0, (
            f"门控档 {mode.value} 没有账单编码，会被写成 -1.0（缺失），必须同步补映射"
        )


def test_bill_code_marks_missing_only_when_unset() -> None:
    """-1 的语义是“未用判停器”。没设档位时才允许是 -1。"""
    assert TurnMetrics().bill()["eou_mode"] == -1.0
