"""预合成 filler（体感优化）：注入点、计量纪律、缺项降级。

三条要钉住的（顺序就是重要性）：

1. **口径B 一分不省**：filler 播了，`caliber_b_ms` 仍只算「真实回答的首片」——不因为它
   前面多了一段预合成音频就变小。那是冒领，本仓最忌讳的一类错（见 `core/filler.py`）。
2. **体感首音变早**：`filler_used = 1`、`first_sound` 打点存在，且排在第一片**真实**
   音频之前。
3. **缺项降级**：库没配 / 目录坏 / 空转写 → 行为与不开一模一样（不抛、不播、字段如实）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice_pipeline.core.filler import FillerBank
from voice_pipeline.core.scheduler.base import AudioFragment
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import TurnEnd
from voice_pipeline.server.session import Session

TRANSCRIPT = "上海今天多少度"
REPLY = "今天晴，气温 22 度。适合出门。"
# 恰好一个 200ms 分片（24k PCM16）与两个分片
CLIP_A = b"\x11" * 9600
CLIP_B = b"\x22" * 19200


def make_bank(tmp_path: Path, clips: dict[str, bytes]) -> FillerBank:
    """写一个最小可用的 filler 目录并加载。"""
    (tmp_path / "a.pcm").write_bytes(clips["a"])
    if "b" in clips:
        (tmp_path / "b.pcm").write_bytes(clips["b"])
    entries = [{"text": "嗯", "file": "a.pcm"}]
    if "b" in clips:
        entries.append({"text": "好的", "file": "b.pcm"})
    (tmp_path / "index.json").write_text(
        json.dumps({"sample_rate": 24000, "clips": entries}, ensure_ascii=False), encoding="utf-8"
    )
    bank = FillerBank.load(tmp_path)
    assert bank is not None
    return bank


def build_session(
    filler: FillerBank | None = None, transcript: str = TRANSCRIPT, cpu_ms: float = 0.0
) -> Session:
    """cpu_ms > 0 时 SimTts 会在合成里真 sleep——用来验证「filler 不等合成」这条时序
    （cpu_ms=0 时一切都在同一个瞬间，量不出先后）。"""
    return Session(
        mode="serial",
        asr=SimAsr(transcript=transcript),
        llm=SimLlm(reply=REPLY),
        tts=SimTts(chunk_per_sentence=2, cpu_ms=cpu_ms),
        filler=filler,
    )


async def run_turn(session: Session) -> tuple[list[bytes], dict[str, float]]:
    await session.on_audio(b"\x00" * 3200)
    frags: list[bytes] = []
    bill: dict[str, float] = {}
    async for item in session.finish():
        if isinstance(item, AudioFragment):
            frags.append(item.data)
            continue
        if isinstance(item, TurnEnd):
            bill = item.metrics
    return frags, bill


# ---- 库本身 ----------------------------------------------------------------


async def test_bank_loads_and_rotates(tmp_path: Path) -> None:
    bank = make_bank(tmp_path, {"a": CLIP_A, "b": CLIP_B})
    assert len(bank) == 2
    assert bank.next_clip() is not None and bank.next_clip() is not None
    # 轮转：两次之后回到第一条（不是每次都同一句）
    assert bank.next_clip().pcm == CLIP_A  # type: ignore[union-attr]


async def test_bank_degrades_to_none(tmp_path: Path) -> None:
    assert FillerBank.load(tmp_path / "nope") is None  # 目录不存在
    (tmp_path / "index.json").write_text("{ 这不是 json", encoding="utf-8")
    assert FillerBank.load(tmp_path) is None  # index 坏了
    (tmp_path / "index.json").write_text('{"clips": []}', encoding="utf-8")
    assert FillerBank.load(tmp_path) is None  # 没有 clip


async def test_from_env_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_bank(tmp_path, {"a": CLIP_A})  # 先把目录备好，再验证开关语义
    monkeypatch.delenv("VOICE_FILLER", raising=False)
    assert FillerBank.from_env() is None  # 默认关
    monkeypatch.setenv("VOICE_FILLER", "1")
    monkeypatch.setenv("VOICE_FILLER_DIR", str(tmp_path))
    env_bank = FillerBank.from_env()
    assert env_bank is not None and env_bank.directory == tmp_path
    monkeypatch.setenv("VOICE_FILLER", "0")
    assert FillerBank.from_env() is None  # 显式关


# ---- 注入点与计量纪律 ------------------------------------------------------


async def test_filler_off_keeps_bill_clean() -> None:
    frags, bill = await run_turn(build_session())
    assert frags, "没有下行音频，用例本身失效"
    assert frags[0] != CLIP_A
    assert "first_sound" not in bill
    assert bill["first_sound_ms"] == -1.0  # 缺失如实留空，不填 0
    assert bill["filler_used"] == 0.0


async def test_filler_prepends_and_never_shifts_caliber_b(tmp_path: Path) -> None:
    """filler 排在真实回答之前，且**不改变口径B 的含义**——它仍是真实首片的时刻。

    用 cpu_ms=60 的 SimTts 把「合成要花时间」写进时序：filler 等到那一刻已经播完，
    而口径B 仍要等合成——两个数分开，正是这套计量的目的。
    """
    bank = make_bank(tmp_path, {"a": CLIP_A})
    frags, bill = await run_turn(build_session(bank, cpu_ms=60))

    # ① 第一片就是 filler（逐字节相同），说明注入点在真音频之前
    assert frags[0] == CLIP_A
    # ② 真实回答的音频紧随其后（SimTts 每句 2 片）——filler 没有把回答挤掉
    assert len(frags) > 1
    assert bill["filler_used"] == 1.0
    assert "first_sound" in bill
    # ③ 计量纪律：体感首音不等合成，口径B（真实首片）仍在等
    assert bill["first_sound_ms"] < 20.0
    assert bill["caliber_b_ms"] >= 40.0
    assert bill["first_sound_ms"] <= bill["caliber_b_ms"]
    # ④ 关掉 filler 跑同一轮：口径B 仍是同一量级——它没被 filler「改善」
    _, bill_off = await run_turn(build_session(cpu_ms=60))
    assert bill_off["caliber_b_ms"] >= 40.0
    assert bill_off["first_sound_ms"] == -1.0


async def test_empty_transcript_gets_no_filler(tmp_path: Path) -> None:
    """空转写不起轮：不发音、也不该冒出 filler（否则 AI 会对空气说「嗯，我看一下」）。"""
    bank = make_bank(tmp_path, {"a": CLIP_A})
    frags, bill = await run_turn(build_session(bank, transcript=""))
    assert frags == []
    assert bill["filler_used"] == 0.0
    assert "first_sound" not in bill


async def test_filler_fragments_use_negative_seq(tmp_path: Path) -> None:
    """filler 分片必须全为负序号。

    `Session.run_turn` 用 `seq >= 0` 识别「真实回答的首片」来记 `tx.first_audio` 那根
    span；filler 若占到 0 号，诊断那一格就挂到了 filler 上（2026-10-17 修正）。
    """
    bank = make_bank(tmp_path, {"a": CLIP_A})
    session = build_session(bank)
    await session.on_audio(b"\x00" * 3200)
    seqs: list[int] = []
    async for item in session.finish():
        if isinstance(item, AudioFragment):
            seqs.append(item.seq)
    assert seqs, "没有下行音频，用例本身失效"
    assert seqs[0] < 0, f"首个 filler 分片的 seq 必须为负，实际 {seqs[0]}"
    assert all(s > 0 for s in seqs[1:]), f"真实回答分片必须为正序号：{seqs}"
