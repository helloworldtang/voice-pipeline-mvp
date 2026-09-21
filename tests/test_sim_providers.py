"""sim 三替身的确定性：CI 里零模型零网络，输出必须可预测。"""

from __future__ import annotations

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts


async def test_sim_asr_returns_scripted_transcript() -> None:
    asr = SimAsr(transcript="上海今天多少度")
    await asr.push_audio(b"\x00" * 320)
    assert await asr.finalize() == "上海今天多少度"
    assert asr.frames == 1


async def test_sim_llm_splits_on_sentence_end() -> None:
    llm = SimLlm(reply="今天晴，气温 22 度。适合出门。")
    pieces = [p async for p in llm.astream("天气")]
    assert pieces == ["今天晴，气温 22 度。", "适合出门。"]


async def test_sim_llm_yields_trailing_fragment() -> None:
    llm = SimLlm(reply="好的")
    assert [p async for p in llm.astream("x")] == ["好的"]


async def test_sim_tts_chunk_size_matches_pcm16_math() -> None:
    tts = SimTts(sample_rate=24000, chunk_ms=200, chunk_per_sentence=2)
    chunks = [c async for c in tts.synth("今天晴。")]

    assert len(chunks) == 2
    # 24000Hz × 200ms × 2 bytes/sample = 9600 bytes
    assert all(len(c) == 9600 for c in chunks)
    assert all(c == b"\x00" * 9600 for c in chunks)
