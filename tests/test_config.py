"""配置：文章第三板「参数抠字节」的落点，参数必须可配置且默认值与文档一致。"""

from __future__ import annotations

from voice_pipeline.config import Settings


def test_audio_defaults_match_design() -> None:
    s = Settings()
    assert s.in_sample_rate == 16000
    assert s.out_sample_rate == 24000
    assert s.up_chunk_ms == 100
    assert s.down_chunk_ms == 200


def test_settings_accepts_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VOICE_DOWN_CHUNK_MS", "120")
    assert Settings().down_chunk_ms == 120
