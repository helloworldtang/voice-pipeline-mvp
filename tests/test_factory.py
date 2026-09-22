"""provider 工厂：mode → 三件套的组装契约。

这里只测「不需要模型」的部分；real 模式没有模型就跳过，CI 零模型零网络照样绿。
"""

from __future__ import annotations

import pytest

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.factory import (
    DEFAULT_ASR_ENGINE,
    DEFAULT_TTS_ENGINE,
    build_providers,
    resolve_tts_engine,
)
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts


def test_sim_mode_returns_sim_trio() -> None:
    providers = build_providers("sim")
    assert isinstance(providers.asr, SimAsr)
    assert isinstance(providers.llm, SimLlm)
    assert isinstance(providers.tts, SimTts)


def test_unknown_mode_rejects_loudly() -> None:
    with pytest.raises(ValueError, match="unknown provider mode"):
        build_providers("nope")


def test_default_asr_engine_is_measured_not_assumed() -> None:
    """默认走的必须是本机实测更快的那个，不是文档里先写的那个。"""
    assert DEFAULT_ASR_ENGINE == "transducer"


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/asr-zipformer/joiner.int8.onnx").is_file(),
    reason="real models not downloaded",
)
def test_real_mode_builds_sherpa_trio() -> None:
    providers = build_providers("real")
    assert type(providers.asr).__name__ == "TransducerStreamingAsr"
    assert type(providers.tts).__name__ == "MeloTts"


def test_default_tts_engine_is_measured_not_assumed() -> None:
    """默认 TTS 必须是实测更快的 melo，不是原先写死在工厂里的 kokoro。

    依据 `scripts/probe_tts.py`（n=5 取 p50，同一批文本）：
      单字（固定开销）  kokoro 812ms  /  melo 255ms
      7 字              kokoro 1285ms /  melo 862ms
      12 字             kokoro 1689ms /  melo 1297ms
    TTS 是端到端口径B 最大的一格（旧默认下实测 ~1440ms，比判停 100ms 与
    LLM 首句 379ms 加起来还大），换引擎是这一格唯一不伤功能的降法。
    防回退：有人觉得 kokoro 音色好就改回去，代价是每轮多 400ms，且那份
    「音色更好」没有实测支撑 —— 要改就先跑 probe_tts 拿数字。
    """
    assert DEFAULT_TTS_ENGINE == "melo"


def test_tts_engine_env_overrides_default_for_ab() -> None:
    """env 必须能覆盖默认：端到端 A/B 的服务端跑在独立进程里，只能靠 env 拧这个旋钮。"""
    assert resolve_tts_engine() == DEFAULT_TTS_ENGINE
    assert resolve_tts_engine("kokoro") == "kokoro"
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("VOICE_PIPELINE_TTS_ENGINE", "kokoro")
        assert resolve_tts_engine() == "kokoro"
        # 显式参数优先级高于 env
        assert resolve_tts_engine("melo") == "melo"
    finally:
        monkeypatch.undo()
