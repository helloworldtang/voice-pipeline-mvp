"""provider 工厂：mode → 三件套的组装契约。

这里只测「不需要模型」的部分；real 模式没有模型就跳过，CI 零模型零网络照样绿。
"""

from __future__ import annotations

import pytest

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.factory import DEFAULT_ASR_ENGINE, build_providers
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
    assert type(providers.tts).__name__ == "KokoroTts"
