"""provider 组装工厂：mode → (asr, llm, tts)。

加新引擎只改这里，调用方（Session / WS）不动。real 三件套延迟加载——模型
加起来近 1s，不该在 import 时付费。
"""

from __future__ import annotations

from dataclasses import dataclass

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts

DEFAULT_ASR_ENGINE = "transducer"  # 本机实测：出词粒度 300ms，优于 paraformer 的 600ms


@dataclass(frozen=True, slots=True)
class ProviderSet:
    asr: AsrProvider
    llm: LlmProvider
    tts: TtsProvider


def build_providers(mode: str = "sim", asr_engine: str = DEFAULT_ASR_ENGINE) -> ProviderSet:
    if mode == "sim":
        return ProviderSet(asr=SimAsr(), llm=SimLlm(), tts=SimTts())

    if mode == "real":
        from voice_pipeline.providers import sherpa as real

        engine = {
            "paraformer": real.ParaformerStreamingAsr,
            "transducer": real.TransducerStreamingAsr,
        }[asr_engine]
        # TTS 4 线程是本机实测最优点（2 线程 665ms → 4 线程 624ms，8 线程反而退步）
        return ProviderSet(asr=engine(), llm=SimLlm(), tts=real.KokoroTts(num_threads=4))

    msg = f"unknown provider mode: {mode}"
    raise ValueError(msg)
