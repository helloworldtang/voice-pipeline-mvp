"""provider 组装工厂：mode → (asr, llm, tts)。

加新引擎只改这里，调用方（Session / WS）不动。real 三件套延迟加载——模型
加起来近 1s，不该在 import 时付费。

real 模式的 LLM 用 `llm_engine` 选：
  - "auto"（默认）：有 DASHSCOPE_API_KEY 就上真 qwen，没有就退回 SimLlm **并告警**
  - "qwen"：强制真 LLM，缺 key 直接抛错
  - "sim"：强制替身
**为什么退回要告警**：端到端账单里 LLM 那一格是 0ms，静默退回会让"本机侧下界"
被当成真实数字。这条日志是数字可信度的护栏。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from voice_pipeline.core.eou import FusedEou, GatingMode
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.eou.smart_turn import SmartTurnEou
from voice_pipeline.providers.llm.qwen_llm import API_KEY_ENV, DEFAULT_MODEL, QwenLlm
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts

logger = logging.getLogger(__name__)

DEFAULT_ASR_ENGINE = "transducer"  # 本机实测：出词粒度 300ms，优于 paraformer 的 600ms
DEFAULT_LLM_MODEL = DEFAULT_MODEL  # qwen-flash：本机实测首完整句 522ms，快档里最快
# 神经档的「确认静音窗」：模型说完就判，不等静音。扫参结论见 docs/实验报告.md §5
DEFAULT_NEURAL_SILENCE_MS = 100.0  # 短确认窗：扫参见 docs/实验报告.md §5
DEFAULT_NEURAL_FALLBACK_MS = 800.0  # 模型不吐 complete 时的兜底，防无限等


@dataclass(frozen=True, slots=True)
class ProviderSet:
    asr: AsrProvider
    llm: LlmProvider
    tts: TtsProvider


def build_llm(engine: str = "auto", model: str = DEFAULT_LLM_MODEL) -> LlmProvider:
    """按 engine 组装 LLM。真 LLM 需要 DASHSCOPE_API_KEY。"""
    if engine == "sim":
        return SimLlm()
    if engine == "qwen":
        return QwenLlm(model=model)
    if engine == "auto":
        if os.environ.get(API_KEY_ENV):
            return QwenLlm(model=model)
        logger.warning(
            "real 模式没有 %s，LLM 退回 SimLlm：端到端数字不含 LLM 首 token，"
            "只是本机侧下界，不要对外当真实延迟报",
            API_KEY_ENV,
        )
        return SimLlm()
    msg = f"unknown llm engine: {engine}"
    raise ValueError(msg)


def build_providers(
    mode: str = "sim",
    asr_engine: str = DEFAULT_ASR_ENGINE,
    llm_engine: str = "auto",
    llm_model: str = DEFAULT_LLM_MODEL,
) -> ProviderSet:
    if mode == "sim":
        return ProviderSet(asr=SimAsr(), llm=SimLlm(), tts=SimTts())

    if mode == "real":
        from voice_pipeline.providers import sherpa as real

        engine = {
            "paraformer": real.ParaformerStreamingAsr,
            "transducer": real.TransducerStreamingAsr,
        }[asr_engine]
        # TTS 4 线程是本机实测最优点（2 线程 665ms → 4 线程 624ms，8 线程反而退步）
        return ProviderSet(
            asr=engine(),
            llm=build_llm(llm_engine, llm_model),
            tts=real.KokoroTts(num_threads=4),
        )

    msg = f"unknown provider mode: {mode}"
    raise ValueError(msg)


def build_eou(
    gating: GatingMode = GatingMode.NEURAL,
    sample_rate: int = 16000,
    *,
    neural: bool | None = None,
    min_silence_neural_ms: float = DEFAULT_NEURAL_SILENCE_MS,
    neural_fallback_ms: float = DEFAULT_NEURAL_FALLBACK_MS,
) -> FusedEou:
    """组装 EOU 判停器（M2b）。

    默认用模型无关的启发式 L2(韵律)/L3(语义)；`gating=NEURAL` 或 `neural=True` 时
    把 L2 换成 Smart Turn v3 神经网络版（D2）。**缺权重必须告警并退回 conservative**——
    静默退回会把启发式的数字当成神经版的数字报出去。
    """
    detector: SmartTurnEou | None = None
    want_neural = neural if neural is not None else (gating is GatingMode.NEURAL)
    if want_neural:
        try:
            detector = SmartTurnEou(sample_rate=sample_rate)
        except FileNotFoundError as exc:
            logger.warning(
                "神经判停器不可用（%s），退回到 conservative 档：账单里的 eou_mode 会写 "
                "conservative，不要当成神经版的数字报",
                exc,
            )
            return FusedEou(gating=GatingMode.CONSERVATIVE, sample_rate=sample_rate)
    return FusedEou(
        gating=gating,
        sample_rate=sample_rate,
        neural=detector,
        min_silence_neural_ms=min_silence_neural_ms,
        neural_fallback_ms=neural_fallback_ms,
    )
