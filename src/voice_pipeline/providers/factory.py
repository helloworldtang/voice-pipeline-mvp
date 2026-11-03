"""provider 组装工厂：mode → (asr, llm, tts)。

加新引擎只改这里，调用方（Session / WS）不动。real 三件套延迟加载——模型
加起来近 1s，不该在 import 时付费。

real 模式的 LLM 用 `llm_engine` 选（可用 `VOICE_LLM_ENGINE` 覆盖）：
  - "auto"（默认）：有 DASHSCOPE_API_KEY 就上真 qwen；缺 key 时若有 mlx-lm 就降级到本地，
    否则退回 SimLlm **并告警**
  - "qwen"：强制云 LLM，缺 key 直接抛错
  - "mlx"：强制本地（MLX，仅 macOS/arm64）；缺依赖直接抛错
  - "sim"：强制替身
**为什么退回要告警**：端到端账单里 LLM 那一格是 0ms，静默退回会让"本机侧下界"
被当成真实数字。这条日志是数字可信度的护栏。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
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
# TTS 引擎：2026-09-22 实测（`scripts/probe_tts.py`，n=5 取 p50）——
#   melo    单字（固定开销）255ms，边际 +37ms/字；12 字句 1297ms
#   kokoro  单字（固定开销）812ms，边际 +47ms/字；12 字句 1689ms
# 同一句换引擎省 390~630ms，而 TTS 是端到端口径B 里最大的一格（旧默认 kokoro 实测 1302ms，
# 比判停 100ms + LLM 首句 379ms 加起来还大）。melo 的代价写在它的 docstring 里：
# 采样率非原生（需重采样）、单音色、音量偏低（RMS 峰值 0.031 vs kokoro 0.080）。
DEFAULT_TTS_ENGINE = "melo"
# 神经档的「确认静音窗」：模型说完就判，不等静音。扫参结论见 docs/实验报告.md §5
DEFAULT_NEURAL_SILENCE_MS = 100.0  # 短确认窗：扫参见 docs/实验报告.md §5
DEFAULT_NEURAL_FALLBACK_MS = 800.0  # 模型不吐 complete 时的兜底，防无限等


@dataclass(frozen=True, slots=True)
class ProviderSet:
    asr: AsrProvider
    llm: LlmProvider
    tts: TtsProvider


def resolve_llm_engine(explicit: str | None = None) -> str:
    """定 LLM 引擎：显式（非 auto）> `VOICE_LLM_ENGINE` > auto。

    与 `resolve_tts_engine` 同一套理由：**单个**引擎是部署配置，不上 wire 协议；
    A/B 与服务端探针要把引擎当旋钮拧，只能走 env。例外是 `PROFILES` 的两值
    策略档（`/ws?profile=`）——进协议的是档不是引擎，见其 docstring。
    """
    if explicit and explicit != "auto":
        return explicit
    return os.environ.get("VOICE_LLM_ENGINE", explicit or "auto").strip() or "auto"


def _mlx_available() -> bool:
    """本地 MLX 是否可用（仅判断依赖，不加载模型——加载是秒级的）。"""
    import importlib.util

    try:
        return importlib.util.find_spec("mlx_lm") is not None
    except (ImportError, ValueError):  # pragma: no cover - 异常环境才走到
        return False


def _build_mlx() -> LlmProvider:
    """本地 MLX：`VOICE_LLM_PROCESS=1` 走独立子进程（消 GIL 争用），否则进程内。"""
    if os.environ.get("VOICE_LLM_PROCESS", "").strip() == "1":
        from voice_pipeline.providers.llm.mlx_proc_llm import MlxProcLlm

        return MlxProcLlm()
    from voice_pipeline.providers.llm.mlx_llm import MlxLlm

    return MlxLlm()


def build_llm(engine: str = "auto", model: str = DEFAULT_LLM_MODEL) -> LlmProvider:
    """按 engine 组装 LLM。真 LLM 需要 DASHSCOPE_API_KEY；本地需 mlx-lm。"""
    engine = resolve_llm_engine(engine)
    if engine == "sim":
        return SimLlm()
    if engine == "qwen":
        return QwenLlm(model=model)
    if engine == "mlx":
        return _build_mlx()
    if engine == "auto":
        if os.environ.get(API_KEY_ENV):
            return QwenLlm(model=model)
        if _mlx_available():
            logger.warning(
                "没有 %s，LLM 降级到本地 MLX：延迟更低但答案更笼统（账单 llm=MlxLlm）",
                API_KEY_ENV,
            )
            return _build_mlx()
        logger.warning(
            "real 模式没有 %s，LLM 退回 SimLlm：端到端数字不含 LLM 首 token，"
            "只是本机侧下界，不要对外当真实延迟报",
            API_KEY_ENV,
        )
        return SimLlm()
    msg = f"unknown llm engine: {engine}"
    raise ValueError(msg)


def resolve_tts_engine(explicit: str | None = None) -> str:
    """定 TTS 引擎：显式参数 > 环境变量 > 默认。

    环境变量 `VOICE_PIPELINE_TTS_ENGINE` 存在的唯一理由是 A/B：端到端 bench 的
    服务端跑在独立进程里（`bench_ws.py` 起 uvicorn 子进程），要把引擎当旋钮拧
    只能走 env。**单个引擎不进 WS 协议** —— gating/grace_ms 是每次会话都可能
    不同的实验旋钮，引擎不是。例外是 `PROFILES` 的两值策略档（`/ws?profile=`）：
    进协议的是档（cloud/latency）不是具体引擎，bench 路径仍走 env。
    """
    if explicit:
        return explicit
    return os.environ.get("VOICE_PIPELINE_TTS_ENGINE", DEFAULT_TTS_ENGINE)


@dataclass(frozen=True, slots=True)
class EngineProfile:
    """一档「策略档」= 定死的 (LLM, TTS) 引擎对。

    客户端经 `/ws?profile=` 挑政策（全云能力 / 响应时间优先），不挑具体引擎——
    单个引擎仍是部署配置不进 wire 协议，进协议的只有这个两值档。可用性由
    部署侧决定（`profile_error` 预检），客户端选了本机不具备的档会被明确拒绝。
    """

    llm_engine: str
    tts_engine: str


PROFILES: dict[str, EngineProfile] = {
    # 全云：qwen-flash + DashScope 流式 TTS（LLM 与 TTS 共用 DASHSCOPE_API_KEY）
    "cloud": EngineProfile(llm_engine="qwen", tts_engine="dashscope"),
    # 响应时间优先：本机 mlx 1.5B + 讯飞 v2 流式（首包 p50 137ms，快档 ~50ms）。
    # 命中投机时轮次 16~31ms（README 10-28 复核），代价是答案比 qwen 笼统。
    "latency": EngineProfile(llm_engine="mlx", tts_engine="xfyun"),
}


def profile_error(name: str) -> str | None:
    """该档在本机是否可用。None=可用；返回字符串=拒绝原因（含缺失项的 env 名）。

    全部是廉价检查（env 存在性 + importlib），不加载模型、不建网络连接——
    必须发生在占槽/建会话之前。不可用就明确拒绝、绝不静默降级回 env 档：
    那会让 cloud/latency 的数字混进同一份账单（静默降级是最坏失败形态）。
    """
    profile = PROFILES.get(name)
    if profile is None:
        return f"unknown profile: {name} (可选 {sorted(PROFILES)})"
    if profile is PROFILES["cloud"]:
        # QwenLlm 与 DashScopeTts 共用同一个 key，一处检查覆盖 LLM+TTS
        if not os.environ.get(API_KEY_ENV, "").strip():
            return f"cloud 档需要 {API_KEY_ENV}（LLM 与 TTS 共用），本服务未配置"
        return None
    # latency 档：MlxLlm 构造不抛（加载都在 preload），必须显式预检依赖——
    # 不检的话要到首轮 preload 才炸，晚失败正是要消灭的形态
    if not _mlx_available():
        return "latency 档需要本地 mlx-lm 依赖（仅 macOS/arm64），本服务未安装"
    from voice_pipeline.providers.tts.xfyun_tts import APP_ID_ENV, KEY_ENV, SECRET_ENV

    missing = [
        env
        for env in (APP_ID_ENV, KEY_ENV, SECRET_ENV)
        if not os.environ.get(env, "").strip()
    ]
    if missing:
        return f"latency 档的 TTS 需要讯飞凭证 {', '.join(missing)}"
    return None


def build_providers(
    mode: str = "sim",
    asr_engine: str = DEFAULT_ASR_ENGINE,
    llm_engine: str = "auto",
    llm_model: str = DEFAULT_LLM_MODEL,
    tts_engine: str | None = None,
) -> ProviderSet:
    if mode == "sim":
        return ProviderSet(asr=SimAsr(), llm=SimLlm(), tts=SimTts())

    if mode == "real":
        from voice_pipeline.providers import sherpa as real

        engine = {
            "paraformer": real.ParaformerStreamingAsr,
            "transducer": real.TransducerStreamingAsr,
        }[asr_engine]
        tts_name = resolve_tts_engine(tts_engine)
        # TTS 分流：dashscope 走云端流式（2C x86 上本地 melo RTF≈14.5 判死，
        # 见 docs/ecs-baseline.md）；其余走本地 sherpa OfflineTts。
        tts: TtsProvider
        if tts_name == "dashscope":
            tts = _cached_tts_cloud()
        elif tts_name == "xfyun":
            # 讯飞 v2 流式：实测服务端首包 p50 137ms（DashScope 344~421），
            # 但连接不可复用，每条句子各自建连（握手 ~38ms）。见 providers/tts/xfyun_tts.py。
            from voice_pipeline.providers.tts.xfyun_tts import XfyunTts

            tts = XfyunTts()
        else:
            tts_cls = real.TTS_ENGINES.get(tts_name)
            if tts_cls is None:
                msg = (
                    f"unknown tts engine: {tts_name} "
                    f"(可选 {list(real.TTS_ENGINES)}、dashscope、xfyun)"
                )
                raise ValueError(msg)
            # TTS 4 线程是本机实测最优点（2 线程 665ms → 4 线程 624ms，8 线程反而退步）
            tts = _cached_tts_local(tts_cls)
        return ProviderSet(
            asr=_build_asr_shared(engine),
            llm=build_llm(llm_engine, llm_model),
            tts=tts,
        )

    msg = f"unknown provider mode: {mode}"
    raise ValueError(msg)


# ---- provider 缓存 ----------------------------------------------------------
# Provider 每 session.begin 重建（Session.__init__），而模型加载是秒级的：
# 不缓存的话浏览器每次刷新都重付 1-3s，还把「begin 阻塞接收循环」放大成体感卡顿
# （ECS 实测 begin 迟到 2.4-2.9s，docs/ecs-baseline.md）。
# 缓存粒度按「重资产」拆：TTS 引擎实例 / LLM / ASR recognizer 跨连接共享，
# ASR wrapper 与 stream 每连接新建（携带会话状态，不可共享）。
# 并发代价：共享的 ORT session 上推理串行——单用户试点无碍，多用户再议。

_LOCAL_TTS_CACHE: dict[str, TtsProvider] = {}
_CLOUD_TTS: TtsProvider | None = None
_ASR_RECOGNIZER_CACHE: dict[str, object] = {}


def _cached_tts_local(tts_cls: Callable[..., object]) -> TtsProvider:
    key = tts_cls.__name__
    if key not in _LOCAL_TTS_CACHE:
        tts: TtsProvider = tts_cls(num_threads=4)  # type: ignore[assignment]
        _LOCAL_TTS_CACHE[key] = tts
    return _LOCAL_TTS_CACHE[key]


def _cached_tts_cloud() -> TtsProvider:
    global _CLOUD_TTS
    if _CLOUD_TTS is None:
        from voice_pipeline.providers.tts.dashscope_tts import DashScopeTts

        _CLOUD_TTS = DashScopeTts()
    return _CLOUD_TTS


def _build_asr_shared(engine: Callable[..., AsrProvider]) -> AsrProvider:
    """ASR：recognizer（模型权重）按引擎名缓存共享，stream 每次新建。"""
    key = engine.__name__
    recognizer = _ASR_RECOGNIZER_CACHE.get(key)
    if recognizer is None:
        asr = engine()  # 首次：正常构造（内含 recognizer 创建）
        _ASR_RECOGNIZER_CACHE[key] = asr._recognizer  # type: ignore[attr-defined]
        return asr
    return engine(recognizer=recognizer)


def build_eou(
    gating: GatingMode = GatingMode.CONSERVATIVE,
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

    ⚠️ 默认档必须与 Session / FusedEou / SessionBegin 三处**保持一致**（第四次定档
    conservative，2026-10-04 真人验收：neural 在真人半秒句内停顿下抢话 3/3）。
    这里漏改不会报错，只会让「直接调 build_eou 的脚本」静默跑在另一个档上，
    数字却混进同一份账单——`test_default_gating_is_conservative_across_layers` 钉住。
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
