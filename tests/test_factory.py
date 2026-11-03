"""provider 工厂：mode → 三件套的组装契约。

这里只测「不需要模型」的部分；real 模式没有模型就跳过，CI 零模型零网络照样绿。
"""

from __future__ import annotations

import pytest

from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.factory import (
    DEFAULT_ASR_ENGINE,
    DEFAULT_TTS_ENGINE,
    PROFILES,
    build_providers,
    profile_error,
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


def test_dashscope_engine_routes_to_cloud_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    """tts=dashscope → 云端流式 provider（session.begin 不再加载本地 TTS 模型）。"""
    from voice_pipeline.providers.tts.dashscope_tts import DashScopeTts

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-not-used")
    providers = build_providers("real", tts_engine="dashscope")
    assert isinstance(providers.tts, DashScopeTts)
    assert providers.tts.tts_streams is True


def test_unknown_tts_engine_lists_dashscope() -> None:
    with pytest.raises(ValueError, match="dashscope"):
        build_providers("real", tts_engine="nope")


@pytest.mark.skipif(
    not __import__("pathlib").Path("models/asr-zipformer/joiner.int8.onnx").is_file(),
    reason="real models not downloaded",
)
def test_real_providers_are_cached_across_builds() -> None:
    """provider 缓存：TTS 引擎与 ASR recognizer 跨 build 共享，ASR wrapper 各自独立。

    不缓存的话每次 session.begin 重付 1-3s 模型加载，浏览器每次刷新都卡
    （ECS 实测 begin 迟到 2.4-2.9s，docs/ecs-baseline.md）。
    """
    import voice_pipeline.providers.factory as F

    a = F.build_providers("real")
    b = F.build_providers("real")
    assert a.tts is b.tts
    assert a.asr is not b.asr  # 会话状态不可共享
    assert a.asr._recognizer is b.asr._recognizer  # type: ignore[attr-defined]


def test_profiles_are_fixed_pairs() -> None:
    """档位组合定死：cloud=(qwen, dashscope)、latency=(mlx, xfyun)。

    这是「政策」不是「旋钮」——客户端挑的是全云/响应时间优先，不是具体引擎。
    有人"顺手"调档（比如 latency 的 TTS 换成 melo），数字口径就变了，而账单里
    只落 profile 名——要改先拿实测数字说话（风格同
    test_default_tts_engine_is_measured_not_assumed）。
    """
    assert set(PROFILES) == {"cloud", "latency"}
    assert (PROFILES["cloud"].llm_engine, PROFILES["cloud"].tts_engine) == ("qwen", "dashscope")
    assert (PROFILES["latency"].llm_engine, PROFILES["latency"].tts_engine) == ("mlx", "xfyun")


def test_profile_error_unknown_name() -> None:
    """未知档名必须明确拒绝（bad_profile），不能静默当默认档处理。"""
    reason = profile_error("nope")
    assert reason is not None
    assert "unknown profile" in reason


def test_profile_error_cloud_needs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """cloud 档一个检查覆盖 LLM+TTS：QwenLlm 与 DashScopeTts 共用 DASHSCOPE_API_KEY。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    reason = profile_error("cloud")
    assert reason is not None
    assert "DASHSCOPE_API_KEY" in reason

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-not-used")
    assert profile_error("cloud") is None


def test_profile_error_latency_needs_mlx_and_xfyun(monkeypatch: pytest.MonkeyPatch) -> None:
    """latency 档三项预检：mlx 依赖（MlxLlm 构造不抛，必须显式检）+ 讯飞三凭证。"""
    import voice_pipeline.providers.factory as factory

    monkeypatch.setattr(factory, "_mlx_available", lambda: False)
    monkeypatch.setenv("XFYUN_TTS_APPID", "app")
    monkeypatch.setenv("XFYUN_TTS_API_KEY", "key")
    monkeypatch.setenv("XFYUN_TTS_API_SECRET", "secret")
    reason = profile_error("latency")
    assert reason is not None
    assert "mlx" in reason

    monkeypatch.setattr(factory, "_mlx_available", lambda: True)
    monkeypatch.delenv("XFYUN_TTS_APPID", raising=False)
    reason = profile_error("latency")
    assert reason is not None
    assert "XFYUN_TTS_APPID" in reason

    monkeypatch.setenv("XFYUN_TTS_APPID", "app")
    assert profile_error("latency") is None
