"""真 LLM provider：解析纪律与装配契约。

**零网络零模型**——这里测的是「思考链必须被丢掉」和「缺 key 必须响亮失败」两条纪律，
不是 qwen 的在线行为。联网实测走 `scripts/probe_llm.py`。
"""

from __future__ import annotations

import json

import pytest

from voice_pipeline.providers.factory import build_llm
from voice_pipeline.providers.llm.qwen_llm import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM,
    QwenLlm,
    build_messages,
    iter_content,
)
from voice_pipeline.providers.llm.sim_llm import SimLlm


def _sse(content: str | None, reasoning: str | None = None) -> str:
    delta: dict[str, object] = {}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if content is not None:
        delta["content"] = content
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def test_iter_content_drops_reasoning_chain() -> None:
    """推理模型的思考链绝不能进 TTS——念出来是事故，不是延迟问题。"""
    lines = [
        _sse(content=None, reasoning="我们需要"),
        _sse(content=None, reasoning="回答用户"),
        _sse(content="今天晴。"),
    ]
    assert list(iter_content(lines)) == ["今天晴。"]


def test_iter_content_stops_at_done() -> None:
    lines = [_sse("前一句。"), "data: [DONE]", _sse("DONE 之后不该再被读到。")]
    assert list(iter_content(lines)) == ["前一句。"]


def test_iter_content_ignores_junk_lines() -> None:
    lines = ["", ": heartbeat", "data: {not json}", _sse("有效。")]
    assert list(iter_content(lines)) == ["有效。"]


def test_default_model_is_the_measured_fastest() -> None:
    """默认档位必须是实测最快的那档，不是名字最好听的那档。

    qwen3.8-flash 也是推理型（首可播 2771ms），别被 flash 后缀骗了。
    """
    assert DEFAULT_MODEL == "qwen-flash"


def test_missing_api_key_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺 key 必须抛错，不能静默退回替身——否则账单里 LLM 那一格会假装是 0。"""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        QwenLlm()


def test_auto_engine_falls_back_to_sim_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺 key 且**没有本地 mlx**才落到替身。装了 `--extra local` 的机器上 auto
    会降级到 MlxLlm（文档语义，账单 llm 字段可判读）——所以这里要把 mlx
    探测一并钉死，否则测试随本机依赖摆动（2026-10-27 装了 extra 后实锤）。"""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr("voice_pipeline.providers.factory._mlx_available", lambda: False)
    assert isinstance(build_llm("auto"), SimLlm)


def test_auto_engine_prefers_local_mlx_over_sim_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺 key 但有 mlx：降级到本地（比替身诚实——替身会让账单假装含 LLM）。"""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr("voice_pipeline.providers.factory._mlx_available", lambda: True)
    llm = build_llm("auto")
    assert type(llm).__name__ == "MlxLlm"


def test_auto_engine_uses_real_llm_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """有 key 就不能再偷偷用替身——这是「端到端数字含 LLM」的唯一保证。"""
    monkeypatch.setenv(API_KEY_ENV, "sk-test-not-used")
    llm = build_llm("auto")
    assert isinstance(llm, QwenLlm)
    assert llm.model == DEFAULT_MODEL


def test_explicit_sim_engine_never_touches_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "sk-test-not-used")
    assert isinstance(build_llm("sim"), SimLlm)


def test_unknown_llm_engine_rejects_loudly() -> None:
    with pytest.raises(ValueError, match="unknown llm engine"):
        build_llm("nope")


def test_build_messages_inserts_history_between_system_and_user() -> None:
    history = [
        {"role": "user", "content": "第一句问话"},
        {"role": "assistant", "content": "第一轮的回答。"},
    ]
    messages = build_messages(
        "系统提示", history, "第二句问话", today_str="今天的日期是2026年10月29日。"
    )
    assert messages == [
        {"role": "system", "content": "系统提示\n今天的日期是2026年10月29日。"},
        {"role": "user", "content": "第一句问话"},
        {"role": "assistant", "content": "第一轮的回答。"},
        {"role": "user", "content": "第二句问话"},
    ]


def test_build_messages_without_history_is_first_turn() -> None:
    fixed = {"today_str": "今天的日期是2026年10月29日。"}
    assert build_messages("系统提示", None, "问话", **fixed) == [
        {"role": "system", "content": "系统提示\n今天的日期是2026年10月29日。"},
        {"role": "user", "content": "问话"},
    ]
    assert build_messages("系统提示", [], "问话", **fixed) == [
        {"role": "system", "content": "系统提示\n今天的日期是2026年10月29日。"},
        {"role": "user", "content": "问话"},
    ]


def test_build_messages_anchors_today_with_weekday() -> None:
    """日期锚定（2026-10-10 真人验收实锤：问「今天几号」答 2024 年且被纠正仍坚持）。
    锚定行追加在 system 末尾、不改原文；每次请求动态生成，跨午夜的长会话不会拿旧日期。"""
    import datetime as dt

    messages = build_messages("系统提示", None, "今天几号", today=dt.datetime(2026, 10, 29))
    assert messages[0]["content"] == "系统提示\n今天的日期是2026年10月29日，星期四。"
    # 不传 today 时用当前日期（只验结构，不验具体值——那是墙钟）
    live = build_messages("系统提示", None, "问话")
    assert "\n今天的日期是" in live[0]["content"]


def test_default_system_forbids_claiming_actions() -> None:
    """宣称类约束（真人验收：「我想订会议室」→答「已为您预订」，本仓无工具层）。"""
    assert "不要声称已经完成" in DEFAULT_SYSTEM


def test_build_messages_passes_history_verbatim() -> None:
    """history 属于调用方（Session），装配不许增删改——原样透传。"""
    history = [{"role": "user", "content": "问话"}]
    build_messages("系统提示", history, "下一句")
    assert history == [{"role": "user", "content": "问话"}]


def test_system_prompt_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """首句字数是剩余唯一性能杠杆（~79ms/字），必须免改代码可调。"""
    monkeypatch.setenv(API_KEY_ENV, "sk-test-not-used")
    monkeypatch.setenv("VOICE_LLM_SYSTEM", "十个字以内回答。")
    llm = QwenLlm()
    assert llm._system == "十个字以内回答。"
    # 显式参数优先于 env
    assert QwenLlm(system="自定义。")._system == "自定义。"
    monkeypatch.delenv("VOICE_LLM_SYSTEM")
    assert QwenLlm()._system == DEFAULT_SYSTEM
