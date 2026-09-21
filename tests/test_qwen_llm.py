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
    QwenLlm,
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
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert isinstance(build_llm("auto"), SimLlm)


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
